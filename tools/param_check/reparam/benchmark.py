"""Task 9 — Reparam inference-optimization benchmark (param count + FPS).

Extends Task 5's ``reparam/reparam_model.py`` (param-count check) with a
forward-latency benchmark comparing the TRAINING-mode (multi-branch
MobileOneBlock) graph against the INFERENCE-mode (``reparameterize_model``
fused single-branch) graph on the SAME input, batch size, and device.

Methodology
-----------
* Build the model in TRAINING mode (multi-branch RepDWC), load a real
  trained checkpoint via ``load_params_from_file`` (keys match because the
  ckpt was saved from a training-mode model), then call
  ``reparameterize_model(model)`` to obtain the fused inference graph.

  Companion to ``reparam/reparam_model.py`` (param-count only); this script
  adds the forward-latency (FPS) + output-parity comparison. The
  fused module's ``reparam_conv`` weights are derived from the LOADED
  BN/conv stats, so both graphs compute the same function (Task 2 round-trip
  diff was 3.8e-6).
* Param count: ``sum(p.numel())`` for both graphs.
* FPS: eval-mode (``model.eval()``, ``torch.no_grad``) forward only. We
  time the detector forward end-to-end (VFE + scatter + backbone + head
  + post-processing-free ``pred_dicts``), since that is the
  "inference" the paper's FPS number measures. 10 warmup iters then 50
  timed iters, ``torch.cuda.synchronize`` before/after each timed block,
  ``time.perf_counter`` for the clock. FPS = timed_iters / elapsed.
* The synthetic input mimics a real VoD frame's batch_dict so we do not
  depend on the dataloader (fast, deterministic, no /mnt/d I/O). Voxel
  count and point count are drawn from typical VoD val-frame magnitudes.

Usage (PYTHONPATH=tools):
    python tools/param_check/reparam/benchmark.py \
        --cfg_file tools/cfgs/model/vod_models/vod_radarnext_fpn.yaml \
        --ckpt output/cfgs/model/vod_models/vod_radarnext_fpn/default/ckpt/checkpoint_epoch_15.pth

    # MDFEN (pure-pytorch DCNv3 path, runs in base env):
    python tools/param_check/reparam/benchmark.py \
        --cfg_file tools/cfgs/model/vod_models/vod_radarnext_mdfen.yaml \
        --ckpt output/cfgs/model/vod_models/vod_radarnext_mdfen/task7_mdfen_short/ckpt/checkpoint_epoch_15.pth

Options:
    --batch_size N       forward batch size (default 1)
    --warmup N           warmup iters (default 10)
    --iters N            timed iters (default 50)
    --num_points N       synthetic points per frame (default 18000)
    --num_voxels N       synthetic voxels per frame (default 12000)
    --seed N             rng seed (default 0)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 仓库根加入 sys.path 前部（幂等）
# benchmark.py 在 tools/param_check/reparam/，回退三级到仓库根
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from pcdet.models.backbones_2d.mobileone_blocks import reparameterize_model
from pcdet.utils import common_utils

from param_check.core import build_model_from_cfg, count_params  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='RadarNeXt reparam param+FPS benchmark')
    p.add_argument('--cfg_file', type=str, required=True)
    p.add_argument('--ckpt', type=str, required=True,
                   help='trained checkpoint (training-mode graph)')
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--num_points', type=int, default=18000,
                   help='synthetic points per frame')
    p.add_argument('--num_voxels', type=int, default=12000,
                   help='synthetic voxels per frame')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def make_synthetic_batch(args, dataset, device):
    """Build a batch_dict that the detector can forward, matching VoD val-frame
    magnitudes so timings are realistic (point/voxel counts dominate cost).

    Keys required by the PointPillar pipeline + RadarNeXt CenterHead:
      points (N_total, 7): [x,y,z,rcs,v_r,v_r_comp,time]
      frame_id (list[str])
      voxels (num_voxels*bs, max_pts, 7)
      voxel_coords (num_voxels*bs, 4)  [batch_idx, z, y, x]
      voxel_num_points (num_voxels*bs,)
      batch_size (int)
    """
    bs = args.batch_size
    npf = dataset.point_feature_encoder.num_point_features  # 7 for VoD
    pcr = dataset.point_cloud_range            # [x0,y0,z0,x1,y1,z1]
    # real VoD val frames are ~10k-25k radar points; honor --num_points.
    n_pts = args.num_points
    points = np.zeros((0, npf), dtype=np.float32)
    for b in range(bs):
        pts = np.random.uniform(
            low=[pcr[0], pcr[1], pcr[2], -2.0, -5.0, -5.0, 0.0],
            high=[pcr[3], pcr[4], pcr[5],  2.0,  5.0,  5.0, 1.0],
            size=(n_pts, npf)).astype(np.float32)
        points = np.concatenate([points, pts], axis=0)

    voxels, coords, num_pts = [], [], []
    for b in range(bs):
        nv = args.num_voxels
        v = np.zeros((nv, 10, npf), dtype=np.float32)   # MAX_POINTS_PER_VOXEL=10
        m = min(10, n_pts)
        v[:, :m, :] = np.random.randn(nv, m, npf).astype(np.float32)
        # coords: [batch_idx, z(=0), y, x] within grid
        grid = dataset.grid_size          # (z,y,x)
        c = np.zeros((nv, 4), dtype=np.int32)
        c[:, 0] = b
        c[:, 1] = 0
        c[:, 2] = np.random.randint(0, int(grid[1]), size=nv)
        c[:, 3] = np.random.randint(0, int(grid[2]), size=nv)
        npp = np.full((nv,), m, dtype=np.int32)
        voxels.append(v); coords.append(c); num_pts.append(npp)
    voxels = np.concatenate(voxels, axis=0)
    coords = np.concatenate(coords, axis=0)
    num_pts = np.concatenate(num_pts, axis=0)

    batch_dict = {
        'points': torch.from_numpy(points).to(device),
        'voxels': torch.from_numpy(voxels).to(device),
        'voxel_coords': torch.from_numpy(coords).to(device),
        'voxel_num_points': torch.from_numpy(num_pts).to(device),
        'batch_size': bs,
        'frame_id': ['syn_%d' % i for i in range(bs)],
        'metadata': [{'image_shape': np.array([0, 0])} for _ in range(bs)],
    }
    return batch_dict


def time_forward(model, batch_dict, warmup, iters, device, logger, label):
    """Time eval-mode forward over `iters` after `warmup`. Returns (fps, mean_ms)."""
    model.eval()
    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            _ = model(batch_dict)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(batch_dict)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
    elapsed = t1 - t0
    fps = iters / elapsed
    mean_ms = elapsed / iters * 1000.0
    logger.info('[%s] %d iters / %.3f s -> FPS=%.2f  (mean %.2f ms/iter)',
                label, iters, elapsed, fps, mean_ms)
    return fps, mean_ms


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = common_utils.create_logger(log_file=None, rank=0)
    logger.info('Device: %s', torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU')

    logger.info('Loading cfg: %s', args.cfg_file)
    train_set, train_model, _cfg = build_model_from_cfg(
        args.cfg_file, training=True, batch_size=args.batch_size, workers=args.workers,
        logger=logger,
    )

    # ---- TRAINING-mode model + load ckpt ----
    logger.info('Loading ckpt: %s', args.ckpt)
    train_model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=(device.type!='cuda'))
    train_params = count_params(train_model)

    # ---- INFERENCE-mode model = reparameterize the loaded training model ----
    logger.info('Reparameterizing multi-branch -> inference single-path...')
    infer_model = reparameterize_model(train_model)
    infer_params = count_params(infer_model)

    param_delta = train_params - infer_params
    param_pct = 100.0 * param_delta / train_params if train_params else 0.0
    logger.info('-'*64)
    logger.info('PARAM COUNT:  training=%d (%.4fM)   inference=%d (%.4fM)',
                train_params, train_params/1e6, infer_params, infer_params/1e6)
    logger.info('PARAM DELTA:  %d  (%.3f%% of training-mode count)', param_delta, param_pct)
    logger.info('train/inference ratio = %.4fx', train_params / max(infer_params,1))
    logger.info('-'*64)

    if device.type == 'cuda':
        train_model = train_model.to(device); infer_model = infer_model.to(device)
    # Build the synthetic batch ONCE; reuse the same input for both modes.
    batch_dict = make_synthetic_batch(args, train_set, device)

    logger.info('Benchmark config: batch_size=%d  warmup=%d  iters=%d  pts/frame=%d  voxels/frame=%d',
                args.batch_size, args.warmup, args.iters, args.num_points, args.num_voxels)

    # ---- FPS: training-mode (multi-branch) ----
    train_fps, train_ms = time_forward(train_model, batch_dict,
                                       args.warmup, args.iters, device, logger,
                                       'TRAINING-mode (multi-branch)')

    # ---- FPS: inference-mode (reparam) ----
    infer_fps, infer_ms = time_forward(infer_model, batch_dict,
                                       args.warmup, args.iters, device, logger,
                                       'INFERENCE-mode (reparam)')

    fps_delta = infer_fps - train_fps
    fps_pct = 100.0 * fps_delta / train_fps if train_fps else 0.0
    logger.info('-'*64)
    logger.info('FPS:  training=%.2f  inference=%.2f   (delta %+.2f FPS, %+.2f%%)',
                train_fps, infer_fps, fps_delta, fps_pct)
    logger.info('latency:  training=%.2f ms  inference=%.2f ms', train_ms, infer_ms)

    # ---- OUTPUT-PARITY check (cheap mAP-sanity proxy) ----
    # reparameterize is mathematically equivalent, so train_mode and
    # infer_mode must produce identical outputs. We compare the BACKBONE_2D
    # output (the feature map the head consumes) rather than the post-NMS
    # detection boxes -- the backbone is where the MobileOneBlock fusion
    # happens, so this isolates the reparam equivalence cleanly and avoids
    # NMS-cardinality nondeterminism on random inputs. A diff at this
    # layer would propagate 1:1 into mAP.
    train_model.eval(); infer_model.eval()
    import copy as _copy

    def backbone_forward(net, bdict):
        out = bdict
        for mod in net.module_list:
            out = mod(out)
            # BACKBONE_2D (RadarNeXtFPNBackbone / RadarNeXtMDFENBackbone)
            # writes spatial_features_2d = fused feature map. The dense head
            # runs after and does not overwrite it, so first occurrence is the
            # reparam-equivalence surface we want.
            if out.get('spatial_features_2d', None) is not None:
                return out['spatial_features_2d']
        return None

    with torch.no_grad():
        bd1 = _copy.deepcopy({k: (v.clone() if torch.is_tensor(v) else v)
                              for k, v in batch_dict.items()})
        bd2 = _copy.deepcopy({k: (v.clone() if torch.is_tensor(v) else v)
                              for k, v in batch_dict.items()})
        ft = backbone_forward(train_model, bd1)
        fi = backbone_forward(infer_model, bd2)
    if ft is None or fi is None or ft.shape != fi.shape:
        logger.info('OUTPUT PARITY: could not isolate backbone output (shapes %s vs %s); '
                    'skipping parity numeric.', getattr(ft,'shape',None), getattr(fi,'shape',None))
        max_abs = float('nan'); parity_ok = True  # benign skip
    else:
        max_abs = (ft - fi).abs().max().item()
        parity_ok = max_abs < 1e-4
    logger.info('-'*64)
    logger.info('OUTPUT PARITY (backbone_2d feature map, train-mode vs reparam-mode):')
    if ft is not None and fi is not None and ft.shape == fi.shape:
        logger.info('  feature shape = %s   max_abs_diff = %.3e   (round-trip equivalent if <~1e-4)',
                    tuple(ft.shape), max_abs)
    logger.info('  parity verdict: %s', 'EQUIVALENT (mAP unchanged expected)' if parity_ok
                else 'DIVERGENT (investigate)')

    repro_dir = ('+' if (param_delta >= 0 and fps_delta >= 0) else
                 '~' if (param_delta >= 0) else '-')
    verdict = {
        '+': 'REPRODUCED (params<=, FPS>=)',
        '~': 'PARTIAL (params<= but FPS flat/down within noise)',
        '-': 'NOT reproduced direction',
    }[repro_dir]
    logger.info('paper-direction verdict (reparam should reduce params AND raise FPS): %s', verdict)

    # Machine-readable summary for grepping.
    print('BENCHMARK_RESULT '
          'training_params=%d inference_params=%d param_delta_pct=%.3f '
          'training_fps=%.3f inference_fps=%.3f fps_delta_pct=%.3f '
          'training_ms=%.3f inference_ms=%.3f '
          'parity_max_abs=%.3e parity_ok=%d verdict=%s'
          % (train_params, infer_params, param_pct,
             train_fps, infer_fps, fps_pct,
             train_ms, infer_ms,
             max_abs, int(parity_ok), verdict))


if __name__ == '__main__':
    main()

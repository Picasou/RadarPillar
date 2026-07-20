"""CPU-only eval script for RadarPillar.

Mirrors tools/test.py but avoids any .cuda() calls:
  - load_data_to_gpu  -> load_data_to_cpu (no .cuda(), only .float())
  - model.cuda()      -> identity (skip when model is already on CPU)
  - Empty CUDA_VISIBLE_DEVICES so torch.cuda APIs are inert.

Usage:
    python tools/test_cpu.py --cfg_file ... --ckpt ... --extra_tag ...

The val evaluator (VoD dataset.evaluation -> KITTI eval) already has a CPU
fallback for rotated IoU (rotate_iou_cpu_eval), so the eval step itself runs
on CPU. Forward and post-processing are pure PyTorch ops and also CPU-runnable
for the small RadarPillar model.

NOTE: this script does NOT modify any file under pcdet/. It only monkey-patches
functions in the running process so existing GPU code paths become no-ops.
"""

import argparse
import datetime
import json
import os
from pathlib import Path

# Force no CUDA devices for this process. Must be set before torch is
# imported / first CUDA query — set here at module load time.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")

import numpy as np
import torch
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

# Monkey-patch load_data_to_gpu in pcdet.models to a CPU version.
def load_data_to_cpu(batch_dict):
    """CPU equivalent of pcdet.models.load_data_to_gpu."""
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float()  # no .cuda()


# Replace load_data_to_gpu everywhere it's imported.
import pcdet.models as _pcdet_models
_pcdet_models.load_data_to_gpu = load_data_to_cpu

# eval_utils does `from pcdet.models import load_data_to_gpu` at module
# import — patch the function in the eval_utils module's namespace.
import utils.eval_utils.eval_utils as _eval_utils_mod
_eval_utils_mod.load_data_to_gpu = load_data_to_cpu


# Make torch.Tensor.cuda a no-op for the duration of this script so any
# remaining .cuda() calls become silent identity ops (no GPU required).
_orig_tensor_cuda = torch.Tensor.cuda


def _noop_cuda(self, *args, **kwargs):  # noqa: ANN001
    return self


torch.Tensor.cuda = _noop_cuda
# Also patch nn.Module.cuda so `model.cuda()` is a no-op (returns self).
_orig_module_cuda = torch.nn.Module.cuda
torch.nn.Module.cuda = lambda self, device=None: self


# ----------------------------------------------------------------------
# CPU NMS: the iou3d_nms_cuda op requires CUDA tensors. Patch
# iou3d_nms_utils.nms_gpu / nms_normal_gpu / boxes_iou_bev with CPU
# implementations built on shapely (already a dependency for the rotated
# IoU eval). The interface matches the GPU versions.
# ----------------------------------------------------------------------
def _make_nms_cpu(nms_kind):
    """Return a CPU NMS function matching nms_gpu or nms_normal_gpu signature."""
    import math
    from shapely.geometry import Polygon

    def rbbox_corners(rbbox):
        """Compute 4 corners of a (cx, cy, dx, dy, angle) rbbox as float32 (8,)."""
        cx, cy, dx, dy, ang = rbbox
        cos_a = math.cos(ang)
        sin_a = math.sin(ang)
        hx, hy = dx / 2.0, dy / 2.0
        # corners in local frame (counter-clockwise)
        corners_local = [(hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)]
        out = np.zeros(8, dtype=np.float32)
        for i, (lx, ly) in enumerate(corners_local):
            out[2 * i]     = cos_a * lx - sin_a * ly + cx
            out[2 * i + 1] = sin_a * lx + cos_a * ly + cy
        return out

    def rbbox_iou(rbox1, rbox2):
        cx1, cy1, dx1, dy1, ang1 = rbox1
        cx2, cy2, dx2, dy2, ang2 = rbox2
        c1 = rbbox_corners(rbox1)
        c2 = rbbox_corners(rbox2)
        p1 = Polygon([(c1[2*i], c1[2*i+1]) for i in range(4)])
        p2 = Polygon([(c2[2*i], c2[2*i+1]) for i in range(4)])
        if not p1.is_valid:
            p1 = p1.buffer(0)
        if not p2.is_valid:
            p2 = p2.buffer(0)
        a1 = dx1 * dy1
        a2 = dx2 * dy2
        inter = p1.intersection(p2).area
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    if nms_kind == 'nms':
        # Standard NMS: greedily suppress overlapping boxes by IoU > thresh.
        # boxes is (N, 7) [x, y, z, dx, dy, dz, heading]; BEV uses [x, y, dx, dy, heading].
        def nms_cpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
            assert boxes.shape[1] == 7, "nms_cpu expects (N, 7) boxes"
            order = scores.sort(descending=True)[1]
            if pre_maxsize is not None:
                order = order[:pre_maxsize]
            boxes_full = boxes[order].cpu().numpy().astype(np.float32)
            bev_idx = [0, 1, 3, 4, 6]
            boxes_sorted = boxes_full[:, bev_idx]
            n = boxes_sorted.shape[0]
            suppressed = np.zeros(n, dtype=bool)
            keep = []
            for i in range(n):
                if suppressed[i]:
                    continue
                keep.append(i)
                for j in range(i + 1, n):
                    if suppressed[j]:
                        continue
                    iou = rbbox_iou(boxes_sorted[i], boxes_sorted[j])
                    if iou > thresh:
                        suppressed[j] = True
            keep_idx = torch.as_tensor(keep, dtype=torch.long)
            return order[keep_idx], None
        return nms_cpu
    else:
        # nms_normal: same as nms for our purposes (BEV rotated NMS).
        def nms_normal_cpu(boxes, scores, thresh, **kwargs):
            return nms_kind and None  # unreachable; placeholder
        # Recursive call to get standard nms logic
        return _make_nms_cpu('nms')


def _boxes_iou_bev_cpu(boxes_a, boxes_b):
    """CPU equivalent of iou3d_nms_utils.boxes_iou_bev."""
    import math
    from shapely.geometry import Polygon
    a = boxes_a.cpu().numpy().astype(np.float32)
    b = boxes_b.cpu().numpy().astype(np.float32)
    n, m = a.shape[0], b.shape[0]
    out = np.zeros((n, m), dtype=np.float32)

    def rbbox_corners(rbbox):
        cx, cy, dx, dy, ang = rbbox
        cos_a = math.cos(ang)
        sin_a = math.sin(ang)
        hx, hy = dx / 2.0, dy / 2.0
        corners_local = [(hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)]
        out_c = np.zeros(8, dtype=np.float32)
        for i, (lx, ly) in enumerate(corners_local):
            out_c[2 * i]     = cos_a * lx - sin_a * ly + cx
            out_c[2 * i + 1] = sin_a * lx + cos_a * ly + cy
        return out_c

    for i in range(n):
        c1 = rbbox_corners(a[i])
        p1 = Polygon([(c1[2*k], c1[2*k+1]) for k in range(4)])
        if not p1.is_valid:
            p1 = p1.buffer(0)
        a1 = a[i, 3] * a[i, 4]
        for j in range(m):
            c2 = rbbox_corners(b[j])
            p2 = Polygon([(c2[2*k], c2[2*k+1]) for k in range(4)])
            if not p2.is_valid:
                p2 = p2.buffer(0)
            a2 = b[j, 3] * b[j, 4]
            inter = p1.intersection(p2).area
            union = a1 + a2 - inter
            out[i, j] = inter / union if union > 0 else 0.0
    return torch.from_numpy(out)


# Patch iou3d_nms_utils with CPU implementations.
from pcdet.ops.iou3d_nms import iou3d_nms_utils as _iou3d_nms_utils
_iou3d_nms_utils.nms_gpu = _make_nms_cpu('nms')
_iou3d_nms_utils.nms_normal_gpu = _make_nms_cpu('nms')
_iou3d_nms_utils.boxes_iou_bev = _boxes_iou_bev_cpu


def _boxes_iou3d_cpu(boxes_a, boxes_b):
    """CPU equivalent of iou3d_nms_utils.boxes_iou3d_gpu.

    boxes_a, boxes_b: (N, 7), (M, 7) [x, y, z, dx, dy, dz, heading].
    Returns (N, M) IoU.
    """
    overlaps_bev = _boxes_iou_bev_cpu(boxes_a[:, [0, 1, 3, 4, 6]], boxes_b[:, [0, 1, 3, 4, 6]])

    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)
    return iou3d


_iou3d_nms_utils.boxes_iou3d_gpu = _boxes_iou3d_cpu


# ----------------------------------------------------------------------
# Fast rotated IoU using OpenCV's rotatedRectangleIntersection (avoids the
# very slow shapely-based rotate_iou_cpu_eval used by KITTI eval).
# ----------------------------------------------------------------------
import cv2 as _cv2


def _rotate_iou_cpu_eval_fast(boxes, query_boxes, criterion=-1):
    """Fast NumPy + OpenCV implementation of rotated 2D IoU.

    boxes:        (N, 5) [cx, cy, dx, dy, heading]
    query_boxes:  (K, 5) [cx, cy, dx, dy, heading]
    Returns:      (N, K) IoU matrix (float32).

    criterion: -1 (IoU) / 0 (overlap with boxes) / 1 (overlap with query)
               / 2 (intersection area only — matches KITTI eval usage).
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)
    N, K = boxes.shape[0], query_boxes.shape[0]
    out = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return out

    areas_a = (boxes[:, 2] * boxes[:, 3]).astype(np.float32)
    areas_b = (query_boxes[:, 2] * query_boxes[:, 3]).astype(np.float32)

    for i in range(N):
        cx, cy, w, h, ang_deg = (
            float(boxes[i, 0]),
            float(boxes[i, 1]),
            float(boxes[i, 2]),
            float(boxes[i, 3]),
            float(np.degrees(boxes[i, 4])),
        )
        rect_a = ((cx, cy), (w, h), ang_deg)
        a1 = areas_a[i]
        for j in range(K):
            qx, qy, qw, qh, qang_deg = (
                float(query_boxes[j, 0]),
                float(query_boxes[j, 1]),
                float(query_boxes[j, 2]),
                float(query_boxes[j, 3]),
                float(np.degrees(query_boxes[j, 4])),
            )
            rect_b = ((qx, qy), (qw, qh), qang_deg)
            try:
                ret, intersect = _cv2.rotatedRectangleIntersection(rect_a, rect_b)
            except _cv2.error:
                ret = 0
                intersect = None
            if ret == 1 and intersect is not None:
                inter_area = float(_cv2.contourArea(intersect))
            elif ret == 2:
                # One rectangle fully inside the other.
                inter_area = float(min(a1, areas_b[j]))
            else:
                inter_area = 0.0
            a2 = areas_b[j]
            if criterion == -1:
                ua = a1 + a2 - inter_area
                out[i, j] = inter_area / ua if ua > 0 else 0.0
            elif criterion == 0:
                out[i, j] = inter_area / a1 if a1 > 0 else 0.0
            elif criterion == 1:
                out[i, j] = inter_area / a2 if a2 > 0 else 0.0
            else:
                out[i, j] = inter_area
    return out


# Patch the rotate_iou GPU op to use the fast cv2 path on CPU.
# This is the function the KITTI eval (eval.py) calls for bev/d3 box IoU.
import pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou as _rotate_iou_mod
_rotate_iou_mod.rotate_iou_gpu_eval = _rotate_iou_cpu_eval_fast
_rotate_iou_mod.rotate_iou_cpu_eval = _rotate_iou_cpu_eval_fast


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser (CPU eval)')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--output_root', type=str, default=None, help='override output_dir')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt_cpu(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    """CPU version of test.eval_single_ckpt.

    Differences vs GPU version:
      - load_params_from_file(to_cpu=True) so checkpoint lands on CPU.
      - No model.cuda() call (model is already on CPU after construction).
      - load_data_to_gpu is patched globally to a CPU variant.
    """
    # load checkpoint onto CPU
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)

    # eval on CPU
    ret_dict = _eval_utils_mod.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )

    # Dump structured result for tools/visualize_eval.py
    try:
        log_eval = sorted(eval_output_dir.glob('log_eval_*.txt'))
        summary_str = log_eval[-1].read_text(errors='ignore') if log_eval else ''
        results = {'ret_dict': {k: float(v) for k, v in (ret_dict or {}).items()},
                   'summary_str': summary_str, 'epoch_id': str(epoch_id)}
        (eval_output_dir / 'results.json').write_text(
            json.dumps(results, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.warning('Failed to dump results.json: %s' % e)


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    if args.output_root:
        output_dir = Path(args.output_root)
    else:
        output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        ckpt_name = Path(args.ckpt).name if args.ckpt is not None else ''
        num_list = re.findall(r'\d+', ckpt_name)
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging (CPU eval)**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s (CPU eval mode)', gpu_list)
    logger.info('NUMBA_DISABLE_CUDA=%s', os.environ.get('NUMBA_DISABLE_CUDA', '0'))

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    with torch.no_grad():
        if args.eval_all:
            # In eval_all mode, repeat_eval_ckpt also calls model.cuda(). Skip for now.
            raise NotImplementedError('eval_all is not supported in CPU mode')
        else:
            eval_single_ckpt_cpu(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == '__main__':
    import re  # local import; only needed in __main__
    main()
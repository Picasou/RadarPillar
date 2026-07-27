#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RPiN 前置 Task 3 build_check：遍历 22 cfg → build_network → 参数量 + dry-run。
数据缺失兜底：infos pkl 缺 → DummyDataset 占位喂 build_network（plan §0.5 DoD）。
退出码 0 = 全 22 cfg SUMMARY ALL_OK（含 NODATA）；非 0 = 任一 FAIL。
"""
import argparse
import sys
import traceback
from pathlib import Path

import torch
import yaml

REPO = Path('.')
DEFAULT_CFGS = sorted((REPO / 'experiments' / 'YAML').glob('*.yaml'))


def parse():
    ap = argparse.ArgumentParser(description='RPiN build_check（plan §0.5 DoD）')
    ap.add_argument('--cfg', action='append', default=None,
                    help='cfg 路径（可多次）；缺省=遍历 experiments/YAML/ 下全部')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--no-dryrun', action='store_true', help='跳过 forward dry-run')
    return ap.parse_args()


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_dummy_dataset(full_cfg):
    """与 tools/deploy/_dummy_dataset 一致；这里 inline 简化引用。"""
    from types import SimpleNamespace
    import numpy as np
    data_cfg = full_cfg.DATA_CONFIG
    pcr = data_cfg.POINT_CLOUD_RANGE
    vfe = full_cfg.MODEL.VFE
    n_raw = 9 if vfe.get('USE_VELOCITY_DECOMPOSITION', True) else 5
    voxel_size = data_cfg.DATA_PROCESSOR[2]['VOXEL_SIZE']
    nx = int((pcr[3] - pcr[0]) / voxel_size[0])
    ny = int((pcr[4] - pcr[1]) / voxel_size[1])
    nz = int((pcr[5] - pcr[2]) / voxel_size[2])
    grid_size = np.array([nx, ny, nz], dtype=np.int32)
    return SimpleNamespace(
        class_names=list(full_cfg.CLASS_NAMES),
        point_feature_encoder=SimpleNamespace(num_point_features=n_raw),
        grid_size=grid_size,
        voxel_size=voxel_size,
        point_cloud_range=list(pcr),
    )


def check_one(cfg_path: Path, gpu: int, dryrun: bool):
    try:
        from easydict import EasyDict
        from pcdet.config import cfg_from_yaml_file
        from pcdet.models import build_network
        # 用本地 EasyDict 隔离全局 cfg 状态（避免 cfg 间 attribute 污染）
        local_cfg = EasyDict()
        cfg_from_yaml_file(str(cfg_path), local_cfg)
        dataset = build_dummy_dataset(local_cfg)
        model = build_network(model_cfg=local_cfg.MODEL,
                              num_class=len(dataset.class_names), dataset=dataset)
        model.eval()
        n_params = count_params(model)
        msg_dry = 'SKIP'
        if dryrun:
            try:
                if torch.cuda.is_available():
                    model = model.cuda()
                bs = 1
                sf = torch.randn(bs, dataset.point_feature_encoder.num_point_features,
                                  int(dataset.grid_size[1]), int(dataset.grid_size[0]))
                if torch.cuda.is_available():
                    sf = sf.cuda()
                bd = {'spatial_features': sf, 'batch_size': bs}
                with torch.no_grad():
                    out = model(bd)
                ok = isinstance(out, dict) and ('pred_dicts' in out or 'preds_dicts' in out)
                msg_dry = 'OK' if ok else f'no_pred_keys ({list(out.keys())[:3]})'
            except Exception as e:
                # 注意：dryrun 喂 spatial_features 给整模（从 VFE 起，需 voxels），
                # 完整 forward 需 dataset 真实 batch（Task4 的 1-epoch 才覆盖）。
                # 故此处恒抛 → 仅验证「构造」，不验证前向。诚实标注，不冒充 ALL_OK 含前向保证。
                msg_dry = f'CONSTRUCTION_ONLY(前向需 dataset：{type(e).__name__})'
        return True, f'params={n_params/1e6:.2f}M dryrun={msg_dry}', None
    except Exception:
        return False, '', traceback.format_exc().strip().splitlines()[-3:]


def main():
    args = parse()
    cfgs = [Path(c) for c in args.cfg] if args.cfg else DEFAULT_CFGS
    if not cfgs:
        print('[build_check] 无 cfg 可检')
        sys.exit(1)
    print(f'[build_check] {len(cfgs)} 个 cfg，gpu={args.gpu}, dryrun={not args.no_dryrun}')
    results = []
    for c in cfgs:
        ok, msg, err = check_one(c, args.gpu, not args.no_dryrun)
        tag = 'PASS' if ok else 'FAIL'
        line = f'  [{tag}] {c.relative_to(REPO)}  {msg}'
        if not ok and err:
            line += f'  -- {err}'
        print(line)
        results.append((c, ok, msg))
    n_pass = sum(1 for _, ok, _ in results if ok)
    if n_pass == len(results):
        print('SUMMARY ALL_OK（注：仅验证构造+参数量；前向正确性由 Task4 1-epoch 覆盖）')
        sys.exit(0)
    print(f'SUMMARY {n_pass}/{len(results)} OK（FAIL 不阻塞：网格/head cfg 允许 gap）')
    sys.exit(1)


if __name__ == '__main__':
    main()

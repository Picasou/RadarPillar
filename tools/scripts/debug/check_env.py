#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RPiN prerequisites Task 1：环境自检。
退出码 0 = 全过；打印 GPU 型号 + 显存（供 Task 4 估 bs）+ 数据状态。
用法: conda activate <env> && python tools/scripts/check_env.py
"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'VoD' / 'view_of_delft_PUBLIC' / 'radar_5frames'

fails = []
optional_missing = []  # 可选依赖（仅 deploy/export 用），缺失不阻塞核心环境


def check(name, fn, optional=False):
    try:
        msg = fn()
        print(f'  [OK]   {name}' + (f' — {msg}' if msg else ''))
        return True
    except Exception as e:
        if optional:
            print(f'  [WARN] {name}（可选，仅 deploy/export）— {type(e).__name__}: {e}')
            optional_missing.append(name)
            return False
        print(f'  [FAIL] {name} — {type(e).__name__}: {e}')
        fails.append(name)
        return False


def ck_python():
    v = sys.version_info
    assert v >= (3, 7), f'Python {v.major}.{v.minor} < 3.7'
    return f'{v.major}.{v.minor}.{v.micro}'


def ck_torch_cuda():
    import torch
    assert torch.cuda.is_available(), 'CUDA 不可用'
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f'torch {torch.__version__} / {name} / {mem:.1f} GiB'


def ck_import(mod):
    def _f():
        m = importlib.import_module(mod)
        return getattr(m, '__version__', '')
    return _f


def ck_data():
    for sub in ['vod_infos_train.pkl', 'vod_infos_val.pkl']:
        assert (DATA / sub).exists(), f'缺 {sub}'
    for sub in ['training/velodyne', 'training/label_2', 'training/calib', 'ImageSets']:
        assert (DATA / sub).exists(), f'缺 {sub}'
    n_train = len(list((DATA / 'training' / 'velodyne').glob('*.bin')))
    return f'infos pkl ✓ / raw velodyne {n_train} 帧 / ImageSets ✓'


print('=== RPiN prerequisites check_env ===')
check('Python>=3.7', ck_python)
check('torch+CUDA+GPU', ck_torch_cuda)
check('spconv', ck_import('spconv'))
check('pcdet', ck_import('pcdet'))
check('onnx', ck_import('onnx'), optional=True)
check('onnxruntime', ck_import('onnxruntime'), optional=True)
check('VoD 数据', ck_data)

if fails:
    print(f'=== FAIL: {len(fails)} 项核心未过: {fails} ===')
    sys.exit(1)
if optional_missing:
    print(f'=== ALL OK（核心环境就绪；可选项缺失不影响训练: {optional_missing}）===')
else:
    print('=== ALL OK ===')
sys.exit(0)

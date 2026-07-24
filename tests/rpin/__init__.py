"""RPiN 前置模块契约测试（DoD2：补齐 commit 491ae7a 缺失的测试文件）。

覆盖 7 个新模块的「不可猜约束」+ 关键 bug 修复回归：
  - PP* 4-bug：SecondFPN 返 list 取[0] / 构造无 input_channels / 多尺度大→小 / num_bev=sum(OUT_CHANNELS)
  - head_2d：H2 train→eval 不崩 + H3 height 按类填充（非跨类均值）
  - VDC：compensate_motion 纯函数 + RADAR_FEATURE_ORDER 常量 + time_scale 量纲

CPU 友好（无 CUDA 依赖），跑法：python -m pytest tests/rpin/ -q
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
from easydict import EasyDict


def _load_model_cfg(tag):
    from pcdet.config import cfg_from_yaml_file
    local_cfg = EasyDict()
    cfg_from_yaml_file(os.path.join(_ROOT, 'experiments', 'YAML', f'{tag}.yaml'), local_cfg)
    return local_cfg.MODEL

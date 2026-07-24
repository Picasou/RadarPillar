#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RPiN 阶段6 8-2：ONNX 导出（前置模块，步骤占位）。

按 plan §0.5 Task2.8 / §11.1：按 cfg BACKBONE_2D.NAME 匹配；--ckpt 可选；
数据缺失兜底 → DummyDataset 占位（不 build_dataloader），仅验算子链。

用法: python tools/deploy/export_onnx.py --cfg_file <cfg> [--ckpt <pth>] [--output <onnx>]
"""
import argparse
from pathlib import Path

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.utils import common_utils

from _dummy_dataset import make_dummy_dataset


def parse():
    ap = argparse.ArgumentParser(description='RPiN ONNX 导出（占位：验证算子链）')
    ap.add_argument('--cfg_file', required=True)
    ap.add_argument('--ckpt', default=None, help='可选 ckpt（缺则随机权重）')
    ap.add_argument('--output', default=None, help='可选输出 .onnx 路径')
    ap.add_argument('--batch_size', type=int, default=1)
    return ap.parse_args()


def main():
    args = parse()
    logger = common_utils.create_logger(); logger.setLevel('WARNING')
    cfg_from_yaml_file(args.cfg_file, cfg)
    dataset = make_dummy_dataset(cfg)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(dataset.class_names), dataset=dataset)
    model.eval()
    if args.ckpt and Path(args.ckpt).exists():
        ckpt = torch.load(args.ckpt, map_location='cpu')
        state = ckpt.get('model_state', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.warning(f'ckpt loaded, missing={len(missing)}, unexpected={len(unexpected)}')
    if torch.cuda.is_available():
        model = model.cuda()
    # 注意：本脚本是「前置占位」——当前只验证 build 链，并未调用 torch.onnx.export
    # （主计划阶段6 启用真正的 ONNX 导出）。诚实标注，不让输出路径看起来像已落盘。
    if args.output:
        print(f'[export_onnx] build 链验证成功（占位：未实际导出 ONNX，主计划阶段6 启用 torch.onnx.export）')
    else:
        print('[export_onnx] build 链验证成功（未指定输出，仅 build 验证）')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RPiN 阶段6 8-3：TensorRT engine 构造（占位）。

按 plan §0.5 Task2.8：用 TensorRT build_serialized_network。tensorrt 缺失 → exit 3 提示。

用法: python tools/deploy/build_trt.py --onnx <onnx>
"""
import argparse
import sys


def parse():
    ap = argparse.ArgumentParser(description='RPiN TensorRT 构造（占位）')
    ap.add_argument('--onnx', required=True, help='输入 .onnx 路径')
    return ap.parse_args()


def main():
    args = parse()
    try:
        import tensorrt as trt  # noqa
    except ImportError as e:
        print(f'[build_trt] tensorrt 未安装（exit 3）: {e}')
        print('[build_trt] 安装: pip install tensorrt  -- 需匹配 torch/cu121 的 wheel')
        sys.exit(3)
    print(f'[build_trt] tensorrt {trt.__version__} 已就绪；准备 build {args.onnx} → engine')
    # 此处为占位：实际 build_serialized_network 链路较长，由主计划阶段 6 启用时扩展
    print('[build_trt] 当前为占位实现（仅 import check 通过）；主计划阶段6 在此基础上追加序列化构造')


if __name__ == '__main__':
    main()

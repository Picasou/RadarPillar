#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSR 数据 infos + gt_database 生成入口。
对齐 plan Task 5: class_names 硬编码 ['1','4','5'](dataset yaml 无此 key);
data_path / save_path 都指 /mnt/d/DataSet/11111111111(与 msr_dataset.yaml 的 DATA_PATH / INFO_PATH 解析路径一致)。
"""
import argparse
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.msr.msr_dataset import create_msr_infos


def main():
    parser = argparse.ArgumentParser(description='Generate MSR dataset infos and gt_database')
    parser.add_argument('--cfg_file', type=str,
                        default='tools/cfgs/dataset/msr_dataset.yaml',
                        help='dataset config yaml')
    parser.add_argument('--data_path', type=str,
                        default='/mnt/d/DataSet/11111111111',
                        help='root data path; also used as save_path (must match YAML INFO_PATH resolution)')
    args = parser.parse_args()

    # class_names 硬编码:dataset yaml 无此 key;类别名即 LABELS type 字段的 str() 值
    class_names = ['1', '4', '5']

    # 加载 dataset cfg(仓库规范:直接用模块级 cfg,cfg_from_yaml_file 就地修改)
    cfg_from_yaml_file(args.cfg_file, cfg)
    dataset_cfg = cfg

    data_path = Path(args.data_path)
    save_path = Path(args.data_path)  # save_path 与 data_path 同:与 YAML INFO_PATH 解析一致

    print('=== create_msr_data ===')
    print('cfg_file    :', args.cfg_file)
    print('data_path   :', data_path)
    print('save_path   :', save_path)
    print('class_names :', class_names)

    create_msr_infos(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        data_path=data_path,
        save_path=save_path,
        workers=4,
    )
    print('=== done ===')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VoD 数据 infos + gt_database 生成入口。
对齐 plan Task 1: class_names 硬编码 [Car, Pedestrian, Cyclist]（dataset yaml 无此 key）；
data_path / save_path 都指 radar_5frames（与 vod_dataset_radar.yaml 的 DATA_PATH/INFO_PATH 解析路径一致）。
"""
import argparse
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.vod.vod_dataset import create_vod_infos


def main():
    parser = argparse.ArgumentParser(description='Generate VoD dataset infos and gt_database')
    parser.add_argument('--cfg_file', type=str,
                        default='tools/cfgs/dataset/vod_dataset_radar.yaml',
                        help='dataset config yaml')
    parser.add_argument('--data_path', type=str,
                        default='data/VoD/view_of_delft_PUBLIC/radar_5frames',
                        help='root data path; also used as save_path (must match YAML INFO_PATH resolution)')
    args = parser.parse_args()

    # class_names 硬编码：dataset yaml 无此 key，且必须与 OpenPCDet VoD dataset 顺序一致
    class_names = ['Car', 'Pedestrian', 'Cyclist']

    # 加载 dataset cfg（仓库规范：直接用模块级 cfg，cfg_from_yaml_file 就地修改）
    cfg_from_yaml_file(args.cfg_file, cfg)
    dataset_cfg = cfg

    data_path = Path(args.data_path)
    save_path = Path(args.data_path)  # save_path 与 data_path 同：与 YAML INFO_PATH 解析一致

    print('=== create_vod_data ===')
    print('cfg_file    :', args.cfg_file)
    print('data_path   :', data_path)
    print('save_path   :', save_path)
    print('class_names :', class_names)

    create_vod_infos(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        data_path=data_path,
        save_path=save_path,
        workers=4,
    )
    print('=== done ===')


if __name__ == '__main__':
    main()

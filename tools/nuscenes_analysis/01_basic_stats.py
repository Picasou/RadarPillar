"""nuScenes 雷达数据集基础元信息统计。

输入：nuScenes 数据集目录 + version
输出：终端报告 + reports/basic_stats.json

用法：
    python tools/nuscenes_analysis/01_basic_stats.py
    python tools/nuscenes_analysis/01_basic_stats.py --version v1.0-trainval
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from nuscenes.nuscenes import NuScenes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataroot', default='data/nuscenes/v1.0-mini')
    ap.add_argument('--version', default='v1.0-mini')
    ap.add_argument('--out', default='reports/basic_stats.json')
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # 1) 顶层表数量
    n_scene = len(nusc.scene)
    n_sample = len(nusc.sample)
    n_sample_data = len(nusc.sample_data)
    n_anno = len(nusc.sample_annotation)
    n_instance = len(nusc.instance)

    # 2) 23 个原始类别出现频次（devkit 1.0.5 直接给出 category_name）
    name_counter = Counter(a['category_name'] for a in nusc.sample_annotation)

    # 3) 雷达文件计数：每个 sample 都应有 5 个 RADAR_*
    radar_chan = [
        'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
        'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT',
    ]
    n_radar_files = 0
    n_radar_missing = 0
    for s in nusc.sample:
        for ch in radar_chan:
            if ch not in s['data']:
                n_radar_missing += 1
                continue
            sd_tok = s['data'][ch]
            path, _, _ = nusc.get_sample_data(sd_tok)
            if Path(path).exists():
                n_radar_files += 1
            else:
                n_radar_missing += 1

    # 4) 每个 sample 的平均标注数
    avg_anno = round(n_anno / max(n_sample, 1), 2)

    report = {
        'version': args.version,
        'n_scene': n_scene,
        'n_sample': n_sample,
        'n_sample_data': n_sample_data,
        'n_annotation': n_anno,
        'n_instance': n_instance,
        'n_radar_files_present': n_radar_files,
        'n_radar_files_missing': n_radar_missing,
        'avg_anno_per_frame': avg_anno,
        'category_raw_dist': dict(name_counter.most_common()),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'\n[ok] report saved -> {out}')


if __name__ == '__main__':
    main()

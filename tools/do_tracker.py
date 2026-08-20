#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tracker 全链路跑真实序列数据 (loader→detector→filter→match→manager→visualizer)。

用法(仓库根):
  python tools/do_tracker.py [--cfg tracker/cfg/cfg.yaml]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracker.tracker import Tracker


def main():
    parser = argparse.ArgumentParser(description='run tracker full chain on seq data')
    parser.add_argument('--cfg', type=str, default='tracker/cfg/cfg.yaml')
    args = parser.parse_args()

    t0 = time.time()
    trk = Tracker(args.cfg)
    print('=== tracker init done (%.1fs), %d seqs ===' % (time.time() - t0, len(trk.cfg.DATA.paths)))
    trk.run()
    print('=== tracker done in %.1fs ===' % (time.time() - t0))


if __name__ == '__main__':
    main()

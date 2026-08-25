# -*-coding:utf-8-*-
"""GT bin 加载验证: 合成 gt_radar_1200/1201 写读往返 + 无标注目录返回空。

用法: python tracker/tests/test_gt_loader.py
"""
import os
import sys
import tempfile
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from types import SimpleNamespace

from tracker.loader import Loader, _GT_HEAD_DTYPE, _GT_REC_DTYPE


def _write_gt_bins(seq_dir, frames):
    """写合成 GT: frames = [ [ (id,x,y,type,width,length,heading,is_ghost,is_unpassable), ... ], ... ]"""
    gt_dir = os.path.join(seq_dir, 'gt.default')
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(os.path.join(seq_dir, 'radar.default'), exist_ok=True)
    heads = np.zeros(len(frames), dtype=_GT_HEAD_DTYPE)
    rec_list = []
    for i, fr in enumerate(frames):
        heads[i] = (1, 0, len(fr), 0)
        for gid, x, y, gtype, w, l, hd, ghost, unpass in fr:
            rec_list.append((gid, round(x * 100), round(y * 100), 0, 0, 0,
                             round(hd * 100), round(w * 100), round(l * 100), 150,
                             gtype, 90, ghost, unpass, 1, 12, 48))
    np.array(rec_list, dtype=_GT_REC_DTYPE).tofile(os.path.join(gt_dir, 'gt_radar_1201.00000.bin'))
    heads.tofile(os.path.join(gt_dir, 'gt_radar_1200.00000.bin'))


def _make_loader():
    return Loader(SimpleNamespace(RUN=SimpleNamespace(vds=None)))


def test_gt_roundtrip():
    assert _GT_HEAD_DTYPE.itemsize == 8 and _GT_REC_DTYPE.itemsize == 27, "GT bin 布局尺寸偏离 spec"
    with tempfile.TemporaryDirectory() as tmp:
        seq = os.path.join(tmp, 'seq')
        _write_gt_bins(seq, [
            [(1, 30.94, 8.45, 1, 2.0, 5.0, -28.88, 0, 0), (3, 2.36, 2.10, 4, 0.7, 1.8, 0.0, 1, 1)],
            [(1, 31.0, 8.50, 1, 2.0, 5.0, -28.0, 0, 0)],
        ])
        gts_list = _make_loader()._load_GTs(os.path.join(seq, 'radar.default'))
        ok = len(gts_list) == 2
        f0 = gts_list[0].Lst
        ok = ok and gts_list[0].num == 2 and gts_list[1].num == 1
        ok = ok and (f0[0].id, f0[0].type) == (1, 1) and (f0[1].id, f0[1].type) == (3, 4)
        ok = ok and abs(f0[0].x - 30.94) < 1e-6 and abs(f0[0].y - 8.45) < 1e-6
        ok = ok and abs(f0[0].heading - (-28.88)) < 1e-6
        ok = ok and abs(f0[0].width - 2.0) < 1e-6 and abs(f0[0].length - 5.0) < 1e-6
        ok = ok and abs(f0[1].x - 2.36) < 1e-6
        # 新字段: type_confi/isghost/ispassable(¬is_unpassable)/is_attention/point_count/point_quality
        ok = ok and (f0[0].type_confi, f0[0].isghost, f0[0].ispassable) == (90, 0, 1)
        ok = ok and (f0[0].is_attention, f0[0].point_count, f0[0].point_quality) == (1, 12, 48)
        ok = ok and (f0[1].isghost, f0[1].ispassable) == (1, 0)
        print('  %s GT bin 往返: 帧数/字段/缩放/heading/新字段' % ('✓' if ok else '✗'))
        return ok


def test_gt_missing_dir():
    with tempfile.TemporaryDirectory() as tmp:
        seq = os.path.join(tmp, 'seq')
        os.makedirs(os.path.join(seq, 'radar.default'))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            gts_list = _make_loader()._load_GTs(os.path.join(seq, 'radar.default'))
        ok = gts_list == [] and len(w) == 1
        print('  %s 无 gt.default -> [] + warning' % ('✓' if ok else '✗'))
        return ok


if __name__ == '__main__':
    results = [test_gt_roundtrip(), test_gt_missing_dir()]
    print('RESULT:', 'PASS' if all(results) else 'FAIL')
    sys.exit(0 if all(results) else 1)

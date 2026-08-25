# -*-coding:utf-8-*-
"""evaluator 验证: 单帧匹配 + TP/FP/FN/MOTA/MOTP/IDSW/Frag 判定 + 类别映射排除。

用法: python tracker/tests/test_evaluator.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from types import SimpleNamespace

from tracker.evaluator import Evaluator


def _cfg():
    return SimpleNamespace(EVALUATE=SimpleNamespace(match_dist=2.0, report=0, template='default'),
                           METRICS=SimpleNamespace(show={}))


def _gt(gid, x, y, gtype):
    return SimpleNamespace(id=gid, x=x, y=y, type=gtype)


def _trk(tid, x, y, ttype):
    return SimpleNamespace(id=tid, x_m=x, y_m=y, type=ttype)


def _ev():
    return Evaluator(_cfg(), class_names=['Car', 'Pedestrian', 'Cyclist'])


def test_match_frame_basic():
    ev = _ev()
    m, ut, ug = ev.match_frame([_gt(1, 10.0, 0.0, 1)], [_trk(7, 10.5, 0.0, 1)])
    ok = len(m) == 1 and abs(m[0][2] - 0.5) < 1e-9 and not ut and not ug
    m2, ut2, ug2 = ev.match_frame([_gt(1, 10.0, 0.0, 1)], [_trk(7, 20.0, 0.0, 1)])
    ok = ok and not m2 and len(ut2) == 1 and len(ug2) == 1      # 超门限
    m3, ut3, ug3 = ev.match_frame([_gt(1, 10.0, 0.0, 2)], [_trk(7, 10.0, 0.0, 1)])
    ok = ok and not m3 and len(ut3) == 1 and len(ug3) == 1      # 类别不符
    m4, _, ug4 = ev.match_frame([_gt(1, 10.0, 0.0, 5)], [_trk(7, 10.0, 0.0, 1)])
    ok = ok and not m4 and not ug4                                # Truck 排除: 不计 FN
    print('  %s match_frame: 门限/类别/排除类' % ('✓' if ok else '✗'))
    return ok


def test_perfect_track():
    ev = _ev()
    seq = [(_gts := None, None)]  # placeholder replaced below
    frames = []
    for i in range(5):
        gts = SimpleNamespace(num=1, Lst=[_gt(1, 10.0 + i, 0.0, 1)])
        frames.append((gts, [_trk(7, 10.0 + i, 0.0, 1)]))
    res = ev.evaluate([frames])
    t = res['total']
    ok = (t['tp'] == 5 and t['fp'] == 0 and t['fn'] == 0
          and t['idsw'] == 0 and t['frag'] == 0 and abs(t['mota'] - 1.0) < 1e-9)
    print('  %s 完美跟踪: TP=5 MOTA=1.0 IDSW=Frag=0' % ('✓' if ok else '✗'))
    return ok


def test_idswitch_and_frag():
    ev = _ev()
    frames = [
        (SimpleNamespace(num=1, Lst=[_gt(1, 10.0, 0.0, 1)]), [_trk(7, 10.0, 0.0, 1)]),
        (SimpleNamespace(num=1, Lst=[_gt(1, 11.0, 0.0, 1)]), [_trk(9, 11.0, 0.0, 1)]),   # 换号
        (SimpleNamespace(num=1, Lst=[_gt(1, 12.0, 0.0, 1)]), []),                          # 断 1 帧
        (SimpleNamespace(num=1, Lst=[_gt(1, 13.0, 0.0, 1)]), [_trk(9, 13.0, 0.0, 1)]),   # 接回
    ]
    res = ev.evaluate([frames])
    t = res['total']
    ok = t['idsw'] == 1 and t['frag'] == 1 and t['fn'] == 1
    print('  %s IDSW=1(换号) Frag=1(断后重接) FN=1' % ('✓' if ok else '✗'))
    return ok


def test_per_class():
    ev = _ev()
    frames = [(SimpleNamespace(num=2, Lst=[_gt(1, 10.0, 0.0, 1), _gt(2, 30.0, 5.0, 2)]),
               [_trk(7, 10.0, 0.0, 1)])]          # 只配到 Car, 行人漏
    res = ev.evaluate([frames])
    t, c1, c2 = res['total'], res['per_class'][1], res['per_class'][2]
    ok = t['tp'] == 1 and t['fn'] == 1 and c1['tp'] == 1 and c2['fn'] == 1
    print('  %s per-class 分账 (Car TP / Ped FN)' % ('✓' if ok else '✗'))
    return ok


if __name__ == '__main__':
    results = [test_match_frame_basic(), test_perfect_track(),
               test_idswitch_and_frag(), test_per_class()]
    print('RESULT:', 'PASS' if all(results) else 'FAIL')
    sys.exit(0 if all(results) else 1)

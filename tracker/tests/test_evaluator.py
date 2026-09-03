"""evaluator 单测: 单帧匹配 / 计账 / 全局指标 / 在线离线口径一致性 / 报告产物."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tracker.schemas import GT, GTs, Trk, TrkHistory
from tracker.evaluator import Evaluator, METRIC_KEYS


def make_gt(id=1, type=1, x=0.0, y=0.0, vx=1.0, vy=0.0, isghost=0):
    return GT(x=x, y=y, z=0.0, vx=vx, vy=vy, length=4.0, width=2.0, height=1.5,
              heading=0.0, type=type, isghost=isghost, ispassable=1, id=id)


def make_trk(id=1, type=1, x=0.0, y=0.0, vx=1.0, vy=0.0):
    return Trk(
        x_m=x, y_m=y, z_m=0, vx_mps=vx, vy_mps=vy, doppler_mps=0,
        ax_mps2=0, ay_mps2=0, heading_deg=0, yaw_rate_degs=0,
        id=id, width_m=2, height_m=1.5, length_m=4, lifetime_s=1,
        x_std_m=0, y_std_m=0, z_std_m=0, vx_std_mps=0, vy_std_mps=0,
        ax_std_mps2=0, ay_std_mps2=0, xy_pos_cov=0, xy_vel_cov=0, xy_acc_cov=0,
        width_std_m=0, height_std_m=0, length_std_m=0,
        heading_std_deg=0, yaw_rate_std_degs=0,
        type=type, type_confi=80, obstacle_prob=1, existence_prob=90,
        motion_status=1, measurement_status=0, passable_status=0,
        rel_vel=0, rel_acc=0, cov=np.zeros((4, 4)), history=TrkHistory())


def make_ev(tmp=None, report=0):
    cfg = SimpleNamespace(
        EVALUATE=SimpleNamespace(match_dist=2.0, report=report, template='default'),
        METRICS=SimpleNamespace(enable=0, show={}),
        MANAGER=SimpleNamespace(dt=0.1),
        DATA=SimpleNamespace(paths=[str(tmp / 'seqA'), str(tmp / 'seqB')] if tmp else []),
        MODEL=SimpleNamespace(cfg=''),
    )
    return Evaluator(cfg, class_names=['Car', 'Pedestrian', 'Cyclist'])


def frame_of(gts):
    return SimpleNamespace(gts=GTs(num=len(gts), Lst=list(gts)))


# ---- 单帧匹配 ----

def test_match_frame_class_and_gate():
    ev = make_ev()
    gts = [make_gt(id=1, type=1, x=0, y=0), make_gt(id=2, type=2, x=10, y=0)]
    trks = [make_trk(id=10, type=1, x=0.5, y=0), make_trk(id=11, type=1, x=9, y=0)]
    m, ut, ug = ev.match_frame(gts, trks)
    assert len(m) == 1 and m[0][0].id == 1 and m[0][1].id == 10
    assert m[0][2] == pytest.approx(0.5)
    assert [t.id for t in ut] == [11]      # Car trk 近 Ped gt, 类别约束不配
    assert [g.id for g in ug] == [2]


def test_match_frame_greedy_nearest():
    ev = make_ev()
    gts = [make_gt(id=1, x=0, y=0), make_gt(id=2, x=1.5, y=0)]
    trks = [make_trk(id=10, x=1.6, y=0), make_trk(id=11, x=0.1, y=0)]
    m, ut, ug = ev.match_frame(gts, trks)
    assert {(g.id, t.id) for g, t, _ in m} == {(1, 11), (2, 10)}
    assert not ut and not ug


def test_match_frame_distance_gate():
    ev = make_ev()
    gts = [make_gt(id=1, x=0, y=0)]
    trks = [make_trk(id=10, x=2.5, y=0)]   # 超 2m 门限
    m, ut, ug = ev.match_frame(gts, trks)
    assert m == [] and len(ut) == 1 and len(ug) == 1


# ---- 逐帧计账 ----

def _run_seq(ev, frames, key='s'):
    ev._reset_seq()
    ev._seq_key = key
    for gts, trks in frames:
        ev._step(gts, trks)
    ev._accum_global()
    return ev._finalize(ev._acc, ev._cls_acc)


def test_ids_frag_mota():
    ev = make_ev()
    gt = make_gt(id=1, type=1, x=0, y=0)
    frames = [
        ([gt], [make_trk(id=10)]),          # f0 配 A
        ([gt], [make_trk(id=10)]),          # f1 配 A
        ([gt], [make_trk(id=11)]),          # f2 换 B -> ids=1
        ([gt], []),                         # f3 漏 -> fn=1
        ([gt], [make_trk(id=11)]),          # f4 复配 B -> frag=1
    ]
    met = _run_seq(ev, frames)
    assert met['tp'] == 4 and met['fn'] == 1 and met['fp'] == 0
    assert met['ids'] == 1 and met['frag'] == 1
    assert met['mota'] == pytest.approx(1 - (1 + 0 + 1) / 5)
    assert met['motp'] == pytest.approx(0.0)   # 逐帧同位配对
    # IDF1: pair(1,10)=2, pair(1,11)=2 -> idtp=2; gt=5, trk=4
    assert met['idf1'] == pytest.approx(4 / 9)
    # DetA=4/5, AssA=(2/4*2)/4=0.25, HOTA=sqrt(0.2)
    assert met['deta'] == pytest.approx(0.8)
    assert met['assa'] == pytest.approx(0.25)
    assert met['hota'] == pytest.approx(0.2 ** 0.5)


def test_ghost_and_unknown_label_filtered():
    ev = make_ev()
    gts = [make_gt(id=1, type=1), make_gt(id=2, type=1, isghost=1), make_gt(id=3, type=5)]
    met = _run_seq(ev, [(gts, [])])
    assert met['fn'] == 1          # 仅非 ghost 且类表内的 Car gt 入账
    assert met['fp'] == 0 and met['tp'] == 0
    assert met['mota'] == pytest.approx(0.0)


def test_velocity_metrics_gate():
    ev = make_ev()
    # gt 静止 (|v|<0.5): 方向指标不计, 幅值误差照计
    gt = make_gt(id=1, type=1, vx=0.0, vy=0.0)
    trk = make_trk(id=10, vx=0.2, vy=0.0)
    met = _run_seq(ev, [([gt], [trk])])
    assert met['vne'] == pytest.approx(0.2)
    assert met['vae'] != met['vae']          # nan: 无有效方向样本
    # gt 反向运动: 角度误差 180°
    gt2 = make_gt(id=1, type=1, vx=2.0, vy=0.0)
    trk2 = make_trk(id=10, vx=-2.0, vy=0.0)
    met2 = _run_seq(ev, [([gt2], [trk2])])
    assert met2['vae'] == pytest.approx(180.0)
    assert met2['vir'] == pytest.approx(1.0)
    assert met2['vaie'] == pytest.approx(180.0)


def test_vde_delay_detection():
    ev = make_ev()
    # gt 速度线性爬坡, trk 恒滞后 1 帧 -> VDE = +0.1s
    frames = []
    for fi in range(8):
        vx = 1.0 + 0.5 * fi
        frames.append(([make_gt(id=1, type=1, x=fi * 1.0, vx=vx)],
                       [make_trk(id=10, x=fi * 1.0, vx=1.0 + 0.5 * max(fi - 1, 0))]))
    met = _run_seq(ev, frames)
    assert met['vde'] == pytest.approx(0.1)
    assert met['vse'] == met['vse']          # 有值或 nan 均可, 不崩即可


# ---- 在线/离线口径一致性 ----

def _build_seqs():
    def seq1():
        g1 = make_gt(id=1, type=1, x=0, y=0, vx=2, vy=0)
        g2 = make_gt(id=2, type=2, x=20, y=5, vx=0, vy=1)
        out = []
        for fi in range(12):
            gts = [g1, g2] if fi != 6 else [g1]          # f6 g2 漏标 -> fn
            trks = [make_trk(id=10, type=1, x=(0 if fi % 2 == 0 else 30), y=0, vx=2, vy=0),
                    make_trk(id=20, type=2, x=20, y=5.5, vx=0, vy=1)]
            if fi == 5:
                trks.append(make_trk(id=30, type=1, x=-5, y=-5))   # FP
            out.append((gts, trks))
        return out

    def seq2():
        out = []
        for fi in range(10):
            vx = 1.0 + 0.4 * fi
            gts = [make_gt(id=5, type=2, x=fi * 1.0, y=0, vx=vx)]
            trks = [make_trk(id=50, type=2, x=fi * 1.0, y=0.1,
                             vx=1.0 + 0.4 * max(fi - 1, 0))] if fi != 4 else []
            out.append((gts, trks))
        return out

    return [('seqA', seq1()), ('seqB', seq2())]


def test_online_offline_consistency():
    seqs = _build_seqs()
    ev1 = make_ev()
    for name, frames in seqs:
        ev1.on_seq_start('/data/%s' % name)
        for gts, trks in frames:
            ev1.online(frame_of(gts), trks)
        ev1.on_seq_end('/data/%s' % name)
    r1 = ev1.on_dataset_end()

    ev2 = make_ev()
    r2 = ev2.evaluate([('/data/%s' % n, f) for n, f in seqs])

    assert set(r1['per_seq']) == set(r2['per_seq']) == {'seqA', 'seqB'}
    for k in METRIC_KEYS:
        assert r1['dataset'][k] == pytest.approx(r2['dataset'][k], nan_ok=True), k
    for name in r1['per_seq']:
        for k in METRIC_KEYS:
            assert r1['per_seq'][name][k] == pytest.approx(r2['per_seq'][name][k], nan_ok=True), k


def test_dataset_aggregation_math():
    # dataset 账 = 序列原料合并后统一 finalize (非指标平均)
    seqs = _build_seqs()
    ev = make_ev()
    r = ev.evaluate([('/data/%s' % n, f) for n, f in seqs])
    ds = r['dataset']
    assert ds['tp'] == r['per_seq']['seqA']['tp'] + r['per_seq']['seqB']['tp']
    assert ds['fn'] == r['per_seq']['seqA']['fn'] + r['per_seq']['seqB']['fn']
    assert ds['fp'] == r['per_seq']['seqA']['fp'] + r['per_seq']['seqB']['fp']


# ---- 回归: 审查修复项 ----

def test_dataset_id_no_cross_seq_collision():
    # 两序列各自完美跟踪但 gt/trk 数值 id 相同 -> dataset IDF1/AssA/HOTA 应全为 1.0
    ev = make_ev()
    hist = []
    for name, tid in (('seqA', 1), ('seqB', 2)):
        frames = [([make_gt(id=1, type=1, x=fi * 0.5)],
                   [make_trk(id=tid, x=fi * 0.5)]) for fi in range(10)]
        hist.append(('/d/%s' % name, frames))
    r = ev.evaluate(hist)
    for k in ('idf1', 'assa', 'hota'):
        assert r['dataset'][k] == pytest.approx(1.0), k
    assert r['per_seq']['seqA']['idf1'] == pytest.approx(1.0)


def test_vde_lead_detection():
    # 线性速度剖面 (末端钳位破坏纯偏移歧义) + trk 超前 1 帧 -> VDE = -0.1s
    # 注: 纯线性 ramp 对任意时移相关系数恒 1 (原理性歧义), tie-break 保守取 |lag| 最小
    ev = make_ev()
    frames = []
    for fi in range(10):
        vx = 1.0 + 0.4 * fi
        frames.append(([make_gt(id=1, type=1, x=fi * 1.0, vx=vx)],
                       [make_trk(id=10, x=fi * 1.0, vx=1.0 + 0.4 * min(fi + 1, 9))]))
    met = _run_seq(ev, frames)
    assert met['vde'] == pytest.approx(-0.1)


def test_duplicate_gt_id_single_ids_step():
    # 同帧重复 gt.id (异常标注): _gt_map 帧末统一提交, ids 单帧至多计 1 次
    ev = make_ev()
    gts = [make_gt(id=1, x=0, y=0), make_gt(id=1, x=5, y=0)]
    trks = [make_trk(id=10, x=0, y=0), make_trk(id=11, x=5, y=0)]
    ev._reset_seq()
    ev._seq_key = 's'
    ev._step(gts, trks)                        # 首帧 prev 不存在 -> ids=0
    ev._step(gts, trks)                        # 次帧 prev=11 (末条), 仅末条判切换
    met = ev._finalize(ev._acc, ev._cls_acc)
    assert met['ids'] <= 1
    assert met['tp'] == 4


def test_curve_no_gt_first_frame(tmp_path):
    # 首帧无 GT 有 FP: 曲线 MOTA 应为 nan (断点) 而非 -1e11, 不崩
    ev = make_ev(tmp=tmp_path, report=1)
    frames = [([], [make_trk(id=10, x=0, y=0)])]                # f0: 无 GT 有 FP
    frames += [([make_gt(id=1, type=1, x=fi * 0.5)], [])
               for fi in range(5)]                              # f1-5: 有 GT 无 trk
    ev.evaluate([(str(tmp_path / 'seqA'), frames)])
    assert (tmp_path / 'eval.default' / 'seqA_curve.png').exists()


# ---- 报告产物 ----

def test_report_and_curve_files(tmp_path):
    ev = make_ev(tmp=tmp_path, report=1)
    frames = [([make_gt(id=1, type=1, x=fi * 0.5, vx=1)],
               [make_trk(id=10, x=fi * 0.5 + 0.2, vx=1)]) for fi in range(6)]
    ev.evaluate([(str(tmp_path / 'seqA'), frames)])
    out_dir = tmp_path / 'eval.default'
    assert (out_dir / 'metrics_default.txt').exists()
    assert (out_dir / 'seqA_curve.png').exists()
    txt = (out_dir / 'metrics_default.txt').read_text(encoding='utf-8')
    assert 'DATASET (1 seqs)' in txt and 'seqA (6 frames)' in txt

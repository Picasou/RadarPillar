"""manager 单测: 重复航迹合并 / 出界删轨 / 近距建轨抑制 / ID 池满告警."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tracker.schemas import Obj, Trk, TrkHistory
from tracker.manager import TrackerManager


def make_manager(**kw):
    man = SimpleNamespace(
        birth_heat=3, death_heat=3, prob_output=80,
        birth_pos_std=0.5, birth_vel_std=5.0,
        merge_dist=kw.get('merge_dist', 1.0),
        birth_min_range=kw.get('birth_min_range', 2.0),
    )
    return TrackerManager(SimpleNamespace(MANAGER=man))


def make_trk(id=1, x=0.0, y=0.0, existence=50, lifetime=1.0, ms=0):
    return Trk(
        x_m=x, y_m=y, z_m=0, vx_mps=0, vy_mps=0, doppler_mps=0,
        ax_mps2=0, ay_mps2=0, heading_deg=0, yaw_rate_degs=0,
        id=id, width_m=2, height_m=1, length_m=4, lifetime_s=lifetime,
        x_std_m=0, y_std_m=0, z_std_m=0, vx_std_mps=0, vy_std_mps=0,
        ax_std_mps2=0, ay_std_mps2=0, xy_pos_cov=0, xy_vel_cov=0, xy_acc_cov=0,
        width_std_m=0, height_std_m=0, length_std_m=0,
        heading_std_deg=0, yaw_rate_std_degs=0,
        type=1, type_confi=0, obstacle_prob=1, existence_prob=existence,
        motion_status=1, measurement_status=ms, passable_status=0,
        rel_vel=0, rel_acc=0, cov=np.zeros((4, 4)), history=TrkHistory())


def make_obj(x=0.0, y=0.0):
    return Obj(id=0, x=x, y=y, length=4.0, width=2.0, type=1, score=0.9)


def test_merge_keeps_better_track():
    m = make_manager()
    a = make_trk(id=1, x=10.0, y=0.0, existence=90)
    b = make_trk(id=2, x=10.4, y=0.0, existence=50)      # 0.4m < 1.0 → 重复
    trks = [a, b]
    m._man_merge_trks(trks)
    assert [t.id for t in trks] == [1]


def test_merge_keeps_both_when_apart():
    m = make_manager()
    a = make_trk(id=1, x=10.0, y=0.0, existence=90)
    b = make_trk(id=2, x=30.0, y=0.0, existence=50)      # 20m 远 → 各留
    trks = [a, b]
    m._man_merge_trks(trks)
    assert [t.id for t in trks] == [1, 2]


def test_merge_tie_break_lifetime():
    m = make_manager()
    a = make_trk(id=1, x=0.0, y=0.0, existence=80, lifetime=0.5)
    b = make_trk(id=2, x=0.3, y=0.0, existence=80, lifetime=4.0)   # 平分 → 寿命长者留
    trks = [a, b]
    m._man_merge_trks(trks)
    assert [t.id for t in trks] == [2]


def test_bound_delete():
    m = make_manager()
    m.set_bound([0, -25, -3, 200, 25, 3])
    inside = make_trk(id=1, x=100.0, y=0.0)
    outside = make_trk(id=2, x=210.0, y=0.0)             # x 超 200 → 出界删
    trks = [inside, outside]
    m._man_delete_trks(trks)
    assert [t.id for t in trks] == [1]
    assert m.bound == (0, 200, -25, 25)


def test_birth_min_range_suppress():
    m = make_manager(birth_min_range=2.0)
    trks = []
    m._man_create_trks([make_obj(x=1.0, y=1.0), make_obj(x=5.0, y=0.0)], trks, 0.1)
    assert [t.id for t in trks] == [1]                   # 近距 (r≈1.41m) 不建轨


def test_id_pool_full_warns(capsys):
    m = make_manager()
    trks = [SimpleNamespace(id=i) for i in range(1, 101)]    # 池满
    n0 = len(trks)
    m._man_create_trks([make_obj(x=50.0, y=0.0)], trks, 0.1)
    assert len(trks) == n0
    assert 'ID 池' in capsys.readouterr().out


def test_birth_carries_z_height_score():
    m = make_manager(birth_min_range=0.0)
    obj = make_obj(x=50.0, y=0.0)
    obj.z, obj.height = 1.5, 1.8
    trks = []
    m._man_create_trks([obj], trks, 0.1)
    assert trks[0].z_m == 1.5 and trks[0].height_m == 1.8 and trks[0].det_score == 0.9

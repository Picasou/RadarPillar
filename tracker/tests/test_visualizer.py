"""验证 visualizer: cfg.VISUAL 解析/校验 + 角点几何 + PNG/GIF 落盘."""
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tracker.schemas import Cfg, FRAME, FrameProc, GT, GTs, Objs, Obj, PTs, Trk, TrkHistory, VDD
from tracker import visualizer
from tracker.visualizer import Visualizer, _box_corners

CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cfg', 'cfg.yaml')


def make_cfg(**visual_overrides):
    """
    测试配置: 真实 cfg.yaml 加载后按需覆写 VISUAL 字段
    """
    cfg = Cfg.get_cfg(CFG_PATH)
    for k, v in visual_overrides.items():
        setattr(cfg.VISUAL, k, v)
    return cfg


def make_trk(x, y, heading_deg=30.0, type_idx=1, tid=7, obstacle_prob=1):
    return Trk(
        x_m=x, y_m=y, z_m=0, vx_mps=1.0, vy_mps=0.0, doppler_mps=0,
        ax_mps2=0, ay_mps2=0, heading_deg=heading_deg, yaw_rate_degs=0,
        id=tid, width_m=2, height_m=1, length_m=4, lifetime_s=1.0,
        x_std_m=0, y_std_m=0, z_std_m=0, vx_std_mps=0, vy_std_mps=0,
        ax_std_mps2=0, ay_std_mps2=0, xy_pos_cov=0, xy_vel_cov=0, xy_acc_cov=0,
        width_std_m=0, height_std_m=0, length_std_m=0,
        heading_std_deg=0, yaw_rate_std_degs=0,
        type=type_idx, type_confi=80, obstacle_prob=obstacle_prob, existence_prob=90,
        motion_status=1, measurement_status=0, passable_status=0,
        rel_vel=0, rel_acc=0, cov=np.zeros((4, 4)), history=TrkHistory(),
    )


def make_frame(frame_id='000001', n_pts=20):
    rng = np.random.default_rng(0)
    pts = np.stack([rng.uniform(5, 40, n_pts), rng.uniform(-15, 15, n_pts),
                    rng.uniform(-1, 1, n_pts), rng.uniform(-10, 10, n_pts),
                    rng.uniform(-5, 5, n_pts), rng.uniform(-8, 8, n_pts),
                    np.zeros(n_pts)], axis=1).astype(np.float32)
    frame = FRAME(
        gts=GTs(num=1, Lst=[GT(x=20, y=5, z=0, vx=0, vy=0, length=4, width=2,
                                height=1.5, heading=0.5, type=1, isghost=0, ispassable=1)]),
        pts=PTs(num=0, Lst=[]),
        vdd=VDD(speed_ms=10.0, yaw_rate=0.0, gear=0),
        objs=Objs(num=1, Lst=[Obj(id=0, x=25, y=-5, vx=0, vy=0, doppler=0,
                                   length=4, width=2, heading=-0.3, type=2, score=0.9)]),
        frame_id=frame_id,
        proc=FrameProc(points=pts),
    )
    return frame


# ---------------- cfg 解析与校验 ----------------

def test_cfg_visual_parse_and_valid():
    cfg = Cfg.get_cfg(CFG_PATH)
    assert cfg.isvalid()
    assert cfg.VISUAL.enable == 1
    assert cfg.VISUAL.save == 2
    assert cfg.VISUAL.label == 1
    assert set(cfg.VISUAL.show) == {'points', 'tracks', 'objs', 'gts'}


def test_cfg_visual_save_out_of_range():
    cfg = make_cfg(save=4)
    with pytest.raises(ValueError):
        cfg.isvalid()


# ---------------- 角点几何 ----------------

def test_box_corners_axis_aligned():
    c = _box_corners(0, 0, 4, 2, 0.0)
    assert c.shape == (4, 2)
    assert np.allclose(c[:, 0].max(), 2.0)   # length/2 沿 x
    assert np.allclose(c[:, 1].max(), 1.0)   # width/2 沿 y


def test_box_corners_rotated_90():
    c = _box_corners(0, 0, 4, 2, np.pi / 2)
    assert np.allclose(c[:, 0].max(), 1.0)   # 转 90° 后 length 沿 y
    assert np.allclose(c[:, 1].max(), 2.0)


# ---------------- 出图落盘 ----------------

def test_run_png(tmp_path, monkeypatch):
    monkeypatch.setattr(visualizer, 'OUT_ROOT', tmp_path)
    viz = Visualizer(make_cfg(enable=1, save=2))
    viz.begin_seq('seqA')
    viz.run(make_frame(), [make_trk(20, 5)])
    png = tmp_path / 'seqA' / '000001.png'
    assert png.exists()


def test_run_enable_off(tmp_path, monkeypatch):
    monkeypatch.setattr(visualizer, 'OUT_ROOT', tmp_path)
    viz = Visualizer(make_cfg(enable=0, save=3))
    viz.begin_seq('seqA')
    viz.run(make_frame(), [make_trk(20, 5)])
    viz.on_seq_end()
    assert not (tmp_path / 'seqA').exists()
    assert not list(tmp_path.glob('*.gif'))


def test_gif(tmp_path, monkeypatch):
    monkeypatch.setattr(visualizer, 'OUT_ROOT', tmp_path)
    viz = Visualizer(make_cfg(enable=1, save=1))
    viz.begin_seq('seqB')
    viz.run(make_frame('000001'), [make_trk(20, 5)])
    viz.run(make_frame('000002'), [make_trk(21, 6)])
    viz.on_seq_end()
    gif = tmp_path / 'seqB.gif'
    assert gif.exists()
    with Image.open(gif) as im:
        assert im.n_frames == 2


def test_save0_no_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(visualizer, 'OUT_ROOT', tmp_path)
    viz = Visualizer(make_cfg(enable=1, save=0))
    viz.begin_seq('seqC')
    viz.run(make_frame(), [make_trk(20, 5)])
    viz.on_seq_end()
    assert list(tmp_path.iterdir()) == []

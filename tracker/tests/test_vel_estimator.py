"""速度量测链单测: TrkHistory 逐帧环形语义 / head-tail 窗选择 / 冷启动 / 常速收敛 / coast 缺口 / 加速跟踪 / 噪声抑制 / ego 对地口径 / meas_dim 口径 + dim=2 二次滤波."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tracker.schemas import HISTORY_LEN, Trk, TrkHistory, VDD
from tracker.updater import VelEstimator
from tracker.utils.common import c_trk_compensate
from tracker.filter import AlphaBetaFilter, KalmanFilter


def make_est(window_s=1.0, alpha=0.5, beta=0.5):
    return VelEstimator(window_s=window_s, alpha=alpha, beta=beta)


def make_trk(birth_xy=(0.0, 0.0), vx=0.0, vy=0.0):
    trk = Trk(
        x_m=birth_xy[0], y_m=birth_xy[1], z_m=0,
        vx_mps=vx, vy_mps=vy, doppler_mps=0,
        ax_mps2=0, ay_mps2=0,
        heading_deg=0, yaw_rate_degs=0,
        id=1, width_m=2, height_m=1, length_m=4, lifetime_s=0,
        x_std_m=0, y_std_m=0, z_std_m=0, vx_std_mps=0, vy_std_mps=0,
        ax_std_mps2=0, ay_std_mps2=0, xy_pos_cov=0, xy_vel_cov=0, xy_acc_cov=0,
        width_std_m=0, height_std_m=0, length_std_m=0,
        heading_std_deg=0, yaw_rate_std_degs=0,
        type=0, type_confi=0, obstacle_prob=0, existence_prob=0,
        motion_status=0, measurement_status=0, passable_status=0,
        rel_vel=0, rel_acc=0,
        cov=np.zeros((4, 4)),
        history=TrkHistory(),
    )
    trk.history.push(birth_xy[0], birth_xy[1], 0.0, 0.0, 0.0)   # 出生状态入史 (帧 0)
    return trk


def frame(trk, est, xy, dt=0.1, coast=False):
    """
    单帧驱动 (模拟生产顺序): 滤波后状态=xy → measure (history 至上一帧) → α-β 二次滤波 → 入史;
    coast 帧无新量测, 状态按当前速度外推 (预测态入史, 索引=帧)
    """
    if coast:
        xy = (trk.x_m + trk.vx_mps * dt, trk.y_m + trk.vy_mps * dt)
    trk.x_m, trk.y_m = xy
    v = est.measure(trk, dt) if not coast else None
    if v is not None:
        est.smooth(trk, v, dt)
    trk.history.push(trk.x_m, trk.y_m, trk.vx_mps, trk.vy_mps, trk.heading_deg)
    return v


# ---- TrkHistory 逐帧环形语义 ----

def test_history_push_within_capacity():
    h = TrkHistory()
    for i in range(5):
        h.push(float(i), -float(i), float(i) * 0.1, -float(i) * 0.1, float(i))
    assert h.tail_idx == 5
    assert h.x_history[4] == 4.0 and h.y_history[4] == -4.0
    assert h.vx_history[4] == pytest.approx(0.4) and h.heading_history[4] == 4.0


def test_history_push_shift_when_full():
    h = TrkHistory()
    for i in range(HISTORY_LEN):
        h.push(float(i), 0.0, 0.0, 0.0, 0.0)
    h.head_idx = 5
    h.push(float(HISTORY_LEN), 0.0, 0.0, 0.0, 0.0)              # 满容再压 → 整体左移
    assert h.tail_idx == HISTORY_LEN
    assert h.x_history[0] == 1.0                                # 最老点被挤掉
    assert h.x_history[-1] == float(HISTORY_LEN)                # 最新点居尾
    assert h.head_idx == 4                                      # 窗头随左移同步减一


# ---- head-tail 窗选择 (索引=帧) ----

def test_measure_window_selection():
    # 逐帧史 x(t)=t+t², window_s=0.3s@0.1s → 窗头=last-3 帧的点
    trk = make_trk(birth_xy=(0.0, 0.0))
    est = make_est(window_s=0.3)
    for k in range(1, 6):                                       # 帧 1..5, last=5
        trk.history.push(1.0 * k + k * k, 0.0, 0.0, 0.0, 0.0)
    v = est.measure(trk, 0.1)
    j0 = 5 - 3                                                  # round(0.3/0.1)=3
    assert trk.history.head_idx == j0
    xh, xt = 1.0 * j0 + j0 * j0, 1.0 * 5 + 25.0
    assert v == pytest.approx(((xt - xh) / 0.3, 0.0))


def test_measure_window_floor_two_points():
    # window_s < 1 帧 → 保底 2 点: 窗头钳到次老点
    trk = make_trk(birth_xy=(0.0, 0.0))
    est = make_est(window_s=0.05)
    for k in range(1, 4):
        trk.history.push(1.0 * k, 0.0, 0.0, 0.0, 0.0)           # x=t·10 m/s (k·0.1s → 1.0·k)
    v = est.measure(trk, 0.1)
    assert trk.history.head_idx == 2
    assert v == pytest.approx((10.0, 0.0))


def test_measure_single_point_returns_none():
    trk = make_trk(birth_xy=(0.0, 0.0))                         # 仅出生点
    assert make_est().measure(trk, 0.1) is None


# ---- 冷启动 / 常速收敛 / coast 缺口 / 加速滞后 / 噪声抑制 ----

def test_two_point_cold_start():
    trk = make_trk(birth_xy=(0.0, 0.0))
    est = make_est()
    v1 = frame(trk, est, (1.0, -0.2))                           # 首 frame: 史中仅出生点 → None
    assert v1 is None
    v2 = frame(trk, est, (2.0, -0.4))                           # 次 frame: 出生→上帧差分
    assert v2 == pytest.approx((10.0, -2.0))
    assert (trk.vx_mps, trk.vy_mps) == pytest.approx((10.0, -2.0))   # 冷启动直写 trk 速度
    assert trk.vel_init is True and trk.ax_mps2 == 0.0


def test_constant_velocity_converges():
    trk = make_trk()
    est = make_est()
    v_true = np.array([10.0, 2.0])
    p = np.zeros(2)
    for _ in range(40):
        p = p + v_true * 0.1
        frame(trk, est, (p[0], p[1]))
    assert np.hypot(trk.vx_mps - 10.0, trk.vy_mps - 2.0) < 0.1


def test_coast_gap_not_skewed():
    # 混合量测帧与 coast 帧 (coast 帧预测态入史, 窗跨缺口仍可差分); 真实位置逐帧恒速前进
    trk = make_trk()
    est = make_est()
    v_true = 8.0
    t = 0.0
    for meas in [True, True, False, False, True, False, True, True]:
        t += 0.1
        if meas:
            frame(trk, est, (v_true * t, 0.0))
        else:
            frame(trk, est, None, coast=True)                   # 状态按当前速度外推
    assert np.hypot(trk.vx_mps - 8.0, trk.vy_mps - 0.0) < 0.3   # 缺口不歪速度


def test_acceleration_lag_bounded():
    trk = make_trk()
    est = make_est(window_s=1.0)
    a = 1.0
    p, v_true, lag = 0.0, 0.0, []
    for _ in range(60):
        v_true += a * 0.1
        p += v_true * 0.1
        frame(trk, est, (p, 0.0))
        lag.append(v_true - trk.vx_mps)
    assert max(lag[-10:]) < 0.5 * a * 1.0 + 0.5                 # 稳态滞后 ~ a·T/2 量级


def test_noise_suppression():
    rng = np.random.default_rng(42)
    v_true = np.array([12.0, 0.0])
    sigma = 0.3
    trk = make_trk()
    est = make_est()
    p = np.zeros(2)
    errs = []
    for _ in range(100):
        p = p + v_true * 0.1
        z = p + rng.normal(0.0, sigma, 2)
        frame(trk, est, (z[0], z[1]))
        errs.append(np.hypot(trk.vx_mps - 12.0, trk.vy_mps - 0.0))
    assert float(np.median(errs)) < 1.5                         # 相邻帧差分口径此噪声 ~4.2 m/s


# ---- 运动自车 ego 补偿 (差分口径=对地速度, history 随 c_trk_compensate 整体补偿) ----

def test_ego_motion_world_static_target():
    trk = make_trk(birth_xy=(50.0, 2.0))
    est = make_est()
    vdd = VDD(speed_ms=5.0, yaw_rate=0.0, gear=1)               # 自车 5 m/s 前进
    x = 50.0
    frame(trk, est, (x, 2.0))
    for _ in range(25):
        c_trk_compensate(trk, vdd, 0.1)                         # history + trk 状态整体推到当前 ego 系
        x -= 0.5                                                # 世界系静止目标在 ego 系观测后退
        frame(trk, est, (x, 2.0))
    assert trk.vx_mps == pytest.approx(0.0, abs=0.15)           # 对地静止 → v≈0 (非相对口径 -5)
    assert trk.vy_mps == pytest.approx(0.0, abs=0.15)


def test_ego_motion_world_moving_target():
    trk = make_trk(birth_xy=(50.0, 0.0))
    est = make_est()
    vdd = VDD(speed_ms=5.0, yaw_rate=0.0, gear=1)
    x = 50.0
    frame(trk, est, (x, 0.0))
    for _ in range(25):
        c_trk_compensate(trk, vdd, 0.1)
        x -= 0.2                                                # 对地 3 m/s, ego 5 m/s → 观测每帧 -0.2
        frame(trk, est, (x, 0.0))
    assert trk.vx_mps == pytest.approx(3.0, abs=0.15)           # 输出对地速度 3 (非相对口径 -2)


# ---- meas_dim 口径 + dim=2 速度二次滤波 ----

def test_filter_meas_dim():
    assert AlphaBetaFilter(alpha=0.85, beta=0.2).meas_dim == 2
    assert KalmanFilter(dim=2, q_acc=1.0, r=np.eye(2)).meas_dim == 2
    assert KalmanFilter(dim=4, q_acc=1.0, r=np.eye(4)).meas_dim == 4


def test_dim2_velocity_secondary_filter():
    # KF dim=2 只吃位置量测; KF 更新后 smooth 以 v_meas 为量测二次滤波速度 (trk.vx 收敛对地真值)
    kf = KalmanFilter(dim=2, q_acc=1.0, r=np.eye(2) * 0.5)
    est = make_est()
    trk = make_trk()
    trk.cov = np.diag([0.25, 0.25, 25.0, 25.0])                 # 出生协方差非零 (cov=0 时 KF 增益恒 0)
    v_true = np.array([6.0, -1.0])
    p = np.zeros(2)
    for _ in range(50):
        kf._predict(trk, 0.1)                                   # 生产同序: 每帧先时间更新 (x+=vx·dt)
        p = p + v_true * 0.1
        v = est.measure(trk, 0.1)
        kf.update(trk, SimpleNamespace(x=p[0], y=p[1], vx=0.0, vy=0.0), 0.1)   # 模拟滤波位置更新
        if v is not None:                                       # 生产同款 guard: 不可差分帧跳过
            est.smooth(trk, v, 0.1)                             # KF 后速度二次滤波
        trk.history.push(trk.x_m, trk.y_m, trk.vx_mps, trk.vy_mps, trk.heading_deg)   # 滤波后状态入史
    assert np.hypot(trk.vx_mps - 6.0, trk.vy_mps + 1.0) < 0.3

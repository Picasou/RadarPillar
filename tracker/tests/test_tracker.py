"""验证 utils.common 的核心逻辑: trk 状态补偿 + 跨帧反向补偿。"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import VDD, FRAME, Trk, TrkState, TrkHistory, PTs, GTs, Objs
from utils.common import compensate_trks, compensate_frame_forward


def make_vdd(speed_ms=10.0, yaw_rate=0.0, gear=0):
    return VDD(speed_ms=speed_ms, yaw_rate=yaw_rate, gear=gear)


def make_trk(x, y, vx, vy, h=0, history_states=None):
    h_obj = TrkHistory()
    if history_states:
        h_obj.states = history_states
    return Trk(
        x_m=x, y_m=y, z_m=0,
        vx_mps=vx, vy_mps=vy,
        ax_mps2=0, ay_mps2=0,
        heading_deg=h, yaw_rate_degs=0,
        id=1, width_m=2, height_m=1, length_m=4, lifetime_s=0,
        x_std_m=0, y_std_m=0, z_std_m=0, vx_std_mps=0, vy_std_mps=0,
        ax_std_mps2=0, ay_std_mps2=0, xy_pos_cov=0, xy_vel_cov=0, xy_acc_cov=0,
        width_std_m=0, height_std_m=0, length_std_m=0,
        heading_std_deg=0, yaw_rate_std_degs=0,
        type=0, type_confi=0, obstacle_prob=0, existence_prob=0,
        motion_status=0, measurement_status=0, passable_status=0,
        rel_vel=0, rel_acc=0,
        cov=np.zeros((4, 4)),
        history=h_obj,
    )


def make_empty_frame():
    return FRAME(
        gts=GTs(num=0, Lst=[]),
        pts=PTs(num=0, Lst=[]),
        vdd=make_vdd(),
        objs=Objs(num=0, Lst=[]),
    )


def test_straight_compensation():
    trk = make_trk(x=100, y=50, vx=10, vy=5, h=90)
    compensate_trks([trk], make_vdd(speed_ms=10.0, yaw_rate=0.0), 0.1)
    assert trk.x_m == 99, f"x_m=99, got {trk.x_m}"
    assert trk.y_m == 50
    assert trk.vx_mps == 10
    assert trk.vy_mps == 5
    assert trk.heading_deg == 90
    print(f"  [PASS] straight: x 100->{trk.x_m}")


def test_turning_compensation():
    trk = make_trk(x=100, y=50, vx=10, vy=5, h=90)
    speed, yr, cycle_s = 10.0, 0.1, 0.1
    compensate_trks([trk], make_vdd(speed_ms=speed, yaw_rate=yr), cycle_s)
    dx, wt = speed * cycle_s, yr * cycle_s
    cos_wt, sin_wt = np.cos(wt), np.sin(wt)
    ex  = (100 - dx) * cos_wt + 50 * sin_wt
    ey  = -(100 - dx) * sin_wt + 50 * cos_wt
    evx = 10 * cos_wt + 5 * sin_wt
    evy = -10 * sin_wt + 5 * cos_wt
    eh  = int(round((90 + np.degrees(wt)) % 360))
    assert abs(trk.x_m - ex) <= 1
    assert abs(trk.y_m - ey) <= 1
    assert abs(trk.vx_mps - evx) <= 1
    assert abs(trk.vy_mps - evy) <= 1
    assert trk.heading_deg == eh
    print(f"  [PASS] turning: x 100->{trk.x_m}, y 50->{trk.y_m}, h 90->{trk.heading_deg}")


def test_empty_trks():
    compensate_trks([], make_vdd(), 0.1)
    print("  [PASS] empty trks list safe")


def test_zero_cycle():
    trk = make_trk(x=100, y=50, vx=10, vy=5, h=90)
    compensate_trks([trk], make_vdd(speed_ms=10.0, yaw_rate=0.1), 0.0)
    assert trk.x_m == 100 and trk.y_m == 50
    print("  [PASS] cycle_s=0 is no-op")


def test_compensate_frame_forward_straight():
    cycle_s = 0.1
    xy = np.array([[30.0, 0.0], [40.0, 5.0]], dtype=np.float32)
    intermediates = [make_empty_frame() for _ in range(3)]
    for f in intermediates:
        f.vdd = make_vdd(speed_ms=10.0, yaw_rate=0.0)
    xy2 = compensate_frame_forward(xy.copy(), intermediates, cycle_s)
    expected = xy - np.array([[3.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    assert np.allclose(xy2, expected)
    print(f"  [PASS] 3-step straight: x {xy[:,0]} -> {xy2[:,0]}")


def test_compensate_frame_forward_turning():
    cycle_s, yr, speed = 0.1, 0.1, 5.0
    xy = np.array([[10.0, 0.0]], dtype=np.float32)
    intermediates = [make_empty_frame() for _ in range(2)]
    for f in intermediates:
        f.vdd = make_vdd(speed_ms=speed, yaw_rate=yr)

    def step(x, y, dx, wt):
        c, s = np.cos(wt), np.sin(wt)
        return (x - dx) * c + y * s, -(x - dx) * s + y * c
    dx, wt = speed * cycle_s, yr * cycle_s
    x1, y1 = step(10.0, 0.0, dx, wt)
    ex, ey = step(x1, y1, dx, wt)

    xy2 = compensate_frame_forward(xy.copy(), intermediates, cycle_s)
    assert abs(xy2[0, 0] - ex) < 1e-4
    assert abs(xy2[0, 1] - ey) < 1e-4
    print(f"  [PASS] 2-step turning: ({xy[0,0]},{xy[0,1]}) -> ({xy2[0,0]:.4f},{xy2[0,1]:.4f})")


def test_history_cumulative_straight():
    trk = make_trk(x=100, y=50, vx=10, vy=5, h=90)
    compensate_trks([trk], make_vdd(speed_ms=10.0, yaw_rate=0.0), 0.1)
    assert trk.history.wt   == 0.0
    assert abs(trk.history.dx   - 1.0) < 1e-6
    assert trk.history.dy   == 0.0
    assert abs(trk.history.dist - 1.0) < 1e-6
    print(f"  [PASS] history cumulative (straight): dx={trk.history.dx}, dist={trk.history.dist}")


def test_history_cumulative_turning():
    trk = make_trk(x=100, y=50, vx=10, vy=5, h=90)
    speed, yr, cycle_s = 10.0, 0.1, 0.1
    compensate_trks([trk], make_vdd(speed_ms=speed, yaw_rate=yr), cycle_s)
    assert abs(trk.history.wt   - yr * cycle_s) < 1e-6
    assert abs(trk.history.dx   - speed * cycle_s) < 1e-6
    assert abs(trk.history.dist - speed * cycle_s) < 1e-6
    print(f"  [PASS] history cumulative (turning): wt={trk.history.wt:.4f}, dx={trk.history.dx:.4f}")


def test_history_states_compensation():
    from dataclasses import replace
    original = [
        TrkState(x=20.0, y=3.0, vx=8.0, vy=2.0, heading=85.0),
        TrkState(x=15.0, y=1.0, vx=7.0, vy=1.5, heading=88.0),
    ]
    snapshot = [replace(s) for s in original]
    trk = make_trk(x=100, y=50, vx=10, vy=5, h=90, history_states=original)
    speed, yr, cycle_s = 10.0, 0.1, 0.1
    compensate_trks([trk], make_vdd(speed_ms=speed, yaw_rate=yr), cycle_s)

    dx, wt = speed * cycle_s, yr * cycle_s
    cos_wt, sin_wt = np.cos(wt), np.sin(wt)
    heading_delta = int(round(np.degrees(wt)))

    for orig, s in zip(snapshot, trk.history.states):
        ex  = (orig.x - dx) * cos_wt + orig.y * sin_wt
        ey  = -(orig.x - dx) * sin_wt + orig.y * cos_wt
        evx = orig.vx * cos_wt + orig.vy * sin_wt
        evy = -orig.vx * sin_wt + orig.vy * cos_wt
        eh  = (orig.heading + heading_delta) % 360
        assert abs(s.x - ex) < 1e-4, f"state x: {s.x} vs {ex}"
        assert abs(s.y - ey) < 1e-4, f"state y: {s.y} vs {ey}"
        assert abs(s.vx - evx) < 1e-4, f"state vx: {s.vx} vs {evx}"
        assert abs(s.vy - evy) < 1e-4, f"state vy: {s.vy} vs {evy}"
        assert abs(s.heading - eh) < 1e-4, f"state h: {s.heading} vs {eh}"
    print(f"  [PASS] history.states 5D compensated ({len(snapshot)} states)")


if __name__ == '__main__':
    print("=== [utils.common] trk 状态补偿 + 跨帧补偿 验证 ===")
    print("\n--- compensate_trks ---")
    test_straight_compensation()
    test_turning_compensation()
    test_empty_trks()
    test_zero_cycle()
    print("\n--- compensate_frame_forward ---")
    test_compensate_frame_forward_straight()
    test_compensate_frame_forward_turning()
    print("\n--- compensate_trks history ---")
    test_history_cumulative_straight()
    test_history_cumulative_turning()
    test_history_states_compensation()
    print("\n=== ALL PASS ===")

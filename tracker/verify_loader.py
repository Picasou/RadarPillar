# -*-coding:utf-8-*-
"""端到端验证: 实例化 Loader, 对比其 schemas 输出与 S4 warden_a1 加载结果的物理值。
逐帧逐主变量对比。用法: python verify_loader.py [数据父目录]
"""
import sys
import os
from types import SimpleNamespace

ROOT = r'C:\Users\Yangf\Desktop\RadarPillar'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tracker'))
sys.path.insert(0, r'C:\Users\Yangf\Desktop\S4\warden-swa-S4')

import numpy as np

# ---- S4 真值加载 (warden_a1.py 实际使用的方式) ----
from lib.file_rpk import get_a1_bin_path, load_data_from_bin_a1

# ---- 新 Loader ----
from loader import Loader
from schemas import Cfg

# 数据路径: 命令行传入 > 默认
DATA_PARENT = (sys.argv[1] if len(sys.argv) > 1
               else r'C:\Users\Yangf\Desktop\IF_DATA\20260423_140057_562')


def build_dummy_cfg():
    """构造最小 Cfg (Loader.getframes 不依赖 cfg, 仅 fallback 用)。"""
    vds = SimpleNamespace(wheelbase_m=4.5, x_pos_m=0.0, y_pos_m=0.0, z_pos_m=0.0, cycle_s=0.1)
    run = SimpleNamespace(vds=vds)
    cfg = SimpleNamespace(RUN=run)
    return cfg


def _close(err, tol=1e-4):
    """float32 容差判定: S4 c_float(32位) vs schemas Python float(64位) 的固有 ULP 差异。"""
    return err <= tol


def compare_vds(loader, s4_vds):
    print("\n=== [VDS] 静态参数 对比 ===")
    new_vds = loader.getvds(DATA_PARENT)
    diffs = {
        'wheelbase_m': abs(s4_vds.wheelbase_m - new_vds.wheelbase_m),
        'x_pos_m':     abs(s4_vds.xpos     - new_vds.x_pos_m),
        'y_pos_m':     abs(s4_vds.ypos     - new_vds.y_pos_m),
        'z_pos_m':     abs(s4_vds.zpos     - new_vds.z_pos_m),
        'rotation_rad':abs(s4_vds.rotation - new_vds.rotation_rad),
        'orientation': abs(s4_vds.orientation - new_vds.oritation),
    }
    new_attr = {'x_pos_m': 'x_pos_m', 'y_pos_m': 'y_pos_m', 'z_pos_m': 'z_pos_m',
                'rotation_rad': 'rotation_rad', 'orientation': 'oritation',
                'wheelbase_m': 'wheelbase_m'}
    for k, v in diffs.items():
        s4v = getattr(s4_vds, _s4key(k))
        newv = getattr(new_vds, new_attr[k])
        print(f"  {k:14s}: S4={s4v!r}, New={newv!r}, err={v:.6e}")
    ok = all(_close(v) for v in diffs.values())
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return ok


def _s4key(k):
    return {'x_pos_m': 'xpos', 'y_pos_m': 'ypos', 'z_pos_m': 'zpos',
            'rotation_rad': 'rotation'}.get(k, k)


def compare_vdd(frames, s4_vdd):
    print("\n=== [VDD] 动态参数 逐帧对比 ===")
    n = min(len(s4_vdd), len(frames))
    print(f"帧数: S4={len(s4_vdd)}, New={len(frames)}, 对比 {n}")
    if n == 0:
        print("RESULT: SKIP (无数据)")
        return True
    max_e_speed, max_e_yaw = 0.0, 0.0
    max_i_speed, max_i_yaw = -1, -1
    for i in range(n):
        es = abs(s4_vdd[i].hostVelocity_mps - frames[i].vdd.speed_ms)
        ey = abs(s4_vdd[i].vehicleYawRate_radps - frames[i].vdd.yaw_rate)
        if es > max_e_speed: max_e_speed, max_i_speed = es, i
        if ey > max_e_yaw:   max_e_yaw,   max_i_yaw   = ey, i
    print(f"  首帧 [0]: speed S4={s4_vdd[0].hostVelocity_mps:.6f} vs New={frames[0].vdd.speed_ms:.6f}, "
          f"yaw S4={s4_vdd[0].vehicleYawRate_radps:.6f} vs New={frames[0].vdd.yaw_rate:.6f}")
    print(f"  最大误差: speed={max_e_speed:.6e} @frame{max_i_speed}, "
          f"yaw={max_e_yaw:.6e} @frame{max_i_yaw}")
    ok = _close(max_e_speed) and _close(max_e_yaw)
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return ok


def compare_pts(frames, s4_det):
    print("\n=== [PTs] 点云 逐帧×逐点 对比 ===")
    n = min(len(s4_det), len(frames))
    print(f"帧数: S4={len(s4_det)}, New={len(frames)}, 对比 {n}")
    num_mm = 0
    max_e = {'range': 0.0, 'ang': 0.0, 'elv': 0.0, 'dpl': 0.0}
    max_i = {'range': -1, 'ang': -1, 'elv': -1, 'dpl': -1}
    for i in range(n):
        if s4_det[i].num != frames[i].pts.num:
            num_mm += 1
            continue
        for j in range(s4_det[i].num):
            s = s4_det[i].dets[j]
            t = frames[i].pts.Lst[j]
            er = abs(s.range_m - t.range_m);   ea = abs(s.ang_rad - t.ang_rad)
            ee = abs(s.elv_rad - t.elv_rad);   ed = abs(s.doppler_mps - t.doppler_mps)
            if er > max_e['range']: max_e['range'], max_i['range'] = er, i
            if ea > max_e['ang']:   max_e['ang'],   max_i['ang']   = ea, i
            if ee > max_e['elv']:   max_e['elv'],   max_i['elv']   = ee, i
            if ed > max_e['dpl']:   max_e['dpl'],   max_i['dpl']   = ed, i
    print(f"  点数不一致帧数: {num_mm}/{n}")
    print(f"  首帧 [0] 第0点: range S4={s4_det[0].dets[0].range_m:.6f} vs New={frames[0].pts.Lst[0].range_m:.6f}")
    print(f"  最大误差: range={max_e['range']:.6e}@f{max_i['range']}, "
          f"ang={max_e['ang']:.6e}@f{max_i['ang']}, "
          f"elv={max_e['elv']:.6e}@f{max_i['elv']}, "
          f"dpl={max_e['dpl']:.6e}@f{max_i['dpl']}")
    ok = num_mm == 0 and all(_close(v) for v in max_e.values())
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return ok


def compare_trk(s4_trk):
    print("\n=== [Trks] 跟踪目标 逐帧×逐目标 对比 ===")
    loader = Loader(build_dummy_cfg())
    trks_list = loader._load_TKs(os.path.join(DATA_PARENT, 'radar.default'))
    n = min(len(s4_trk), len(trks_list))
    print(f"帧数: S4={len(s4_trk)}, New={len(trks_list)}, 对比 {n}")
    num_mm = 0
    keys = ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'len', 'wid', 'hdg']
    max_e = {k: 0.0 for k in keys}
    max_i = {k: -1 for k in keys}
    for i in range(n):
        if s4_trk[i].numEntries != trks_list[i].num:
            num_mm += 1
            continue
        for j in range(s4_trk[i].numEntries):
            s = s4_trk[i].tracks[j]
            t = trks_list[i].Lst[j]
            vals = {
                'x':   abs(s.centerPosX - t.x_m),
                'y':   abs(s.centerPosY - t.y_m),
                'vx':  abs(s.vx        - t.vx_mps),
                'vy':  abs(s.vy        - t.vy_mps),
                'ax':  abs(s.ax        - t.ax_mps2),
                'ay':  abs(s.ay        - t.ay_mps2),
                'len': abs(s.length    - t.length_m),
                'wid': abs(s.width     - t.width_m),
                'hdg': abs(s.heading_rad - np.radians(t.heading_deg)),
            }
            for k in keys:
                if vals[k] > max_e[k]: max_e[k], max_i[k] = vals[k], i
    print(f"  目标数不一致帧数: {num_mm}/{n}")
    if n > 0 and s4_trk[0].numEntries > 0:
        s, t = s4_trk[0].tracks[0], trks_list[0].Lst[0]
        print(f"  首帧 [0] 第0目标: x S4={s.centerPosX:.4f} vs New={t.x_m:.4f}, "
              f"vx S4={s.vx:.4f} vs New={t.vx_mps:.4f}")
    print("  最大误差:")
    for k in keys:
        print(f"    {k:4s}: {max_e[k]:.6e} @frame{max_i[k]}")
    ok = num_mm == 0 and all(_close(max_e[k]) for k in keys)
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return ok


if __name__ == '__main__':
    print("=" * 64)
    print("端到端验证 (实例化 Loader, 对比物理值)")
    print(f"数据: {DATA_PARENT}")
    print("=" * 64)

    bin_path = get_a1_bin_path(DATA_PARENT)
    print(f"S4 定位 bin 目录: {bin_path}")
    s4_vds, s4_vdd, s4_det, s4_trk = load_data_from_bin_a1(bin_path)

    loader = Loader(build_dummy_cfg())
    frames = loader.getframes(DATA_PARENT).Lst

    ok_vds = compare_vds(loader, s4_vds)
    ok_vdd = compare_vdd(frames, s4_vdd)
    ok_pts = compare_pts(frames, s4_det)
    ok_trk = compare_trk(s4_trk)

    print("\n" + "=" * 64)
    print(f"汇总: VDS={'PASS' if ok_vds else 'FAIL'}  VDD={'PASS' if ok_vdd else 'FAIL'}  "
          f"PTs={'PASS' if ok_pts else 'FAIL'}  Trks={'PASS' if ok_trk else 'FAIL'}")
    print("=" * 64)
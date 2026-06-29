# -*-coding:utf-8-*-
"""端到端验证: 实例化 Loader, 全字段对比其 schemas 输出与 S4 warden_a1 加载结果。

用法: python tracker/tests/test_loader.py [数据父目录]
"""
import sys
import os
from types import SimpleNamespace

ROOT = r'C:\Users\Yangf\Desktop\RadarPillar'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tracker'))
sys.path.insert(0, r'C:\Users\Yangf\Desktop\S4\warden-swa-S4')

import numpy as np

from lib.file_rpk import get_a1_bin_path, load_data_from_bin_a1
from loader import Loader
from schemas import Cfg

DATA_PARENT = (sys.argv[1] if len(sys.argv) > 1
               else r'C:\Users\Yangf\Desktop\IF_DATA\20260423_140057_562')


def build_dummy_cfg():
    vds = SimpleNamespace(wheelbase_m=4.5, x_pos_m=0.0, y_pos_m=0.0, z_pos_m=0.0, cycle_s=0.1)
    run = SimpleNamespace(vds=vds)
    return SimpleNamespace(RUN=run)


def _close(err, tol=1e-4):
    return err <= tol


# ==================================================
# VDS: 全字段对比 (6 个)
# ==================================================
def compare_vds(loader, s4_vds):
    print("\n=== [VDS] 静态参数 全字段对比 ===")
    new_vds = loader.getvds(DATA_PARENT)
    fields = [
        ('wheelbase_m',  s4_vds.wheelbase_m,  new_vds.wheelbase_m),
        ('x_pos_m',      s4_vds.xpos,          new_vds.x_pos_m),
        ('y_pos_m',      s4_vds.ypos,          new_vds.y_pos_m),
        ('z_pos_m',      s4_vds.zpos,          new_vds.z_pos_m),
        ('rotation_rad', s4_vds.rotation,      new_vds.rotation_rad),
        ('orientation',  s4_vds.orientation,   new_vds.oritation),
    ]
    fail = []
    for name, s, n in fields:
        err = abs(s - n)
        marker = '✓' if _close(err) else '✗'
        print(f"  {marker} {name:14s}: S4={s!r}, New={n!r}, err={err:.6e}")
        if not _close(err):
            fail.append(name)
    print("RESULT: PASS" if not fail else f"RESULT: FAIL ({fail})")
    return not fail


# ==================================================
# VDD: 全字段对比 (3 个 × N 帧, 从 B_2021 原始读取以保留 driveGearEngaged)
# ==================================================
def compare_vdd(frames, s4_vdd):
    from lib.rw_struct import struct_read, B_2021
    print("\n=== [VDD] 动态参数 全字段×逐帧 对比 ===")
    radar_dir = os.path.join(DATA_PARENT, 'radar.default')
    s4_vdd_raw = struct_read(os.path.join(radar_dir, '2021.00000.bin'), B_2021)
    n = min(len(s4_vdd_raw), len(frames))
    print(f"帧数: S4={len(s4_vdd_raw)}, New={len(frames)}, 对比 {n}")
    fields = [
        ('speed_ms', 'hostVelocity_mps',     'speed_ms'),
        ('yaw_rate',  'vehicleYawRate_radps', 'yaw_rate'),
        ('gear',      'driveGearEngaged',     'gear'),
    ]
    max_e = {f[0]: (0.0, -1) for f in fields}
    for i in range(n):
        for name, s4_attr, new_attr in fields:
            err = abs(getattr(s4_vdd_raw[i], s4_attr) - getattr(frames[i].vdd, new_attr))
            if err > max_e[name][0]:
                max_e[name] = (err, i)
    fail = []
    for name, _, _ in fields:
        err, fi = max_e[name]
        if name == 'gear':
            ok = err == 0
        else:
            ok = _close(err)
        marker = '✓' if ok else '✗'
        print(f"  {marker} {name:10s}: max_err={err:.6e} @frame{fi}")
        if not ok:
            fail.append(name)
    print("RESULT: PASS" if not fail else f"RESULT: FAIL ({fail})")
    return not fail


# ==================================================
# PTs: 全字段对比 (13 个 × N 帧 × N 点)
# ==================================================
def compare_pts(frames, s4_det):
    print("\n=== [PTs] 点云 全字段×逐帧×逐点 对比 ===")
    n = min(len(s4_det), len(frames))
    print(f"帧数: S4={len(s4_det)}, New={len(frames)}, 对比 {n}")
    fields = [
        ('beam',                'beam',                    'beam',                False),
        ('extra_cnt',           'extra_cnt',               'extra_cnt',           False),
        ('exist_confidence',    'exist_confidence',        'exist_confidence',    False),
        ('doppler_anti_amb',    'doppler_anti_amb_confi',  'doppler_anti_amb_confi', False),
        ('id',                  'id',                      'id',                  False),
        ('flags',               'flags',                   'flags',               False),
        ('rcs',                 'rcs',                     'rcs',                 False),
        ('snr',                 'snr',                     'snr',                 False),
        ('frame',               'frame',                   'frame',               False),
        ('range_m',             'range_m',                 'range_m',             True),
        ('ang_rad',             'ang_rad',                 'ang_rad',             True),
        ('elv_rad',             'elv_rad',                 'elv_rad',             True),
        ('doppler_mps',         'doppler_mps',             'doppler_mps',         True),
    ]
    max_e = {f[0]: (0.0, -1, -1) for f in fields}
    num_mm = 0
    for i in range(n):
        if s4_det[i].num != frames[i].pts.num:
            num_mm += 1
            continue
        for j in range(s4_det[i].num):
            s = s4_det[i].dets[j]
            t = frames[i].pts.Lst[j]
            for name, s4_attr, new_attr, is_float in fields:
                err = abs(getattr(s, s4_attr) - getattr(t, new_attr))
                cur = max_e[name]
                if err > cur[0]:
                    max_e[name] = (err, i, j)
    fail = []
    if num_mm:
        fail.append(f'pts_count_mismatch({num_mm})')
    print(f"  点数不一致帧数: {num_mm}/{n}")
    print("  全字段最大误差:")
    for name, s4_attr, new_attr, is_float in fields:
        err, fi, fj = max_e[name]
        ok = (not is_float and err == 0) or (is_float and _close(err))
        marker = '✓' if ok else '✗'
        print(f"    {marker} {name:18s}: max_err={err:.6e} @f{fi}.p{fj}")
        if not ok:
            fail.append(name)
    print("RESULT: PASS" if not fail else f"RESULT: FAIL ({fail})")
    return not fail


# ==================================================
# Objs: 全字段对比 (loader 已做 ego 补偿, 物理量不强制 byte-equal)
# ==================================================
def compare_objs(s4_trk):
    print("\n=== [Objs] 原始目标 全字段×逐帧×逐目标 对比 (loader 已做 ego 补偿) ===")
    loader = Loader(build_dummy_cfg())
    objs_list = loader._load_objs(os.path.join(DATA_PARENT, 'radar.default'))
    n = min(len(s4_trk), len(objs_list))
    print(f"帧数: S4={len(s4_trk)}, New={len(objs_list)}, 对比 {n}")
    fields = [
        ('id',         'id',             'id',           False, None),
        ('x',          'centerPosX',     'x',            True,  None),
        ('y',          'centerPosY',     'y',            True,  None),
        ('vx',         'vx',             'vx',           True,  None),
        ('vy',         'vy',             'vy',           True,  None),
        ('length',     'length',         'length',       True,  None),
        ('width',      'width',          'width',        True,  None),
        ('heading',    'heading_rad',    'heading',      True,  'rad'),
        ('type',       'objTypClass',    'type',         False, None),
        ('isghost',    None,             'isghost',      False, None),
        ('ispassable', None,             'ispassable',   False, None),
    ]
    max_e = {f[0]: (0.0, -1, -1) for f in fields}
    num_mm = 0
    for i in range(n):
        if s4_trk[i].numEntries != objs_list[i].num:
            num_mm += 1
            continue
        for j in range(s4_trk[i].numEntries):
            s = s4_trk[i].tracks[j]
            t = objs_list[i].Lst[j]
            for name, s4_attr, new_attr, is_float, s4_unit in fields:
                if s4_attr is None:
                    continue
                sv = getattr(s, s4_attr)
                if s4_unit == 'rad':
                    sv = np.degrees(sv)  # S4 是弧度, 转成度与 loader 比
                err = abs(sv - getattr(t, new_attr))
                cur = max_e[name]
                if err > cur[0]:
                    max_e[name] = (err, i, j)
    fail = []
    if num_mm:
        fail.append(f'objs_count_mismatch({num_mm})')
    print(f"  目标数不一致帧数: {num_mm}/{n}")
    print("  全字段最大偏差 (含 ego 补偿, 物理量限 <5m):")
    for name, s4_attr, new_attr, is_float, s4_unit in fields:
        if s4_attr is None:
            print(f"    - {name:11s}: S4 无此字段, loader 默认 0")
            continue
        err, fi, fj = max_e[name]
        if not is_float:
            ok = err == 0
            limit_str = '==0'
        else:
            ok = err < 5.0
            limit_str = '<5m'
        marker = '✓' if ok else '✗'
        print(f"    {marker} {name:11s}: max_diff={err:.4f} @f{fi}.o{fj} (限 {limit_str})")
        if not ok:
            fail.append(name)
    print("RESULT: PASS" if not fail else f"RESULT: FAIL ({fail})")
    return not fail


if __name__ == '__main__':
    print("=" * 64)
    print(f"端到端验证 (全字段) 数据: {DATA_PARENT}")
    print("=" * 64)
    bin_path = get_a1_bin_path(DATA_PARENT)
    print(f"S4 定位: {bin_path}")
    s4_vds, s4_vdd, s4_det, s4_trk = load_data_from_bin_a1(bin_path)

    loader = Loader(build_dummy_cfg())
    frames = loader.getframes(DATA_PARENT).Lst

    ok = []
    ok.append(('VDS',  compare_vds(loader, s4_vds)))
    ok.append(('VDD',  compare_vdd(frames, s4_vdd)))
    ok.append(('PTs',  compare_pts(frames, s4_det)))
    ok.append(('Objs', compare_objs(s4_trk)))

    print("\n" + "=" * 64)
    for name, passed in ok:
        print(f"  {name:6s}: {'PASS' if passed else 'FAIL'}")
    all_pass = all(p for _, p in ok)
    print("=" * 64)
    print("ALL PASS" if all_pass else "SOME FAIL")
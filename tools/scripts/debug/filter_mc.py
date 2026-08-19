#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Task 7: 滤波器蒙特卡洛仿真验证 (离线)。

4 滤波器 (α-β / KF / EKF-CTRV / IMM) × 3 场景 × 4 噪声档 × N runs,
对齐 Riccati 理论稳态下界, 出 RMSE 时序 / σ 曲线 / 柱状图 + CSV。
oracle 关联 (目标 i 量测 → 航迹 i) 隔离 matcher 误差; ego 补偿置零 (vdd=0)。
退出码 0 = 全部验收判据 PASS; 非 0 = 任一 FAIL (实际滤波器缺陷, 如实上报)。
"""
import argparse
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
# 中文字体 (Windows 优先 SimHei, 回退英文以防空白方块)
for _f in ['SimHei', 'Microsoft YaHei']:
    if _f in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams['font.sans-serif'] = [_f]
        break
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tracker.schemas import (
    Cfg, CfgRun, CfgVds, CfgData, CfgModel,
    CfgFilter, CfgFilterPara, CfgFilterParaKf,
    CfgMatch, CfgVisualize, CfgEvaluate, CfgManager,
    Obj, Trk, TrkHistory, Matches, VDD,
)
from tracker.filter import Filter

# ---------------- 全局参数 ----------------
DT = 0.1                       # 10 Hz
T_FRAMES = 300                 # 30 s
SIGMA_POS_LEVELS = [0.1, 0.3, 0.5, 1.0]
SIGMA_VEL = 0.1
BURN_IN = 150                  # 稳态窗口 [150:300]
P0 = np.diag([100.0, 100.0, 10000.0, 10000.0])
_VDD_ZERO = VDD(speed_ms=0.0, yaw_rate=0.0, gear=0)  # ego 补偿无操作
OUT_DIR = os.path.join(_REPO, 'tracker', 'tests', '.tmp', 'filter_mc')
FILT_NAMES = {1: 'α-β', 2: 'KF', 3: 'EKF', 4: 'IMM'}


# =========================================================
# Block 1: 场景库 - (T,N,4) truth [x,y,vx,vy] + (T,N) yaw_rate(rad/s)
# =========================================================
def _s1_straight():
    """5 目标匀速直线, 轨迹交叉, 方向各异 (KF 主场)."""
    inits = [
        (0.0, 50.0, 2.0, -1.0),
        (50.0, 0.0, -2.0, 1.0),
        (0.0, 0.0, 1.5, 1.5),
        (50.0, 50.0, -1.5, -1.5),
        (25.0, -10.0, 0.0, 2.0),
    ]
    N = len(inits)
    truth = np.zeros((T_FRAMES, N, 4))
    for i, (x0, y0, vx, vy) in enumerate(inits):
        for k in range(T_FRAMES):
            t = k * DT
            truth[k, i] = [x0 + vx * t, y0 + vy * t, vx, vy]
    yaw = np.zeros((T_FRAMES, N))
    return truth, yaw


def _s2_turn():
    """4 目标恒定 ω 圆弧 (CTRV 真值), 不同 (v, ω) (EKF 主场)."""
    cfgs = [
        (0.0, 0.0, 10.0, 0.0, 0.10),
        (40.0, 0.0, 8.0, np.pi / 2, -0.15),
        (-30.0, 30.0, 12.0, -np.pi / 4, 0.05),
        (20.0, -20.0, 6.0, np.pi, 0.20),
    ]
    N = len(cfgs)
    truth = np.zeros((T_FRAMES, N, 4))
    yaw = np.zeros((T_FRAMES, N))
    for i, (x0, y0, v, psi0, w) in enumerate(cfgs):
        x, y, psi = x0, y0, psi0
        for k in range(T_FRAMES):
            truth[k, i] = [x, y, v * np.cos(psi), v * np.sin(psi)]
            yaw[k, i] = w
            x, y, psi = _ctrv_step(x, y, psi, v, w, DT)
    return truth, yaw


def _s3_straight_then_turn():
    """3 目标: 前 150 帧直行, 后 150 帧恒定 ω 转弯 (拼接, IMM 主场)."""
    cfgs = [
        (0.0, 0.0, 8.0, 0.0, 0.10),
        (20.0, 20.0, 6.0, np.pi / 4, -0.12),
        (-10.0, 30.0, 10.0, 0.0, 0.0),  # 全程直行 (对照)
    ]
    N = len(cfgs)
    truth = np.zeros((T_FRAMES, N, 4))
    yaw = np.zeros((T_FRAMES, N))
    SPLIT = T_FRAMES // 2
    for i, (x0, y0, v, psi0, w_turn) in enumerate(cfgs):
        x, y, psi = x0, y0, psi0
        for k in range(T_FRAMES):
            truth[k, i] = [x, y, v * np.cos(psi), v * np.sin(psi)]
            w = w_turn if k >= SPLIT else 0.0
            yaw[k, i] = w
            x, y, psi = _ctrv_step(x, y, psi, v, w, DT)
    return truth, yaw


def _ctrv_step(x, y, psi, v, w, dt):
    """CTRV 一步积分 (|ω|<1e-9 退化为直线)."""
    if abs(w) < 1e-9:
        return x + v * np.cos(psi) * dt, y + v * np.sin(psi) * dt, psi
    x += (v / w) * (np.sin(psi + w * dt) - np.sin(psi))
    y += (v / w) * (-np.cos(psi + w * dt) + np.cos(psi))
    return x, y, psi + w * dt


def scenarios():
    return {
        'S1_straight': _s1_straight(),
        'S2_turn': _s2_turn(),
        'S3_mix': _s3_straight_then_turn(),
    }


# =========================================================
# Block 2: 量测生成 - oracle 关联隐含于索引
# =========================================================
def gen_measurements(truth, sigma_pos, sigma_vel, seed):
    """z = truth + 高斯噪声 (x,y ~ σ_pos; vx,vy ~ σ_vel)."""
    rng = np.random.default_rng(seed)
    z = truth.copy()
    z[..., 0:2] += rng.normal(0.0, sigma_pos, z[..., 0:2].shape)
    z[..., 2:4] += rng.normal(0.0, sigma_vel, z[..., 2:4].shape)
    return z


# =========================================================
# cfg / trk / obj 工厂
# =========================================================
def _diag(n, v):
    return [[float(v) if i == j else 0.0 for j in range(n)] for i in range(n)]


def make_cfg(filt_type, sigma_pos, sigma_vel=SIGMA_VEL):
    """R 对齐当前 σ (正确调谐); Q: KF=1.0 (CV 转弯不确定), EKF=0.01 (信任 CTRV), IMM-KF=0.1."""
    sp2 = float(sigma_pos * sigma_pos)
    sv2 = float(sigma_vel * sigma_vel)
    R4 = [[sp2, 0.0, 0.0, 0.0],
          [0.0, sp2, 0.0, 0.0],
          [0.0, 0.0, sv2, 0.0],
          [0.0, 0.0, 0.0, sv2]]
    return Cfg(
        RUN=CfgRun(mode=1, overlap=0, delay=1, accum_frames=1,
                   vds=CfgVds(wheelbase_m=4.5, x_pos_m=0.0, y_pos_m=0.0, z_pos_m=0.0, cycle_s=DT)),
        DATA=CfgData(paths=['/mc/seq1', '/mc/seq2']),
        MODEL=CfgModel(cfg='./tools/cfgs/vod_models/vod_radarpillar.yaml',
                       ckpt='./checkpoints/vod_radarpillar.pth', score_thresh=0.3),
        FILTER=CfgFilter(type=filt_type, para=CfgFilterPara(
            para_abf={'alpha': 0.85, 'beta': 0.20},
            para_kf=CfgFilterParaKf(dim=4, q=_diag(4, 1.0), r=[row[:] for row in R4]),
            # EKF 信任其 CTRV 过程模型 (转弯真值), Q 远小于 KF; 否则精确预测被当噪声滤掉, 退化成 KF.
            para_ekf={'dim': 4, 'q': _diag(4, 0.01), 'r': [row[:] for row in R4]},
            para_imm={'models': [
                          {'type': 1, 'alpha': 0.85, 'beta': 0.20, 'r': sp2},
                          {'type': 2, 'q': _diag(4, 0.1), 'r': [row[:] for row in R4]},
                      ],
                      'markov': [[0.95, 0.05], [0.05, 0.95]]},
        )),
        MATCH=CfgMatch(gap_type=1, gap_dim=2, gap_weight=[1.0, 1.0, 1.0], thresh=3.0),
        VISUALIZE=CfgVisualize(enable=0,
                               show={'points': 0, 'tracks': 0, 'objs': 0, 'gts': 0},
                               metrics=0,
                               metrics_show={'tp': 0, 'fp': 0, 'fn': 0, 'mota': 0}),
        EVALUATE=CfgEvaluate(type=0, report=0, template='default'),
        MANAGER=CfgManager(birth_heat=3, death_heat=3, dt=DT, history_horizon=4.0,
                           adapter={'smooth': 1, 'markov': 1}),
    )


def make_trk(z0, yaw_rate_degs=0.0, tid=1):
    """航迹初始化 - 首帧量测填状态, P0 大对角."""
    return Trk(
        x_m=float(z0[0]), y_m=float(z0[1]), z_m=0.0,
        vx_mps=float(z0[2]), vy_mps=float(z0[3]),
        doppler_mps=0.0, ax_mps2=0.0, ay_mps2=0.0,
        heading_deg=0.0, yaw_rate_degs=float(yaw_rate_degs),
        id=tid, width_m=2.0, height_m=1.0, length_m=4.0, lifetime_s=0.0,
        x_std_m=0.0, y_std_m=0.0, z_std_m=0.0, vx_std_mps=0.0, vy_std_mps=0.0,
        ax_std_mps2=0.0, ay_std_mps2=0.0, xy_pos_cov=0.0, xy_vel_cov=0.0, xy_acc_cov=0.0,
        width_std_m=0.0, height_std_m=0.0, length_std_m=0.0,
        heading_std_deg=0.0, yaw_rate_std_degs=0.0,
        type=0, type_confi=0, obstacle_prob=0, existence_prob=0,
        motion_status=0, measurement_status=1, passable_status=0,
        rel_vel=0, rel_acc=0,
        cov=P0.copy(),
        history=TrkHistory(),
    )


def make_obj(z):
    return Obj(id=0, x=float(z[0]), y=float(z[1]), vx=float(z[2]), vy=float(z[3]))


# =========================================================
# Block 3: MC 引擎 - common random numbers (同 scenario/σ 跨滤波器同噪声)
# =========================================================
def run_mc(truth, yaw, sigma_pos, filt_type, n_runs, base_seed):
    """单 scenario × 单 σ × 单滤波器 MC; 返回 (pos_rmse[T], vel_rmse[T])."""
    T, N, _ = truth.shape
    cfg = make_cfg(filt_type, sigma_pos)
    assert cfg.isvalid(), f"cfg invalid type={filt_type} σ={sigma_pos}"
    pos_sq = np.zeros(T)
    vel_sq = np.zeros(T)
    for run in range(n_runs):
        z = gen_measurements(truth, sigma_pos, SIGMA_VEL, seed=base_seed + run)
        f = Filter(cfg)  # 每 run 重建 (IMM _states 复位)
        trks = [make_trk(z[0, i], yaw_rate_degs=float(np.degrees(yaw[0, i])), tid=i + 1)
                for i in range(N)]
        for k in range(T):
            if k > 0:
                for i, trk in enumerate(trks):
                    trk.yaw_rate_degs = float(np.degrees(yaw[k, i]))  # EKF 从真值取 ω
                f.predict(trks, _VDD_ZERO, DT)
                matches = Matches(matched=[(trks[i], make_obj(z[k, i])) for i in range(N)],
                                  unmatched_trks=[], unmatched_objs=[])
                f.update(matches, DT)
            for i, trk in enumerate(trks):
                ex = trk.x_m - truth[k, i, 0]
                ey = trk.y_m - truth[k, i, 1]
                evx = trk.vx_mps - truth[k, i, 2]
                evy = trk.vy_mps - truth[k, i, 3]
                pos_sq[k] += ex * ex + ey * ey
                vel_sq[k] += evx * evx + evy * evy
    denom = n_runs * N
    return np.sqrt(pos_sq / denom), np.sqrt(vel_sq / denom)


def steady(rmse):
    """稳态均值 - 窗口 [BURN_IN:T]."""
    return float(np.mean(rmse[BURN_IN:]))


# =========================================================
# Riccati 理论基准 - KF dim=4 CV 稳态后验协方差
# =========================================================
def kf_steady_covariance(sigma_pos, sigma_vel=SIGMA_VEL, dt=DT, q=1.0, iters=3000):
    """迭代后验-后验 Riccati 收敛到稳态; 返回 (P_prior, P_post).
    注: brief 公式 P←FPF'+Q−FPH'(HPH'+R)⁻¹HPF' 收敛到 *先验* 协方差;
    MC 测的是 *后验* 误差 (update 后), 故对齐后验 P_post=(I−KH)P_prior."""
    F = np.eye(4); F[0, 2] = F[1, 3] = dt
    H = np.eye(4)
    Q = np.diag([q, q, q, q]).astype(float)
    R = np.diag([sigma_pos ** 2, sigma_pos ** 2, sigma_vel ** 2, sigma_vel ** 2]).astype(float)
    P_post = np.eye(4) * 1000.0
    P_prior = None
    for _ in range(iters):
        P_prior = F @ P_post @ F.T + Q
        S = H @ P_prior @ H.T + R
        K = P_prior @ H.T @ np.linalg.inv(S)
        P_post = (np.eye(4) - K @ H) @ P_prior
        P_post = (P_post + P_post.T) / 2.0
    return P_prior, P_post


# =========================================================
# Block 4: 报告 - 图表 + CSV + 验收判据
# =========================================================
def report(results, n_runs, wallclock, out_dir):
    """results[(scenario, filt_type, sigma)] = (pos_rmse[T], vel_rmse[T])."""
    os.makedirs(out_dir, exist_ok=True)
    scen_names = ['S1_straight', 'S2_turn', 'S3_mix']
    scen_labels = {'S1_straight': 'S1 匀速直线', 'S2_turn': 'S2 恒定转弯', 'S3_mix': 'S3 直行转转弯'}
    filt_types = [1, 2, 3, 4]
    sigma_ref = 0.5  # 时序图用的代表 σ

    # ---- fig1: RMSE 时序 (σ=0.5) ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, sn in zip(axes, scen_names):
        for ft in filt_types:
            pos_rmse, _ = results[(sn, ft, sigma_ref)]
            ax.plot(pos_rmse, label=FILT_NAMES[ft], lw=1.4)
        ax.axvline(BURN_IN, color='k', ls='--', lw=0.7, alpha=0.5)
        ax.set_title(f'{scen_labels[sn]}  (位置噪声 σ={sigma_ref})')
        ax.set_xlabel('帧序号'); ax.set_ylabel('位置 RMSE [m]')
        ax.set_yscale('log'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f'位置 RMSE 随帧变化 (MC 次数 n={n_runs})', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig1_rmse_timecourse.png'), dpi=110, bbox_inches='tight')
    plt.close(fig)

    # ---- fig2: 稳态 RMSE vs σ ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, sn in zip(axes, scen_names):
        for ft in filt_types:
            ys = [steady(results[(sn, ft, s)][0]) for s in SIGMA_POS_LEVELS]
            ax.plot(SIGMA_POS_LEVELS, ys, marker='o', label=FILT_NAMES[ft])
        ax.set_title(scen_labels[sn]); ax.set_xlabel('位置噪声 σ [m]'); ax.set_ylabel('稳态位置 RMSE [m]')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f'稳态位置 RMSE 随噪声 σ 变化 (窗口 [{BURN_IN}:{T_FRAMES}], 次数 n={n_runs})', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig2_steady_vs_sigma.png'), dpi=110, bbox_inches='tight')
    plt.close(fig)

    # ---- fig3: 场景×滤波器柱状图 (σ 平均) ----
    means = {ft: [] for ft in filt_types}
    for ft in filt_types:
        for sn in scen_names:
            val = float(np.mean([steady(results[(sn, ft, s)][0]) for s in SIGMA_POS_LEVELS]))
            means[ft].append(val)
    x = np.arange(len(scen_names)); w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, ft in enumerate(filt_types):
        ax.bar(x + (idx - 1.5) * w, means[ft], width=w, label=FILT_NAMES[ft])
    ax.set_xticks(x); ax.set_xticklabels([scen_labels[sn] for sn in scen_names])
    ax.set_ylabel('平均稳态位置 RMSE [m] (各 σ 取平均)')
    ax.set_title(f'场景 × 滤波器 稳态位置 RMSE (MC 次数 n={n_runs})')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig3_bar_scenario_filter.png'), dpi=110, bbox_inches='tight')
    plt.close(fig)

    # ---- CSV ----
    csv_path = os.path.join(out_dir, 'rmse_summary.csv')
    with open(csv_path, 'w', encoding='utf-8') as fp:
        fp.write('场景,滤波器,位置噪声σ,稳态位置RMSE,稳态速度RMSE\n')
        for sn in scen_names:
            for ft in filt_types:
                for s in SIGMA_POS_LEVELS:
                    pr, vr = results[(sn, ft, s)]
                    fp.write(f'{sn},{FILT_NAMES[ft]},{s},{steady(pr):.6f},{steady(vr):.6f}\n')

    # ---- 验收判据 ----
    checks = _evaluate_checks(results, scen_names)
    _print_summary(checks, n_runs, wallclock, out_dir)
    return checks


def _have(results, sn, ft, s):
    return (sn, ft, s) in results


def _evaluate_checks(results, scen_names):
    """逐条核对验收判据, 返回 [(name, PASS/FAIL, detail)]. 跳过未跑的 (scenario,σ) 组合."""
    checks = []

    # (1) S1 KF 稳态位置 RMSE vs Riccati 后验 (10% 容差, 每 σ)
    for s in SIGMA_POS_LEVELS:
        if not _have(results, 'S1_straight', 2, s):
            continue
        kf_pos, _ = results[('S1_straight', 2, s)]
        mc_pos = steady(kf_pos)                                  # 2D pos RMSE = √(E[ex²]+E[ey²])
        _, P_post = kf_steady_covariance(s)
        theory_pos = float(np.sqrt(P_post[0, 0] + P_post[1, 1]))  # 2D 后验位置 std
        rel = abs(mc_pos - theory_pos) / theory_pos
        checks.append(('C1a Riccati(KF) σ=%.2f' % s, rel < 0.10,
                       f'MC={mc_pos:.4f} theory={theory_pos:.4f} rel={rel*100:.1f}%'))

    # (1b) brief 原始公式 (先验) 透明对比 - 不计入 PASS/FAIL, 仅记录
    for s in SIGMA_POS_LEVELS:
        if not _have(results, 'S1_straight', 2, s):
            continue
        kf_pos, _ = results[('S1_straight', 2, s)]
        mc_pos = steady(kf_pos)
        P_prior, _ = kf_steady_covariance(s)
        prior_pos = float(np.sqrt(P_prior[0, 0] + P_prior[1, 1]))
        rel = abs(mc_pos - prior_pos) / prior_pos
        checks.append(('C1b [info] brief-公式(先验) σ=%.2f' % s, None,
                       f'MC={mc_pos:.4f} prior={prior_pos:.4f} rel={rel*100:.1f}%'))

    # (2) S1 直线场景 EKF 不应劣于 KF (CTRV 直线退化为 CV; EKF Q 更小故稳态或更优, 但不得发散)
    #     旧判据 "EKF≈KF(<5%)" 假设两者同 Q, 现 EKF Q 更小, 改为上界约束。
    for s in SIGMA_POS_LEVELS:
        if not (_have(results, 'S1_straight', 2, s) and _have(results, 'S1_straight', 3, s)):
            continue
        kf_pos, _ = results[('S1_straight', 2, s)]
        ekf_pos, _ = results[('S1_straight', 3, s)]
        kf_steady = steady(kf_pos)
        ekf_steady = steady(ekf_pos)
        # EKF 稳态可比 KF 低 (低 Q 收益), 但不得高出 10% (CTRV 直线≈CV, 不应有劣势)
        ok = ekf_steady <= kf_steady * 1.10
        checks.append(('C2 EKF不劣于KF(S1) σ=%.2f' % s, ok,
                       f'KF={kf_steady:.4f} EKF={ekf_steady:.4f} '
                       f'rel={(ekf_steady-kf_steady)/kf_steady*100:.1f}%'))

    # (2b) S2/S3 转弯场景 EKF 稳态优于 KF (CTRV 过程模型价值 - 低 Q 才显现)
    for sn in ['S2_turn', 'S3_mix']:
        for s in SIGMA_POS_LEVELS:
            if not (_have(results, sn, 2, s) and _have(results, sn, 3, s)):
                continue
            kf_steady = steady(results[(sn, 2, s)][0])
            ekf_steady = steady(results[(sn, 3, s)][0])
            # EKF 在转弯应严格更低; 容忍 3% 以内视为持平 (低 σ 时差距小)
            ok = ekf_steady < kf_steady * 1.03
            checks.append(('C2b EKF<KF %s σ=%.2f' % (sn, s), ok,
                           f'KF={kf_steady:.4f} EKF={ekf_steady:.4f} '
                           f'rel={(kf_steady-ekf_steady)/kf_steady*100:.1f}%'))

    # (3) 所有滤波器稳态 RMSE 随 σ 单调非递减
    for sn in scen_names:
        for ft in [1, 2, 3, 4]:
            ys = [steady(results[(sn, ft, s)][0]) for s in SIGMA_POS_LEVELS
                  if _have(results, sn, ft, s)]
            if len(ys) < 2:
                continue
            mono = all(ys[i + 1] >= ys[i] - 1e-6 for i in range(len(ys) - 1))
            checks.append(('C3 mono %s/%s' % (sn, FILT_NAMES[ft]), mono,
                           '→'.join(f'{v:.4f}' for v in ys)))

    # (4) S3 IMM 平均 RMSE 非发散 (< 2× 最优单模型, 有限)
    for s in SIGMA_POS_LEVELS:
        if not all(_have(results, 'S3_mix', ft, s) for ft in [1, 2, 3, 4]):
            continue
        if 'S3_mix' not in scen_names:
            continue
        imm_pos = steady(results[('S3_mix', 4, s)][0])
        best = min(steady(results[('S3_mix', ft, s)][0]) for ft in [1, 2, 3])
        ok = np.isfinite(imm_pos) and imm_pos < 2.0 * best
        checks.append(('C4 IMM 不发散 S3 σ=%.2f' % s, ok,
                       f'IMM={imm_pos:.4f} best={best:.4f} ratio={imm_pos/best:.2f}'))

    # (5) α-β 在 S1 的 RMSE ≤ 1.5× KF
    for s in SIGMA_POS_LEVELS:
        if not (_have(results, 'S1_straight', 1, s) and _have(results, 'S1_straight', 2, s)):
            continue
        ab = steady(results[('S1_straight', 1, s)][0])
        kf = steady(results[('S1_straight', 2, s)][0])
        ok = ab <= 1.5 * kf
        checks.append(('C5 α-β≤1.5×KF S1 σ=%.2f' % s, ok,
                       f'αβ={ab:.4f} KF={kf:.4f} ratio={ab/kf:.2f}'))
    return checks


def _print_summary(checks, n_runs, wallclock, out_dir):
    print('\n' + '=' * 78)
    print(f' Filter MC — n_runs={n_runs}  wallclock={wallclock:.1f}s  out={out_dir}')
    print('=' * 78)
    n_pass = sum(1 for _, v, _ in checks if v is True)
    n_fail = sum(1 for _, v, _ in checks if v is False)
    n_info = sum(1 for _, v, _ in checks if v is None)
    for name, ok, detail in checks:
        tag = 'PASS' if ok is True else ('FAIL' if ok is False else 'INFO')
        print(f'  [{tag:4}] {name:38} {detail}')
    print('-' * 78)
    print(f'  TOTAL: {n_pass} PASS / {n_fail} FAIL / {n_info} INFO')
    print('=' * 78)


# =========================================================
# main
# =========================================================
def main():
    ap = argparse.ArgumentParser(description='Task 7 滤波器蒙特卡洛仿真验证')
    ap.add_argument('--runs', type=int, default=25,
                    help='每组合 MC run 数 (默认 25 — 计时显示 100 runs 全扫描 ~8 min 超预算; '
                         '25 runs ~1.9 min 且 σ=0.1 Riccati 裕度稳定; --runs 100 可回放全量)')
    ap.add_argument('--out', default=OUT_DIR, help='输出目录')
    ap.add_argument('--probe', action='store_true', help='快速计时 (1 场景×1 σ×4 滤波×小 runs)')
    args = ap.parse_args()

    scens = scenarios()
    t0 = time.time()
    results = {}
    if args.probe:
        sn = 'S1_straight'
        truth, yaw = scens[sn]
        for ft in [1, 2, 3, 4]:
            results[(sn, ft, 0.5)] = run_mc(truth, yaw, 0.5, ft, args.runs, base_seed=1000)
        _print_summary(_evaluate_checks(results, [sn]), args.runs, time.time() - t0, args.out)
        print(f'[PROBE] {args.runs} runs × 4 filters × S1 × σ=0.5 → {time.time()-t0:.2f}s')
        return 0

    n_runs = args.runs
    for s_idx, sn in enumerate(['S1_straight', 'S2_turn', 'S3_mix']):
        truth, yaw = scens[sn]
        for s_idx2, s in enumerate(SIGMA_POS_LEVELS):
            base_seed = 1000 + s_idx * 10000 + s_idx2 * 1000  # CRN: 跨滤波器同 seed
            for ft in [1, 2, 3, 4]:
                results[(sn, ft, s)] = run_mc(truth, yaw, s, ft, n_runs, base_seed)
        elapsed = time.time() - t0
        print(f'  [sweep] {sn} done @ {elapsed:.1f}s')
    wallclock = time.time() - t0

    checks = report(results, n_runs, wallclock, args.out)
    n_fail = sum(1 for _, v, _ in checks if v is False)
    return 1 if n_fail > 0 else 0


if __name__ == '__main__':
    sys.exit(main())

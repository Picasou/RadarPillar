"""
滤波蒙特卡洛可视化 - 出两张落地图:
  1) RMSE 收敛曲线: 三场景下 4 滤波器位置/速度 RMSE (同一张图, 多子图)
  2) 轨迹图: 单次 trial 的真值 + 量测 + 4 滤波估计 (每场景一张)

复用 tools/scripts/debug/filter_mc.py 的场景/噪声/cfg/MC 逻辑, 只加可视化。
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 仓库根加入 sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tracker.schemas import Trk, TrkHistory, Obj, Matches
from tracker.filter import Filter
# 复用旧 MC 脚本的场景生成 / 噪声 / cfg / 状态构造 (已经验证过的逻辑)
import importlib.util
_spec = importlib.util.spec_from_file_location('filter_mc', os.path.join(_HERE, 'filter_mc.py'))
filter_mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(filter_mc)

# ---- matplotlib 中文 ----
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.path.join(_ROOT, 'tracker', 'tests', '.tmp', 'filter_mc_plots')
os.makedirs(OUT_DIR, exist_ok=True)

# 4 滤波器: type, 显示名, 颜色, 线型
FILTERS = [
    (1, 'α-β',  '#d62728', '--'),
    (2, 'KF',   '#1f77b4', '-'),
    (3, 'EKF',  '#2ca02c', '-'),
    (4, 'IMM',  '#9467bd', '-'),
]
SCENARIOS = [('S1_直线', filter_mc._s1_straight),
             ('S2_转弯', filter_mc._s2_turn),
             ('S3_混合', filter_mc._s3_straight_then_turn)]
SIGMA = 0.3           # 出图用单一噪声档 (RMSE 曲线/轨迹)
N_RUNS = 30           # MC 次数 (出图用, 比 100 快)


def run_single(truth, yaw, sigma_pos, filt_type, seed):
    """单次 trial: 返回 (est[T,N,4], z_meas[T,N,4]). est 含全帧逐帧估计."""
    T, N, _ = truth.shape
    cfg = filter_mc.make_cfg(filt_type, sigma_pos)
    z = filter_mc.gen_measurements(truth, sigma_pos, filter_mc.SIGMA_VEL, seed=seed)
    f = Filter(cfg)
    trks = [filter_mc.make_trk(z[0, i], yaw_rate_degs=float(np.degrees(yaw[0, i])), tid=i + 1)
            for i in range(N)]
    est = np.zeros((T, N, 4))
    for k in range(T):
        f.predict(trks, filter_mc._VDD_ZERO, filter_mc.DT)
        matched = [(trks[i], filter_mc.make_obj(z[k, i])) for i in range(N)]
        f.update(Matches(matched=matched, unmatched_trks=[]), filter_mc.DT)
        for i, trk in enumerate(trks):
            est[k, i] = [trk.x_m, trk.y_m, trk.vx_mps, trk.vy_mps]
    return est, z


def run_mc_curve(truth, yaw, sigma_pos, filt_type, n_runs, base_seed=0):
    """多 trial 平均, 返回每帧 RMSE (pos[T], vel[T])."""
    T, N, _ = truth.shape
    pos_sq = np.zeros(T)
    vel_sq = np.zeros(T)
    for run in range(n_runs):
        est, _ = run_single(truth, yaw, sigma_pos, filt_type, seed=base_seed + run)
        de = est - truth
        pos_sq += np.sum(de[..., 0] ** 2 + de[..., 1] ** 2, axis=1)
        vel_sq += np.sum(de[..., 2] ** 2 + de[..., 3] ** 2, axis=1)
    n = N * n_runs
    return np.sqrt(pos_sq / n), np.sqrt(vel_sq / n)


# ============ 图 1: RMSE 收敛曲线 ============
def plot_rmse_curves():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for col, (sname, gen) in enumerate(SCENARIOS):
        truth, yaw = gen()
        T = truth.shape[0]
        t = np.arange(T) * filter_mc.DT
        for ftype, fname, color, ls in FILTERS:
            pos_r, vel_r = run_mc_curve(truth, yaw, SIGMA, ftype, N_RUNS)
            axes[0, col].plot(t, pos_r, label=fname, color=color, ls=ls, lw=1.5)
            axes[1, col].plot(t, vel_r, label=fname, color=color, ls=ls, lw=1.5)
        axes[0, col].set_title(f'{sname}  (σ={SIGMA})')
        axes[0, col].set_ylabel('位置 RMSE (m)')
        axes[0, col].grid(alpha=0.3)
        axes[1, col].set_ylabel('速度 RMSE (m/s)')
        axes[1, col].set_xlabel('时间 (s)')
        axes[1, col].grid(alpha=0.3)
    axes[0, 0].legend(loc='upper right', fontsize=9)
    fig.suptitle('滤波器蒙特卡洛 RMSE 收敛曲线 (各场景 4 滤波对比)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, 'fig1_rmse_curves.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'[图1] {out}')


# ============ 图 2: 单次 trial 轨迹 ============
def plot_trajectories(seed=7):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for col, (sname, gen) in enumerate(SCENARIOS):
        ax = axes[col]
        truth, yaw = gen()
        N = truth.shape[1]
        # 真值
        for i in range(N):
            ax.plot(truth[:, i, 0], truth[:, i, 1], color='black', lw=2.0, alpha=0.7,
                    label='真值' if i == 0 else None)
        # 取一个 trial 的量测 (取第 0 个目标做散点示意, 避免过密)
        z_all = filter_mc.gen_measurements(truth, SIGMA, filter_mc.SIGMA_VEL, seed=seed)
        for i in range(N):
            ax.scatter(z_all[:, i, 0], z_all[:, i, 1], s=5, color='gray', alpha=0.5,
                       marker='.', label='量测' if i == 0 else None)
        # 4 滤波估计 (单 trial)
        for ftype, fname, color, ls in FILTERS:
            est, _ = run_single(truth, yaw, SIGMA, ftype, seed=seed)
            for i in range(N):
                ax.plot(est[:, i, 0], est[:, i, 1], color=color, ls=ls, lw=1.2, alpha=0.9,
                        label=fname if i == 0 else None)
        ax.set_title(f'{sname}  (σ={SIGMA}, seed={seed})')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(loc='best', fontsize=7, ncol=2)
    fig.suptitle('单次蒙特卡洛目标轨迹 (真值 + 量测 + 4 滤波估计)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, 'fig2_trajectories.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'[图2] {out}')


if __name__ == '__main__':
    print(f'输出目录: {OUT_DIR}  | σ={SIGMA}, MC={N_RUNS} runs')
    plot_rmse_curves()
    plot_trajectories()
    print('完成.')

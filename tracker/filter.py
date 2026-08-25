"""滤波 predict/update - α-β / KF / EKF / IMM 统一入口，predict 内部完成 ego 补偿."""
from __future__ import annotations
import numpy as np

from .schemas import Cfg, Obj, Trk
from .utils.common import c_trk_compensate


"""
辅助工具
"""
def _get_state(trk: Trk) -> tuple[np.ndarray, np.ndarray]:
    """读取 trk 4 维运动学状态 [x,y,vx,vy] 与 4×4 协方差。状态恒 4 维。"""
    return (np.array([trk.x_m, trk.y_m, trk.vx_mps, trk.vy_mps]),
            trk.cov.copy())


def _write_state(trk: Trk, x: np.ndarray, P: np.ndarray) -> None:
    """4 维状态与协方差写回 trk, 并由对角线同步 std 字段。状态恒 4 维。"""
    trk.x_m = float(x[0])
    trk.y_m = float(x[1])
    trk.vx_mps = float(x[2])
    trk.vy_mps = float(x[3])
    trk.cov = P
    trk.x_std_m = float(np.sqrt(P[0, 0]))
    trk.y_std_m = float(np.sqrt(P[1, 1]))
    trk.vx_std_mps = float(np.sqrt(P[2, 2]))
    trk.vy_std_mps = float(np.sqrt(P[3, 3]))


def _get_z(obj: Obj, dim: int) -> np.ndarray:
    """组装量测向量 - dim=2 取位置(真 2 维量测), dim=4 取位置+速度。dim 是量测维。"""
    if dim == 2:
        return np.array([obj.x, obj.y])
    return np.array([obj.x, obj.y, obj.vx, obj.vy])


"""
filter 数据核
"""
def abf_predict(x: np.ndarray, dt: float) -> np.ndarray:
    """常速度外推 4 维状态 [x,y,vx,vy] - 位置按速度积分, 返回新数组."""
    x_new = x.copy()
    x_new[:2] += x_new[2:] * dt
    return x_new


def abf_update(x: np.ndarray, z: np.ndarray, alpha: float, beta: float, dt: float,
               r_var: float | None = None) -> tuple[np.ndarray, float]:
    """α-β 量测修正 - 位置 α 修正/速度 β/dt 修正; 似然为 len(z) 维全量测残差各向同性高斯 (与 KF 同维尺度)."""
    if r_var is not None and r_var <= 0:
        raise ValueError(f"abf_update r_var must be > 0, got {r_var}")
    x_new = x.copy()
    r_pos = z[:2] - x_new[:2]                 # α-β 修正仅用位置残差
    r_full = z - x_new[:len(z)]               # 似然用修正前 (先验 x) 的全量测残差, 在 += 之前计算
    x_new[:2] += alpha * r_pos
    if dt > 0:
        x_new[2:] += beta * r_pos / dt
    if r_var is None:
        return x_new, 0.0
    n = len(z)
    ll = (2.0 * np.pi * r_var) ** (-n / 2) * np.exp(-0.5 * float(r_full @ r_full) / r_var)
    return x_new, float(ll)


def kf_predict(x: np.ndarray, P: np.ndarray, dt: float, Q: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """常速度 4 维状态外推 - F 恒 4×4 含 x+=vx·dt。状态维固定, dim(量测维)不影响预测。"""
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    x_new = F @ x
    P_new = F @ P @ F.T + Q
    return x_new, P_new


def kf_update(x: np.ndarray, P: np.ndarray, z: np.ndarray, H: np.ndarray, R: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, float]:
    """KF 量测更新 - 返回 (后验状态, 对称后验协方差, 量测似然 N(z; Hx, S))."""
    r = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x_new = x + K @ r
    P_new = (np.eye(len(x)) - K @ H) @ P
    P_new = (P_new + P_new.T) / 2.0
    _, logdet = np.linalg.slogdet(S)
    maha = float(r @ np.linalg.solve(S, r))
    ll = float(np.exp(-0.5 * (len(z) * np.log(2.0 * np.pi) + logdet + maha)))
    return x_new, P_new, ll


def _ctrv_f(x: np.ndarray, dt: float, yaw_rate_rad: float) -> np.ndarray:
    vx, vy = float(x[2]), float(x[3])
    v = float(np.hypot(vx, vy))
    theta = yaw_rate_rad * dt
    ct, st = np.cos(theta), np.sin(theta)
    c, s = vx / v, vy / v                      # cosψ, sinψ
    a = (s * ct + c * st) - s                  # sin(ψ+θ) − sinψ
    b = c - (c * ct - s * st)                  # cosψ − cos(ψ+θ)
    p = (c * a + s * b) / yaw_rate_rad
    q = (s * a - c * b) / yaw_rate_rad
    return np.array([
        [1.0, 0.0,  p,   q],
        [0.0, 1.0, -q,   p],
        [0.0, 0.0,  ct, -st],
        [0.0, 0.0,  st,  ct],
    ])


def ctrv_predict(x: np.ndarray, P: np.ndarray, dt: float, yaw_rate_rad: float, Q: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    if abs(yaw_rate_rad) < 1e-6:
        return kf_predict(x, P, dt, Q)
    vx, vy = float(x[2]), float(x[3])
    v = float(np.hypot(vx, vy))
    if v < 1e-6:
        P_new = P + Q
        return x.copy(), (P_new + P_new.T) / 2.0
    w = yaw_rate_rad
    theta = w * dt
    ct, st = np.cos(theta), np.sin(theta)
    c, s = vx / v, vy / v
    c2 = c * ct - s * st                       # cos(ψ+θ)
    s2 = s * ct + c * st                       # sin(ψ+θ)
    x_new = x.copy()
    x_new[0] += (v / w) * (s2 - s)
    x_new[1] += (v / w) * (c - c2)
    x_new[2] = v * c2
    x_new[3] = v * s2
    F = _ctrv_f(x, dt, w)
    P_new = F @ P @ F.T + Q
    return x_new, (P_new + P_new.T) / 2.0


"""
滤波类
"""
class _TemplateFilter:
    """所有滤波器的基类"""
    def get_R(self):
        raise NotImplementedError

    def get_Q(self):
        raise NotImplementedError

    def predict(self, trks: list[Trk], vdd, cycle_s: float) -> None:
        for trk in trks:
            c_trk_compensate(trk, vdd, cycle_s)
            self._predict(trk, cycle_s)
        self._prune(trks)

    def _predict(self, trk: Trk, cycle_s: float) -> None:
        raise NotImplementedError

    def _prune(self, trks: list[Trk]) -> None:
        pass

    def reset(self) -> None:
        """序列边界重置: 无状态滤波器空实现, 有状态子类 (IMM) 覆写."""
        pass

    def update(self, trk: Trk, obj: Obj, cycle_s: float) -> None:
        self._update(trk, obj, cycle_s)

    def _update(self, trk: Trk, obj: Obj, cycle_s: float) -> None:
        raise NotImplementedError


class AlphaBetaFilter(_TemplateFilter):
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def get_R(self):
        pass

    def get_Q(self):
        pass

    def _predict(self, trk: Trk, cycle_s: float) -> None:
        x = np.array([trk.x_m, trk.y_m, trk.vx_mps, trk.vy_mps])
        x = abf_predict(x, cycle_s)
        trk.x_m, trk.y_m = float(x[0]), float(x[1])
        trk.vx_mps, trk.vy_mps = float(x[2]), float(x[3])

    def _update(self, trk: Trk, obj: Obj, cycle_s: float) -> None:
        x = np.array([trk.x_m, trk.y_m, trk.vx_mps, trk.vy_mps])
        x, _ = abf_update(x, _get_z(obj, 2), self.alpha, self.beta, cycle_s)
        trk.x_m, trk.y_m = float(x[0]), float(x[1])
        trk.vx_mps, trk.vy_mps = float(x[2]), float(x[3])
        

class KalmanFilter(_TemplateFilter):
    def __init__(self, dim: int, q, r):
        self.dim = dim                      # 量测维: 2=仅(x/y), 4=(x/y/vx/vy)
        self.q = q
        self.r = r
        self.H = np.eye(4)[:2].copy() if dim == 2 else np.eye(4)

    def get_R(self):
        R = np.asarray(self.r, dtype=float)
        if self.dim == 2 and R.shape == (4, 4):
            R = R[:2, :2].copy()
        return R

    def get_Q(self):
        return np.asarray(self.q, dtype=float)

    def _predict(self, trk: Trk, cycle_s: float) -> None:
        x, P = _get_state(trk)
        x, P = kf_predict(x, P, cycle_s, self.get_Q())
        _write_state(trk, x, P)

    def _update(self, trk: Trk, obj: Obj, cycle_s: float) -> None:
        z = _get_z(obj, self.dim)
        x, P = _get_state(trk)
        x, P, _ = kf_update(x, P, z, self.H, self.get_R())
        _write_state(trk, x, P)
        

class EkfFilter(_TemplateFilter):
    def __init__(self, dim: int, q, r):
        self.dim = dim                      # 量测维: 2=仅(x/y), 4=(x/y/vx/vy); 状态恒 4 维走 CTRV
        self.q = q
        self.r = r
        if dim not in (2, 4):
            raise ValueError(f"EkfFilter supports dim in (2, 4), got {dim}")
        self.H = np.eye(4)[:2].copy() if dim == 2 else np.eye(4)

    def get_R(self):
        R = np.asarray(self.r, dtype=float)
        if self.dim == 2 and R.shape == (4, 4):
            R = R[:2, :2].copy()
        return R

    def get_Q(self):
        return np.asarray(self.q, dtype=float)

    def _predict(self, trk: Trk, cycle_s: float) -> None:
        w_rad = trk.yaw_rate_degs * np.pi / 180.0
        x, P = _get_state(trk)
        x, P = ctrv_predict(x, P, cycle_s, w_rad, self.get_Q())
        _write_state(trk, x, P)

    def _update(self, trk: Trk, obj: Obj, cycle_s: float) -> None:
        z = _get_z(obj, self.dim)
        x, P = _get_state(trk)
        x, P, _ = kf_update(x, P, z, self.H, self.get_R())
        _write_state(trk, x, P)
        

class ImmFilter(_TemplateFilter):
    def __init__(self, models: list[dict], markov: np.ndarray, meas_dim: int = 4):
        self.models = models
        self.markov = np.asarray(markov, dtype=float)
        self.meas_dim = meas_dim              # 量测维: 2=仅(x/y), 4=(x/y/vx/vy); 状态恒 4 维
        self.H = np.eye(4)[:2].copy() if meas_dim == 2 else np.eye(4)
        self.M = len(models)
        self._states: dict[int, dict] = {}

    def get_R(self):
        return [m.get('r') for m in self.models]

    def get_Q(self):
        return [m.get('q') for m in self.models]

    def _predict(self, trk: Trk, cycle_s: float) -> None:
        # 新 id 均匀概率初始化, bank 取自 trk 当前 4 维状态 (逐模型拷贝)
        if trk.id not in self._states:
            x0, P0 = _get_state(trk)
            banks = [(x0.copy(), P0.copy()) for _ in range(self.M)]
            self._states[trk.id] = {'probs': np.full(self.M, 1.0 / self.M), 'banks': banks}
        # 输入混合 → 各模型 kernel 预测
        entry = self._states[trk.id]
        probs = entry['probs']
        xs = np.stack([b[0] for b in entry['banks']])
        Ps = np.stack([b[1] for b in entry['banks']])
        new_banks = []
        for j, m in enumerate(self.models):
            x_bar, P_bar = self._mix(probs, xs, Ps, j)
            if m['type'] == 1:
                new_banks.append((abf_predict(x_bar, cycle_s), P_bar))
            else:
                new_banks.append(kf_predict(x_bar, P_bar, cycle_s, m['q']))
        entry['banks'] = new_banks
        # 预测(先验)混合态写回 trk: 履行 predict 契约, coast/中间帧航迹不再冻结
        c = probs @ self.markov
        cs = float(c.sum())
        c = c / cs if cs > 0.0 else np.full(self.M, 1.0 / self.M)
        xs_p = np.stack([b[0] for b in new_banks])
        Ps_p = np.stack([b[1] for b in new_banks])
        x_pred = c @ xs_p
        dx_p = xs_p - x_pred
        P_pred = np.einsum('i,ijk->jk', c, Ps_p) + np.einsum('i,ij,ik->jk', c, dx_p, dx_p)
        _write_state(trk, x_pred, (P_pred + P_pred.T) / 2.0)

    def _prune(self, trks: list[Trk]) -> None:
        # 惰性剪枝 - 删除不在当前 trks 的 id
        alive = {trk.id for trk in trks}
        for tid in [k for k in self._states if k not in alive]:
            del self._states[tid]

    def reset(self) -> None:
        # 序列边界重置 - 清空全部 IMM bank (跨序列航迹不延续)
        self._states.clear()

    def _update(self, trk: Trk, obj: Obj, cycle_s: float) -> None:
        entry = self._states.get(trk.id)
        if entry is None:
            return
        probs = entry['probs']
        lambdas = np.zeros(self.M)
        new_banks = []
        z = _get_z(obj, self.meas_dim)       # 量测维统一: 2 维仅位置, 4 维含速度
        for j, m in enumerate(self.models):
            x, P = entry['banks'][j]
            if m['type'] == 1:
                # α-β kernel 只动 x, P 仅在混合中流转; 似然按 len(z) 维各向同性高斯
                x_new, ll = abf_update(x, z, m['alpha'], m['beta'],
                                       cycle_s, r_var=m['r'])
                new_banks.append((x_new, P))
            else:
                x_new, P_new, ll = kf_update(x, P, z, self.H, m['r'])
                new_banks.append((x_new, P_new))
            lambdas[j] = ll
        # 概率更新 probs_j ∝ Λ_j·π_j, 归一化退化时保持上一帧
        unnorm = lambdas * (probs @ self.markov)
        total = float(unnorm.sum())
        if total > 0.0:
            new_probs = unnorm / total
        else:
            new_probs = probs
        # 输出混合
        xs = np.stack([b[0] for b in new_banks])
        Ps = np.stack([b[1] for b in new_banks])
        x_out = new_probs @ xs
        dx = xs - x_out
        P_out = np.einsum('i,ijk->jk', new_probs, Ps) \
            + np.einsum('i,ij,ik->jk', new_probs, dx, dx)
        P_out = (P_out + P_out.T) / 2.0
        entry['probs'] = new_probs
        entry['banks'] = new_banks
        _write_state(trk, x_out, P_out)
        

    def _mix(self, probs: np.ndarray, xs: np.ndarray, Ps: np.ndarray, j: int
             ) -> tuple[np.ndarray, np.ndarray]:
        """输入混合第 j 个模型 - π_j 预测概率加权各 bank 的状态/协方差 (含均值扩散项)."""
        pi_j = float(probs @ self.markov[:, j])
        if pi_j > 0.0:
            w = probs * self.markov[:, j] / pi_j
        else:
            w = np.full(self.M, 1.0 / self.M)
        x_bar = w @ xs
        dx = xs - x_bar
        P_bar = np.einsum('i,ijk->jk', w, Ps) + np.einsum('i,ij,ik->jk', w, dx, dx)
        return x_bar, (P_bar + P_bar.T) / 2.0


"""
调用入口
"""
class Filter:
    """
    滤波器主类 - 按 Cfg.FILTER.type 路由。
      predict:
        in  :(trks, vdd, cycle_s)
        out :trks (预测)

      update :
        in  :(trk, obj, cycle_s)
        out :单航迹量测修正 (循环由上层 update 入口驱动)
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.filter = self._build_filter(cfg)

    def _build_filter(self, cfg: Cfg) -> _TemplateFilter:
        """
        按 Cfg.FILTER.type 路由到具体滤波算法:
            1 = α-β 滤波    (常速度 2 维, 无过程/量测噪声矩阵)
            2 = KF          (线性卡尔曼, dim∈{2,4})
            3 = EKF         (扩展卡尔曼, 非线性观测)
            4 = IMM         (交互多模型, 内部混合 α-β / KF 子模型)
        """
        ftype = cfg.FILTER.type
        para  = cfg.FILTER.para
        if ftype == 1:
            # α-β 滤波
            ab = para.para_abf
            return AlphaBetaFilter(alpha=ab['alpha'], beta=ab['beta'])
        if ftype == 2:
            # 线性 KF
            kf = para.para_kf
            return KalmanFilter(dim=kf.dim, q=kf.q, r=kf.r)
        if ftype == 3:
            # 扩展卡尔曼 EKF
            ekf = para.para_ekf
            return EkfFilter(dim=ekf.get('dim', 4), q=ekf['q'], r=ekf['r'])
        if ftype == 4:
            # 交互多模型 IMM - 子模型参数描述符 (1=α-β {alpha,beta,r}, 2=KF {q,r})
            imm = para.para_imm
            models = []
            for sub_para in imm.get('models', []):
                stype = sub_para.get('type')
                if stype == 1:
                    r = float(sub_para['r'])
                    if r <= 0.0:
                        raise ValueError(f"IMM α-β sub-model r must be > 0, got {r}")
                    models.append({'type': 1, 'alpha': float(sub_para['alpha']),
                                   'beta': float(sub_para['beta']), 'r': r})
                elif stype == 2:
                    models.append({'type': 2, 'q': np.asarray(sub_para['q'], dtype=float),
                                   'r': np.asarray(sub_para['r'], dtype=float)})
                else:
                    raise ValueError(f"Unsupported IMM sub-model type: {stype}")
            markov = np.asarray(imm.get('markov', []), dtype=float)
            m = len(models)
            if markov.shape != (m, m):
                raise ValueError(f"IMM markov shape must be {(m, m)}, got {markov.shape}")
            return ImmFilter(models=models, markov=markov, meas_dim=imm.get('dim', 4))
        raise ValueError(f"Unsupported FILTER.type: {ftype}")

    def predict(self, trks: list[Trk], vdd, cycle_s: float) -> None:
        self.filter.predict(trks, vdd, cycle_s)

    def reset(self) -> None:
        """
        滤波重置: 委托内部滤波器清跨序列状态 (序列边界调用)
        """
        self.filter.reset()

    def update(self, trk: Trk, obj: Obj, cycle_s: float) -> None:
        self.filter.update(trk, obj, cycle_s)

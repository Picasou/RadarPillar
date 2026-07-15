"""滤波预测 - α-β / KF / EKF / IMM 统一入口，predict 内部完成 ego 补偿."""
from __future__ import annotations
import numpy as np

from .schemas import Cfg, Trk
from .utils.common import compensate_trks


class _BasePredictor:
    """所有预测器的基类 - predict: (trks,vdd,cycle_s) → 原地更新 trks; 子类实现 _predict(trks)."""

    def predict(self, trks: list[Trk], vdd, cycle_s: float) -> None:
        # ego 补偿在所有滤波器预测前统一执行
        compensate_trks(trks, vdd, cycle_s)
        self._predict(trks)

    def _predict(self, trks: list[Trk]) -> None:
        raise NotImplementedError


class AlphaBetaPredictor(_BasePredictor):
    """α-β 滤波 (FILTER.type=1) - 常速度 2 维."""

    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def _predict(self, trks: list[Trk]) -> None:
        # TODO: α-β 状态外推 x' = x + v*dt; v' = v + β*(z - x_pred)/dt
        pass


class KalmanPredictor(_BasePredictor):
    """线性 KF (FILTER.type=2) - 状态=[x,y,vx,vy]."""

    def __init__(self, dim: int, q: float, r: float):
        self.dim = dim
        self.q = q
        self.r = r

    def _predict(self, trks: list[Trk]) -> None:
        # TODO: KF 预测 x' = F*x, P' = F*P*F' + Q
        pass


class EkfPredictor(_BasePredictor):
    """扩展 KF (FILTER.type=3) - 非线性观测."""

    def __init__(self, dim: int, q, r):
        self.dim = dim
        self.q = q
        self.r = r

    def _predict(self, trks: list[Trk]) -> None:
        # TODO: EKF 预测 + 雅可比更新
        pass


class ImmPredictor(_BasePredictor):
    """交互多模型 (FILTER.type=4) - 多模型混合."""

    def __init__(self, models: list[_BasePredictor], markov: np.ndarray):
        self.models = models
        self.markov = markov

    def _predict(self, trks: list[Trk]) -> None:
        # TODO: IMM 输入混合 → 各模型预测 → 输出混合
        pass


class Filter:
    """
    滤波器主类 - 按 Cfg.FILTER.type 路由。
      predict: 
        in  :(trks, vdd, cycle_s) 
        out :trks (预测)

      update : 
        in  :(trks - cls)               
        out :trks (更新)
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.predictor = self._build_predictor(cfg)

    def _build_predictor(self, cfg: Cfg) -> _BasePredictor:
        """按 Cfg.FILTER.type 路由到具体滤波算法:
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
            return AlphaBetaPredictor(alpha=ab['alpha'], beta=ab['beta'])
        if ftype == 2:
            # 线性 KF
            kf = para.para_kf
            return KalmanPredictor(dim=kf.dim, q=kf.q, r=kf.r)
        if ftype == 3:
            # 扩展卡尔曼 EKF
            ekf = para.para_ekf
            return EkfPredictor(dim=ekf.get('dim', 4), q=ekf.q, r=ekf.r)
        if ftype == 4:
            # 交互多模型 IMM (子模型可为 α-β / KF)
            imm = para.para_imm
            sub = []
            for sub_para in imm.get('models', []):
                if sub_para.get('type') == 1:
                    sub.append(AlphaBetaPredictor(sub_para['alpha'], sub_para['beta']))
                elif sub_para.get('type') == 2:
                    sub.append(KalmanPredictor(sub_para['dim'], sub_para['q'], sub_para['r']))
            markov = np.array(imm.get('markov', [[0.9, 0.1], [0.1, 0.9]]), dtype=float)
            return ImmPredictor(models=sub, markov=markov)
        raise ValueError(f"Unsupported FILTER.type: {ftype}")

    def predict(self, trks: list[Trk], vdd, cycle_s: float) -> None:
        """统一预测入口 - 内部完成 ego 补偿 + 状态外推."""
        self.predictor.predict(trks, vdd, cycle_s)

    def update(self, matches) -> None:
        """量测更新 - 路由到具体预测器."""
        self.predictor.update(matches) if hasattr(self.predictor, 'update') else None

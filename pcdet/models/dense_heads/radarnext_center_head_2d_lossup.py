"""消融对照 head：RadarNeXtCenterHead2DLossUp（配置：纯 2d + 放大 loc_loss）。

目的：验证 center 头 BEV 优势是否源于「回归维数多 → loc_loss 更大 → 回归梯度更强」。
逻辑：保持 2d 的 6-cat 监督（不预测 z/h），仅把 CODE_WEIGHTS 从 [1.0]*6 放大到
      [8/6]*6 ≈ [1.333]*6，使 6 维 ``loc_loss`` 的量级近似等于 center 头 8 维
      ``.sum()`` 的效果（每维误差等量级假设下，scale 6×1.333 = 8 = center）。

      这是「保守下界」模拟：真实 center 的 z 是绝对高度、误差更大，放大比 > 1.333。
      若 1.333× 纯 2d 仍能把 BEV（尤其 Pedestrian）拉到 center 水平 → loss 放大坐实；
      若拉不动 → 即便真实放大比更大，loss 放大这条路也基本排除。

唯一改动：覆盖 ``self.code_weights``。targets/loss/predict 全继承自 2DNoZ（6-cat，
无 z/h 监督），保证除 loc_loss 量级外与 2d 基线完全一致。
"""
from .radarnext_center_head_2d_noz import RadarNeXtCenterHead2DNoZ


class RadarNeXtCenterHead2DLossUp(RadarNeXtCenterHead2DNoZ):
    """纯 2d head + 放大 CODE_WEIGHTS：6-cat 监督不变，loc_loss 量级对齐 center。"""

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size,
                 point_cloud_range, predict_boxes_when_training=True):
        super().__init__(model_cfg, input_channels, num_class, class_names, grid_size,
                         point_cloud_range, predict_boxes_when_training)
        # 2DNoZ 在 super 中强制 self.code_weights=[1.0]*6；此处覆盖。
        # 放大比 = center 维数(8) / 2d 维数(6)，使 6 维 .sum() ≈ center 8 维 .sum()。
        self.code_weights = [1.0 * 8.0 / 6.0] * 6

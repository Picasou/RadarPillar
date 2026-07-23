"""RPiN 阶段4 6-3：RadarNeXtCenterHead2D。
继承 RadarNeXtCenterHead，cfg 控制把 height 头输出替换为 anchor_bottom_height（去 z，
2D BEV 口径）。box_preds 仍 7 列（x,y,z_anchor,dx,dy,dz,heading）兼容 IoU 与 NMS。
"""
import torch

from .radarnext_center_head import RadarNeXtCenterHead


class RadarNeXtCenterHead2D(RadarNeXtCenterHead):
    """2D BEV head：去 z 预测，z 固定为 anchor_bottom_height（cfg 设）。

    cfg 关键差异（相对于 RadarNeXtCenterHead）：
      COMMON_HEADS 仍含 'height':(1, num_layers)（保持 parent forward/loss 兼容），
        但 forward 在 batch_dict 里将 preds_dict['height'] 替换为 cfg.ANCHOR_BOTTOM_HEIGHTS；
      USE_DIRECTION_CLASSIFIER 可关，rot 退化为单 bin（cfg.ROT_BINS=1）；
      STRIDES=[2] 表示 input 80→160 上采样（plan §0.5 S8）。
    """

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size,
                 point_cloud_range, predict_boxes_when_training=True):
        # 在父类 __init__ 之前塞入/调整 cfg（如果 cfg 不含 height 则补一个 dummy 让 parent 能跑）
        from easydict import EasyDict
        cfg = EasyDict(dict(model_cfg))
        common = dict(cfg.get('COMMON_HEADS', {}))
        if 'height' not in common:
            # 父类无条件使用 preds_dict['height']；补一个 dummy 头
            common['height'] = (1, int(cfg.get('NUM_HM_CONV', 2)))
            cfg.COMMON_HEADS = common
        super().__init__(cfg, input_channels, num_class, class_names, grid_size,
                         point_cloud_range, predict_boxes_when_training)
        # 类别 → anchor bottom height 映射（来自 cfg.ANCHOR_BOTTOM_HEIGHTS）
        self.anchor_bottom_heights = cfg.get('ANCHOR_BOTTOM_HEIGHTS', [-1.78])
        # 2D 评估协议（裁决时观察项，不参与 head*）
        self.bbox_code_size = int(cfg.get('BBOX_CODE_SIZE', 7))

    def _override_height(self, preds_dicts):
        """把每个 task 的 preds_dict['height'] 替换为 cfg.ANCHOR_BOTTOM_HEIGHTS 的均值。

        2D BEV 口径：z 固定为 anchor 高度（去 z 预测）。height 头通道数通常是 1（不分
        类），故用 anchor 表的均值作为常数填充，保持 shape (B, C_h, H, W)。
        """
        h_list = self.anchor_bottom_heights
        mean_h = float(sum(h_list)) / max(1, len(h_list))
        for preds_dict in preds_dicts:
            preds_dict['height'] = torch.full_like(preds_dict['height'], mean_h)

    def forward(self, data_dict):
        out = super().forward(data_dict)
        if self.training:
            return out
        # eval 路径：preds_dicts 在 forward_ret_dict 里；取出来改 height
        if self.forward_ret_dict is not None:
            self._override_height(self.forward_ret_dict)
        return out

    @torch.no_grad()
    def predict(self, preds_dicts, data_dict):
        """在父类 predict 之前先覆盖 height，避免 NMS 用预测的 z。"""
        self._override_height(preds_dicts)
        return super().predict(preds_dicts, data_dict)

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
        """2D BEV 口径：去 z 预测，height 按 cell 预测类别（hm argmax）填该类 anchor 底高。

        H3 修复：旧版用跨类均值（mean(-1.78,-0.6,-0.72)=-1.033）填所有 cell，
        Ped/Cyc 的 z 偏差 >1m → 3D IoU 恒低于阈值、AP 数学上为 0。改为按 cell 预测
        类别 scatter 各类 anchor_bottom_heights（与 AnchorHead 逐类 z 口径一致）。
        height 头 shape (B, 1, H, W)；hm shape (B, C, H, W)，C==len(anchor_bottom_heights)。
        """
        for preds_dict in preds_dicts:
            hm = preds_dict['hm']                       # (B, C, H, W)
            height = preds_dict['height']               # (B, 1, H, W)
            cls = hm.argmax(dim=1)                      # (B, H, W) 每 cell 预测类别
            anchors = torch.tensor(self.anchor_bottom_heights,
                                   device=height.device, dtype=height.dtype)
            preds_dict['height'] = anchors[cls].unsqueeze(1)   # (B, 1, H, W)

    # H2 修复：删除旧 forward 的 eval 分支——它把 forward_ret_dict（训练态的 dict
    # {'preds_dicts':...,'gt_boxes':...}）误传给 _override_height，train 后首次
    # in-process eval（early_stop）迭代 dict 得字符串键 → TypeError 必崩。
    # eval 的 height 覆盖已由下方 predict() override（经父类 forward 虚分发）完整承担，
    # forward 无需 override，直接继承父类。

    @torch.no_grad()
    def predict(self, preds_dicts, data_dict):
        """在父类 predict 之前先按类覆盖 height，避免 NMS 用预测的 z。"""
        self._override_height(preds_dicts)
        return super().predict(preds_dicts, data_dict)

"""RadarPillar anchor head + pillar mask（仅推理后处理优化）。

继承 AnchorHeadSingle。推理时，把落在"非空 pillar"之外的 anchor 的分类预测
置为极负，使 Detector3DTemplate 原有的 score_thresh + NMS 自动将其过滤，
从而减少进入 NMS 的候选框数量。

- 训练 / loss / anchor-GT IoU 匹配：完全不变（mask 仅推理、非 training 时生效）。
- 非空位置来源：batch_dict['voxel_coords']（VFE 体素化阶段即固定，[batch, z, y, x]）。
- 隔离性：本类独立，AnchorHeadSingle 原类不动，老实验 0 影响。
"""
import numpy as np
import torch

from .anchor_head_single import AnchorHeadSingle


class RadarPillarAnchorHeadSingle(AnchorHeadSingle):
    """AnchorHeadSingle + 推理期 pillar mask。"""

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size,
                 point_cloud_range, predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, input_channels=input_channels, num_class=num_class,
            class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training, **kwargs,
        )
        self._point_cloud_range = np.array(point_cloud_range, dtype=np.float64)
        self._grid_size = np.array(grid_size, dtype=np.float64)
        self._voxel_size = (self._point_cloud_range[3:6] - self._point_cloud_range[0:3]) / self._grid_size
        self._voxel_coords = None
        self._anchor_cell_x = None  # 惰性预计算（每个 anchor 落在哪个 pillar cell）
        self._anchor_cell_y = None

    def _ensure_anchor_cells(self):
        """预计算每个 anchor 中心落在哪个 BEV pillar cell（与 batch 无关，固定）。"""
        if self._anchor_cell_x is not None:
            return
        anchors = torch.cat(self.anchors, dim=-3) if isinstance(self.anchors, list) else self.anchors
        anchors_flat = anchors.view(-1, anchors.shape[-1]).detach().cpu()
        ax, ay = anchors_flat[:, 0], anchors_flat[:, 1]
        self._anchor_cell_x = (((ax - self._point_cloud_range[0]) / self._voxel_size[0]).long()).clamp(
            min=0, max=int(self._grid_size[0]) - 1)
        self._anchor_cell_y = (((ay - self._point_cloud_range[1]) / self._voxel_size[1]).long()).clamp(
            min=0, max=int(self._grid_size[1]) - 1)

    def forward(self, data_dict):
        self._voxel_coords = data_dict.get('voxel_coords', None)
        return super().forward(data_dict)

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        batch_cls_preds, batch_box_preds = super().generate_predicted_boxes(
            batch_size, cls_preds, box_preds, dir_cls_preds)
        if (not self.training) and (self._voxel_coords is not None):
            batch_cls_preds = self._mask_non_pillar_anchors(batch_cls_preds, batch_size)
        return batch_cls_preds, batch_box_preds

    @torch.no_grad()
    def _mask_non_pillar_anchors(self, batch_cls_preds, batch_size):
        """非 pillar 位置 anchor 的 cls 预测置极负，等效于让 score_thresh 自动过滤。"""
        self._ensure_anchor_cells()
        device = batch_cls_preds.device
        cell_x = self._anchor_cell_x.to(device)   # (num_anchors,)
        cell_y = self._anchor_cell_y.to(device)
        voxel_coords = self._voxel_coords          # (N,4) [batch, z, y, x]
        max_x, max_y = int(self._grid_size[0]), int(self._grid_size[1])
        out = batch_cls_preds.clone()
        for b in range(batch_size):
            mb = voxel_coords[:, 0] == b
            px = voxel_coords[mb, 3].long()        # pillar 的 x cell 索引
            py = voxel_coords[mb, 2].long()        # pillar 的 y cell 索引
            occ = torch.zeros((max_y, max_x), dtype=torch.bool, device=device)
            valid = (px >= 0) & (px < max_x) & (py >= 0) & (py < max_y)
            occ[py[valid], px[valid]] = True
            anchor_in = occ[cell_y, cell_x]        # (num_anchors,) 该 anchor 是否在 pillar 上
            out[b, ~anchor_in, :] = -1e9
        return out

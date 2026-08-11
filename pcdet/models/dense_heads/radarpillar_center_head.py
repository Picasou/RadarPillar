"""RadarPillar center head + pillar mask（仅推理后处理优化）。

继承 RadarNeXtCenterHead。推理时，在 heatmap 进入 predict 之前，把"非空 pillar"
之外位置的 heatmap logit 压成极负，使 predict→post_processing 的 score_thresh
自动过滤，从而减少进入 NMS 的候选框数量。

- 训练 / loss / get_targets：完全不变（mask 仅推理生效）。
- 非空位置来源：batch_dict['voxel_coords']（VFE 体素化阶段即固定，[batch, z, y, x]）。
- 隔离性：本类独立，RadarNeXtCenterHead 原类不动，老实验 0 影响。
"""
import torch

from .radarnext_center_head import RadarNeXtCenterHead


class RadarPillarCenterHead(RadarNeXtCenterHead):
    """RadarNeXtCenterHead + 推理期 pillar mask。"""

    def forward(self, data_dict):
        voxel_coords = data_dict.get('voxel_coords', None)
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        ret_dicts = []
        for task in self.tasks:
            ret_dicts.append(task(x))

        if self.training:
            self.forward_ret_dict = {
                'preds_dicts': ret_dicts,
                'gt_boxes': data_dict['gt_boxes'],
            }
        else:
            if voxel_coords is not None:
                self._mask_non_pillar_heatmap(ret_dicts, voxel_coords, data_dict['batch_size'])
            data_dict['pred_dicts'] = self.predict(ret_dicts, data_dict)
        return data_dict

    @torch.no_grad()
    def _mask_non_pillar_heatmap(self, ret_dicts, voxel_coords, batch_size):
        """非 pillar 位置的 heatmap logit 压成极负（sigmoid 后≈0，被 score_thresh 过滤）。

        用加法实现：pillar 位置加 0（不变），非 pillar 位置加 -1e9（压制）。
        """
        stride = self.out_size_factor
        NEG = -1e9
        for ret in ret_dicts:
            hm = ret['hm']                            # (B, num_cls, H, W) logit
            _, _, H, W = hm.shape
            device = hm.device
            suppress = hm.new_full((batch_size, H, W), NEG)   # 默认全部压制
            for b in range(batch_size):
                mb = voxel_coords[:, 0] == b
                px = voxel_coords[mb, 3].long() // stride     # 到 feature-map grid
                py = voxel_coords[mb, 2].long() // stride
                valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
                suppress[b, py[valid], px[valid]] = 0.0       # pillar 位置不压制
            ret['hm'] = hm + suppress.unsqueeze(1)            # (B,1,H,W) broadcast

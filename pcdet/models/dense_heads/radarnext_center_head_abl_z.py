"""消融对照 head：RadarNeXtCenterHeadAblZ（配置 X）。

目的：在 2DNoZ（去 z、去 h）基础上，**只加回 z 监督**（保留 height head），
dim 仍 2 通道（不监督 h）。用于隔离 z 监督对 BEV AP 的贡献。

监督 7-cat：offset_xy(2) + z(1) + log(dx,dy)(2) + sin/cos(2)。
predict：z 用 height head 预测值；dim 2 通道，dz 占位 0（BEV 不用）。

对照矩阵（b1 backbone）：
  2d(2DNoZ):     z 无, h 无  → BEV 48.14
  AblZ(本类):    z 有, h 无  → ?
  AblH:          z 无, h 有  → ?
  center(父类):  z 有, h 有  → BEV 51.86
"""
import numpy as np
import torch

from .radarnext_center_head import (
    RadarNeXtCenterHead,
    draw_heatmap_gaussian_np,
    gaussian_radius_np,
)


class RadarNeXtCenterHeadAblZ(RadarNeXtCenterHead):
    """配置 X：2d + z 监督（有 height head，dim 2 通道）。"""

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size,
                 point_cloud_range, predict_boxes_when_training=True):
        from easydict import EasyDict
        cfg = EasyDict(dict(model_cfg))
        common = dict(cfg.get('COMMON_HEADS', {}))
        # 保留 height（z 监督）；dim 强制 2 通道（不监督 h）
        if 'height' not in common:
            common['height'] = (1, 2)
        if 'dim' in common:
            d_classes, d_conv = common['dim']
            common['dim'] = (2, int(d_conv))
        common.pop('iou', None)
        cfg.COMMON_HEADS = common
        cfg.CODE_WEIGHTS = [1.0] * 7   # off_xy(2) + z(1) + lw(2) + rot(2)
        cfg.BBOX_CODE_SIZE = 6
        self.anchor_bottom_heights = list(cfg.get('ANCHOR_BOTTOM_HEIGHTS', [-1.78]))
        cfg.WITH_CORNER = False
        cfg.WITH_REG_IOU = False
        super().__init__(cfg, input_channels, num_class, class_names, grid_size,
                         point_cloud_range, predict_boxes_when_training)
        self.bbox_code_size = 6

    # ------------------------------------------------------------------ #
    # Targets                                                            #
    # ------------------------------------------------------------------ #
    def get_targets_single(self, gt_labels_3d, gt_bboxes_3d):
        """7-cat anno_box：offset_xy + z + log(dx,dy) + sin(heading) + cos(heading)。

        numpy 向量化 (替代逐目标 torch 循环): 与 2DNoZ 同款改造, 消除每目标 ~20 次
        GPU launch/sync (曾致 GPU 25% / 主进程单核 100%)。数值语义与 torch 版
        逐位/1ulp 等价 (对拍 golden: test_2dnoz_targets.py)。
        """
        device = gt_labels_3d.device
        max_objs = int(self.model_cfg.get('MAX_OBJS', 500)) * int(self.model_cfg.get('DENSE_REG', 1))
        gt_annotation_num = 7  # off2 + z1 + lw2 + rot2

        labels_np = gt_labels_3d.detach().cpu().numpy()
        boxes_np = gt_bboxes_3d.detach().cpu().numpy()

        pcr = np.asarray(self.point_cloud_range, dtype=np.float32)
        vs = np.asarray(self.voxel_size, dtype=np.float32)
        osf = np.float32(self.out_size_factor)
        fm_w = int(self.grid_size[0]) // self.out_size_factor
        fm_h = int(self.grid_size[1]) // self.out_size_factor
        min_overlap = float(self.model_cfg.get('GAUSSIAN_OVERLAP', 0.1))
        min_radius = int(self.model_cfg.get('MIN_RADIUS', 2))

        # 按 task 归类 GT: 与 torch 版一致 (先类别分桶, 桶内 GT 原序)
        task_boxes, task_classes = [], []
        flag2 = 0
        for class_name in self.class_names:
            parts_box, parts_cls = [], []
            for local_idx, name in enumerate(class_name):
                hit = labels_np == (local_idx + flag2)
                parts_box.append(boxes_np[hit])
                parts_cls.append(np.full(int(hit.sum()), local_idx + 1, dtype=np.int64))
            task_boxes.append(np.concatenate(parts_box, axis=0))
            task_classes.append(np.concatenate(parts_cls, axis=0))
            flag2 += len(class_name)

        heatmaps, anno_boxes, inds, masks, corner_heatmaps, cat_labels, gt_boxes = \
            [], [], [], [], [], [], []

        for idx in range(len(self.tasks)):
            heatmap = np.zeros((len(self.class_names[idx]), fm_h, fm_w), dtype=np.float32)
            corner_heatmap = np.zeros((1, fm_h, fm_w), dtype=np.float32)

            anno_box = np.zeros((max_objs, gt_annotation_num), dtype=np.float32)
            gt_box = np.zeros((max_objs, 7), dtype=np.float32)

            ind = np.zeros((max_objs), dtype=np.int64)
            mask = np.zeros((max_objs), dtype=np.uint8)
            cat_label = np.zeros((max_objs), dtype=np.int64)

            boxes = task_boxes[idx]
            classes = task_classes[idx]
            num_objs = min(boxes.shape[0], max_objs)

            for k in range(num_objs):
                cls_id = int(classes[k]) - 1

                length = boxes[k, 3] / vs[0] / osf
                width = boxes[k, 4] / vs[1] / osf

                if width > 0 and length > 0:
                    radius = gaussian_radius_np((float(width), float(length)), min_overlap)
                    radius = max(min_radius, int(radius))

                    coor_x = (boxes[k, 0] - pcr[0]) / vs[0] / osf
                    coor_y = (boxes[k, 1] - pcr[1]) / vs[1] / osf
                    cx, cy = int(coor_x), int(coor_y)

                    if not (0 <= cx < fm_w and 0 <= cy < fm_h):
                        continue

                    draw_heatmap_gaussian_np(heatmap[cls_id], cx, cy, radius)

                    rot = boxes[k, 6]

                    ind[k] = cy * fm_w + cx
                    mask[k] = 1
                    cat_label[k] = cls_id
                    # 7-cat: offset_xy + z + log(dx,dy) + sin/cos (与 2DNoZ 区别: 写入 z)
                    anno_box[k] = np.concatenate([
                        np.asarray([coor_x - cx, coor_y - cy], dtype=np.float32),
                        np.asarray([boxes[k, 2]], dtype=np.float32),
                        np.log(boxes[k, 3:5]),
                        np.asarray([np.sin(rot), np.cos(rot)], dtype=np.float32)])
                    gt_box[k] = boxes[k, 0:7]

            heatmaps.append(torch.from_numpy(heatmap).to(device))
            corner_heatmaps.append(torch.from_numpy(corner_heatmap).to(device))
            anno_boxes.append(torch.from_numpy(anno_box).to(device))
            gt_boxes.append(torch.from_numpy(gt_box).to(device))
            masks.append(torch.from_numpy(mask).to(device))
            inds.append(torch.from_numpy(ind).to(device))
            cat_labels.append(torch.from_numpy(cat_label).to(device))

        return heatmaps, anno_boxes, inds, masks, corner_heatmaps, cat_labels, gt_boxes

    # ------------------------------------------------------------------ #
    # Loss                                                               #
    # ------------------------------------------------------------------ #
    def loss_by_feat(self, preds_dicts, gt_boxes_full):
        """7-cat L1：offset_xy + z + log(dx,dy) + sin/cos。"""
        heatmaps, anno_boxes, gt_inds, gt_masks, corner_heatmaps, cat_labels, gt_boxes = \
            self.get_targets(gt_boxes_full)

        losses = {}
        for task_id, preds_dict in enumerate(preds_dicts):
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])
            hm_loss = self.crit(preds_dict['hm'], heatmaps[task_id], gt_inds[task_id],
                                gt_masks[task_id], cat_labels[task_id])
            losses.update({f'{task_id}_hm_loss': hm_loss})

            target_box = anno_boxes[task_id]
            # 7-cat：reg(2) + height(1) + dim(2) + rot(2)
            preds_dict['anno_box'] = torch.cat(
                (preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                 preds_dict['rot']), dim=1)

            box_loss = self.crit_reg(
                preds_dict['anno_box'], gt_masks[task_id], gt_inds[task_id], target_box)
            loc_loss = (box_loss * box_loss.new_tensor(self.code_weights)).sum()

            losses.update({f'{task_id}_loc_loss': loc_loss * self.weight})

        return losses

    # ------------------------------------------------------------------ #
    # Predict                                                            #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, preds_dicts, data_dict):
        """7-cat decode：z 用 height head 预测值；dim 2 通道，dz 占位 0。"""
        rets = []

        post_center_range = list(self.model_cfg.get('POST_CENTER_LIMIT_RANGE',
                                                    list(self.point_cloud_range)))
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
                device=preds_dicts[0]['hm'].device,
            )

        for task_id, preds_dict in enumerate(preds_dicts):
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            batch_hm = torch.sigmoid(preds_dict['hm'])
            batch_dim = torch.exp(preds_dict['dim'])  # (B,H,W,2) → dx, dy
            zeros_dz = torch.zeros_like(batch_dim[..., 0:1])
            batch_dim = torch.cat([batch_dim, zeros_dz], dim=-1)  # (B,H,W,3)

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']  # z 用预测值（与 2DNoZ 区别）
            batch_rot = torch.atan2(batch_rots, batch_rotc)
            batch_iou = torch.ones(
                (batch_hm.shape[0], batch_hm.shape[1], batch_hm.shape[2]),
                dtype=batch_dim.dtype, device=batch_dim.device)

            batch, H, W, num_cls = batch_hm.size()
            batch_reg = batch_reg.reshape(batch, H * W, 2)
            batch_hei = batch_hei.reshape(batch, H * W, 1)
            batch_rot = batch_rot.reshape(batch, H * W, 1)
            batch_dim = batch_dim.reshape(batch, H * W, 3)
            batch_hm = batch_hm.reshape(batch, H * W, num_cls)

            ys, xs = torch.meshgrid(
                torch.arange(0, H), torch.arange(0, W), indexing='ij')
            ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_hm.device).float()
            xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_hm.device).float()
            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]
            xs = xs * self.out_size_factor * self.voxel_size[0] + self.point_cloud_range[0]
            ys = ys * self.out_size_factor * self.voxel_size[1] + self.point_cloud_range[1]

            batch_box_preds = torch.cat(
                [xs, ys, batch_hei, batch_dim, batch_rot], dim=2)
            rets.append(self.post_processing(task_id, batch_box_preds,
                                             batch_hm, post_center_range, batch_iou))

        num_samples = len(rets[0])
        pred_dicts = []
        for i in range(num_samples):
            bboxes = torch.cat([ret[i]['bboxes'] for ret in rets], dim=0) if rets else \
                torch.zeros((0, 7), device=preds_dicts[0]['hm'].device)
            scores = torch.cat([ret[i]['scores'] for ret in rets], dim=0) if rets else \
                torch.zeros((0,), device=preds_dicts[0]['hm'].device)
            labels_list = []
            flag = 0
            for j, num_class in enumerate(self.num_classes):
                labels_list.append(rets[j][i]['labels'] + flag + 1)
                flag += num_class
            labels = torch.cat(labels_list, dim=0) if labels_list else \
                torch.zeros((0,), dtype=torch.int64, device=preds_dicts[0]['hm'].device)

            pred_dicts.append({
                'pred_boxes': bboxes,
                'pred_scores': scores,
                'pred_labels': labels,
            })
        return pred_dicts

"""RPiN head 阶段：RadarNeXtCenterHead2DNoZ。

字面 2D head：完全去 z 通道，输出 [x, y, dx, dy, heading, type] 六元组。
继承 RadarNeXtCenterHead，重写 forward 路径中 3 个接触 height 的方法：
    - get_targets_single  ：写 6-cat anno_box（offset_xy, log_dx, log_dy, sin_cos_h）
    - loss_by_feat        ：拼 6-cat anno_box，不参与 height L1
    - predict             ：decode 时按类填 z（来自 cfg.ANCHOR_BOTTOM_HEIGHTS），
                            保持 box 7 维兼容 NMS/IoU

cfg 关键差异（相对于 RadarNeXtCenterHead）：
    - COMMON_HEADS 不含 'height'；dim 由 3 通道变 2 通道（只预测 l, w）
    - CODE_WEIGHTS = [1.0] * 6
    - BBOX_CODE_SIZE = 5（x, y, dx, dy, heading；评估时 z=anchor_bottom_height 填）
    - ANCHOR_BOTTOM_HEIGHTS = [-1.78, -0.6, -0.72]（Car / Ped / Cyc，按 hm argmax 查）
"""
import torch

from .radarnext_center_head import (
    RadarNeXtCenterHead,
    draw_heatmap_gaussian,
    gaussian_radius,
)


class RadarNeXtCenterHead2DNoZ(RadarNeXtCenterHead):
    """2D head：去 z 通道；6-cat 监督；BEV 评估口径。"""

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size,
                 point_cloud_range, predict_boxes_when_training=True):
        # cfg 调整：把 height 从 common_heads 摘掉（如果还在就删），并把 dim 通道数锁为 2
        from easydict import EasyDict
        cfg = EasyDict(dict(model_cfg))
        common = dict(cfg.get('COMMON_HEADS', {}))
        common.pop('height', None)  # 父类 parent parent 仍会读 height；保险起见显式删
        # dim 必须为 2 通道（l, w），与 6-cat 监督对齐
        if 'dim' in common:
            d_classes, d_conv = common['dim']
            common['dim'] = (2, int(d_conv))
        cfg.COMMON_HEADS = common
        # code_weights 强制 6 个 1.0
        cfg.CODE_WEIGHTS = [1.0] * 6
        cfg.BBOX_CODE_SIZE = 5
        # 类别 -> anchor bottom height（预测时按类填 z，box 7 维兼容 NMS）
        self.anchor_bottom_heights = list(cfg.get('ANCHOR_BOTTOM_HEIGHTS', [-1.78]))
        # IoU/corner 等辅助开关全部关（与 2D 头口径一致；如需开需要重新对齐 7 维 box）
        cfg.WITH_CORNER = False
        cfg.WITH_REG_IOU = False
        if 'iou' in common:
            common.pop('iou')
            cfg.COMMON_HEADS = common
        super().__init__(cfg, input_channels, num_class, class_names, grid_size,
                         point_cloud_range, predict_boxes_when_training)
        # self.code_weights 是 list[1.0]*6（父类从 cfg.CODE_WEIGHTS 读）
        # self.bbox_code_size = 5
        # self.common_heads 不含 height
        # 覆盖 bbox_code_size（父类已从 cfg 读为 5）
        self.bbox_code_size = 5

    # ------------------------------------------------------------------ #
    # Targets                                                            #
    # ------------------------------------------------------------------ #
    def get_targets_single(self, gt_labels_3d, gt_bboxes_3d):
        """写 6-cat anno_box：offset_xy + log(dx,dy) + sin(heading) + cos(heading)。"""
        device = gt_labels_3d.device
        max_objs = int(self.model_cfg.get('MAX_OBJS', 500)) * int(self.model_cfg.get('DENSE_REG', 1))
        grid_size = torch.tensor(self.grid_size, device=device)
        pc_range = torch.tensor(self.point_cloud_range, device=device)
        voxel_size = torch.tensor(self.voxel_size, device=device)
        gt_annotation_num = 6  # 与 code_weights 长度一致：offset2+log_dim2+rot2

        feature_map_size = (grid_size[:2] // self.out_size_factor).int()

        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)

        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks, corner_heatmaps, cat_labels, gt_boxes = \
            [], [], [], [], [], [], []

        for idx in range(len(self.tasks)):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1], feature_map_size[0]))
            corner_heatmap = torch.zeros(
                (1, feature_map_size[1], feature_map_size[0]),
                dtype=torch.float32, device=device)

            anno_box = gt_bboxes_3d.new_zeros((max_objs, gt_annotation_num), dtype=torch.float32)
            gt_box = gt_bboxes_3d.new_zeros((max_objs, 7), dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)
            cat_label = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                length = task_boxes[idx][k][3]
                width = task_boxes[idx][k][4]
                length = length / voxel_size[0] / self.out_size_factor
                width = width / voxel_size[1] / self.out_size_factor

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (width, length),
                        min_overlap=float(self.model_cfg.get('GAUSSIAN_OVERLAP', 0.1)))
                    radius = max(int(self.model_cfg.get('MIN_RADIUS', 2)), int(radius))

                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][1], task_boxes[idx][k][2]

                    coor_x = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor
                    coor_y = (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                    center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)

                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    # 6-cat 目标：offset_x, offset_y, log(dx), log(dy), sin(h), cos(h)
                    # 不写 z 监督（字面 2D 头）；gt_box 仍写 7 维以兼容 IoU 计算时的 z
                    radius = radius // 2
                    rot = task_boxes[idx][k][6]
                    box_dim_2d = task_boxes[idx][k][3:5]  # dx, dy (no dz)
                    box_dim_2d = box_dim_2d.log()

                    new_idx = k
                    x_int, y_int = center_int[0], center_int[1]

                    assert (y_int * feature_map_size[0] + x_int <
                            feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = y_int * feature_map_size[0] + x_int
                    mask[new_idx] = 1
                    cat_label[new_idx] = cls_id

                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x_int, y_int], device=device),
                        box_dim_2d,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0)
                    ])
                    gt_box[new_idx] = task_boxes[idx][k][0:7]

            heatmaps.append(heatmap)
            corner_heatmaps.append(corner_heatmap)
            anno_boxes.append(anno_box)
            inds.append(ind)
            masks.append(mask)
            cat_labels.append(cat_label)
            gt_boxes.append(gt_box)

        return heatmaps, anno_boxes, inds, masks, corner_heatmaps, cat_labels, gt_boxes

    # ------------------------------------------------------------------ #
    # Loss                                                               #
    # ------------------------------------------------------------------ #
    def loss_by_feat(self, preds_dicts, gt_boxes_full):
        """6-cat L1（offset_xy + log_dx_dy + sin_cos_h），无 z 监督。"""
        heatmaps, anno_boxes, gt_inds, gt_masks, corner_heatmaps, cat_labels, gt_boxes = \
            self.get_targets(gt_boxes_full)

        losses = {}
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])
            hm_loss = self.crit(preds_dict['hm'], heatmaps[task_id], gt_inds[task_id],
                                gt_masks[task_id], cat_labels[task_id])
            losses.update({f'{task_id}_hm_loss': hm_loss})

            target_box = anno_boxes[task_id]
            # 6-cat 拼接：offset_xy(2) + log_dx_dy(2) + sin_cos_h(2)
            preds_dict['anno_box'] = torch.cat(
                (preds_dict['reg'], preds_dict['dim'], preds_dict['rot']), dim=1)

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
        """Decode 6-cat 输出，按类填 z（来自 anchor_bottom_heights），拼 7 维 box。"""
        rets = []

        post_center_range = list(self.model_cfg.get('POST_CENTER_LIMIT_RANGE',
                                                    list(self.point_cloud_range)))
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
                device=preds_dicts[0]['hm'].device,
            )

        # 类别 -> anchor bottom height tensor
        anchor_heights = torch.tensor(self.anchor_bottom_heights,
                                      dtype=preds_dicts[0]['hm'].dtype,
                                      device=preds_dicts[0]['hm'].device)

        for task_id, preds_dict in enumerate(preds_dicts):
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            batch_hm = torch.sigmoid(preds_dict['hm'])
            batch_dim = torch.exp(preds_dict['dim'])  # (B, H, W, 2) → dx, dy
            # dz 不输出，置 0 占位（box 仍 7 维兼容 NMS/IoU，但 3D IoU 不会有效；
            # BEV IoU 不看 dz 故不受影响。2D 变体 early_stop 走 bev metric 即可）
            zeros_dz = torch.zeros_like(batch_dim[..., 0:1])
            batch_dim = torch.cat([batch_dim, zeros_dz], dim=-1)  # (B, H, W, 3)

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_rot = torch.atan2(batch_rots, batch_rotc)
            batch_iou = torch.ones(
                (batch_hm.shape[0], batch_hm.shape[1], batch_hm.shape[2]),
                dtype=batch_dim.dtype, device=batch_dim.device)

            batch, H, W, num_cls = batch_hm.size()
            batch_reg = batch_reg.reshape(batch, H * W, 2)
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

            # z 按 hm argmax 查 anchor_heights（每 cell 预测类别 → 查对应 z）
            cls = batch_hm.argmax(dim=-1)  # (B, H*W)
            batch_hei = anchor_heights[cls].unsqueeze(-1)  # (B, H*W, 1)

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


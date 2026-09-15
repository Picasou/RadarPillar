"""RPiN head 2D vel branch：RadarNeXtCenterHead2DTrunkVel（Trunk 全共享 + vel 回归头）。

8-cat 监督: offset_xy + log_lw + sin_cos_h + vx_vy(车体系对地速度, 原值不编码)。
vel 来源: gt_boxes 9 列时取 [:, 7:9]; 仍 7 列(未重生 vel infos)时补 0 并 warning
一次(链路可冒烟, 数值意义待数据侧重生 9 维 infos)。
loss: vel 并入现有 RegLoss L1, code_weights 8 项; 拆 loc_loss/vel_loss 两键独立监控。
predict: box 9 维 [x,y,z,l,w,h,rot,vx,vy]; NMS/eval/IoU 只消费前 7 维——
父类 post_processing 的 [-1] 索引在 9 维下会误取 vy 当 rot, 且空 tensor 硬编码
7 维, 故覆写(rot 恒 index 6, 空 tensor 动态宽)。
"""
import warnings

import numpy as np
import torch

from .radarnext_center_head import draw_heatmap_gaussian_np, gaussian_radius_np
from .radarnext_center_head_2d_trunk import RadarNeXtCenterHead2DTrunk


class RadarNeXtCenterHead2DTrunkVel(RadarNeXtCenterHead2DTrunk):
    """2D Trunk + vel branch：8-cat 监督, 9 维 box 输出。"""

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size,
                 point_cloud_range, predict_boxes_when_training=True):
        super().__init__(model_cfg, input_channels, num_class, class_names,
                         grid_size, point_cloud_range, predict_boxes_when_training)
        # NoZ 基类强制 [1.0]*6, vel 版 8(reg2+dim2+rot2+vel2)
        self.code_weights = [1.0] * 8
        self._warned_no_vel = False

    # ------------------------------------------------------------------ #
    # Targets                                                            #
    # ------------------------------------------------------------------ #
    def get_targets(self, gt_boxes_full):
        """整 batch 监督 target(vel 版列分支): 10 列取 vel, 8 列兼容现状。

        10 列 [xyz,lwh,rot,vx,vy,class] → box 9 维(labels 取末列);
        8 列 [xyz,lwh,rot,class](现状 infos 无 vel) → box 7 维;
        class 恒 1-based → 0-based。基类仅认 8 列, 其余列数静默 labels 全 0, 故覆写。
        """
        batch_size = gt_boxes_full.shape[0]
        last = gt_boxes_full.shape[-1]
        if last == 10:
            gt_bboxes = gt_boxes_full[..., :9]
            gt_labels = gt_boxes_full[..., 9].long() - 1
        elif last == 8:
            gt_bboxes = gt_boxes_full[..., :7]
            gt_labels = gt_boxes_full[..., 7].long() - 1
        else:
            raise ValueError(
                f'vel head gt_boxes 列数异常: {last} (期望 8=7+cls 或 10=7+vel2+cls)')

        sample_labels = [gt_labels[b] for b in range(batch_size)]
        sample_bboxes = [gt_bboxes[b] for b in range(batch_size)]

        results = [self.get_targets_single(lab, box)
                   for lab, box in zip(sample_labels, sample_bboxes)]
        # results: list (B) of 7-tuples; 转置成 per-task batched tensors(同基类)
        keys_count = 7
        transposed = [[[] for _ in range(len(self.tasks))] for _ in range(keys_count)]
        for r in results:
            for ki, per_task_list in enumerate(r):
                for ti, tensor in enumerate(per_task_list):
                    transposed[ki][ti].append(tensor)
        stacked = [[torch.stack(transposed[ki][ti]) for ti in range(len(self.tasks))]
                   for ki in range(keys_count)]
        heatmaps, anno_boxes, inds, masks, corner_heatmaps, cat_labels, gt_boxes = stacked
        return heatmaps, anno_boxes, inds, masks, corner_heatmaps, cat_labels, gt_boxes

    def get_targets_single(self, gt_labels_3d, gt_bboxes_3d):
        """写 8-cat anno_box：6-cat 基础上追加 vx,vy(9 列 GT) 或 0(7 列 GT)。

        numpy 向量化口径与 NoZ 版一致(循环体纯 numpy, 结尾每 task 一次上卡)。
        """
        device = gt_labels_3d.device
        max_objs = int(self.model_cfg.get('MAX_OBJS', 500)) * int(self.model_cfg.get('DENSE_REG', 1))
        gt_annotation_num = 8  # offset2+log_dim2+rot2+vel2, 与 code_weights 长度一致

        labels_np = gt_labels_3d.detach().cpu().numpy()
        boxes_np = gt_bboxes_3d.detach().cpu().numpy()

        # vel 自适应: 9 列真监督; 7 列(旧 infos)补 0, warning 一次
        has_vel = boxes_np.shape[1] >= 9
        if not has_vel and not self._warned_no_vel:
            warnings.warn('MSR vel head: gt_boxes 7 列(未重生 vel infos), vel 监督补 0')
            self._warned_no_vel = True

        pcr = np.asarray(self.point_cloud_range, dtype=np.float32)
        vs = np.asarray(self.voxel_size, dtype=np.float32)
        osf = np.float32(self.out_size_factor)
        fm_w = int(self.grid_size[0]) // self.out_size_factor
        fm_h = int(self.grid_size[1]) // self.out_size_factor
        min_overlap = float(self.model_cfg.get('GAUSSIAN_OVERLAP', 0.1))
        min_radius = int(self.model_cfg.get('MIN_RADIUS', 2))

        # 按 task 归类 GT: 与 NoZ 版一致, 先按类别分桶、桶内保持 GT 原序
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
            # 2D 头无 corner 监督, 恒 0 (与 NoZ 版一致)
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

                # length,width 折成 feature cell (只用于 radius, 同 NoZ 版)
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

                    # 8-cat 目标: 6-cat + vx,vy; dx,dy 用原始米制, vel 原值直传
                    rot = boxes[k, 6]
                    box_dim_2d = boxes[k, 3:5]
                    vel_2d = boxes[k, 7:9] if has_vel else np.zeros(2, dtype=np.float32)

                    ind[k] = cy * fm_w + cx
                    mask[k] = 1
                    cat_label[k] = cls_id
                    anno_box[k] = np.concatenate([
                        np.asarray([coor_x - cx, coor_y - cy], dtype=np.float32),
                        np.log(box_dim_2d),
                        np.asarray([np.sin(rot), np.cos(rot)], dtype=np.float32),
                        np.asarray(vel_2d, dtype=np.float32)])
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
        """8-cat L1（6-cat + vx,vy），拆 loc_loss/vel_loss 两键独立监控。"""
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
            # 8-cat 拼接：offset_xy(2) + log_dx_dy(2) + sin_cos_h(2) + vx_vy(2)
            preds_dict['anno_box'] = torch.cat(
                (preds_dict['reg'], preds_dict['dim'], preds_dict['rot'],
                 preds_dict['vel']), dim=1)

            box_loss = self.crit_reg(
                preds_dict['anno_box'], gt_masks[task_id], gt_inds[task_id], target_box)
            # RegLoss 输出逐 cat 维向量(dim=8): 前 6 归 loc_loss, 后 2 归 vel_loss
            box_loss = box_loss * box_loss.new_tensor(self.code_weights)
            loc_loss = box_loss[0:6].sum()
            vel_loss = box_loss[6:8].sum()

            losses.update({f'{task_id}_loc_loss': loc_loss * self.weight,
                           f'{task_id}_vel_loss': vel_loss * self.weight})

        return losses

    # ------------------------------------------------------------------ #
    # Predict                                                            #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, preds_dicts, data_dict):
        """Decode 8-cat 输出，拼 9 维 box [x,y,z,l,w,h,rot,vx,vy]（vel 直通）。"""
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
            # dz 不输出，置 0 占位（BEV IoU 不看 dz; 3D 口径与 NoZ 版一致恒 0）
            zeros_dz = torch.zeros_like(batch_dim[..., 0:1])
            batch_dim = torch.cat([batch_dim, zeros_dz], dim=-1)  # (B, H, W, 3)

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_rot = torch.atan2(batch_rots, batch_rotc)
            batch_vel = preds_dict['vel']  # (B, H, W, 2) vx, vy 原值直通
            batch_iou = torch.ones(
                (batch_hm.shape[0], batch_hm.shape[1], batch_hm.shape[2]),
                dtype=batch_dim.dtype, device=batch_dim.device)

            batch, H, W, num_cls = batch_hm.size()
            batch_reg = batch_reg.reshape(batch, H * W, 2)
            batch_rot = batch_rot.reshape(batch, H * W, 1)
            batch_dim = batch_dim.reshape(batch, H * W, 3)
            batch_vel = batch_vel.reshape(batch, H * W, 2)
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

            # 9 维: [x,y,z,l,w,h,rot,vx,vy], vel 附尾不进 NMS/IoU
            batch_box_preds = torch.cat(
                [xs, ys, batch_hei, batch_dim, batch_rot, batch_vel], dim=2)
            rets.append(self.post_processing(task_id, batch_box_preds,
                                             batch_hm, post_center_range, batch_iou))

        num_samples = len(rets[0])
        pred_dicts = []
        for i in range(num_samples):
            bboxes = torch.cat([ret[i]['bboxes'] for ret in rets], dim=0) if rets else \
                torch.zeros((0, 9), device=preds_dicts[0]['hm'].device)
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

    # ------------------------------------------------------------------ #
    # Post processing                                                    #
    # ------------------------------------------------------------------ #
    def post_processing(self, task_id, batch_box_preds, batch_hm, post_center_range, batch_iou):
        """Per-sample NMS + rectifier scoring; 9 维 box 全列通过(rot 恒 index 6)。"""
        nms_cfg = self.model_cfg.NMS_CONFIG
        score_threshold = float(self.model_cfg.get('SCORE_THRESHOLD', 0.1))
        batch_size = len(batch_hm)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]
            iou_preds = batch_iou[i].view(-1)
            scores, labels = torch.max(hm_preds, dim=-1)
            score_mask = scores > score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
                & (box_preds[..., :3] <= post_center_range[3:]).all(1)

            mask = distance_mask & score_mask

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]
            iou_preds = torch.clamp(iou_preds[mask], min=0., max=1.)
            rectifier = torch.tensor(self.rectifier[task_id], device=hm_preds.device).to(scores)
            scores = torch.pow(scores, 1 - rectifier[labels]) * torch.pow(iou_preds, rectifier[labels])

            # 动态宽(9 维; 空 mask 后仍保 (0,9))
            selected_boxes = torch.zeros((0, box_preds.shape[-1]),
                                         device=box_preds.device, dtype=box_preds.dtype)
            selected_labels = torch.zeros((0,), dtype=torch.int64, device=labels.device)
            selected_scores = torch.zeros((0,), dtype=scores.dtype, device=scores.device)
            for class_id in range(hm_preds.shape[-1]):
                class_sel = (labels == class_id)
                scores_class = scores[class_sel]
                labels_class = labels[class_sel]
                box_preds_class = box_preds[class_sel]
                # rot 恒 index 6(父类 [-1] 在 9 维下会误取 vy 当 rot)
                boxes_for_nms_class = box_preds_class[:, [0, 1, 2, 3, 4, 5, 6]]
                selected = self._rotate_nms(
                    boxes_for_nms_class, scores_class,
                    thresh=float(nms_cfg.NMS_THRESH),
                    pre_maxsize=int(nms_cfg.NMS_PRE_MAXSIZE),
                    post_max_size=int(nms_cfg.NMS_POST_MAXSIZE),
                )
                selected_boxes = torch.cat((selected_boxes, box_preds_class[selected]), dim=0)
                selected_scores = torch.cat((selected_scores, scores_class[selected]), dim=0)
                selected_labels = torch.cat((selected_labels, labels_class[selected]), dim=0)

            prediction_dict = {
                'bboxes': selected_boxes,
                'scores': selected_scores,
                'labels': selected_labels,
            }
            prediction_dicts.append(prediction_dict)

        return prediction_dicts

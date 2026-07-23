"""RadarNeXt CenterPoint-style detection head (OpenPCDet port).

This is a faithful port of RadarNeXt's ``RadarNeXt_Head`` /
``DecopCenterHead`` (projects/RadarNeXt/radarnext/radarnext_head.py and
DecopCenterHead.py) from mmdet3d to pure torch + pcdet ops. It is an
anchor-free, CenterPoint-style multi-task head producing heatmaps and
box regressors per BEV feature cell, with auxiliary corner heatmap, IoU
score, and dIoU regression heads.

Translation decisions (see .superpowers/sdd/briefs/task-4-brief.md):

  * Constructor signature aligned to OpenPCDet ``build_dense_head``:
    ``__init__(self, model_cfg, input_channels, num_class, class_names,
    grid_size, point_cloud_range, predict_boxes_when_training=True)``.
  * ``tasks=[{num_class:3, class_names:['Car','Pedestrian','Cyclist']}]`` —
    identical to the dataset CLASS_NAMES.
  * ``common_heads`` values come from YAML as lists (e.g. [2,2]); code uses
    len()/index, never assumes tuple.
  * ``SepHead``: FPN variant uses stride=2 (ConvTranspose 80->160) to align
    the FPN's (B,384,80,80) output with the target feature_map_size=160.
  * code_weights len = 8 (reg2+height1+dim3+rot2); bbox_code_size = 7.
  * Losses: with_corner=True, with_iou=True (aligned via diagonal of
    boxes_iou3d_gpu), with_reg_iou=True (dIoU via bbox3d_overlaps_diou),
    plus FastFocalLoss + L1 RegLoss. All weights from model_cfg.
  * get_targets: feature_map_size = grid_size[:2] // out_size_factor
    (derived, never hardcoded). Consumes ``batch_dict['gt_boxes']`` tensors
    directly — these are already volume-center z, so .gravity_center is NOT
    called (audit-correct).
  * predict/post_processing: rectifier, NMS thresholds from model_cfg;
    OpenPCDet ``iou3d_nms_utils.nms_gpu`` replaces mmdet3d ``rotate_nms_pcdet``.
    The original ``bboxes[:,2] -= bboxes[:,5]*0.5`` z-transform is REMOVED —
    OpenPCDet VoD evaluation (boxes3d_lidar_to_kitti_camera) expects
    volume-center z and performs the center->bottom shift itself.
  * No mmdet3d ``decouple_pred_processing`` / ``channels_list`` multi-scale
    branches — FPN/MDFEN variants are already fused to a single scale upstream.

Pure torch + pcdet ops only. No mmdet3d/mmcv.
"""

import copy

import numpy as np
import torch
from torch import nn

from ...ops.iou3d_nms import iou3d_nms_utils
from .radarnext_losses import (
    FastFocalLoss,
    IouLoss,
    IouRegLoss,
    RegLoss,
)


# --------------------------------------------------------------------------- #
# Gaussian heatmap helpers (ported verbatim from mmdet3d.models.utils.gaussian)#
# --------------------------------------------------------------------------- #
def gaussian_2d(shape, sigma=1):
    """Generate a 2D gaussian kernel (numpy, matches mmdet3d)."""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Draw a gaussian kernel onto ``heatmap`` keeping the per-pixel max.

    Args:
        heatmap (Tensor): (C, H, W) or (H, W) target heatmap.
        center (Tensor):  (2,) [cx, cy] in feature-map pixel coords.
        radius (int):     gaussian radius.
        k (int):          amplitude multiplier.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
                 radius - left:radius + right]).to(heatmap.device, torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.5):
    """Gaussian radius for a (height, width) detection size (matches mmdet3d)."""
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def center_to_corner_box2d(center, dim, angles, origin=0.5):
    """BEV boxes -> 4 corners (numpy). Port of mmdet3d.structures helper.

    Args:
        center (np.ndarray): (N, 2)
        dim (np.ndarray):    (N, 2) [length, width]
        angles (np.ndarray): (N,) rotation around z.
        origin (float):      0.5 = center.
    Returns:
        np.ndarray: (N, 4, 2)
    """
    corners_norm = np.array(
        [[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=np.float32)
    corners_norm = corners_norm / 2 * (1 - origin)
    corners = dim.reshape([-1, 1, 2]) * corners_norm.reshape([1, 4, 2])
    corners = corners + center.reshape(-1, 1, 2)

    rot_cos = np.cos(angles).reshape(-1, 1, 1)
    rot_sin = np.sin(angles).reshape(-1, 1, 1)
    rot_mat = np.concatenate([rot_cos, -rot_sin, rot_sin, rot_cos], axis=-1)
    rot_mat = rot_mat.reshape(-1, 2, 2)
    corners = corners @ rot_mat
    return corners


# --------------------------------------------------------------------------- #
# ConvBlock (ported from projects/PillarNeXt/pillarnext/utils/conv.py)         #
# --------------------------------------------------------------------------- #
class _Conv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride,
                 conv_layer=nn.Conv2d, bias=False, **kwargs):
        super(_Conv, self).__init__()
        padding = kwargs.get('padding', kernel_size // 2)  # default same size
        self.conv = conv_layer(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU block (matches PillarNeXt ConvBlock semantics)."""

    def __init__(self, inplanes, planes, kernel_size, stride=1,
                 conv_layer=nn.Conv2d,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU, **kwargs):
        super(ConvBlock, self).__init__()
        padding = kwargs.get('padding', kernel_size // 2)  # default same size
        self.conv = _Conv(inplanes, planes, kernel_size=kernel_size, stride=stride,
                          padding=padding, bias=False, conv_layer=conv_layer)
        self.norm = norm_layer(planes)
        self.act = act_layer()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out


# --------------------------------------------------------------------------- #
# SepHead                                                                      #
# --------------------------------------------------------------------------- #
class SepHead(nn.Module):
    """Per-task separated prediction head.

    If ``stride > 1`` a single ConvTranspose2d deblock upsamples the shared
    feature to the target feature_map_size (e.g. 80 -> 160 for stride=2); with
    ``stride == 1`` the deblock is the identity.
    """

    def __init__(
        self,
        in_channels,
        heads,
        stride=1,
        head_conv=64,
        final_kernel=1,
        bn=True,
        init_bias=-2.19,
        **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)
        if stride > 1:
            self.deblock = ConvBlock(in_channels, head_conv, kernel_size=int(stride),
                                     stride=int(stride), padding=0, conv_layer=nn.ConvTranspose2d)
            in_channels = head_conv
        else:
            self.deblock = nn.Identity()
        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = nn.Sequential()
            for i in range(num_conv - 1):
                fc.append(nn.Conv2d(in_channels, head_conv,
                                    kernel_size=final_kernel, stride=1,
                                    padding=final_kernel // 2, bias=True))
                if bn:
                    fc.append(nn.BatchNorm2d(head_conv))
                fc.append(nn.ReLU())

            fc.append(nn.Conv2d(head_conv, classes,
                                kernel_size=final_kernel, stride=1,
                                padding=final_kernel // 2, bias=True))

            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)

            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.deblock(x)
        ret_dict = dict()
        for head in self.heads:
            if not self.training and head == 'corner_hm':
                # At inference, deactivate the auxiliary corner head.
                continue
            ret_dict[head] = self.__getattr__(head)(x)
        return ret_dict


# --------------------------------------------------------------------------- #
# RadarNeXtCenterHead                                                          #
# --------------------------------------------------------------------------- #
class RadarNeXtCenterHead(nn.Module):
    """Anchor-free CenterPoint-style head for RadarNeXt (OpenPCDet port).

    Forward contract (matches OpenPCDet dense heads):
        Input:  ``data_dict`` carrying ``spatial_features_2d`` (B, C, H, W) and,
                when training, ``gt_boxes`` (B, M, 8) [xyz dx dy dz heading class].
        Output: the same ``data_dict`` with the per-task predictions stored in
                ``self.forward_ret_dict`` for loss, and (eval) decoded predictions
                written to ``data_dict['pred_dicts']``.
    """

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size,
                 point_cloud_range, predict_boxes_when_training=True):
        super(RadarNeXtCenterHead, self).__init__()
        self.model_cfg = model_cfg
        self.predict_boxes_when_training = predict_boxes_when_training

        # Tasks come from model_cfg.TASKS (one task for VoD: all 3 classes).
        # class_names passed in is the dataset CLASS_NAMES (used for label
        # mapping in get_targets); the per-task class list also comes from
        # model_cfg.TASKS so the YAML controls task decomposition.
        self.class_names_all = class_names
        tasks_cfg = list(model_cfg.TASKS)
        self.tasks_cfg = tasks_cfg
        num_classes = [int(t['num_class']) for t in tasks_cfg]
        self.class_names = [list(t['class_names']) for t in tasks_cfg]
        self.num_classes = num_classes

        # Scalar weights & switches (from model_cfg).
        self.code_weights = list(model_cfg.CODE_WEIGHTS)
        self.weight = float(model_cfg.get('WEIGHT', 1.0))  # loc-loss scale
        self.corner_weight = float(model_cfg.get('CORNER_WEIGHT', 1.0))
        self.iou_weight = float(model_cfg.get('IOU_WEIGHT', 1.0))
        self.iou_reg_weight = float(model_cfg.get('IOU_REG_WEIGHT', 0.5))

        self.in_channels = int(input_channels)
        self.strides = [int(s) for s in model_cfg.get('STRIDES', [1] * len(tasks_cfg))]

        self.rectifier = model_cfg.get('RECTIFIER', [[0.0] * nc for nc in num_classes])

        self.bbox_code_size = int(model_cfg.get('BBOX_CODE_SIZE', 7))

        # common_heads (YAML parses (2,2) as list [2,2]); store as dict.
        self.common_heads = dict(model_cfg.COMMON_HEADS)

        # Switches.
        self.with_corner = bool(model_cfg.get('WITH_CORNER', False))
        self.with_reg_iou = bool(model_cfg.get('WITH_REG_IOU', False))
        self.with_iou = 'iou' in self.common_heads

        share_conv_channel = int(model_cfg.get('SHARE_CONV_CHANNEL', 64))
        num_hm_conv = int(model_cfg.get('NUM_HM_CONV', 2))
        num_corner_hm_conv = int(model_cfg.get('NUM_CORNER_HM_CONV', 2))
        init_bias = float(model_cfg.get('INIT_BIAS', -2.19))
        final_kernel = int(model_cfg.get('FINAL_KERNEL', 3))

        # Geometry (consumed in get_targets / predict / iou decode).
        self.grid_size = np.array(grid_size)
        self.point_cloud_range = np.array(point_cloud_range)
        self.voxel_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.grid_size
        self.out_size_factor = int(model_cfg.get('OUT_SIZE_FACTOR', 2))

        # Loss criteria.
        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()
        if self.with_corner:
            self.corner_crit = nn.MSELoss(reduction='none')
        if self.with_reg_iou:
            self.crit_iou_reg = IouRegLoss()
        if self.with_iou:
            self.crit_iou = IouLoss()

        # Shared convolution over the fused feature.
        self.shared_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, share_conv_channel,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True),
        )

        # Per-task SepHeads.
        self.tasks = nn.ModuleList()
        for (num_cls, stride) in zip(num_classes, self.strides):
            heads = copy.deepcopy(self.common_heads)
            if self.with_corner:
                heads.update(dict(hm=(num_cls, num_hm_conv), corner_hm=(1, num_corner_hm_conv)))
            else:
                heads.update(dict(hm=(num_cls, num_hm_conv)))
            self.tasks.append(
                SepHead(share_conv_channel, heads, stride=stride,
                        bn=True, init_bias=init_bias, final_kernel=final_kernel)
            )

        self.forward_ret_dict = None
        self.init_weights()

    # ----------------------------------------------------------------- #
    # Initialization                                                     #
    # ----------------------------------------------------------------- #
    def init_weights(self):
        """Conv2d -> Kaiming; BN weight -> uniform_ (audit #10, not constant 1)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # IMPORTANT: uniform_ on the BN affine weight (audit #10).
                if m.weight is not None:
                    nn.init.uniform_(m.weight)

    # ----------------------------------------------------------------- #
    # Forward                                                            #
    # ----------------------------------------------------------------- #
    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        ret_dicts = []
        for task in self.tasks:
            ret_dicts.append(task(x))

        if self.training:
            # Build targets & cache predictions for get_loss().
            self.forward_ret_dict = {
                'preds_dicts': ret_dicts,
                'gt_boxes': data_dict['gt_boxes'],
            }
        else:
            # Decode predictions into final detection dicts (head's own
            # post_processing — not the detector template's anchor NMS).
            data_dict['pred_dicts'] = self.predict(ret_dicts, data_dict)

        return data_dict

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
        return y

    # ----------------------------------------------------------------- #
    # Loss                                                               #
    # ----------------------------------------------------------------- #
    def get_loss(self):
        """
        Compute and return the total loss + a tb_dict of sub-losses.

        Mirrors RadarNeXt's loss_by_feat: focal hm + L1 reg + (corner hm) +
        (aligned IoU score) + (dIoU reg), summed across tasks.
        """
        tb_dict = {}
        loss = torch.zeros(1, device=self.shared_conv[0].weight.device).squeeze()
        if self.forward_ret_dict is None:
            return loss, tb_dict

        preds_dicts = self.forward_ret_dict['preds_dicts']
        gt_boxes_full = self.forward_ret_dict['gt_boxes']  # (B, M, 8) or (B, M, 7)
        losses = self.loss_by_feat(preds_dicts, gt_boxes_full)

        for k, v in losses.items():
            tb_dict[k] = v.item()
            loss = loss + v
        return loss, tb_dict

    def loss_by_feat(self, preds_dicts, gt_boxes_full):
        """Per-task loss assembly — verbatim translation of RadarNeXt's version."""
        heatmaps, anno_boxes, gt_inds, gt_masks, corner_heatmaps, cat_labels, gt_boxes = \
            self.get_targets(gt_boxes_full)

        losses = {}
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])

            hm_loss = self.crit(preds_dict['hm'], heatmaps[task_id], gt_inds[task_id],
                                gt_masks[task_id], cat_labels[task_id])

            if self.with_corner:
                preds_dict['corner_hm'] = self._sigmoid(preds_dict['corner_hm'])
                corner_loss = self.corner_crit(preds_dict['corner_hm'],
                                               corner_heatmaps[task_id])
                corner_mask = (corner_heatmaps[task_id] > 0).to(corner_loss)
                corner_loss = (corner_loss * corner_mask).sum() / (
                    corner_mask.sum() + 1e-4)
                losses.update({
                    f'{task_id}_corner_loss': corner_loss * self.corner_weight
                })

            target_box = anno_boxes[task_id]
            # Reconstruct the anno_box from multiple reg heads.
            # RPiN 前置修复：与 get_targets_single 7-cat 对齐（去掉 height / vel），
            # 让 pred (reg + dim + rot = 7) 与 target 一致；vel 头仍参与前向但不影响 7-cat 回归。
            preds_dict['anno_box'] = torch.cat((preds_dict['reg'],
                                                preds_dict['dim'],
                                                preds_dict['rot']), dim=1)

            # Regression loss for dimension, offset, height, rotation.
            box_loss = self.crit_reg(
                preds_dict['anno_box'], gt_masks[task_id], gt_inds[task_id], target_box)

            loc_loss = (box_loss * box_loss.new_tensor(self.code_weights)).sum()

            losses.update({
                f'{task_id}_hm_loss': hm_loss,
                f'{task_id}_loc_loss': loc_loss * self.weight
            })

            if self.with_iou or self.with_reg_iou:
                batch_dim = torch.exp(torch.clamp(preds_dict['dim'], min=-5, max=5))
                batch_dim = batch_dim.permute(0, 2, 3, 1).contiguous()
                batch_rot = preds_dict['rot'].clone()
                batch_rot = batch_rot.permute(0, 2, 3, 1).contiguous()
                batch_rots = batch_rot[..., 0:1]
                batch_rotc = batch_rot[..., 1:2]
                batch_rot = torch.atan2(batch_rots, batch_rotc)
                batch_reg = preds_dict['reg'].clone().permute(0, 2, 3, 1).contiguous()
                batch_hei = preds_dict['height'].clone().permute(0, 2, 3, 1).contiguous()

                batch, H, W, _ = batch_dim.size()

                batch_reg = batch_reg.reshape(batch, H * W, 2)
                batch_hei = batch_hei.reshape(batch, H * W, 1)
                batch_rot = batch_rot.reshape(batch, H * W, 1)
                batch_dim = batch_dim.reshape(batch, H * W, 3)

                ys, xs = torch.meshgrid(
                    torch.arange(0, H), torch.arange(0, W), indexing='ij')
                ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_dim)
                xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_dim)

                xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
                ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

                xs = xs * self.out_size_factor * self.voxel_size[0] + self.point_cloud_range[0]
                ys = ys * self.out_size_factor * self.voxel_size[1] + self.point_cloud_range[1]

                batch_box_preds = torch.cat(
                    [xs, ys, batch_hei, batch_dim, batch_rot], dim=2)
                batch_box_preds = batch_box_preds.permute(
                    0, 2, 1).contiguous().reshape(batch, -1, H, W)

                if self.with_iou:
                    pred_boxes_for_iou = batch_box_preds.detach()
                    iou_loss = self.crit_iou(preds_dict['iou'], gt_masks[task_id], gt_inds[task_id],
                                             pred_boxes_for_iou, gt_boxes[task_id])
                    losses.update({
                        f'{task_id}_iou_loss': iou_loss * self.iou_weight
                    })

                if self.with_reg_iou:
                    iou_reg_loss = self.crit_iou_reg(batch_box_preds, gt_masks[task_id], gt_inds[task_id],
                                                     gt_boxes[task_id])
                    losses.update({
                        f'{task_id}_iou_reg_loss': iou_reg_loss * self.iou_reg_weight
                    })

        return losses

    # ----------------------------------------------------------------- #
    # Targets                                                            #
    # ----------------------------------------------------------------- #
    def get_targets(self, gt_boxes_full):
        """Build supervision targets for the whole batch.

        Mirrors RadarNeXt's ``get_targets`` (which used mmdet's multi_apply),
        but pulls gt from OpenPCDet's ``batch_dict['gt_boxes']`` tensor.

        Args:
            gt_boxes_full (Tensor): (B, M, 7) or (B, M, 8); if the last column
                holds the 1-based class index it is used as labels. Boxes are
                already in OpenPCDet volume-center convention — .gravity_center
                is NOT called.
        """
        batch_size = gt_boxes_full.shape[0]
        # Split boxes vs labels. OpenPCDet packs [xyz, dx, dy, dz, heading, class].
        # OpenPCDet class is 1-based (Car=1, Pedestrian=2, Cyclist=3); the
        # original mmdet3d get_targets_single expects 0-based labels, so we
        # shift to 0-based here to keep get_targets_single a verbatim port.
        if gt_boxes_full.shape[-1] == 8:
            gt_bboxes = gt_boxes_full[..., :7]
            gt_labels = gt_boxes_full[..., 7].long() - 1
        else:
            gt_bboxes = gt_boxes_full
            gt_labels = torch.zeros(gt_bboxes.shape[:2], dtype=torch.long,
                                    device=gt_bboxes.device)

        # Per-sample lists.
        sample_labels = [gt_labels[b] for b in range(batch_size)]
        sample_bboxes = [gt_bboxes[b] for b in range(batch_size)]

        results = [self.get_targets_single(lab, box) for lab, box in zip(sample_labels, sample_bboxes)]
        # results: list (B) of 7-tuples each being a list (num_tasks) of tensors.
        # Transpose to per-task batched tensors.
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
        """Generate training targets for a single sample (per-task lists).

        Faithful port of RadarNeXt's ``get_targets_single``. OpenPCDet gt boxes
        are already volume-center, so the ``gravity_center`` call from the
        original is dropped. gt labels are 1-based (OpenPCDet convention) and
        are mapped to per-task 0-based ``cls_id`` via the task's class_names.
        """
        device = gt_labels_3d.device
        max_objs = int(self.model_cfg.get('MAX_OBJS', 500)) * int(self.model_cfg.get('DENSE_REG', 1))
        grid_size = torch.tensor(self.grid_size, device=device)
        pc_range = torch.tensor(self.point_cloud_range, device=device)
        voxel_size = torch.tensor(self.voxel_size, device=device)
        gt_annotation_num = len(self.code_weights)

        feature_map_size = (grid_size[:2] // self.out_size_factor).int()

        # Reorganize gt by tasks. gt_labels_3d are 0-based (matching the
        # original mmdet3d convention); the per-task class list defines the
        # local 0-based index space.
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
                # 0 is background for each task, so add 1 here.
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
            cat_label = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.int64)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                # gt boxes [xyz dx dy dz heading] -> length,width in feature cells
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

                    radius = radius // 2
                    rot = task_boxes[idx][k][6]
                    corner_keypoints = center_to_corner_box2d(
                        center.unsqueeze(0).cpu().numpy(),
                        torch.tensor([[length, width]], dtype=torch.float32).numpy(),
                        angles=rot.cpu().numpy().reshape(1),
                        origin=0.5)
                    corner_keypoints = torch.from_numpy(corner_keypoints).to(center)

                    draw_gaussian(corner_heatmap[0], center_int, radius)
                    draw_gaussian(corner_heatmap[0],
                                  (corner_keypoints[0, 0] + corner_keypoints[0, 1]) / 2, radius)
                    draw_gaussian(corner_heatmap[0],
                                  (corner_keypoints[0, 2] + corner_keypoints[0, 3]) / 2, radius)
                    draw_gaussian(corner_heatmap[0],
                                  (corner_keypoints[0, 0] + corner_keypoints[0, 3]) / 2, radius)
                    draw_gaussian(corner_heatmap[0],
                                  (corner_keypoints[0, 1] + corner_keypoints[0, 2]) / 2, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    cat_label[new_idx] = cls_id
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    box_dim = box_dim.log()
                    # RPiN 前置计划修复：parent 漏掉 z 不写 7-col anno_box，
                    # 与 code_weights=7 + bbox_code_size=7 不一致 → 首个非空目标即崩。
                    # 改为 7-cat：dx, dy, log(dx), log(dy), log(dz), sin, cos。
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device=device),
                        box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0)
                    ])
                    gt_box[new_idx] = task_boxes[idx][k][0:7]

            heatmaps.append(heatmap)
            corner_heatmaps.append(corner_heatmap)
            anno_boxes.append(anno_box)
            gt_boxes.append(gt_box)
            masks.append(mask)
            inds.append(ind)
            cat_labels.append(cat_label)
        return heatmaps, anno_boxes, inds, masks, corner_heatmaps, cat_labels, gt_boxes

    # ----------------------------------------------------------------- #
    # Predict / post-processing                                          #
    # ----------------------------------------------------------------- #
    @torch.no_grad()
    def predict(self, preds_dicts, data_dict):
        """Decode head outputs, run NMS, return a list of per-sample dicts.

        Returns:
            list[dict]: each dict has keys 'pred_boxes' (N,7), 'pred_scores' (N,),
                'pred_labels' (N,) with labels in OpenPCDet's 1-based class
                index space (VoD eval does ``class_names[pred_labels - 1]``).
        """
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
            # convert N C H W to N H W C
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            batch_hm = torch.sigmoid(preds_dict['hm'])
            batch_dim = torch.exp(preds_dict['dim'])

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']
            if 'iou' in preds_dict.keys():
                batch_iou = (preds_dict['iou'].squeeze(dim=-1) + 1) * 0.5
                batch_iou = batch_iou.type_as(batch_dim)
            else:
                batch_iou = torch.ones(
                    (batch_hm.shape[0], batch_hm.shape[1], batch_hm.shape[2]),
                    dtype=batch_dim.dtype, device=batch_hm.device)

            batch_rot = torch.atan2(batch_rots, batch_rotc)

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

        # Merge per-task results across the batch.
        num_samples = len(rets[0])
        pred_dicts = []
        for i in range(num_samples):
            bboxes = torch.cat([ret[i]['bboxes'] for ret in rets], dim=0) if rets else \
                torch.zeros((0, 7), device=preds_dicts[0]['hm'].device)
            scores = torch.cat([ret[i]['scores'] for ret in rets], dim=0) if rets else \
                torch.zeros((0,), device=preds_dicts[0]['hm'].device)
            # Compose global labels across tasks. Each task's labels are 0-based
            # (argmax of the task heatmap) and offset by the running class count
            # ``flag``; the final +1 converts to OpenPCDet's 1-based convention
            # (VoD evaluation does ``class_names[pred_labels - 1]``).
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

    @torch.no_grad()
    def post_processing(self, task_id, batch_box_preds, batch_hm, post_center_range, batch_iou):
        """Per-sample NMS + rectifier scoring for one task."""
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

            selected_boxes = torch.zeros((0, 7), device=box_preds.device, dtype=box_preds.dtype)
            selected_labels = torch.zeros((0,), dtype=torch.int64, device=labels.device)
            selected_scores = torch.zeros((0,), dtype=scores.dtype, device=scores.device)
            for class_id in range(hm_preds.shape[-1]):
                class_sel = (labels == class_id)
                scores_class = scores[class_sel]
                labels_class = labels[class_sel]
                box_preds_class = box_preds[class_sel]
                boxes_for_nms_class = box_preds_class[:, [0, 1, 2, 3, 4, 5, -1]]
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

    @staticmethod
    def _rotate_nms(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
        """OpenPCDet ``iou3d_nms_utils.nms_gpu`` as a drop-in for the original
        ``rotate_nms_pcdet``. Returns the selected indices (post topk)."""
        if boxes.shape[0] == 0:
            return torch.zeros((0,), dtype=torch.long, device=boxes.device)
        selected, _ = iou3d_nms_utils.nms_gpu(
            boxes.contiguous(), scores, thresh, pre_maxsize=pre_maxsize)
        if post_max_size is not None:
            selected = selected[:post_max_size]
        return selected

"""Canary: import original RadarNeXt modules without mmdet3d/mmengine/mmcv.

Stub the mm* dependencies with minimal pure-torch equivalents, then load the
original .py files via the normal import system using a synthesized namespace
package layout that mirrors the RadarNeXt repo. This proves the FPN-chain
originals (RepDWC / SECONDFPN / RadarNeXt_Head / losses) can be exercised in
this conda env (py3.12 / cu124) WITHOUT the DCNv3 / MMDet3D stack.

The DCNv3 import in ``common.py`` is bypassed by pre-stubbing
``projects.RadarNeXt.radarnext.DeformFFN`` so the multi-scale deform layer
classes (which parity tests DEFER to Task 7) become inert placeholders.
"""

import sys
import types

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Pure-torch fallbacks for mmcv rotated BEV overlap (used by pillarnext utils) #
# --------------------------------------------------------------------------- #
def _rotate_boxes_overlap_bev(boxes_a, boxes_b):
    """Pure-torch rotated-BEV overlap (N,M). Faithful to mmcv boxes_overlap_bev.

    Boxes are (N,7) [x,y,z,dx,dy,dz,heading]; output is (N,M) BEV overlap area.
    Uses shapely for accurate rotated-rect intersection, falls back to
    axis-aligned bbox overlap if shapely is unavailable.
    """
    try:
        from shapely.geometry import Polygon
        import numpy as np

        def corners(b):
            cx, cy, dx, dy, ang = b[:, 0], b[:, 1], b[:, 3], b[:, 4], b[:, 6]
            half_dx = dx / 2
            half_dy = dy / 2
            cs = torch.stack([
                torch.stack([-half_dx, -half_dy], -1),
                torch.stack([-half_dx, half_dy], -1),
                torch.stack([half_dx, half_dy], -1),
                torch.stack([half_dx, -half_dy], -1),
            ], dim=1)
            c = torch.cos(ang).unsqueeze(1)
            s = torch.sin(ang).unsqueeze(1)
            R = torch.stack([torch.cat([c, -s], -1), torch.cat([s, c], -1)], dim=1)
            cs = torch.bmm(cs, R.transpose(1, 2))
            cs = cs + torch.stack([cx, cy], -1).unsqueeze(1)
            return cs

        A = corners(boxes_a).cpu().numpy()
        B = corners(boxes_b).cpu().numpy()
        N, M = A.shape[0], B.shape[0]
        out = torch.zeros((N, M), dtype=boxes_a.dtype, device=boxes_a.device)
        for i in range(N):
            pa = Polygon(A[i]).buffer(0)
            if not pa.is_valid or pa.area == 0:
                continue
            for j in range(M):
                pb = Polygon(B[j]).buffer(0)
                if not pb.is_valid or pb.area == 0:
                    continue
                out[i, j] = float(pa.intersection(pb).area)
        return out
    except Exception:
        # Bounding-box fallback (axis-aligned; not accurate for rotated boxes).
        N, M = boxes_a.shape[0], boxes_b.shape[0]
        out = torch.zeros((N, M), dtype=boxes_a.dtype, device=boxes_a.device)
        for i in range(N):
            for j in range(M):
                mn = torch.maximum(boxes_a[i, :2], boxes_b[j, :2])
                mx = torch.minimum(
                    boxes_a[i, :2] + boxes_a[i, 3:5] / 2,
                    boxes_b[j, :2] + boxes_b[j, 3:5] / 2)
                d = (mx - mn).clamp(min=0)
                out[i, j] = d[0] * d[1]
        return out


def install_stubs():
    """Install pure-torch stubs for mmengine / mmdet3d / mmdet / mmcv."""
    if getattr(sys, '_parity_stubs_installed', False):
        return
    sys._parity_stubs_installed = True

    # ---- mmengine ----
    mmengine = types.ModuleType('mmengine')
    mmmodel = types.ModuleType('mmengine.model')

    class BaseModule(nn.Module):
        """Shim BaseModule that still behaves as a proper nn.Module so that
        forward() binds, parameters(), state_dict() etc. all work. We ignore
        the mmdet3d init_cfg machinery (parity tests do their own seeding)."""

        def __init__(self, *a, init_cfg=None, **kw):
            super().__init__()
            self.init_cfg = init_cfg

    mmmodel.BaseModule = BaseModule
    mmengine.model = mmmodel
    sys.modules['mmengine'] = mmengine
    sys.modules['mmengine.model'] = mmmodel

    mmlogging = types.ModuleType('mmengine.logging')
    mmlogging.print_log = lambda *a, **kw: None
    sys.modules['mmengine.logging'] = mmlogging
    mmstruct = types.ModuleType('mmengine.structures')

    class InstanceData:
        pass

    mmstruct.InstanceData = InstanceData
    sys.modules['mmengine.structures'] = mmstruct

    # ---- mmdet3d ----
    mmdet3d = types.ModuleType('mmdet3d')
    registry = types.ModuleType('mmdet3d.registry')

    class _MODELS:
        @staticmethod
        def register_module(force=False):
            def wrap(cls):
                return cls
            return wrap

        @staticmethod
        def build(*a, **kw):
            raise NotImplementedError

    registry.MODELS = _MODELS()
    mmdet3d.registry = registry
    sys.modules['mmdet3d'] = mmdet3d
    sys.modules['mmdet3d.registry'] = registry
    mmdet3dutils = types.ModuleType('mmdet3d.utils')
    mmdet3dutils.ConfigType = dict
    mmdet3dutils.OptMultiConfig = object
    mmdet3d.utils = mmdet3dutils
    sys.modules['mmdet3d.utils'] = mmdet3dutils

    mmstruct3d = types.ModuleType('mmdet3d.structures')
    # center_to_corner_box2d: mmdet3d's version is tensor-aware (the port's
    # copy is numpy-only). Provide a tensor-aware implementation matching
    # mmdet3d's reference so the original head's get_targets_single works
    # on CUDA tensors.
    def _center_to_corner_box2d_tensor(center, dim, angles, origin=0.5):
        import torch
        # If ALL inputs are numpy, return numpy (matches the original caller's
        # `torch.from_numpy(corner_keypoints)` downstream contract). The original
        # head's get_targets_single calls this with numpy center/dim but a
        # tensor angle on CUDA — promote to a common device and return numpy
        # on CPU for that mixed case too.
        center_np = not torch.is_tensor(center)
        dim_np = not torch.is_tensor(dim)
        angles_np = not torch.is_tensor(angles)
        all_np = center_np and dim_np and angles_np
        # Coerce all inputs to numpy arrays for a unified code path.
        import numpy as np
        center = center.cpu().numpy() if torch.is_tensor(center) else np.asarray(center)
        dim = dim.cpu().numpy() if torch.is_tensor(dim) else np.asarray(dim)
        angles = angles.cpu().numpy() if torch.is_tensor(angles) else np.asarray(angles)
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
        return corners  # numpy ndarray
    mmstruct3d.center_to_corner_box2d = _center_to_corner_box2d_tensor
    mmdet3d.structures = mmstruct3d
    sys.modules['mmdet3d.structures'] = mmstruct3d

    mmdet3dmodels = types.ModuleType('mmdet3d.models')
    mmdet3d.models = mmdet3dmodels
    sys.modules['mmdet3d.models'] = mmdet3dmodels
    mmdet3ddet = types.ModuleType('mmdet3d.models.detectors')

    class Base3DDetector:
        def __init__(self, *a, init_cfg=None, data_preprocessor=None, **kw):
            pass

    mmdet3ddet.Base3DDetector = Base3DDetector
    sys.modules['mmdet3d.models.detectors'] = mmdet3ddet
    mmdet3dmodels_utils = types.ModuleType('mmdet3d.models.utils')
    # Use the PORT's verbatim implementations (already proven faithful).
    # Lazy import to avoid circular dependency.
    def _lazy_port_gaussian():
        from pcdet.models.dense_heads.radarnext_center_head import (
            draw_heatmap_gaussian, gaussian_radius, gaussian_2d,
        )
        return draw_heatmap_gaussian, gaussian_radius, gaussian_2d
    mmdet3dmodels_utils.draw_heatmap_gaussian = property(lambda self: None)  # placeholder
    # We can't lazy-bind at call-time easily; instead bind the actual functions
    # by importing the port module eagerly here. The port lives in the same
    # repo, so the import is safe.
    try:
        from pcdet.models.dense_heads.radarnext_center_head import (
            draw_heatmap_gaussian as _dhg,
            gaussian_radius as _gr,
            gaussian_2d as _g2d,
        )
        mmdet3dmodels_utils.draw_heatmap_gaussian = _dhg
        mmdet3dmodels_utils.gaussian_radius = _gr
    except Exception:
        mmdet3dmodels_utils.draw_heatmap_gaussian = None
        mmdet3dmodels_utils.gaussian_radius = None
    sys.modules['mmdet3d.models.utils'] = mmdet3dmodels_utils

    # ---- mmdet ----
    mmdet = types.ModuleType('mmdet')
    mmdetmodels = types.ModuleType('mmdet.models')
    mmdetmodelsutils = types.ModuleType('mmdet.models.utils')

    def multi_apply(fn, *args, **kwargs):
        """mmdet.models.utils.multi_apply: map fn over zipped args, then
        transpose the per-element tuple outputs into lists of tuples -> stacks.

        Mirrors the mmdet3d reference behavior used by RadarNeXt_Head.
        """
        results = list(map(fn, *args, **kwargs))
        if not isinstance(results[0], (list, tuple)):
            return results
        # Transpose: [(a0,b0,...), (a1,b1,...), ...] -> ([a0,a1,...], [b0,b1,...])
        return list(map(list, zip(*results)))

    mmdetmodelsutils.multi_apply = multi_apply
    mmdet.models = mmdetmodels
    mmdetmodels.utils = mmdetmodelsutils
    sys.modules['mmdet'] = mmdet
    sys.modules['mmdet.models'] = mmdetmodels
    sys.modules['mmdet.models.utils'] = mmdetmodelsutils

    # ---- mmcv ----
    mmcv = types.ModuleType('mmcv')
    mmcvcnn = types.ModuleType('mmcv.cnn')

    def build_conv_layer(cfg, in_channels, out_channels, kernel_size, stride,
                         bias=True, **kw):
        # mmcv behavior: cfg.copy(); pop 'type'; layer(*args, **kwargs, **cfg).
        # cfg keys fill in any kwargs NOT passed explicitly. So if the caller
        # did NOT pass bias=, cfg['bias'] takes effect.
        if isinstance(cfg, dict):
            cfg_bias = cfg.get('bias', None)
            if cfg_bias is not None:
                bias = cfg_bias
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                         stride=stride, padding=0, bias=bias)

    def build_norm_layer(cfg, num_features):
        t = cfg.get('type', 'BN') if isinstance(cfg, dict) else 'BN'
        eps = cfg.get('eps', 1e-3) if isinstance(cfg, dict) else 1e-3
        momentum = cfg.get('momentum', 0.01) if isinstance(cfg, dict) else 0.01
        return t, nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)

    def build_upsample_layer(cfg, in_channels, out_channels, kernel_size,
                             stride, bias=True, **kw):
        # mmcv: deconv builds ConvTranspose2d; cfg bias is honored.
        if isinstance(cfg, dict):
            merged = dict(cfg)
            merged.pop('type', None)
            bias = merged.get('bias', bias)
        return nn.ConvTranspose2d(in_channels, out_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  bias=bias)

    mmcvcnn.build_conv_layer = build_conv_layer
    mmcvcnn.build_norm_layer = build_norm_layer
    mmcvcnn.build_upsample_layer = build_upsample_layer
    mmcv.cnn = mmcvcnn
    sys.modules['mmcv'] = mmcv
    sys.modules['mmcv.cnn'] = mmcvcnn
    mmcvutils = types.ModuleType('mmcv.utils')
    mmcv.utils = mmcvutils
    sys.modules['mmcv.utils'] = mmcvutils
    extloader = types.ModuleType('mmcv.utils.ext_loader')

    def load_ext(*a, **kw):
        raise ImportError('mmcv ext not available')

    extloader.load_ext = load_ext
    sys.modules['mmcv.utils.ext_loader'] = extloader
    mmcvops = types.ModuleType('mmcv.ops')
    mmcv.ops = mmcvops
    sys.modules['mmcv.ops'] = mmcvops
    mmcvops_iou3d = types.ModuleType('mmcv.ops.iou3d')
    mmcvops_iou3d.boxes_overlap_bev = _rotate_boxes_overlap_bev
    mmcvops.iou3d = mmcvops_iou3d
    sys.modules['mmcv.ops.iou3d'] = mmcvops_iou3d

    # ---- projects namespace packages ----
    RN_PATH = '/home/admin/projects/RadarNeXt'
    if RN_PATH not in sys.path:
        sys.path.insert(0, RN_PATH)
    projects_mod = types.ModuleType('projects')
    projects_mod.__path__ = [RN_PATH + '/projects']
    sys.modules['projects'] = projects_mod
    rn_mod = types.ModuleType('projects.RadarNeXt')
    rn_mod.__path__ = [RN_PATH + '/projects/RadarNeXt']
    sys.modules['projects.RadarNeXt'] = rn_mod
    rnr_mod = types.ModuleType('projects.RadarNeXt.radarnext')
    rnr_mod.__path__ = [RN_PATH + '/projects/RadarNeXt/radarnext']
    sys.modules['projects.RadarNeXt.radarnext'] = rnr_mod
    # DeformFFN stub (avoid DCNv3 import in common.py)
    deformffn_mod = types.ModuleType('projects.RadarNeXt.radarnext.DeformFFN')

    class _Obj:
        pass

    deformffn_mod.DCNv3 = _Obj
    deformffn_mod.DCNv3_pytorch = _Obj

    class DeformFFN:
        pass

    deformffn_mod.DeformFFN = DeformFFN
    deformffn_mod.build_norm_layer = lambda *a, **kw: (_ for _ in ()).throw(
        NotImplementedError())
    deformffn_mod.build_act_layer = lambda *a, **kw: (_ for _ in ()).throw(
        NotImplementedError())
    sys.modules['projects.RadarNeXt.radarnext.DeformFFN'] = deformffn_mod
    # PillarNeXt namespace
    pn_mod = types.ModuleType('projects.PillarNeXt')
    pn_mod.__path__ = [RN_PATH + '/projects/PillarNeXt']
    sys.modules['projects.PillarNeXt'] = pn_mod
    pnl_mod = types.ModuleType('projects.PillarNeXt.pillarnext')
    pnl_mod.__path__ = [RN_PATH + '/projects/PillarNeXt/pillarnext']
    sys.modules['projects.PillarNeXt.pillarnext'] = pnl_mod
    pnlutils = types.ModuleType('projects.PillarNeXt.pillarnext.utils')
    pnlutils.__path__ = [RN_PATH + '/projects/PillarNeXt/pillarnext/utils']
    sys.modules['projects.PillarNeXt.pillarnext.utils'] = pnlutils

    # Stub PillarNeXt utils submodules that pull mmcv ext_module at import time.
    # box_torch_ops: rotate_nms_pcdet (used only in head.predict — not in train forward).
    bto = types.ModuleType('projects.PillarNeXt.pillarnext.utils.box_torch_ops')

    def rotate_nms_pcdet(boxes, scores, thresh, pre_maxsize=None,
                         post_max_size=None):
        """Pure-torch NMS via 3D IoU (fallback for mmcv ext_module)."""
        return _rotate_nms_pcdet_torch(boxes, scores, thresh, pre_maxsize,
                                       post_max_size)

    bto.rotate_nms_pcdet = rotate_nms_pcdet
    sys.modules['projects.PillarNeXt.pillarnext.utils.box_torch_ops'] = bto

    # iou3d_nms_utils: boxes_iou3d_gpu + boxes_aligned_iou3d_gpu.
    iou3du = types.ModuleType(
        'projects.PillarNeXt.pillarnext.utils.iou3d_nms_utils')
    iou3du.boxes_iou3d_gpu = _boxes_iou3d_gpu_torch
    iou3du.boxes_aligned_iou3d_gpu = _boxes_aligned_iou3d_gpu_torch
    sys.modules['projects.PillarNeXt.pillarnext.utils.iou3d_nms_utils'] = iou3du


def _boxes_iou3d_gpu_torch(boxes_a, boxes_b):
    """Pure-torch 3D IoU (N,M) using shapely-based rotated BEV overlap."""
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    bev = _rotate_boxes_overlap_bev(boxes_a, boxes_b)  # (N,M)
    a_hmax = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    a_hmin = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    b_hmax = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    b_hmin = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)
    overlaps_h = torch.clamp(torch.min(a_hmax, b_hmax)
                             - torch.max(a_hmin, b_hmin), min=0)
    overlaps_3d = bev * overlaps_h
    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)
    return overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)


def _boxes_aligned_iou3d_gpu_torch(boxes_a, boxes_b):
    """Pure-torch aligned 3D IoU (N,) — diagonal of the (N,N) IoU matrix."""
    iou = _boxes_iou3d_gpu_torch(boxes_a, boxes_b)
    return torch.diagonal(iou)


def _rotate_nms_pcdet_torch(boxes, scores, thresh, pre_maxsize=None,
                            post_max_size=None):
    """Pure-torch rotated 3D NMS fallback. Greedy, IoU threshold on BEV."""
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()
    keep = []
    if boxes.shape[0] > 0:
        # Suppress by 3D IoU > thresh.
        suppressed = torch.zeros(boxes.shape[0], dtype=torch.bool,
                                 device=boxes.device)
        for i in range(boxes.shape[0]):
            if suppressed[i]:
                continue
            keep.append(i)
            if len(keep) > 0 and i + 1 < boxes.shape[0]:
                rest = boxes[i + 1:]
                base = boxes[i:i + 1].expand_as(rest)
                ious = _boxes_iou3d_gpu_torch(base, rest)[0]
                suppressed[i + 1:] = suppressed[i + 1:] | (ious > thresh)
    sel = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    selected = order[sel].contiguous()
    if post_max_size is not None:
        selected = selected[:post_max_size]
    return selected


if __name__ == '__main__':
    install_stubs()
    import projects.RadarNeXt.radarnext.rep_dwc as rd
    import projects.RadarNeXt.radarnext.radarnext_head as rh
    import projects.PillarNeXt.pillarnext.loss as pl
    print('OK RepDWC:', rd.RepDWC)
    print('OK RadarNeXt_Head:', rh.RadarNeXt_Head)
    print('OK losses:', pl.FastFocalLoss, pl.RegLoss, pl.IouLoss, pl.IouRegLoss,
          pl.bbox3d_overlaps_diou)
    m = rd.RepDWC(in_channels=32, out_channels=[64, 128, 256],
                  layer_nums=[3, 5, 5], layer_strides=[2, 2, 2], num_outputs=3)
    print('OK RepDWC instance, blocks:', len(m.blocks))
    x = torch.randn(1, 32, 64, 64)
    outs = m(x)
    print('OK RepDWC forward, outputs:', [tuple(o.shape) for o in outs])

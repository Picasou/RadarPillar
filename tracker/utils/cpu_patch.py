"""CPU 推理 patch: NMS/IoU CUDA op → shapely/cv2 CPU 实现, .cuda() → no-op. 取自 tools/test_cpu.py."""
from __future__ import annotations

import numpy as np
import torch


def load_data_to_cpu(batch_dict: dict) -> None:
    """load_data_to_gpu 的 CPU 版: numpy → tensor, 不上 GPU."""
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float()


def _rbbox_corners(rbbox):
    """(cx, cy, dx, dy, angle) → 4 角点 (8,)."""
    import math
    cx, cy, dx, dy, ang = rbbox
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    hx, hy = dx / 2.0, dy / 2.0
    corners_local = [(hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)]
    out = np.zeros(8, dtype=np.float32)
    for i, (lx, ly) in enumerate(corners_local):
        out[2 * i]     = cos_a * lx - sin_a * ly + cx
        out[2 * i + 1] = sin_a * lx + cos_a * ly + cy
    return out


def _make_nms_cpu():
    """BEV rotated NMS 的 CPU 实现, 签名对齐 iou3d_nms_utils.nms_gpu."""
    from shapely.geometry import Polygon

    def rbbox_iou(r1, r2):
        c1, c2 = _rbbox_corners(r1), _rbbox_corners(r2)
        p1 = Polygon([(c1[2*i], c1[2*i+1]) for i in range(4)])
        p2 = Polygon([(c2[2*i], c2[2*i+1]) for i in range(4)])
        if not p1.is_valid:
            p1 = p1.buffer(0)
        if not p2.is_valid:
            p2 = p2.buffer(0)
        a1, a2 = r1[2] * r1[3], r2[2] * r2[3]
        inter = p1.intersection(p2).area
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def nms_cpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
        assert boxes.shape[1] == 7, "nms_cpu expects (N, 7) boxes"
        order = scores.sort(descending=True)[1]
        if pre_maxsize is not None:
            order = order[:pre_maxsize]
        bev = boxes[order].cpu().numpy().astype(np.float32)[:, [0, 1, 3, 4, 6]]
        n = bev.shape[0]
        suppressed = np.zeros(n, dtype=bool)
        keep = []
        for i in range(n):
            if suppressed[i]:
                continue
            keep.append(i)
            for j in range(i + 1, n):
                if not suppressed[j] and rbbox_iou(bev[i], bev[j]) > thresh:
                    suppressed[j] = True
        return order[torch.as_tensor(keep, dtype=torch.long)], None

    return nms_cpu


def _boxes_iou_bev_cpu(boxes_a, boxes_b):
    """boxes_iou_bev 的 CPU 版 (shapely)."""
    from shapely.geometry import Polygon
    a = boxes_a.cpu().numpy().astype(np.float32)
    b = boxes_b.cpu().numpy().astype(np.float32)
    n, m = a.shape[0], b.shape[0]
    out = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        c1 = _rbbox_corners(a[i])
        p1 = Polygon([(c1[2*k], c1[2*k+1]) for k in range(4)])
        if not p1.is_valid:
            p1 = p1.buffer(0)
        a1 = a[i, 3] * a[i, 4]
        for j in range(m):
            c2 = _rbbox_corners(b[j])
            p2 = Polygon([(c2[2*k], c2[2*k+1]) for k in range(4)])
            if not p2.is_valid:
                p2 = p2.buffer(0)
            a2 = b[j, 3] * b[j, 4]
            inter = p1.intersection(p2).area
            union = a1 + a2 - inter
            out[i, j] = inter / union if union > 0 else 0.0
    return torch.from_numpy(out)


def _boxes_iou3d_cpu(boxes_a, boxes_b):
    """boxes_iou3d_gpu 的 CPU 版: BEV IoU × 高度交叠."""
    overlaps_bev = _boxes_iou_bev_cpu(boxes_a[:, [0, 1, 3, 4, 6]], boxes_b[:, [0, 1, 3, 4, 6]])
    a_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    a_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    b_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    b_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)
    overlaps_h = torch.clamp(torch.min(a_max, b_max) - torch.max(a_min, b_min), min=0)
    overlaps_3d = overlaps_bev * overlaps_h
    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)
    return overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)


def apply_cpu_patch() -> None:
    """打 CPU patch: .cuda() no-op + NMS/IoU CUDA op → CPU 实现. 幂等."""
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, device=None: self

    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    nms_cpu = _make_nms_cpu()
    iou3d_nms_utils.nms_gpu = nms_cpu
    iou3d_nms_utils.nms_normal_gpu = nms_cpu
    iou3d_nms_utils.boxes_iou_bev = _boxes_iou_bev_cpu
    iou3d_nms_utils.boxes_iou3d_gpu = _boxes_iou3d_cpu

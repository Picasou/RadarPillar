"""用 pcdet.ops.iou3d_nms.boxes_iou_bev 替代 numba rotate_iou_gpu_eval。

绕开 numba 0.59.1 在 WSL2 + CUDA 596.36 上 cuda.to_device() 的 SIGSEGV
（实测：import cuda OK，但任何 to_device 调用立即 SIGSEGV）。

两条调用入口都被覆盖：
  - pcdet/datasets/kitti/kitti_object_eval_python/eval.py:117
      bev_box_overlap(boxes, qboxes, criterion=-1) → 7D kitti [x, y, z, l, w, h, ry]
  - pcdet/datasets/kitti/kitti_object_eval_python/eval.py:152
      d3_box_overlap → 5D [x, z, l, h, ry]（select [0,2,3,5,6] of 7D）

pcdet iou3d_nms.boxes_iou_bev 输入格式 7D [x, y, z, dx, dy, dz, heading]，与 KITTI
7D 列序相同（dx↔l, dy↔w, dz↔h, heading↔ry），可直接传。5D 路径需要投影到 7D
（dz=0 占位，因为 boxes_iou_bev 只读 x,y,dx,dy,heading）。

criterion 参数：本路径下被忽略，iou3d_nms 总返回 IoU；
对 criterion=2 的 d3_box_overlap 调用（用作 rinc>0 布尔判）仍安全——
只要保证"零↔非零"一致即可。
"""
import numpy as np
import torch

from pcdet.ops.iou3d_nms import iou3d_nms_utils


def _as_torch(arr):
    """接 numpy 或 torch tensor；返回 contiguous float32 CUDA tensor。"""
    if isinstance(arr, np.ndarray):
        t = torch.from_numpy(arr.astype(np.float32, copy=False))
    elif torch.is_tensor(arr):
        t = arr
    else:
        raise TypeError(f"unsupported box type: {type(arr)}")
    if t.device.type != 'cuda':
        t = t.cuda()
    if not t.is_contiguous():
        t = t.contiguous()
    return t.float()


def _pad_5d_to_7d(b):
    """kitti 5D [x, y, dim1, dim2, angle] → pcdet 7D [x, y, 0, dim1, dim2, 0, heading]。

    只用于 BEV 投影（d3_box_overlap criterion!=−1 路径），dz=0 是占位不影响。
    """
    n = b.shape[0]
    out = torch.zeros(n, 7, device=b.device, dtype=torch.float32)
    out[:, [0, 1]] = b[:, [0, 1]]   # cx, cy
    out[:, [3, 4]] = b[:, [2, 3]]   # dim1, dim2 → dx, dy
    out[:, 6] = b[:, 4]             # angle → heading
    return out


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """drop-in replacement；返回 np.ndarray (N, K)。

    criterion 形参保留但本路径下忽略（iou3d_nms 总返回 IoU）。
    d3_box_overlap 调 criterion=2 时，本函数返回的 >0 状态被判为有效；
    """
    b = _as_torch(boxes)
    q = _as_torch(query_boxes)

    if b.shape[1] == q.shape[1] == 7:
        iou_t = iou3d_nms_utils.boxes_iou_bev(b, q)
    elif b.shape[1] == q.shape[1] == 5:
        iou_t = iou3d_nms_utils.boxes_iou_bev(_pad_5d_to_7d(b), _pad_5d_to_7d(q))
    else:
        raise ValueError(
            f"rotate_iou_gpu_eval: shape mismatch boxes={tuple(b.shape)} "
            f"query_boxes={tuple(q.shape)} (expected both 7 or both 5)"
        )

    return iou_t.detach().cpu().numpy()


def install():
    """Monkey-patch 到 kitti eval 的 rotate_iou_gpu_eval 命名空间。

    eval.py 用 `from .rotate_iou import rotate_iou_gpu_eval` 已把原 numba 函数
    绑到 eval 模块级名（line 6）。两处都打，才能确保 bev_box_overlap
    (eval.py:117) 与 d3_box_overlap (eval.py:152) 都命中本函数。

    同时把 _rio._CUDA_OK 置 True，让任何仍走 _rio.rotate_iou_gpu_eval
    的代码不被 CPU fallback 截胡。
    """
    import pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou as _rio
    import pcdet.datasets.kitti.kitti_object_eval_python.eval as _eval_mod

    _rio._CUDA_OK = True
    _eval_mod.rotate_iou_gpu_eval = rotate_iou_gpu_eval
    return True

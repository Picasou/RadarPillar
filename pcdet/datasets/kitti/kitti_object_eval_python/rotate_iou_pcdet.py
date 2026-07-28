"""Drop-in replacement for numba-based rotate_iou via pcdet iou3d_nms.

Why this exists: numba 0.59.1 on this WSL2 + CUDA 596.36 box segfaults inside
`numba.cuda.cudadrv.devices.get_context` on the very first cuda call from any
*decorated* function (jit / njit / cuda.jit). This includes @numba.jit
function *imports* -- the JIT tries to acquire a CUDA context at decoration
time, dies before any user code runs. The fix is to never let the real
rotate_iou.py module execute -- install() inserts a stub into sys.modules
so relative `from .rotate_iou import X` statements resolve to the stub
instead of importing the real file.

Coverage (per ABtest.md RPiN stage-1 failure analysis):
- pcdet/.../eval.py:117 bev_box_overlap -> 7D kitti [x,y,z,l,w,h,ry]
- pcdet/.../eval.py:152 d3_box_overlap  -> 5D [x,z,l,h,ry] (5D slice of 7D)

pcdet's iou3d_nms.boxes_iou_bev input is 7D [x,y,z,dx,dy,dz,heading] which is
column-aligned with KITTI 7D (dx~l, dy~w, dz~h, heading~ry). 5D path pads to
7D with dz=0 because boxes_iou_bev only reads x,y,dx,dy,heading.

`criterion` argument is ignored: iou3d_nms always returns IoU. The
criterion=2 callers (d3_box_overlap) only use the returned value as a `>0`
boolean (per eval.py:128 `if rinc[i,j] > 0`), so any non-negative return
is functionally correct.
"""
import numpy as np
import torch

from pcdet.ops.iou3d_nms import iou3d_nms_utils


def _as_torch(arr):
    """Accept numpy or torch tensor; return contiguous float32 CUDA tensor."""
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
    """kitti 5D [x,y,dim1,dim2,angle] -> pcdet 7D [x,y,0,dim1,dim2,0,heading]."""
    n = b.shape[0]
    out = torch.zeros(n, 7, device=b.device, dtype=torch.float32)
    out[:, [0, 1]] = b[:, [0, 1]]
    out[:, [3, 4]] = b[:, [2, 3]]
    out[:, 6] = b[:, 4]
    return out


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """Drop-in replacement; returns np.ndarray (N, K).

    `criterion` is preserved but ignored -- iou3d_nms always returns IoU.
    criterion=2 (d3_box_overlap driver) only checks >0 so any non-negative
    return is functionally correct.
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
    """Install the stub monkey-patch.

    Two things go wrong if we let pcdet import the real rotate_iou.py:
      1. `@numba.jit` decorators on top-level functions fire at IMPORT time,
         calling `get_current_device` -> numba SIGSEGVs.
      2. `from .rotate_iou import rotate_iou_gpu_eval` then rebinds nothing
         in `eval_mod`, so bev_box_overlap calls the still-broken numba fn.

    Fix: replace sys.modules entry for rotate_iou with a stub that exposes
    only the symbols eval.py needs. Then relative import pulls our stub,
    never executes the real file. Both numba crashes avoided.

    Note: install() must be called BEFORE any `from ... rotate_iou ...`
    import. Tools/test.py at the top auto-installs on startup; any other
    caller should do the same.
    """
    import sys
    import types

    real_name = 'pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou'
    eval_name = 'pcdet.datasets.kitti.kitti_object_eval_python.eval'

    # 1) Drop any cached real module so our stub actually replaces it
    #    (must drop BEFORE building stub, or build stub first)
    cached = sys.modules.pop(real_name, None)
    if cached is not None:
        # we held the real reference; no use for it
        pass

    # 2) Build stub exposing the minimum surface eval.py touches.
    #    MUST NOT `import ...rotate_iou` here -- that would load the real
    #    file (whose @numba.jit decorators trigger compile_device ->
    #    SIGSEGV on this box).
    stub = types.ModuleType(real_name)
    stub._CUDA_OK = True
    stub.rotate_iou_gpu_eval = rotate_iou_gpu_eval
    stub.rotate_iou_cpu_eval = lambda *a, **k: None
    stub.rotate_iou = rotate_iou_gpu_eval
    sys.modules[real_name] = stub

    # 3) If eval was already imported (unlikely but safe), rebind its attr
    eval_mod = sys.modules.get(eval_name)
    if eval_mod is not None:
        eval_mod.rotate_iou_gpu_eval = rotate_iou_gpu_eval

    return True

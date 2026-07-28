"""GPU vs CPU IoU 数值一致性测试。

覆盖两条调用路径：
  - 7D kitti [x, y, z, l, w, h, ry] （bev_box_overlap 入口）
  - 5D kitti [x, y, dim1, dim2, angle] （d3_box_overlap 入口）

两条都用新 GPU wrapper (rotate_iou_pcdet.rotate_iou_gpu_eval)
对照 CPU shapely (rotate_iou_cpu_eval)。
阈值 max abs diff < 1e-4 视为通过；超出则逐元素 dump 头 5 个分歧对以便诊断。

用法：
  bash -lc 'conda activate angle && PYTHONNOUSERSITE=1 python tests/test_rotate_iou_consistency.py'
"""
import os
import sys

# 避免 user-local 坏 torch（详见 memory: torch-user-local-mask）
os.environ.setdefault('PYTHONNOUSERSITE', '1')

# 项目根必须在 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

import numpy as np


def _rng(seed):
    return np.random.default_rng(seed)


def _box_7d(rng, n):
    """kitti 7D [x, y, z, l, w, h, ry]"""
    boxes = np.zeros((n, 7), dtype=np.float32)
    boxes[:, [0, 1]] = rng.uniform(-30.0, 30.0, (n, 2))   # x, y
    boxes[:, 2] = -1.5                                      # z (kitti 相机系)
    boxes[:, [3, 4]] = rng.uniform(2.0, 5.0, (n, 2))       # l, w
    boxes[:, 5] = 1.7                                       # h
    boxes[:, 6] = rng.uniform(-np.pi, np.pi, n)             # ry
    return boxes


def _box_5d(rng, n):
    """kitti 5D [x, y, dim1, dim2, angle]"""
    boxes = np.zeros((n, 5), dtype=np.float32)
    boxes[:, [0, 1]] = rng.uniform(-30.0, 30.0, (n, 2))
    boxes[:, [2, 3]] = rng.uniform(2.0, 5.0, (n, 2))
    boxes[:, 4] = rng.uniform(-np.pi, np.pi, n)
    return boxes


def _assert_close(iou_gpu, iou_cpu, fmt, tol=1e-4):
    diff = np.abs(iou_gpu - iou_cpu)
    max_diff = float(diff.max())
    n_overlap = int((iou_cpu > 0).sum())
    pct_off = float((diff > tol).sum()) / diff.size * 100
    print(f'  fmt={fmt}  shape={iou_gpu.shape}  max_abs_diff={max_diff:.3e}  '
          f'cells>tol={pct_off:.3f}%  overlap_cells={n_overlap}')
    if max_diff > tol:
        # 找出 top-5 最大分歧对
        flat = np.argsort(diff.flatten())[::-1][:5]
        print('  [warn] top-5 大分歧对 (i, j) | cpu | gpu:')
        for idx in flat:
            i, j = divmod(int(idx), diff.shape[1])
            print(f'    ({i:>4}, {j:>4}) cpu={iou_cpu[i,j]:.6f} gpu={iou_gpu[i,j]:.6f}')
        raise AssertionError(f'fmt={fmt} max diff {max_diff} > tol {tol}')
    return max_diff


def main():
    print('[env] python:', sys.executable)
    print('[env] PYTHONNOUSERSITE:', os.environ.get('PYTHONNOUSERSITE', 'unset'))

    # 先装 patch，再 import eval 的原 numba 函数作 CPU 对照
    from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou_pcdet import (
        rotate_iou_gpu_eval as gpu_iou,
        install as _install,
    )
    _install()

    # CPU baseline：原 numba 的 cpu 路径（不依赖 CUDA，只用 shapely）
    from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou import (
        rotate_iou_cpu_eval as cpu_iou,
    )

    print('[ok] imports done; running parity checks...')
    print()

    rng = _rng(42)
    for fmt_name, gen in (('7D', _box_7d), ('5D', _box_5d)):
        for n_gt, n_pred in [(80, 50), (200, 200), (500, 500)]:
            a = gen(rng, n_pred)
            b = gen(rng, n_gt)
            iou_gpu = gpu_iou(a, b, criterion=-1)
            iou_cpu = cpu_iou(a, b, criterion=-1)
            _assert_close(iou_gpu, iou_cpu, f'{fmt_name} pred={n_pred} gt={n_gt}')

    print()
    print('[pass] GPU IoU matches CPU shapely within 1e-4 across both formats')
    return 0


if __name__ == '__main__':
    sys.exit(main())

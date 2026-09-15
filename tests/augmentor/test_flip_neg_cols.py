"""
flip 镜像一致性单测: NEG_POINT_COLS 点速度列取反 + 9列框 vy/vx 自动取反, 其余列不动。
"""
import numpy as np
import pytest

from pcdet.datasets.augmentor.augmentor_utils import random_flip_along_x, random_flip_along_y


def _force_flip_on(seed=0):
    """固定种子使 np.random.choice 输出 True (flip 生效)。"""
    np.random.seed(seed)
    # choice([False,True], p=[.5,.5]) 在 seed=0 下首个输出为 False, 试到 True 的种子
    for s in range(64):
        np.random.seed(s)
        if np.random.choice([False, True], p=[0.5, 0.5]):
            return s
    raise RuntimeError('无有效种子')


def test_flip_x_msr_layout():
    """MSR 6列点[x,y,z,dop_x,dop_y,rcs]+9列框: 镜像后 y/dop_y/vy 取反, dop_x/rcs/vx 不动。"""
    s = _force_flip_on()
    points = np.array([[1., 2., 3., 0.5, -1.5, 10.]])
    boxes = np.array([[1., 2., 0., 4., 2., 1.5, 0.3, 2.0, -3.0]])
    np.random.seed(s)
    b2, p2 = random_flip_along_x(boxes.copy(), points.copy(), neg_point_cols=[4])
    assert p2[0, 1] == -2.           # y 取反
    assert p2[0, 4] == 1.5           # dop_y 取反
    assert p2[0, 3] == 0.5 and p2[0, 5] == 10.   # dop_x / rcs 不动
    assert b2[0, 1] == -2. and b2[0, 6] == -0.3  # 框 y / heading 取反
    assert b2[0, 7] == 2.0 and b2[0, 8] == 3.0   # vx 不动, vy 取反


def test_flip_y_msr_layout():
    """沿y镜像: x/dop_x/vx 取反, dop_y/rcs/vy 不动 (MSR 未用 y-flip, 守对称)。"""
    s = _force_flip_on()
    points = np.array([[1., 2., 3., 0.5, -1.5, 10.]])
    boxes = np.array([[1., 2., 0., 4., 2., 1.5, 0.3, 2.0, -3.0]])
    np.random.seed(s)
    b2, p2 = random_flip_along_y(boxes.copy(), points.copy(), neg_point_cols=[3])
    assert p2[0, 0] == -1. and p2[0, 3] == -0.5  # x / dop_x 取反
    assert p2[0, 4] == -1.5 and p2[0, 5] == 10.  # dop_y / rcs 不动
    assert b2[0, 7] == -2.0 and b2[0, 8] == -3.0  # vx 取反, vy 不动


def test_default_no_neg_cols_unchanged():
    """不传 NEG_POINT_COLS: 行为与旧版一致 (6列点不触发任何点速度取反)。"""
    s = _force_flip_on()
    points = np.array([[1., 2., 3., 0.5, -1.5, 10.]])
    boxes = np.array([[1., 2., 0., 4., 2., 1.5, 0.3, 2.0, -3.0]])
    np.random.seed(s)
    b2, p2 = random_flip_along_x(boxes.copy(), points.copy())
    assert p2[0, 4] == -1.5   # dop_y 未取反 (显式不配置则维持旧语义)

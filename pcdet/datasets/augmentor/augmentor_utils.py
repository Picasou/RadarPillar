import numpy as np

from ...utils import common_utils


def random_flip_along_x(gt_boxes, points, neg_point_cols=()):
    """
    沿x镜像(y→-y): 框y/heading/vy取反, 点y取反; neg_point_cols 额外取反点速度列(如 dop_y_gnd)。

    Args:
        gt_boxes : (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points   : (M, 3 + C)
        neg_point_cols: 该镜像下需取反的点列索引 (笛卡尔y向速度列, 数据集布局各异故显式传入)
    Returns:
    Note: 旧守卫分支(gt>7列时点col6取反)为历史7列雷达布局保留, 6列 MSR 点云不触发;
          MSR 镜像一致性靠 neg_point_cols (vel 训练增强启用前提)。
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]
            if points.shape[1] >= 7:
                points[:, 6] = -points[:, 6]

        for c in neg_point_cols:
            points[:, c] = -points[:, c]

    return gt_boxes, points


def random_flip_along_y(gt_boxes, points, neg_point_cols=()):
    """
    沿y镜像(x→-x): 框x/heading/vx处理, 点x取反; neg_point_cols 额外取反点速度列(如 dop_x_gnd)。

    Args:
        gt_boxes : (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points   : (M, 3 + C)
        neg_point_cols: 该镜像下需取反的点列索引 (笛卡尔x向速度列)
    Returns:
    Note: 同 random_flip_along_x, 历史守卫分支保留, 布局差异靠 neg_point_cols 显式声明。
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]
            if points.shape[1] >= 7:
                points[:, 5] = -points[:, 5]

        for c in neg_point_cols:
            points[:, c] = -points[:, c]

    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    Note: rotate_points_along_z already only rotates x,y,z (indices 0:3) and
          preserves all other features. The additional velocity rotation below
          is only needed when gt_boxes carry velocity columns (shape > 7).
          For radar data (gt_boxes=7 cols), points[:, 5:7] are [v_r_comp, time],
          NOT [vx, vy], so the velocity rotation must be skipped.
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]
        if points.shape[1] >= 7:
            velocity = np.hstack((points[:, 5:7], np.zeros((points.shape[0], 1), dtype=points.dtype)))
            velocity = common_utils.rotate_points_along_z(velocity[np.newaxis, :, :], np.array([noise_rotation]))[0]
            points[:, 5:7] = velocity[:, 0:2]

    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points

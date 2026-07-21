"""
VoD eval 单帧可视化: 左侧 BEV (radar 点云 + GT + Pred 旋转框) + 右侧相机图像 (GT + Pred 3D cube)。
颜色 / 标签 / 范围 / 风格与 visualize_bev.py 保持一致。
"""
import pickle
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

# 复用 visualize_bev 的色板
CLASS_COLORS_GT = {
    "Car": "#2ecc71", "Pedestrian": "#3498db", "Cyclist": "#e74c3c",
}
CLASS_COLORS_PRED = {
    "Car": "#27ae60", "Pedestrian": "#2980b9", "Cyclist": "#c0392b",
}
BEV_XLIM = (0, 52)
BEV_YLIM = (-26, 26)


# ══════════════════════════════════════════════════════════════
#  工具
# ══════════════════════════════════════════════════════════════
def _box_corners_2d(cx, cy, l, w, heading):
    """BEV 4 角点。"""
    corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    return corners @ rot.T + np.array([cx, cy])


def _draw_box_bev(ax, cx, cy, l, w, heading, color, linestyle="-", linewidth=2.5, alpha=1.0):
    """BEV 旋转框：颜色表示类别 + 线型区分 GT/Pred。
       不在每个 box 旁写 category 文本 — 仅画框本身。
       类别靠图例 (legend) 表达。"""
    corners = _box_corners_2d(cx, cy, l, w, heading)
    poly = Polygon(corners, fill=False, edgecolor=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    ax.add_patch(poly)
    # 朝向：物体中心指向 heading 方向画一段轴线，便于看出方向
    dx = np.cos(heading) * l * 0.5
    dy = np.sin(heading) * l * 0.5
    ax.plot([cx, cx + dx], [cy, cy + dy], color=color, linewidth=1.2, alpha=alpha)


def _lidar_boxes_to_corners_3d(boxes_lidar):
    """lidar (N,7) [x,y,z,dx,dy,dz,h] -> (N,8,3). 内联实现, 避免 pcdet 依赖。"""
    if boxes_lidar.shape[0] == 0:
        return np.zeros((0, 8, 3), dtype=np.float64)
    x, y, z, dx, dy, dz, h = boxes_lidar.T
    zeros, ones = np.zeros_like(x), np.ones_like(x)
    corners = np.stack([
        np.stack([x - dx/2, y - dy/2, z - dz/2], -1),
        np.stack([x + dx/2, y - dy/2, z - dz/2], -1),
        np.stack([x + dx/2, y + dy/2, z - dz/2], -1),
        np.stack([x - dx/2, y + dy/2, z - dz/2], -1),
        np.stack([x - dx/2, y - dy/2, z + dz/2], -1),
        np.stack([x + dx/2, y - dy/2, z + dz/2], -1),
        np.stack([x + dx/2, y + dy/2, z + dz/2], -1),
        np.stack([x - dx/2, y + dy/2, z + dz/2], -1),
    ], axis=1)  # (N, 8, 3)
    # 在 xy 平面绕 z 旋转 heading: 用 (N,3,3) rot 直接构造
    c, s = np.cos(h), np.sin(h)
    rot = np.empty((x.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = c;  rot[:, 0, 1] = -s; rot[:, 0, 2] = 0
    rot[:, 1, 0] = s;  rot[:, 1, 1] =  c; rot[:, 1, 2] = 0
    rot[:, 2, 0] = 0;  rot[:, 2, 1] =  0; rot[:, 2, 2] = 1
    rel = corners - np.stack([x, y, z], -1)[:, None, :]
    rotated = np.einsum("nij,npj->npi", rot, rel)
    return rotated + np.stack([x, y, z], -1)[:, None, :]


def _project_corners_to_image(corners_3d, calib):
    """(N,8,3) -> (N,8,2) 像素坐标。用 calib.lidar_to_img (P2 @ Tr_velo_to_cam @ R0 @ X)。"""
    if corners_3d.shape[0] == 0:
        return np.zeros((0, 8, 2), dtype=np.float64)
    pts_img, _ = calib.lidar_to_img(corners_3d.reshape(-1, 3))
    return pts_img.reshape(-1, 8, 2)


def _draw_3d_cube_on_image(img, corners_2d, color, linestyle="-", line_thickness=2):
    """在 BGR 图像上画 3D 立方体 12 棱。
    linestyle:
      "-"  : GT (实线)
      "--" : Pred (虚线，简单用 dash 通过 custom drawing)
    """
    if corners_2d.shape[0] == 0:
        return img
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # bottom
             (4, 5), (5, 6), (6, 7), (7, 4),  # top
             (0, 4), (1, 5), (2, 6), (3, 7)]  # verticals
    is_dash = (linestyle == "--")
    for (i, j) in edges:
        p1 = corners_2d[i].astype(int)
        p2 = corners_2d[j].astype(int)
        if is_dash:
            # 在 OpenCV 里手动实现虚线：每 8 px 一段、3 px 间隔
            dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
            length = max(int(np.hypot(dx, dy)), 1)
            ux, uy = dx / length, dy / length
            pos = 0
            while pos < length:
                start = (int(p1[0] + ux * pos), int(p1[1] + uy * pos))
                end_pos = min(pos + 6, length)
                end = (int(p1[0] + ux * end_pos), int(p1[1] + uy * end_pos))
                cv2.line(img, start, end, color, line_thickness, lineType=cv2.LINE_AA)
                pos += 10
        else:
            cv2.line(img, tuple(p1), tuple(p2), color, line_thickness, lineType=cv2.LINE_AA)
    return img


# ══════════════════════════════════════════════════════════════
#  主合成
# ══════════════════════════════════════════════════════════════
def compose_one_frame(frame_id, points, image, calib,
                      gt_boxes_lidar, gt_names,
                      pred_boxes_lidar, pred_names, pred_scores,
                      output_path, score_thresh=0.1):
    """合成单帧: BEV(左) + 相机(右) PNG。

    Args:
        frame_id: str/int
        points: (N,7) radar 点云 (用于 BEV)
        image: HxWx3 BGR (OpenCV 读取) 或 RGB
        calib: calibration_kitti.Calibration 实例
        gt_boxes_lidar / gt_names: (N,7) / list[str]
        pred_boxes_lidar / pred_names / pred_scores: 同上 + (N,)
        output_path: 输出 PNG
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # image 可能是 BGR (cv2) 或 RGB (skimage)。统一假设输入是 BGR, 画完后转 RGB 给 matplotlib。
    img_bgr = image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    fig, (ax_bev, ax_img) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={"width_ratios": [1.4, 1.8]})

    # ---- BEV (左) ----
    if points is not None and points.shape[0] > 0:
        rcs = points[:, 3]
        ax_bev.scatter(points[:, 0], points[:, 1], c=rcs, cmap="viridis", s=4, alpha=0.6, zorder=1)

    n_gt, n_pred_filt = 0, 0
    if gt_boxes_lidar is not None and gt_boxes_lidar.shape[0] > 0:
        for i, name in enumerate(gt_names):
            color = CLASS_COLORS_GT.get(name, "#95a5a6")
            x, y, _, dx, dy, _, h = gt_boxes_lidar[i]
            # GT 用实线
            _draw_box_bev(ax_bev, x, y, dx, dy, h, color, linestyle="-", linewidth=2.0, alpha=0.9)
            n_gt += 1

    if pred_boxes_lidar is not None and pred_boxes_lidar.shape[0] > 0:
        for i, (name, sc) in enumerate(zip(pred_names, pred_scores)):
            if sc < score_thresh:
                continue
            color = CLASS_COLORS_PRED.get(name, "#7f8c8d")
            x, y, _, dx, dy, _, h = pred_boxes_lidar[i]
            # 预测用虚线
            _draw_box_bev(ax_bev, x, y, dx, dy, h, color, linestyle="--", linewidth=1.6, alpha=0.8)
            n_pred_filt += 1

    ax_bev.set_xlim(BEV_XLIM); ax_bev.set_ylim(BEV_YLIM)
    ax_bev.set_aspect("equal")
    ax_bev.set_xlabel("X (forward) [m]"); ax_bev.set_ylabel("Y (left) [m]")
    # title: 颜色映射说明 + 数量统计
    ax_bev.set_title(
        f"BEV  ·  Color → Class  (Green=Car, Blue=Pedestrian, Red=Cyclist)\n"
        f"GT solid | Pred dashed  ·  GT: {n_gt}  ·  Pred (sc≥{score_thresh:.2f}): {n_pred_filt}",
        fontsize=10
    )
    ax_bev.grid(True, alpha=0.3)
    ax_bev.plot(0, 0, marker="^", color="white", markersize=10, markeredgecolor="black", zorder=5)

    # ---- 图像 (右) ----
    # GT: 实线 3D cube on image (用作 ground-truth 对照)
    if gt_boxes_lidar is not None and gt_boxes_lidar.shape[0] > 0:
        gt_corners3d = _lidar_boxes_to_corners_3d(gt_boxes_lidar[:, :7])
        gt_corners2d = _project_corners_to_image(gt_corners3d, calib)
        for i, name in enumerate(gt_names):
            color_rgb = CLASS_COLORS_GT.get(name, "#95a5a6")
            color_bgr = (int(color_rgb[5:7], 16), int(color_rgb[3:5], 16), int(color_rgb[1:3], 16))
            _draw_3d_cube_on_image(img_bgr, gt_corners2d[i], color_bgr, linestyle="-", line_thickness=2)

    # 注意: Pred 框不在 camera image 上画 — 仅在左侧 BEV 中以虚线显示
    # 避免遮挡车辆、人体，便于直接看 BEV 评估结果

    ax_img.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax_img.set_title(f"Frame {frame_id}  ·  camera image (GT only)")
    ax_img.axis("off")

    # 图例 — 三个 group: GT (实线) 和 Pred (虚线), 每种颜色 = 一个类别
    legend_elements = [
        # GT = 实线
        Line2D([0], [0], color="#2ecc71", lw=3, ls="-", label="GT  ·  Car  (green)"),
        Line2D([0], [0], color="#3498db", lw=3, ls="-", label="GT  ·  Pedestrian  (blue)"),
        Line2D([0], [0], color="#e74c3c", lw=3, ls="-", label="GT  ·  Cyclist  (red)"),
        # Pred = 虚线 (颜色对应类别加深版, 但仍然 green/blue/red)
        Line2D([0], [0], color="#27ae60", lw=3, ls="--", label="Pred  ·  Car"),
        Line2D([0], [0], color="#2980b9", lw=3, ls="--", label="Pred  ·  Pedestrian"),
        Line2D([0], [0], color="#c0392b", lw=3, ls="--", label="Pred  ·  Cyclist"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True, edgecolor="gray")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ══════════════════════════════════════════════════════════════
#  数据加载工具 (供顶层 driver 调用)
# ══════════════════════════════════════════════════════════════
def iter_sample_ids_uniform(infos_pkl, n_samples, seed=42):
    """读 vod_infos_*.pkl, 等距取 n_samples 个 frame_id。"""
    with open(infos_pkl, "rb") as f:
        infos = pickle.load(f)
    ids = [str(info["point_cloud"]["lidar_idx"]) for info in infos]
    if n_samples >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    idx = np.linspace(0, len(ids) - 1, n_samples).astype(int)
    idx = sorted({int(i) for i in idx})
    return [ids[i] for i in idx]


def load_frame_assets(dataroot, split, frame_id):
    """读一帧: radar 点云 + 相机图像 + calib + GT boxes (lidar)。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from pcdet.utils import calibration_kitti

    root_split = Path(dataroot) / ("training" if split != "test" else "testing")

    # radar
    pts = np.fromfile(str(root_split / "velodyne" / f"{frame_id}.bin"), dtype=np.float32).reshape(-1, 7)

    # image (cv2 BGR)
    img_path = root_split / "image_2" / f"{frame_id}.jpg"
    if not img_path.exists():
        img_path = root_split / "image_2" / f"{frame_id}.png"
    img = cv2.imread(str(img_path))

    # calib
    calib = calibration_kitti.Calibration(root_split / "calib" / f"{frame_id}.txt")

    # GT: 读原始 label_2/*.txt + 走 vod_dataset 的 KITTI 相机->lidar 转换
    # 这里用 pcdet 的 box_utils 直接转, 避免依赖整个 vod_dataset
    from pcdet.utils import box_utils
    gt_boxes_lidar, gt_names = [], []
    label_path = root_split / "label_2" / f"{frame_id}.txt"
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                p = line.split()
                if not p:
                    continue
                if p[0] not in ("Car", "Pedestrian", "Cyclist"):
                    continue
                # KITTI label: h, w, l, x_cam, y_cam, z_cam, ry
                h, w, l = float(p[8]), float(p[9]), float(p[10])
                x, y, z = float(p[11]), float(p[12]), float(p[13])
                ry = float(p[14])
                cam_box = np.array([[x, y, z, l, h, w, ry]], dtype=np.float32)
                lidar_box = box_utils.boxes3d_kitti_camera_to_lidar(cam_box, calib)[0]
                gt_boxes_lidar.append(lidar_box)
                gt_names.append(p[0])

    gt_boxes_lidar = np.array(gt_boxes_lidar, dtype=np.float64) if gt_boxes_lidar else np.zeros((0, 7))
    return pts, img, calib, gt_boxes_lidar, gt_names


def lookup_predictions_for_frame(result_pkl, frame_id):
    """从 result.pkl (test.py 落盘的 det_annos list) 找指定 frame_id 的预测。"""
    import pickle
    with open(result_pkl, "rb") as f:
        det_annos = pickle.load(f)
    for d in det_annos:
        if str(d.get("frame_id")) == str(frame_id):
            return (np.asarray(d["boxes_lidar"], dtype=np.float64),
                    list(d["name"]),
                    np.asarray(d["score"], dtype=np.float64))
    return np.zeros((0, 7)), [], np.array([])

#!/usr/bin/env python3
"""
nuScenes 数据集可视化工具
左侧: BEV 俯视图 — 雷达点云 + GT boxes(实线) + 预测 boxes(虚线), 不同类别不同颜色
右侧: 6 个相机图像 — 3D box 投影到图像, 按前/后/左/右方位排列
"""

import argparse
import random
import sys
from pathlib import Path

# 自动添加项目根目录到 sys.path（visual_utils → utils → tools → RadarPillar）
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud, Box
from pcdet.datasets.nuscenes.nuscenes_utils import map_name_from_general_to_detection

# ── 类别颜色 ──────────────────────────────────────────────────
# 检测类别（来自 map_name_from_general_to_detection）
DET_CLASSES = ["car", "truck", "bus", "trailer", "construction_vehicle",
               "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier"]

CLASS_COLORS_GT = {
    "car":                 "#2ecc71",
    "truck":               "#e67e22",
    "bus":                 "#9b59b6",
    "trailer":             "#1abc9c",
    "construction_vehicle":"#95a5a6",
    "bicycle":             "#e74c3c",
    "motorcycle":          "#c0392b",
    "pedestrian":          "#3498db",
    "traffic_cone":        "#f1c40f",
    "barrier":             "#e74c3c",
}

CLASS_COLORS_PRED = {
    "car":                 "#27ae60",
    "truck":               "#d35400",
    "bus":                 "#8e44ad",
    "trailer":             "#16a085",
    "construction_vehicle":"#7f8c8d",
    "bicycle":             "#c0392b",
    "motorcycle":          "#a93226",
    "pedestrian":          "#2980b9",
    "traffic_cone":        "#d4ac0d",
    "barrier":             "#c0392b",
}

# 相机排列顺序：前向从左到右（FL / F / FR），后向从左到右（BL / B / BR）
CAM_ORDER = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
             "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

CAM_TITLES = {
    "CAM_FRONT":       "Front",
    "CAM_FRONT_RIGHT": "Front Right",
    "CAM_BACK_RIGHT":  "Back Right",
    "CAM_BACK":        "Back",
    "CAM_BACK_LEFT":   "Back Left",
    "CAM_FRONT_LEFT":  "Front Left",
}

# ── BEV 绘图 ──────────────────────────────────────────────────

def get_box_bev_corners(box: Box):
    """获取 BEV 俯视图下旋转矩形的 4 个角点 (4, 2)。"""
    corners = box.bottom_corners()  # (3, 4)
    return corners[:2].T  # (4, 2)


def draw_bev_box(ax, corners, color, linestyle="-", linewidth=2, alpha=1.0, label=None):
    """在 BEV 图上绘制旋转矩形。
    nuScenes box.bottom_corners() 顺序 (4 角, z=bottom):
      0=front-bottom-left,  1=front-bottom-right,
      2=rear-bottom-right,  3=rear-bottom-left
    (逆时针绕 z 轴, z 向上)
    - corners[0]→corners[1]: 沿 w 短边 (front 边)
    - corners[1]→corners[2]: 沿 l 长边 (heading 方向, box forward)
    - corners[3]→corners[0]: 沿 l 长边 (heading 反向? No — 跟 local x 同向)
    实测: corners[3]→corners[0] 方向 = local x (heading)
    """
    poly = plt.Polygon(corners, fill=False, edgecolor=color,
                       linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    ax.add_patch(poly)
    # heading: corners[3]→corners[0] 方向 (沿 l 长边, box 前向 = local x)
    cx = corners[:, 0].mean()
    cy = corners[:, 1].mean()
    dx = corners[0, 0] - corners[3, 0]
    dy = corners[0, 1] - corners[3, 1]
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx, dy = dx / norm * 1.5, dy / norm * 1.5
    ax.arrow(cx, cy, dx, dy, head_width=0.5, head_length=0.3,
             fc=color, ec=color, alpha=alpha, linewidth=1)
    if label:
        ax.text(cx, cy, label, fontsize=5, color=color,
                ha="center", va="bottom", alpha=alpha)


def collect_all_radar_points(nusc, sample, dataroot):
    """汇集一帧所有雷达通道的点云（ego 坐标系 = 自车后轴中心）。
    nuScenes 中 calibrated_sensor.rotation 是 sensor→ego 旋转（验证: R @ (p_sensor - t) 给出正确位置）,
    即 p_ego = R @ (p_sensor - trans).
    """
    RADAR_CH = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
                "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]
    all_pts = []
    for ch in RADAR_CH:
        if ch not in sample["data"]:
            continue
        sd = nusc.get("sample_data", sample["data"][ch])
        p = RadarPointCloud.from_file(str(Path(dataroot) / sd["filename"]))
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        R_sensor_to_ego = Quaternion(cs["rotation"]).rotation_matrix
        trans_sensor_in_ego = np.array(cs["translation"])
        pts_sensor = p.points[:3, :]
        # p_ego = R @ (p_sensor - t)
        pts_ego = R_sensor_to_ego @ (pts_sensor - trans_sensor_in_ego[:, None])
        rcs = p.points[5:6, :]
        all_pts.append(np.vstack([pts_ego, rcs]))
    if not all_pts:
        return np.zeros((4, 0))
    return np.hstack(all_pts)


def global_to_ego_matrix(nusc, sample):
    """获取该 sample 对应的 global→ego 变换矩阵 (ego 原点 = 自车后轴中心)。
    注意 nuScenes 中 ego_pose 描述的是自车后轴中心的 global 位姿;
    而 LIDAR_TOP 等 sensor 的 calibrated_sensor.translation 是 sensor 在 ego 中的偏移.
    """
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    rot_ep = Quaternion(ep["rotation"]).rotation_matrix  # global→ego
    trans_ep = np.array(ep["translation"])
    T = np.eye(4)
    T[:3, :3] = rot_ep.T
    T[:3, 3] = -rot_ep.T @ trans_ep
    return T


def get_sample_gt_boxes_ego(nusc, sample):
    """获取一帧所有 GT box（ego 坐标系，过滤 ignore 类别）。
    nuScenes 中 ann['rotation'] 描述的是 **box→global** 旋转（验证: Box.corners() = R @ template + center）。
    ego_pose rotation 也是 **ego→global** 旋转。
    所以 R_box_to_ego = R_ego_to_global.T @ R_box_to_global
    对应 quaternion: q_ego_to_box = q_global_to_ego * q_box_to_global
    """
    boxes = []
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    q_ego_to_global = Quaternion(ep["rotation"])
    trans_ep = np.array(ep["translation"])
    R_ego_to_global = q_ego_to_global.rotation_matrix
    R_global_to_ego = R_ego_to_global.T
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        det_cls = map_name_from_general_to_detection.get(ann["category_name"], "ignore")
        if det_cls == "ignore":
            continue
        # center: global → ego
        center_global = np.array(ann["translation"])
        center_ego = R_global_to_ego @ (center_global - trans_ep)
        # orientation: 直接用 box→ego 矩阵
        R_box_to_global = Quaternion(ann["rotation"]).rotation_matrix
        R_box_to_ego = R_global_to_ego @ R_box_to_global
        q_box_to_ego = Quaternion(matrix=R_box_to_ego)
        box = Box(center_ego.tolist(), ann["size"], q_box_to_ego)
        box.det_cls = det_cls
        box.ann_token = ann_token
        boxes.append(box)
    return boxes


def get_sample_gt_boxes(nusc, sample):
    """保留旧函数以兼容（全局坐标系），但 BEV 调用 get_sample_gt_boxes_ego。"""
    return get_sample_gt_boxes_ego(nusc, sample)


def _ego_to_screen(ego_pts, R):
    """ego 坐标 → matplotlib 屏幕坐标 (X=左, Y=上) → 旋转 90° 映射:
       屏幕 X  ↔  ego -Y  (left/right 反向)
       屏幕 Y  ↔  ego +X  (forward)
    """
    return (-ego_pts[:, 1], ego_pts[:, 0])


def draw_bev(ax, nusc, sample, dataroot, pred_boxes=None, score_thresh=0.1,
             bev_range=None, min_range=20, pad=5):
    """绘制 BEV 俯视图。
    ego 坐标系: X=forward(屏幕↑), Y=left(屏幕←), 自车在原点。
    数据在 ego 坐标中, 内部旋转 90° 渲染到 matplotlib 屏幕。
    """
    # 汇总雷达点云
    radar_pts = collect_all_radar_points(nusc, sample, dataroot)
    n_radar = int(radar_pts.shape[1])

    # 自适应 xlim/ylim
    if bev_range is None:
        xs, ys = [], []
        if n_radar > 0:
            xs.append(radar_pts[0]); ys.append(radar_pts[1])
        gt_boxes = get_sample_gt_boxes(nusc, sample)
        for b in gt_boxes:
            corners = get_box_bev_corners(b)
            xs.append(corners[:, 0]); ys.append(corners[:, 1])
        if xs and ys:
            x_all = np.concatenate(xs); y_all = np.concatenate(ys)
            max_abs = max(abs(x_all).max(), abs(y_all).max())
            R = max(max_abs + pad, min_range)
        else:
            R = min_range
        gt_boxes = get_sample_gt_boxes(nusc, sample)
    else:
        R = bev_range
        gt_boxes = get_sample_gt_boxes(nusc, sample)
    n_gt = len(gt_boxes)

    # 绘制雷达点（按 RCS 着色）
    sc = None
    if n_radar > 0:
        sx, sy = _ego_to_screen(np.column_stack([radar_pts[0], radar_pts[1]]), R)
        sc = ax.scatter(sx, sy, c=radar_pts[3], cmap="viridis",
                        s=4, alpha=0.6, zorder=1)

    # 绘制 GT boxes（实线）
    for box in gt_boxes:
        color = CLASS_COLORS_GT.get(box.det_cls, "#95a5a6")
        corners_ego = get_box_bev_corners(box)
        sx, sy = _ego_to_screen(corners_ego, R)
        corners_screen = np.column_stack([sx, sy])
        draw_bev_box(ax, corners_screen, color, linestyle="-", linewidth=2, alpha=0.9)

    # 绘制预测 boxes（虚线）
    n_pred = 0
    if pred_boxes is not None and len(pred_boxes) > 0:
        for pbox in pred_boxes:
            if pbox.get("score", 1.0) < score_thresh:
                continue
            color = CLASS_COLORS_PRED.get(pbox["det_cls"], "#7f8c8d")
            box = Box(pbox["translation"], pbox["size"], Quaternion(pbox["rotation"]))
            corners_ego = get_box_bev_corners(box)
            sx, sy = _ego_to_screen(corners_ego, R)
            corners_screen = np.column_stack([sx, sy])
            draw_bev_box(ax, corners_screen, color, linestyle="--", linewidth=1.5, alpha=0.7)
            n_pred += 1

    # 自车 (后轴中心, 在屏幕中心)
    ax.plot(0, 0, marker="^", color="white", markersize=10,
            markeredgecolor="black", zorder=5)
    # 自车朝向箭头: 屏幕 Y 方向 = ego forward (+X)
    ax.arrow(0, 0, 0, 3, head_width=0.8, head_length=0.4,
             fc="white", ec="black", linewidth=1, zorder=6)

    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect("equal")
    # 在 ego 坐标轴上的标签 (屏幕上是 -Y, X)
    ax.set_xlabel("Y [m] (left ← | right →)")
    ax.set_ylabel("X [m] (forward ↑)")
    ax.set_title(f"BEV View  (n_radar_pts={n_radar}, n_gt={n_gt}, n_pred={n_pred}, range=±{R:.0f}m)")
    ax.grid(True, alpha=0.3)

    # 简化图例
    present_gt = sorted(set(b.det_cls for b in gt_boxes))
    legend_elements = [
        Line2D([0], [0], color=CLASS_COLORS_GT.get(c, "#95a5a6"),
               linewidth=2, label=f"GT {c}")
        for c in present_gt
    ]
    if pred_boxes is not None and n_pred > 0:
        present_pred = sorted(set(b["det_cls"] for b in pred_boxes
                                  if b.get("score", 1.0) >= score_thresh))
        legend_elements += [
            Line2D([0], [0], color=CLASS_COLORS_PRED.get(c, "#7f8c8d"),
                   linewidth=2, linestyle="--", label=f"Pred {c}")
            for c in present_pred
        ]
    if legend_elements:
        ax.legend(handles=legend_elements, fontsize=7, loc="upper left")

    return sc


# ── 相机图像 + 3D box 投影 ──────────────────────────────────────

def box_in_image(box, intrinsic, imsize, vis_level=1):
    """检查 box 是否在图像可见范围内（简化版）。"""
    corners_3d = box.corners()  # (3, 8)
    # world → sensor
    # 此函数假设 box 已经在 sensor 坐标系
    corners_img = view_points(corners_3d, intrinsic, normalize=True)
    w, h = imsize
    visible = ((corners_img[0] > 0) & (corners_img[0] < w) &
               (corners_img[1] > 0) & (corners_img[1] < h) &
               (corners_img[2] > 0))
    return visible.any()


def view_points(points, view, normalize=True):
    """投影 3D 点到图像平面（nuScenes 兼容）。"""
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    view = view[:3, :3] if normalize else view[:3, :4]
    if points.shape[0] == 3:
        pts = view[:3, :3] @ points
    else:
        pts = view[:3, :4] @ np.vstack([points, np.ones((1, points.shape[1]))])
    if normalize:
        pts[:2, :] /= pts[2:3, :] + 1e-6
    return pts


def render_sample_camera(ax, nusc, sample, cam_channel, dataroot, gt_boxes):
    """渲染单路相机图像 + 3D box 投影（gt_boxes 假定为 ego 坐标系）。"""
    sd = nusc.get("sample_data", sample["data"][cam_channel])
    img_path = str(Path(dataroot) / sd["filename"])
    im = cv2.imread(img_path)
    if im is None:
        ax.text(0.5, 0.5, f"Image not found:\n{img_path}",
                transform=ax.transAxes, ha="center", va="center", fontsize=8)
        ax.set_title(CAM_TITLES.get(cam_channel, cam_channel), fontsize=9)
        return
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]

    # 相机内参
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    intrinsic = np.array(cs["camera_intrinsic"])

    # nuScenes 约定: cs.translation/rotation 是 sensor 在 ego 系的位姿
    # 构建 ego→sensor 矩阵 (4x4)
    sensor_T = np.eye(4)
    sensor_T[:3, :3] = Quaternion(cs["rotation"]).rotation_matrix
    sensor_T[:3, 3] = cs["translation"]
    ego_to_sensor = np.linalg.inv(sensor_T)

    for box in gt_boxes:
        # box 已在 ego 坐标系 → 复制到 sensor 坐标（修改 center 和 orientation）
        corners_ego = box.corners()  # (3, 8)
        ones = np.ones((1, corners_ego.shape[1]))
        corners_ego_h = np.vstack([corners_ego, ones])
        corners_sensor_h = ego_to_sensor @ corners_ego_h
        corners_sensor = corners_sensor_h[:3]
        z_min = corners_sensor[2].min()
        if z_min < 0.1:  # box 在 sensor 后方或很近
            continue

        # 投影到图像
        corners_img = view_points(corners_sensor, intrinsic, normalize=True)  # (3, 8)

        # 只绘制部分在图像内的 box
        visible = ((corners_img[0] > 0) & (corners_img[0] < w) &
                   (corners_img[1] > 0) & (corners_img[1] < h) &
                   (corners_sensor[2] > 0))
        if not visible.any():
            continue

        color = CLASS_COLORS_GT.get(box.det_cls, "#95a5a6")
        color_rgb = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

        # 绘制 3D box 的 12 条边
        corners_img_int = corners_img[:2].T.astype(np.int32)  # (8, 2)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # top
            (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
        ]
        for i, j in edges:
            pt1 = corners_img_int[i]
            pt2 = corners_img_int[j]
            # 跳过完全在图像外的边
            if (pt1[0] < -200 or pt1[0] > w + 200 or pt1[1] < -200 or pt1[1] > h + 200 or
                pt2[0] < -200 or pt2[0] > w + 200 or pt2[1] < -200 or pt2[1] > h + 200):
                continue
            cv2.line(im, tuple(pt1), tuple(pt2), color_rgb, 2)

        # 标注类别（顶部角点投影）
        # 顶面中心作为标注位置
        top_center_ego = np.array(box.center) + np.array([0, 0, box.wlh[2] / 2])
        top_center_h = ego_to_sensor @ np.array([top_center_ego[0], top_center_ego[1], top_center_ego[2], 1])
        if top_center_h[2] > 0.1:
            tx, ty = (intrinsic @ top_center_h[:3] / top_center_h[2])[:2]
            tx, ty = int(tx), int(ty)
            if 0 < tx < w and 0 < ty < h:
                cv2.putText(im, box.det_cls, (tx, ty - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 1, cv2.LINE_AA)

    ax.imshow(im)
    ax.set_title(CAM_TITLES.get(cam_channel, cam_channel), fontsize=9)
    ax.axis("off")


# ── 主入口 ─────────────────────────────────────────────────────

def visualize_sample(nusc, sample_token, dataroot, output_dir,
                     pred_file=None, score_thresh=0.1, bev_range=55):
    """可视化单帧数据。"""
    sample = nusc.get("sample", sample_token)
    gt_boxes = get_sample_gt_boxes(nusc, sample)

    # 加载预测（可选）
    pred_boxes = None
    if pred_file and Path(pred_file).exists():
        import pickle
        with open(pred_file, "rb") as f:
            pred_data = pickle.load(f)
        # 根据实际 pred 格式适配
        # 这里假设 pred_data 是 dict: token -> dict with 'boxes_lidar', 'score', 'name'
        if isinstance(pred_data, dict) and sample_token in pred_data:
            p = pred_data[sample_token]
            pred_boxes = []
            for i in range(len(p.get("boxes_lidar", []))):
                bx = p["boxes_lidar"][i]
                pred_boxes.append({
                    "translation": [bx[0], bx[1], bx[2]],
                    "size": [bx[4], bx[3], bx[5]],  # lwh → whl
                    "rotation": Quaternion(axis=[0, 0, 1], angle=bx[6]).q.tolist(),
                    "det_cls": p["name"][i] if "name" in p else "car",
                    "score": p["score"][i] if "score" in p else 1.0,
                })

    # 创建 figure: 左 BEV + 右 6 相机
    fig = plt.figure(figsize=(28, 14))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.6, 1, 1, 1],
                          hspace=0.15, wspace=0.1)

    # 左侧 BEV（占满左侧 2 行）
    ax_bev = fig.add_subplot(gs[:, 0])
    sc = draw_bev(ax_bev, nusc, sample, dataroot, pred_boxes=pred_boxes,
                  score_thresh=score_thresh, bev_range=bev_range)
    if sc is not None:
        plt.colorbar(sc, ax=ax_bev, label="RCS [dBsm]", shrink=0.6, pad=0.02)

    # 右侧 6 个相机：前排 FL/F/FR，后排 BL/B/BR
    cam_axes = [
        fig.add_subplot(gs[0, 1]),  # FRONT_LEFT
        fig.add_subplot(gs[0, 2]),  # FRONT
        fig.add_subplot(gs[0, 3]),  # FRONT_RIGHT
        fig.add_subplot(gs[1, 1]),  # BACK_LEFT
        fig.add_subplot(gs[1, 2]),  # BACK
        fig.add_subplot(gs[1, 3]),  # BACK_RIGHT
    ]

    for cam_ch, ax_cam in zip(CAM_ORDER, cam_axes):
        render_sample_camera(ax_cam, nusc, sample, cam_ch, dataroot, gt_boxes)

    # 以时间戳命名：nuScenes timestamp 是 unix epoch (s.us)
    ts = sample["timestamp"]
    ts_str = f"{ts:.6f}".replace(".", "_")
    fig.suptitle(f"Sample: {sample_token[:12]} | timestamp={ts_str}", fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = Path(output_dir) / f"vis_t{ts_str}_{sample_token[:12]}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="nuScenes 可视化工具")
    parser.add_argument("--dataroot", type=str, default="/mnt/d/DATASET_PART",
                        help="nuScenes 数据集根目录")
    parser.add_argument("--version", type=str, default="v1.0-mini",
                        help="nuScenes 数据集版本")
    parser.add_argument("--output_dir", type=str, default="output/vis_nuscenes",
                        help="输出目录")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="随机可视化帧数")
    parser.add_argument("--sample_tokens", type=str, nargs="*", default=None,
                        help="指定 sample token（优先于随机采样）")
    parser.add_argument("--pred_file", type=str, default=None,
                        help="预测结果 pkl 文件路径（可选）")
    parser.add_argument("--score_thresh", type=float, default=0.1,
                        help="预测 box 置信度阈值")
    parser.add_argument("--bev_range", type=float, default=55,
                        help="BEV 显示范围（米）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    print(f"Loading {args.version} from {args.dataroot}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    if args.sample_tokens:
        tokens = args.sample_tokens
    else:
        random.seed(args.seed)
        tokens = random.sample([s["token"] for s in nusc.sample],
                               min(args.n_samples, len(nusc.sample)))

    print(f"Visualizing {len(tokens)} samples...")
    for i, token in enumerate(tokens):
        print(f"  [{i+1}/{len(tokens)}] {token[:12]}...")
        visualize_sample(nusc, token, args.dataroot, args.output_dir,
                         pred_file=args.pred_file,
                         score_thresh=args.score_thresh,
                         bev_range=args.bev_range)

    print(f"Done. Output in {args.output_dir}/")


if __name__ == "__main__":
    main()

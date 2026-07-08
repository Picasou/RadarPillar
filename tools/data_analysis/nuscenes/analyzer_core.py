#!/usr/bin/env python3
"""nuScenes 数据分析核心模块

每种分析函数自包含（加载 + 计算），统一签名：
    analyze_xxx(source_type, source_obj, cfg) -> dict

对外暴露的分析函数：
    - analyze_basic
    - analyze_radar_distribution
    - analyze_gt_stats
    - analyze_bev_heatmap
    - analyze_gt_pointcloud
    - analyze_pts_in_gt_boxes
    - plot_all
"""
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════
#  常量定义
# ══════════════════════════════════════════════════════════════
RADAR_CHANNELS = [
    "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT",
]

DETECTION_CLASSES = [
    "car", "truck", "bus", "trailer", "construction_vehicle",
    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier",
]

# nuScenes category → 检测类映射 (devkit 模式用)
ATTR_TO_DET = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus": "bus",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.emergency": "car",
    "vehicle.other": "ignore",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.worker": "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
    "human.pedestrian.stroller": "pedestrian",
    "human.pedestrian.wheelchair": "pedestrian",
    "human.pedestrian.other": "pedestrian",
    "human.cyclist": "ignore",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "movable_object.other": "ignore",
    "static_object.bicycle_rack": "ignore",
}

# pkl 模式字段映射
RADAR_FIELD_TO_INDEX = {
    "x": 0, "y": 1, "z": 2,
    "rcs": 5,
    "vx": 6, "vy": 7,
}
RADAR_USED_FIELDS = ["x", "y", "z", "rcs", "vx", "vy"]


# ══════════════════════════════════════════════════════════════
#  数据源检测
# ══════════════════════════════════════════════════════════════
def detect_source(dataroot, pkl_root=None):
    """自动检测数据源类型。

    Returns:
        "pkl": 有 pkl 文件
        "devkit": 有 nuScenes 原始数据 (v1.0-mini / v1.0-trainval)
    """
    root = Path(dataroot)
    pkl_root = Path(pkl_root) if pkl_root else root

    # 检查 pkl
    pkl_files = list(pkl_root.glob("nuscenes_infos_radar_1sweeps_*.pkl"))
    if pkl_files:
        return "pkl"

    # 检查 devkit 原始数据
    for v in ["v1.0-mini", "v1.0-trainval", "v1.0-test"]:
        if (root / v).exists():
            return "devkit"

    raise ValueError(f"无法识别数据源: {dataroot}, 未找到 pkl 或 nuScenes 原始数据")


# ══════════════════════════════════════════════════════════════
#  数据加载器 - pkl 模式
# ══════════════════════════════════════════════════════════════
def load_infos(pkl_paths):
    """加载多个 pkl 文件并合并。"""
    infos = []
    for p in pkl_paths:
        with open(p, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list):
            infos.extend(data)
        else:
            infos.append(data)
    return infos


def get_pkl_files(dataroot, pkl_root=None):
    """获取 pkl 文件列表。"""
    root = Path(pkl_root) if pkl_root else Path(dataroot)
    pkl_files = sorted(root.glob("nuscenes_infos_radar_1sweeps_*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"未找到 pkl 文件: {root}/nuscenes_infos_radar_1sweeps_*.pkl")
    return pkl_files


def load_radar_points_pkl(info, root_path):
    """pkl 模式: 加载 5 通道雷达点云到 ego 系。

    Returns: (N, 7) [x, y, z, rcs, vx, vy, time]
    """
    from pcdet.datasets.nuscenes.nuscenes_radar_dataset import NuScenesRadarDataset

    used_indices = [RADAR_FIELD_TO_INDEX[f] for f in RADAR_USED_FIELDS]
    T_per_ch = info.get("radar_T_ego_sensor", {})

    channel_points = []
    for ch, rel_path in info.get("radar_channels", {}).items():
        p = root_path / rel_path
        if p.exists():
            channel_points.append(
                NuScenesRadarDataset.load_one_radar_pcd(str(p), used_indices, T_per_ch.get(ch))
            )

    if not channel_points:
        fallback = root_path / info.get("radar_path", "")
        if fallback and fallback.exists():
            channel_points.append(
                NuScenesRadarDataset.load_one_radar_pcd(str(fallback), used_indices, None)
            )

    if not channel_points:
        return np.zeros((0, 7), dtype=np.float32)

    points = np.concatenate(channel_points, axis=0)
    points[:, -1] = 0.0  # time = 0
    return points


def load_gt_boxes_pkl(info):
    """pkl 模式: 加载 GT boxes。

    Returns: boxes (N, 7), names (list)
    """
    boxes = info.get("gt_boxes")
    names = info.get("gt_names")
    if boxes is None or names is None:
        return None, None
    return boxes, names


# ══════════════════════════════════════════════════════════════
#  数据加载器 - devkit 模式
# ══════════════════════════════════════════════════════════════
def load_radar_points_devkit(nusc, sample):
    """devkit 模式: 加载 5 通道雷达点到 ego 系。

    Returns: (N, 3) [x, y, z]
    """
    from nuscenes.utils.data_classes import RadarPointCloud
    from pyquaternion import Quaternion

    all_pts = []
    sd_ref = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", sd_ref["ego_pose_token"])

    for ch in RADAR_CHANNELS:
        if ch not in sample["data"]:
            continue
        sd = nusc.get("sample_data", sample["data"][ch])
        pcd_path = os.path.join(nusc.dataroot, sd["filename"])
        if not os.path.exists(pcd_path):
            continue

        pc = RadarPointCloud.from_file(pcd_path)
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        pc.rotate(Quaternion(cs["rotation"]).rotation_matrix)
        pc.translate(np.array(cs["translation"]))

        all_pts.append(pc.points[:3, :].T)

    if all_pts:
        return np.concatenate(all_pts, axis=0).astype(np.float64)
    return np.zeros((0, 3), dtype=np.float64)


def load_gt_boxes_devkit(nusc, sample):
    """devkit 模式: 加载 GT boxes 到 ego 系。

    Returns: boxes (M, 7), names (list)
    """
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion

    sd_ref = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", sd_ref["ego_pose_token"])
    rot_g = Quaternion(ego_pose["rotation"])
    T = np.eye(4)
    T[:3, :3] = rot_g.rotation_matrix.T
    T[:3, 3] = -rot_g.rotation_matrix.T @ np.array(ego_pose["translation"])

    boxes, names = [], []
    for ann_tok in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_tok)
        det_cls = ATTR_TO_DET.get(ann["category_name"], "ignore")
        if det_cls == "ignore":
            continue

        b = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))
        center_ego = T[:3, :3] @ b.center + T[:3, 3]
        wlh = np.array([b.wlh[1], b.wlh[0], b.wlh[2]])  # (l, w, h)
        yaw_ego = Quaternion(T[:3, :3] @ b.rotation_matrix).yaw_pitch_roll[0]

        boxes.append([*center_ego, *wlh, yaw_ego])
        names.append(det_cls)

    if boxes:
        return np.array(boxes, dtype=np.float64), names
    return np.zeros((0, 7), dtype=np.float64), []


def get_nuscenes(dataroot, version="v1.0-mini"):
    """加载 NuScenes 数据集对象。"""
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version=version, dataroot=dataroot, verbose=False)


# ══════════════════════════════════════════════════════════════
#  批量加载辅助函数
# ══════════════════════════════════════════════════════════════
def _iter_samples(source_type, source_obj, cfg):
    """根据数据源类型，返回 (info/sample, index) 的迭代器。"""
    max_samples = cfg.get("max_samples", 0) if isinstance(cfg, dict) else cfg.get("max_samples", 0)

    if source_type == "pkl":
        infos = source_obj
        items = infos[:max_samples] if max_samples else infos
        return items
    else:
        nusc = source_obj
        samples = nusc.sample[:max_samples] if max_samples else nusc.sample
        return samples


def _load_radar_points(source_type, source_obj, cfg):
    """批量加载雷达点云。

    Returns: list[np.ndarray] — 每帧的雷达点
    """
    items = _iter_samples(source_type, source_obj, cfg)
    radar_points_list = []

    if source_type == "pkl":
        root = Path(cfg.get("pkl_root", cfg.get("dataroot", ".")) if isinstance(cfg, dict)
                     else getattr(cfg, "pkl_root", getattr(cfg, "dataroot", ".")))
        for info in tqdm(items, desc="Loading radar (pkl)"):
            pts = load_radar_points_pkl(info, root)
            radar_points_list.append(pts)
    else:
        nusc = source_obj
        for sample in tqdm(items, desc="Loading radar (devkit)"):
            pts = load_radar_points_devkit(nusc, sample)
            radar_points_list.append(pts)

    return radar_points_list


def _load_gt_data(source_type, source_obj, cfg, load_velocity=False):
    """批量加载 GT 数据。

    Returns: list[dict] — 每帧 {"boxes": ndarray, "names": list, "velocities": list|None}
    """
    items = _iter_samples(source_type, source_obj, cfg)
    gt_data_list = []

    if source_type == "pkl":
        for info in tqdm(items, desc="Loading GT (pkl)"):
            boxes, names = load_gt_boxes_pkl(info)
            if boxes is not None and len(names) > 0:
                gt_data_list.append({
                    "boxes": boxes,
                    "names": names,
                    "velocities": [None] * len(names),
                })
    else:
        nusc = source_obj
        for sample in tqdm(items, desc="Loading GT (devkit)"):
            boxes, names = load_gt_boxes_devkit(nusc, sample)
            if boxes.shape[0] > 0:
                gt_data_list.append({
                    "boxes": boxes,
                    "names": names,
                    "velocities": [None] * len(names),
                })

    return gt_data_list


def _load_radar_gt_pairs(source_type, source_obj, cfg):
    """批量加载 (radar, GT) 配对数据。

    Returns: list[tuple[np.ndarray, dict]]
    """
    items = _iter_samples(source_type, source_obj, cfg)
    pairs = []

    if source_type == "pkl":
        root = Path(cfg.get("pkl_root", cfg.get("dataroot", ".")) if isinstance(cfg, dict)
                     else getattr(cfg, "pkl_root", getattr(cfg, "dataroot", ".")))
        for info in tqdm(items, desc="Loading radar+GT (pkl)"):
            pts = load_radar_points_pkl(info, root)
            boxes, names = load_gt_boxes_pkl(info)
            if boxes is not None and len(names) > 0:
                pairs.append((pts, {"boxes": boxes, "names": names}))
    else:
        nusc = source_obj
        for sample in tqdm(items, desc="Loading radar+GT (devkit)"):
            pts = load_radar_points_devkit(nusc, sample)
            boxes, names = load_gt_boxes_devkit(nusc, sample)
            if boxes.shape[0] > 0:
                pairs.append((pts, {"boxes": boxes, "names": names}))

    return pairs


# ══════════════════════════════════════════════════════════════
#  Point-in-Box
# ══════════════════════════════════════════════════════════════
try:
    from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False


def count_pts_in_boxes(points_xyz, boxes_7col):
    """统计每个 box 内的点数。"""
    if boxes_7col.shape[0] == 0 or points_xyz.shape[0] == 0:
        return np.zeros(boxes_7col.shape[0], dtype=np.int64)

    if HAS_CUDA:
        mask = points_in_boxes_cpu(points_xyz.astype(np.float32), boxes_7col[:, :7].astype(np.float32))
        return mask.sum(axis=1).astype(np.int64)

    # NumPy fallback
    counts = np.zeros(boxes_7col.shape[0], dtype=np.int64)
    points_xyz = points_xyz.astype(np.float64)
    boxes = boxes_7col.astype(np.float64)

    for i in range(boxes.shape[0]):
        cx, cy, cz, dx, dy, dz, heading = boxes[i, :7]
        pts = points_xyz - np.array([cx, cy, cz])
        cos_h, sin_h = np.cos(-heading), np.sin(-heading)
        x_rot = pts[:, 0] * cos_h - pts[:, 1] * sin_h
        y_rot = pts[:, 0] * sin_h + pts[:, 1] * cos_h
        z_rot = pts[:, 2]
        inside = (np.abs(x_rot) <= dx / 2) & (np.abs(y_rot) <= dy / 2) & (np.abs(z_rot) <= dz / 2)
        counts[i] = inside.sum()

    return counts


# ══════════════════════════════════════════════════════════════
#  分析器实现 (自包含: 加载 + 计算)
# ══════════════════════════════════════════════════════════════
def analyze_basic(source_type, source_obj, cfg=None):
    """基础统计: scene/sample/annotation/instance 数量。"""
    from pcdet.datasets.nuscenes.nuscenes_utils import map_name_from_general_to_detection

    if source_type == "pkl":
        infos = source_obj
        n_sample = len(infos)
        # 统计类别
        cat_counter = defaultdict(int)
        for info in infos:
            for name in info.get("gt_names", []):
                if name != "ignore":
                    cat_counter[name] += 1
        return {
            "n_sample": n_sample,
            "n_anno": sum(cat_counter.values()),
            "cat": dict(sorted(cat_counter.items(), key=lambda x: -x[1])),
        }
    else:
        # devkit
        nusc = source_obj
        cat_counter = defaultdict(int)
        for ann in nusc.sample_annotation:
            name = map_name_from_general_to_detection.get(ann["category_name"], "ignore")
            if name != "ignore":
                cat_counter[name] += 1
        return {
            "n_scene": len(nusc.scene),
            "n_sample": len(nusc.sample),
            "n_anno": len(nusc.sample_annotation),
            "n_inst": len(nusc.instance),
            "cat": dict(sorted(cat_counter.items(), key=lambda x: -x[1])),
        }


def analyze_radar_distribution(source_type, source_obj, cfg):
    """雷达点云分布: range, RCS, velocity magnitude。"""
    radar_points_list = _load_radar_points(source_type, source_obj, cfg)

    rng_all, rcs_all, vmag_all = [], [], []

    for pts in radar_points_list:
        if pts.shape[0] == 0:
            continue
        # range
        rng = np.linalg.norm(pts[:, :3], axis=1)
        rng_all.append(rng)
        # RCS (index 3 if 7-col, or not available in devkit mode)
        if pts.shape[1] >= 4:
            rcs_all.append(pts[:, 3])
        # vmag
        if pts.shape[1] >= 6:
            vmag = np.linalg.norm(pts[:, 4:6], axis=1)
            vmag_all.append(vmag)

    def st(arr):
        if len(arr) == 0:
            return {"mean": 0, "p50": 0, "p95": 0, "max": 0}
        arr = np.concatenate(arr)
        return {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        }

    return {
        "n_frames": len(radar_points_list),
        "range": st(rng_all),
        "rcs": st(rcs_all) if rcs_all else {},
        "vmag": st(vmag_all) if vmag_all else {},
    }


def analyze_gt_stats(source_type, source_obj, cfg):
    """GT 统计: 类别数量、距离、速度、尺寸。"""
    gt_data_list = _load_gt_data(source_type, source_obj, cfg)

    cls_count = defaultdict(int)
    rng_by_cls = defaultdict(list)
    spd_by_cls = defaultdict(list)
    sz_by_cls = defaultdict(list)

    for gt_data in gt_data_list:
        boxes, names = gt_data["boxes"], gt_data["names"]
        velocities = gt_data.get("velocities", [None] * len(names))

        for i, name in enumerate(names):
            if name == "ignore":
                continue
            cls_count[name] += 1
            # range (xy 距离)
            rng = np.linalg.norm(boxes[i, :2])
            rng_by_cls[name].append(float(rng))
            # speed
            v = velocities[i]
            if v is not None and not np.isnan(v).any():
                spd_by_cls[name].append(float(np.linalg.norm(v[:2])))
            # size (dx, dy, dz)
            sz_by_cls[name].append(boxes[i, 3:6].tolist())

    def avg(d):
        return float(np.mean(d)) if d else 0.0

    return {
        "n_gt_boxes": sum(len(g["names"]) for g in gt_data_list),
        "counts": dict(sorted(cls_count.items(), key=lambda x: -x[1])),
        "range": {c: avg(rng_by_cls[c]) for c in cls_count},
        "speed": {c: avg(spd_by_cls[c]) for c in cls_count},
        "size": {c: np.mean(sz_by_cls[c], 0).tolist() for c in cls_count if sz_by_cls[c]},
    }


def analyze_bev_heatmap(source_type, source_obj, cfg, range_m=50, bins=100):
    """BEV 热力图: GT 中心点在 ego 坐标系下的分布。"""
    gt_data_list = _load_gt_data(source_type, source_obj, cfg)

    centers = []
    for gt_data in gt_data_list:
        boxes, names = gt_data["boxes"], gt_data["names"]
        for i, name in enumerate(names):
            if name != "ignore":
                centers.append(boxes[i, :2])  # cx, cy
    centers = np.array(centers) if centers else np.zeros((0, 2))

    H, xedges, yedges = np.histogram2d(
        centers[:, 0], centers[:, 1],
        bins=bins, range=[[-range_m, range_m], [-range_m, range_m]]
    )
    return {"n": len(centers), "hist": H.T.tolist()}


def analyze_gt_pointcloud(source_type, source_obj, cfg):
    """GT box 内预计算的 radar/lidar 点数 (仅 pkl 模式)。"""
    if source_type != "pkl":
        return None

    infos = source_obj
    stats = defaultdict(lambda: {"radar": [], "lidar": []})
    for info in infos:
        names = info.get("gt_names", [])
        n_radar = info.get("num_radar_pts", [])
        n_lidar = info.get("num_lidar_pts", [])
        for name, nr, nl in zip(names, n_radar, n_lidar):
            if name != "ignore":
                stats[name]["radar"].append(int(nr))
                stats[name]["lidar"].append(int(nl))

    summary = {}
    for cls, d in stats.items():
        radar = np.array(d["radar"])
        lidar = np.array(d["lidar"])
        summary[cls] = {
            "n_boxes": len(radar),
            "radar": _metrics(radar),
            "lidar": _metrics(lidar),
        }
    return summary


def analyze_pts_in_gt_boxes(source_type, source_obj, cfg):
    """GT box 内实际加载的 5ch 雷达点数 (point-in-box)。"""
    radar_gt_pairs = _load_radar_gt_pairs(source_type, source_obj, cfg)

    stats = defaultdict(list)

    for radar_pts, gt_data in radar_gt_pairs:
        boxes, names = gt_data["boxes"], gt_data["names"]
        if boxes.shape[0] == 0:
            continue

        counts = count_pts_in_boxes(radar_pts[:, :3], boxes)
        for i, name in enumerate(names):
            if name != "ignore":
                stats[name].append(int(counts[i]))

    summary = {}
    for cls, arr in stats.items():
        arr = np.array(arr)
        summary[cls] = {"n_boxes": len(arr), **_metrics(arr)}
    return summary


def _metrics(arr):
    """计算统计指标。"""
    if len(arr) == 0:
        return {"mean": 0, "median": 0, "p95": 0, "frac_zero": 0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "frac_zero": float((arr == 0).mean()),
    }


# ══════════════════════════════════════════════════════════════
#  可视化 (可选)
# ══════════════════════════════════════════════════════════════
def plot_all(report, out_dir):
    """生成所有可视化图表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TODO: 按需实现各类图表
    # - GT class 柱状图
    # - radar range/rcs/vmag 分布
    # - BEV heatmap
    # - boxplot / CDF

    print(f"Plots saved to {out_dir}/")

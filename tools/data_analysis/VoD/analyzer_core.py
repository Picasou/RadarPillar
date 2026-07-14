#!/usr/bin/env python3
"""VoD (View of Delft) 4D-radar 数据分析核心模块

镜像 tools/data_analysis/nuscenes 的功能结构，适配 VoD radar_5frames 数据集。

设计要点:
    - 数据源统一为 pcdet 生成的 vod_infos_{split}.pkl（含 gt_boxes_lidar 与
      预计算 num_points_in_gt），点云从 velodyne/*.bin 实时读取。
    - 分析产出为"可合并的原始数组"(analyze_shard)，支持分片(shard)并行后再
      合并(merge_partials)，最后统一计算统计量与绘图。

回答三个问题 (扩展版):
    1. 平均每帧点云数         -> compute_report.frame_points.*
    2. GT 类别数量统计        -> compute_report.gt_class_counts
                                + 14 类原始 label_2 标签分布 (configurable)
    3. 每类目标点云个数       -> compute_report.per_class_pointcloud
                                (pkl FOV 预计算 + 全点云实时 point-in-box)

扩展:
    4. 7 通道逐字段统计       -> compute_report.radar_field_stats
                                (range, rcs, v_r, v_r_comp, |v|, time, z)
    5. 距离分箱 / 高度分布    -> compute_report.distance_bins / z_distribution
    6. BEV 热力图数据         -> compute_report.bev_heatmap
    7. 目标按距离分箱点数     -> compute_report.per_class_by_distance
"""
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ══════════════════════════════════════════════════════════════
#  常量定义
# ══════════════════════════════════════════════════════════════
RADAR_FEATURE_ORDER = ["x", "y", "z", "rcs", "v_r", "v_r_comp", "time"]
FIELD_IDX = {f: i for i, f in enumerate(RADAR_FEATURE_ORDER)}

# 检测类别 (与 vod_radarpillar 模型训练一致)
DETECTION_CLASSES = ["Car", "Pedestrian", "Cyclist"]

# 距离分箱 (m)
DISTANCE_BINS = [0, 10, 20, 30, 50, 100, 200]

# 截断 / 遮挡分桶
TRUNCATION_LEVELS = [(0.0, 0.15, "none"), (0.15, 0.30, "partial"), (0.30, 0.50, "mostly"), (0.50, 1.01, "heavy")]
OCCLUSION_LEVELS = [0, 1, 2, 3]  # 0=visible, 1=partly, 2=mostly, 3=unknown

# BEV 热力图
BEV_RANGE_M = 50
BEV_BINS = 100


# ══════════════════════════════════════════════════════════════
#  Point-in-Box
# ══════════════════════════════════════════════════════════════
try:
    import torch
    from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
    HAS_OP = True
except Exception:
    HAS_OP = False


def count_pts_in_boxes(xyz, boxes):
    if boxes.shape[0] == 0 or xyz.shape[0] == 0:
        return np.zeros(boxes.shape[0], dtype=np.int64)
    if HAS_OP:
        idx = points_in_boxes_cpu(
            torch.from_numpy(xyz[:, :3].astype(np.float32)),
            torch.from_numpy(boxes[:, :7].astype(np.float32)),
        ).numpy()
        return idx.sum(axis=1).astype(np.int64)
    counts = np.zeros(boxes.shape[0], dtype=np.int64)
    p = xyz[:, :3].astype(np.float64)
    b = boxes.astype(np.float64)
    for i in range(b.shape[0]):
        cx, cy, cz, dx, dy, dz, h = b[i, :7]
        q = p - np.array([cx, cy, cz])
        c, s = np.cos(-h), np.sin(-h)
        xr = q[:, 0] * c - q[:, 1] * s
        yr = q[:, 0] * s + q[:, 1] * c
        zr = q[:, 2]
        counts[i] = ((np.abs(xr) <= dx / 2) & (np.abs(yr) <= dy / 2) & (np.abs(zr) <= dz / 2)).sum()
    return counts


# ══════════════════════════════════════════════════════════════
#  数据加载
# ══════════════════════════════════════════════════════════════
def load_infos(dataroot, split):
    p = Path(dataroot) / f"vod_infos_{split}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"未找到 infos: {p}")
    with open(p, "rb") as f:
        return pickle.load(f)


def split_root(dataroot, split):
    return Path(dataroot) / ("training" if split != "test" else "testing")


def load_points(root_split, idx):
    return np.fromfile(str(root_split / "velodyne" / f"{idx}.bin"), dtype=np.float32).reshape(-1, 7)


def scan_raw_labels(label_dir):
    """扫一遍原始 label_2/*.txt, 返回 14 类 (含 DontCare) 计数。"""
    c = Counter()
    n_files = 0
    for fp in Path(label_dir).glob("*.txt"):
        n_files += 1
        for line in open(fp):
            parts = line.split()
            if parts:
                c[parts[0]] += 1
    return n_files, dict(c)


# ══════════════════════════════════════════════════════════════
#  分片分析
# ══════════════════════════════════════════════════════════════
def _per_frame_field_stats(pts):
    """每帧 7 通道逐点统计 -> per-frame dict。"""
    n = pts.shape[0]
    if n == 0:
        return {"n": 0}
    rng = np.linalg.norm(pts[:, :3], axis=1)
    res = {
        "n": int(n),
        "range": rng,                 # 用于全局合并
        "rcs": pts[:, 3],
        "v_r": pts[:, 4],
        "v_r_comp": pts[:, 5],
        "time": pts[:, 6],
        "z": pts[:, 2],
    }
    res["vmag"] = np.abs(pts[:, 4])  # 径向速度绝对值 (替代 vx/vy 不存在的环境)
    return res


def analyze_shard(infos, root_split, shard=0, nshards=1, limit=0, desc=""):
    """对 infos 的一个分片做统计，输出可合并的原始数组。"""
    from tqdm import tqdm

    sub = infos[shard::nshards] if nshards > 1 else infos
    if limit:
        sub = sub[:limit]

    ppf = []                       # 每帧点数
    ppf_class = defaultdict(int)   # 每帧目标数 (3 类)
    bev_centers = []               # GT 中心 BEV 坐标 (cx, cy)
    field_arrays = defaultdict(list)   # 每帧 7 通道数组 (用于合并后算分位)

    per = defaultdict(lambda: {
        "npig": [], "live": [], "range": [], "size": [],
        "truncated": [], "occluded": [], "difficulty": [],
        "range_bin": [],   # 距离分桶 (0..len(DISTANCE_BINS)-2)
    })

    for info in tqdm(sub, desc=desc or "analyze"):
        idx = info["point_cloud"]["lidar_idx"]
        pts = load_points(root_split, idx)
        n = pts.shape[0]
        ppf.append(n)
        fst = _per_frame_field_stats(pts)
        for k, v in fst.items():
            if k != "n":
                field_arrays[k].append(v)
        field_arrays["n_per_frame"].append(n)

        annos = info.get("annos", {})
        names = annos.get("name", np.array([]))
        boxes = annos.get("gt_boxes_lidar", np.zeros((0, 7)))
        npig = annos.get("num_points_in_gt", None)
        truncated = annos.get("truncated", np.array([]))
        occluded = annos.get("occluded", np.array([]))
        difficulty = annos.get("difficulty", np.array([]))

        if boxes.shape[0] == 0:
            continue

        live = count_pts_in_boxes(pts[:, :3], boxes)
        for k in range(boxes.shape[0]):
            name = str(names[k]) if k < len(names) else "DontCare"
            if name not in DETECTION_CLASSES:
                continue

            ppf_class[name] += 1
            r = float(np.linalg.norm(boxes[k, :2]))
            bev_centers.append((float(boxes[k, 0]), float(boxes[k, 1])))

            d = per[name]
            d["live"].append(int(live[k]))
            if npig is not None and k < len(npig) and npig[k] >= 0:
                d["npig"].append(int(npig[k]))
            d["range"].append(r)
            d["size"].append([float(boxes[k, 3]), float(boxes[k, 4]), float(boxes[k, 5])])
            # 距离分桶
            d["range_bin"].append(int(np.searchsorted(DISTANCE_BINS, r, side="right") - 1))
            # 截断 / 遮挡 / 难度
            if k < len(truncated):
                d["truncated"].append(float(truncated[k]))
            if k < len(occluded):
                d["occluded"].append(int(occluded[k]))
            if k < len(difficulty):
                d["difficulty"].append(int(difficulty[k]))

    return {
        "n_frames": len(sub),
        "points_per_frame": ppf,
        "per_frame_class_count": dict(ppf_class),
        "bev_centers": bev_centers,
        "field_arrays": dict(field_arrays),  # 全点云 7 通道 (合并用)
        "per_class": {k: dict(v) for k, v in per.items()},
    }


def merge_partials(parts):
    """合并多个 analyze_shard 的输出。"""
    ppf = []
    ppf_class = defaultdict(int)
    bev_centers = []
    field_arrays = defaultdict(list)
    per = defaultdict(lambda: {
        "npig": [], "live": [], "range": [], "size": [],
        "truncated": [], "occluded": [], "difficulty": [], "range_bin": [],
    })
    nf = 0
    for pt in parts:
        nf += pt.get("n_frames", len(pt.get("points_per_frame", [])))
        ppf.extend(pt.get("points_per_frame", []))
        for k, v in pt.get("per_frame_class_count", {}).items():
            ppf_class[k] += v
        bev_centers.extend(pt.get("bev_centers", []))
        for k, v in pt.get("field_arrays", {}).items():
            field_arrays[k].extend(v)
        for cls, d in pt.get("per_class", {}).items():
            for key in per[cls]:
                per[cls][key].extend(d.get(key, []))
    return {
        "n_frames": nf,
        "points_per_frame": ppf,
        "per_frame_class_count": dict(ppf_class),
        "bev_centers": bev_centers,
        "field_arrays": dict(field_arrays),
        "per_class": {k: dict(v) for k, v in per.items()},
    }


# ══════════════════════════════════════════════════════════════
#  统计量计算
# ══════════════════════════════════════════════════════════════
def _m(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {"mean": 0, "median": 0, "p5": 0, "p25": 0, "p75": 0, "p95": 0, "max": 0, "frac_zero": 0, "n": 0}
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p5": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
        "frac_zero": float((a == 0).mean()),
        "n": int(a.size),
    }


def compute_report(merged, name, raw_label_counts=None, raw_label_files=0):
    """由合并后的原始数组 + 14 类原始标签计数 -> 最终报告。"""
    ppf = np.asarray(merged["points_per_frame"], dtype=np.float64)
    frame = {
        "n_frames": int(merged["n_frames"]),
        "avg_points_per_frame": float(ppf.mean()) if ppf.size else 0,
        "median": float(np.median(ppf)) if ppf.size else 0,
        "p5": float(np.percentile(ppf, 5)) if ppf.size else 0,
        "p95": float(np.percentile(ppf, 95)) if ppf.size else 0,
        "min": int(ppf.min()) if ppf.size else 0,
        "max": int(ppf.max()) if ppf.size else 0,
        "total_points": int(ppf.sum()) if ppf.size else 0,
    }

    # 7 通道逐字段统计
    radar_field_stats = {}
    for k in ("range", "rcs", "v_r", "v_r_comp", "vmag", "time", "z"):
        arr = merged["field_arrays"].get(k, [])
        if arr:
            radar_field_stats[k] = _m(np.concatenate(arr))

    # 距离分桶
    ppf_arr = ppf
    rng_all = np.concatenate(merged["field_arrays"].get("range", [])) if merged["field_arrays"].get("range") else np.array([])
    dist_bins = []
    for i in range(len(DISTANCE_BINS) - 1):
        lo, hi = DISTANCE_BINS[i], DISTANCE_BINS[i + 1]
        in_bin = (rng_all >= lo) & (rng_all < hi)
        cnt = int(in_bin.sum())
        area = np.pi * (hi ** 2 - lo ** 2)
        density = cnt / area if area > 0 else 0
        dist_bins.append({
            "range_m": f"{lo}-{hi}",
            "n_points": cnt,
            "frac": float(cnt / rng_all.size) if rng_all.size else 0,
            "density_pts_per_m2": float(density),
        })

    # BEV 热力图
    centers = np.asarray(merged["bev_centers"], dtype=np.float64) if merged["bev_centers"] else np.zeros((0, 2))
    if centers.shape[0] > 0:
        H, _, _ = np.histogram2d(centers[:, 0], centers[:, 1],
                                 bins=BEV_BINS, range=[[-BEV_RANGE_M, BEV_RANGE_M], [-BEV_RANGE_M, BEV_RANGE_M]])
        bev_heatmap = {"n": int(centers.shape[0]), "hist": H.T.tolist(),
                       "range_m": BEV_RANGE_M, "bins": BEV_BINS}
    else:
        bev_heatmap = {"n": 0, "hist": [], "range_m": BEV_RANGE_M, "bins": BEV_BINS}

    # 类别统计
    per, counts = {}, {}
    for cls in DETECTION_CLASSES:
        d = merged["per_class"].get(cls)
        if not d:
            continue
        n = len(d["live"])
        counts[cls] = n
        # 按距离分桶的点数 (full 口径)
        bin_counts = [[0, 0] for _ in range(len(DISTANCE_BINS) - 1)]  # [cnt_boxes, sum_pts]
        for bi, pt in zip(d["range_bin"], d["live"]):
            if 0 <= bi < len(bin_counts):
                bin_counts[bi][0] += 1
                bin_counts[bi][1] += pt
        per_cls_dist = []
        for i, (cb, sp) in enumerate(bin_counts):
            lo, hi = DISTANCE_BINS[i], DISTANCE_BINS[i + 1]
            per_cls_dist.append({
                "range_m": f"{lo}-{hi}",
                "n_boxes": cb,
                "mean_pts": float(sp / cb) if cb else 0,
            })

        per[cls] = {
            "n_boxes": n,
            "pts_in_box_pkl_fov": _m(d["npig"]),
            "pts_in_box_full": _m(d["live"]),
            "range": _m(d["range"]),
            "size_lwh": {  # 逐维箱型
                "length": _m([s[0] for s in d["size"]]),
                "width":  _m([s[1] for s in d["size"]]),
                "height": _m([s[2] for s in d["size"]]),
            },
            "truncated": _m(d["truncated"]),
            "occluded_counts": dict(Counter(d["occluded"])),
            "difficulty_counts": dict(Counter(d["difficulty"])),
            "points_by_distance": per_cls_dist,
        }

    # 14 类原始分布
    raw_dist = {"n_files": raw_label_files, "counts": raw_label_counts or {}}

    return {
        "name": name,
        "frame_points": frame,
        "gt_class_counts": dict(sorted(counts.items(), key=lambda x: -x[1])),
        "n_gt_boxes": int(sum(counts.values())),
        "per_class_pointcloud": per,
        "per_frame_class_count": merged["per_frame_class_count"],
        "radar_field_stats": radar_field_stats,
        "distance_bins": dist_bins,
        "bev_heatmap": bev_heatmap,
        "raw_label_distribution": raw_dist,
    }


# ══════════════════════════════════════════════════════════════
#  CSV 输出
# ══════════════════════════════════════════════════════════════
def write_csvs(report, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) frame_points.csv
    fp = report["frame_points"]
    (out_dir / "frame_points.csv").write_text(
        "metric,value\n"
        f"n_frames,{fp['n_frames']}\n"
        f"avg_points_per_frame,{fp['avg_points_per_frame']:.4f}\n"
        f"median,{fp['median']}\n"
        f"p5,{fp['p5']}\n"
        f"p95,{fp['p95']}\n"
        f"min,{fp['min']}\n"
        f"max,{fp['max']}\n"
        f"total_points,{fp['total_points']}\n",
        encoding="utf-8")

    # 2) per_class.csv
    lines = ["class,n_boxes,full_mean,full_median,full_p95,full_frac_zero,fov_mean,fov_median,fov_frac_zero,"
             "range_mean,range_p95,length_mean,width_mean,height_mean"]
    for cls, d in report["per_class_pointcloud"].items():
        fl = d["pts_in_box_full"]; fv = d["pts_in_box_pkl_fov"]
        rg = d["range"]; sz = d["size_lwh"]
        lines.append(
            f"{cls},{d['n_boxes']},"
            f"{fl['mean']:.2f},{fl['median']:.0f},{fl['p95']:.0f},{fl['frac_zero']:.3f},"
            f"{fv['mean']:.2f},{fv['median']:.0f},{fv['frac_zero']:.3f},"
            f"{rg['mean']:.2f},{rg['p95']:.2f},"
            f"{sz['length']['mean']:.3f},{sz['width']['mean']:.3f},{sz['height']['mean']:.3f}"
        )
    (out_dir / "per_class.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 3) radar_field_stats.csv
    lines = ["field,mean,median,p5,p25,p75,p95,max,frac_zero,n"]
    for f, m in report["radar_field_stats"].items():
        lines.append(
            f"{f},{m['mean']:.4f},{m['median']:.4f},{m['p5']:.4f},{m['p25']:.4f},"
            f"{m['p75']:.4f},{m['p95']:.4f},{m['max']:.4f},{m['frac_zero']:.3f},{m['n']}"
        )
    (out_dir / "radar_field_stats.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 4) distance_bins.csv
    lines = ["range_m,n_points,frac,density_pts_per_m2"]
    for b in report["distance_bins"]:
        lines.append(f"{b['range_m']},{b['n_points']},{b['frac']:.4f},{b['density_pts_per_m2']:.4f}")
    (out_dir / "distance_bins.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 5) points_by_distance.csv (宽表: 一行一类, 一列一距离桶)
    header = ["class"] + [f"mean_pts_{b['range_m']}m" for b in report["distance_bins"]]
    lines = [",".join(header)]
    for cls, d in report["per_class_pointcloud"].items():
        row = [cls] + [f"{b['mean_pts']:.2f}" for b in d["points_by_distance"]]
        lines.append(",".join(row))
    (out_dir / "points_by_distance.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 6) raw_label_distribution.csv (14 类)
    rl = report["raw_label_distribution"]
    lines = ["class,count"]
    for k, v in sorted(rl["counts"].items(), key=lambda x: -x[1]):
        lines.append(f"{k},{v}")
    (out_dir / "raw_label_distribution.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 7) occlusion_difficulty.csv
    lines = ["class,occluded_0,occluded_1,occluded_2,occluded_3,easy,moderate,hard,unknown"]
    for cls, d in report["per_class_pointcloud"].items():
        oc = d["occluded_counts"]; df = d["difficulty_counts"]
        lines.append(
            f"{cls},"
            f"{oc.get(0,0)},{oc.get(1,0)},{oc.get(2,0)},{oc.get(3,0)},"
            f"{df.get(1,0)},{df.get(2,0)},{df.get(3,0)},{df.get(0,0)}"
        )
    (out_dir / "occlusion_difficulty.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"CSVs saved to {out_dir}/")


# ══════════════════════════════════════════════════════════════
#  可视化
# ══════════════════════════════════════════════════════════════
def plot_all(report, merged, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = report["name"]

    # 1) GT 类别数量
    counts = report["gt_class_counts"]
    if counts:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(list(counts.keys()), list(counts.values()), color=["#4C72B0", "#DD8452", "#55A868"])
        for i, (k, v) in enumerate(counts.items()):
            ax.text(i, v, str(v), ha="center", va="bottom")
        ax.set_title(f"GT class counts ({name})")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / "gt_class_counts.png", dpi=120)
        plt.close(fig)

    # 2) 每帧点数直方图
    ppf = np.asarray(merged["points_per_frame"], dtype=np.float64)
    if ppf.size:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(ppf, bins=50, color="#4C72B0", alpha=0.85)
        ax.axvline(ppf.mean(), color="red", linestyle="--", label=f"mean={ppf.mean():.1f}")
        ax.set_title(f"Points per frame ({name})")
        ax.set_xlabel("points / frame"); ax.set_ylabel("frames"); ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "points_per_frame_hist.png", dpi=120)
        plt.close(fig)

    # 3) 每类目标平均点数 (FOV vs full)
    per = report["per_class_pointcloud"]
    if per:
        classes = list(per.keys())
        fov = [per[c]["pts_in_box_pkl_fov"]["mean"] for c in classes]
        full = [per[c]["pts_in_box_full"]["mean"] for c in classes]
        x = np.arange(len(classes)); w = 0.38
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - w/2, fov, w, label="pkl(FOV)", color="#DD8452")
        ax.bar(x + w/2, full, w, label="full cloud", color="#55A868")
        ax.set_xticks(x); ax.set_xticklabels(classes)
        ax.set_title(f"Mean radar points per object ({name})")
        ax.set_ylabel("mean points / box"); ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "pts_per_box_by_class.png", dpi=120)
        plt.close(fig)

    # 4) BEV 热力图
    bev = report["bev_heatmap"]
    if bev["hist"]:
        H = np.array(bev["hist"])
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(H, origin="lower", extent=[-BEV_RANGE_M, BEV_RANGE_M, -BEV_RANGE_M, BEV_RANGE_M],
                       cmap="hot", aspect="equal")
        ax.set_xlabel("x (m, forward)"); ax.set_ylabel("y (m, left)")
        ax.set_title(f"GT center BEV heatmap ({name}, n={bev['n']})")
        plt.colorbar(im, ax=ax, label="count")
        fig.tight_layout()
        fig.savefig(out_dir / "bev_heatmap.png", dpi=120)
        plt.close(fig)

    # 5) 距离分布 (距离 vs 点数密度)
    db = report["distance_bins"]
    if db:
        labels = [b["range_m"] for b in db]
        vals = [b["density_pts_per_m2"] for b in db]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, vals, color="#4C72B0", alpha=0.85)
        ax.set_title(f"Point density by radial distance ({name})")
        ax.set_xlabel("distance (m)"); ax.set_ylabel("points / m^2")
        fig.tight_layout()
        fig.savefig(out_dir / "distance_density.png", dpi=120)
        plt.close(fig)

    # 6) 7 通道分布 (2x4 子图)
    fields = report["radar_field_stats"]
    if fields:
        fkeys = list(fields.keys())
        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        axes = axes.flatten()
        for i, k in enumerate(fkeys):
            arr = np.concatenate(merged["field_arrays"].get(k, [])) if merged["field_arrays"].get(k) else np.array([])
            ax = axes[i]
            if arr.size:
                arr_clip = arr[(arr >= np.percentile(arr, 1)) & (arr <= np.percentile(arr, 99))]
                ax.hist(arr_clip, bins=50, color="#4C72B0", alpha=0.85)
                ax.set_title(f"{k}  (n={fields[k]['n']})")
            else:
                ax.set_title(f"{k} (empty)")
            ax.tick_params(labelsize=8)
        for j in range(len(fkeys), 8):
            axes[j].axis("off")
        fig.suptitle(f"Radar 7-channel distributions ({name})")
        fig.tight_layout()
        fig.savefig(out_dir / "radar_field_distributions.png", dpi=120)
        plt.close(fig)

    # 7) 每类按距离的点数 (折线)
    if per:
        fig, ax = plt.subplots(figsize=(7, 4))
        for cls, d in per.items():
            xs = [b["range_m"] for b in d["points_by_distance"]]
            ys = [b["mean_pts"] for b in d["points_by_distance"]]
            ax.plot(xs, ys, marker="o", label=cls)
        ax.set_title(f"Mean radar points vs distance ({name})")
        ax.set_xlabel("distance (m)"); ax.set_ylabel("mean points / box")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "points_vs_distance.png", dpi=120)
        plt.close(fig)

    # 8) 14 类原始分布
    rl = report["raw_label_distribution"]["counts"]
    if rl:
        items = sorted(rl.items(), key=lambda x: -x[1])
        labels = [k for k, _ in items]
        vals = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, vals, color="#4C72B0")
        ax.set_title(f"Raw label_2 class distribution ({name}, 14 classes)")
        ax.set_ylabel("count"); plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / "raw_label_distribution.png", dpi=120)
        plt.close(fig)

    print(f"Plots saved to {out_dir}/")


def write_markdown(report, out_dir):
    out_dir = Path(out_dir)
    fp = report["frame_points"]
    L = []
    L.append(f"# VoD radar_5frames 数据集统计 — `{report['name']}`\n")

    L.append("## 1. 平均每帧点云数\n")
    L.append(f"- 帧数: **{fp['n_frames']}**  ·  平均每帧点数: **{fp['avg_points_per_frame']:.1f}**  ·  总点数: {fp['total_points']}")
    L.append(f"- 中位数 / P5 / P95: {fp['median']:.0f} / {fp['p5']:.0f} / {fp['p95']:.0f}")
    L.append(f"- 最小 / 最大: {fp['min']} / {fp['max']}\n")

    L.append("## 2. GT 类别数量 (Car / Pedestrian / Cyclist)\n")
    L.append("| 类别 | 数量 |")
    L.append("| --- | --- |")
    for k, v in report["gt_class_counts"].items():
        L.append(f"| {k} | {v} |")
    L.append(f"| **合计** | **{report['n_gt_boxes']}** |\n")

    L.append("## 3. 每类目标包含的点云数\n")
    L.append("两种口径: `pkl(FOV)` = infos 预计算 (仅相机视场内); `full` = 全点云 point-in-box。\n")
    L.append("| 类别 | box数 | FOV均值 | FOV中位 | FOV空框比 | full均值 | full中位 | full空框比 | 平均距离(m) | 距离P95(m) | 平均尺寸 l×w×h |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cls, d in report["per_class_pointcloud"].items():
        fov, full = d["pts_in_box_pkl_fov"], d["pts_in_box_full"]
        rg, sz = d["range"], d["size_lwh"]
        sz_s = f"{sz['length']['mean']:.2f}×{sz['width']['mean']:.2f}×{sz['height']['mean']:.2f}"
        L.append(
            f"| {cls} | {d['n_boxes']} | {fov['mean']:.2f} | {fov['median']:.0f} | {fov['frac_zero']:.2f} "
            f"| {full['mean']:.2f} | {full['median']:.0f} | {full['frac_zero']:.2f} | {rg['mean']:.1f} | {rg['p95']:.1f} | {sz_s} |"
        )
    L.append("")

    L.append("## 4. 雷达 7 通道逐字段统计\n")
    L.append("| 通道 | 含义 | 均值 | 中位 | P5 | P95 | 最大 | 零值比 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    desc_map = {"range": "到 ego 原点距离(m)", "rcs": "雷达散射截面(dBsm)", "v_r": "原始径向速度(m/s)",
                "v_r_comp": "自车运动补偿后径向速度(m/s)", "vmag": "|v_r| (m/s)", "time": "时间偏移(s)", "z": "高度(m)"}
    for f, m in report["radar_field_stats"].items():
        L.append(f"| {f} | {desc_map.get(f, '')} | {m['mean']:.2f} | {m['median']:.2f} | {m['p5']:.2f} | {m['p95']:.2f} | {m['max']:.2f} | {m['frac_zero']:.3f} |")
    L.append("")

    L.append("## 5. 距离分箱 — 雷达点云密度衰减\n")
    L.append("| 距离(m) | 点数 | 占比 | 密度 (pts/m²) |")
    L.append("| --- | --- | --- | --- |")
    for b in report["distance_bins"]:
        L.append(f"| {b['range_m']} | {b['n_points']} | {b['frac']*100:.1f}% | {b['density_pts_per_m2']:.3f} |")
    L.append("")

    L.append("## 6. 每类目标按距离分箱的平均点数 (full 口径)\n")
    L.append("| 类别 | " + " | ".join([b['range_m'] + 'm' for b in report["distance_bins"]]) + " |")
    L.append("| --- | " + " | ".join(["---"] * len(report["distance_bins"])) + " |")
    for cls, d in report["per_class_pointcloud"].items():
        row = [f"{b['mean_pts']:.2f} (n={b['n_boxes']})" for b in d["points_by_distance"]]
        L.append(f"| {cls} | " + " | ".join(row) + " |")
    L.append("")

    L.append("## 7. BEV 目标中心分布\n")
    L.append(f"- 目标中心数: **{report['bev_heatmap']['n']}**  ·  范围: ±{BEV_RANGE_M} m  ·  网格: {BEV_BINS}×{BEV_BINS}\n")

    L.append("## 8. 原始 label_2 标签分布 (14 类)\n")
    L.append("| 类别 | 数量 |")
    L.append("| --- | --- |")
    for k, v in sorted(report["raw_label_distribution"]["counts"].items(), key=lambda x: -x[1]):
        L.append(f"| {k} | {v} |")
    L.append("")

    L.append("## 9. 截断 / 遮挡 / 难度 (3 类)\n")
    L.append("| 类别 | truncated均值 | occluded:0/1/2/3 | difficulty:easy/mod/hard |")
    L.append("| --- | --- | --- | --- |")
    for cls, d in report["per_class_pointcloud"].items():
        oc = d["occluded_counts"]; df = d["difficulty_counts"]
        L.append(f"| {cls} | {d['truncated']['mean']:.3f} | "
                 f"{oc.get(0,0)}/{oc.get(1,0)}/{oc.get(2,0)}/{oc.get(3,0)} | "
                 f"{df.get(1,0)}/{df.get(2,0)}/{df.get(3,0)} |")
    L.append("")

    # 嵌入图
    for img in ("gt_class_counts.png", "points_per_frame_hist.png", "pts_per_box_by_class.png",
                "bev_heatmap.png", "distance_density.png", "radar_field_distributions.png",
                "points_vs_distance.png", "raw_label_distribution.png"):
        L.append(f"![{img}]({img})\n")

    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Markdown saved to {out_dir}/report.md")

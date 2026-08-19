#!/usr/bin/env python3
"""VoD eval 后处理: 结构化结果 + 损失曲线 + 多帧可视化

用法:
    # 三件套
    python tools/visualize_eval.py --eval_dir <...>/eval/epoch_100/val/val_eval \
        --dataroot /mnt/d/DATASET/VoD/.../radar_5frames \
        --train_log_dir output/cfgs/model/vod_models/vod_radarpillar/<EXTRA_TAG>

    # 只画 loss
    python tools/visualize_eval.py ... --loss_only

    # 只画帧
    python tools/visualize_eval.py ... --frames_only --n_samples 12 --score_thresh 0.2

    # 重新生成 results.json (基于 log_eval_*.txt)
    python tools/visualize_eval.py ... --results_only
"""
import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# 让同目录下的 tools/ 模块可导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.visual_utils.visualize_loss import parse_log, visualize_loss as plot_loss_from_log


# ═══ vod 单帧可视化 (原 utils/visual_utils/visualize_vod_eval.py, 唯一消费方在此内联) ═══
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

CLASS_COLORS_GT = {
    "Car": "#2ecc71", "Pedestrian": "#3498db", "Cyclist": "#e74c3c",
}
CLASS_COLORS_PRED = {
    "Car": "#27ae60", "Pedestrian": "#2980b9", "Cyclist": "#c0392b",
}
BEV_XLIM = (0, 52)
BEV_YLIM = (-26, 26)


def _box_corners_2d(cx, cy, l, w, heading):
    """BEV 4 角点。"""
    corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    return corners @ rot.T + np.array([cx, cy])


def _draw_box_bev(ax, cx, cy, l, w, heading, color, linestyle="-", linewidth=2.5, alpha=1.0):
    """BEV 旋转框：颜色表示类别 + 线型区分 GT/Pred。类别靠图例表达。"""
    corners = _box_corners_2d(cx, cy, l, w, heading)
    poly = Polygon(corners, fill=False, edgecolor=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    ax.add_patch(poly)
    dx = np.cos(heading) * l * 0.5
    dy = np.sin(heading) * l * 0.5
    ax.plot([cx, cx + dx], [cy, cy + dy], color=color, linewidth=1.2, alpha=alpha)


def _lidar_boxes_to_corners_3d(boxes_lidar):
    """lidar (N,7) [x,y,z,dx,dy,dz,h] -> (N,8,3). 内联实现, 避免 pcdet 依赖。"""
    if boxes_lidar.shape[0] == 0:
        return np.zeros((0, 8, 3), dtype=np.float64)
    x, y, z, dx, dy, dz, h = boxes_lidar.T
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
    c, s = np.cos(h), np.sin(h)
    rot = np.empty((x.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = c;  rot[:, 0, 1] = -s; rot[:, 0, 2] = 0
    rot[:, 1, 0] = s;  rot[:, 1, 1] =  c; rot[:, 1, 2] = 0
    rot[:, 2, 0] = 0;  rot[:, 2, 1] =  0; rot[:, 2, 2] = 1
    rel = corners - np.stack([x, y, z], -1)[:, None, :]
    rotated = np.einsum("nij,npj->npi", rot, rel)
    return rotated + np.stack([x, y, z], -1)[:, None, :]


def _project_corners_to_image(corners_3d, calib):
    """(N,8,3) -> (N,8,2) 像素坐标。"""
    if corners_3d.shape[0] == 0:
        return np.zeros((0, 8, 2), dtype=np.float64)
    pts_img, _ = calib.lidar_to_img(corners_3d.reshape(-1, 3))
    return pts_img.reshape(-1, 8, 2)


def _draw_3d_cube_on_image(img, corners_2d, color, linestyle="-", line_thickness=2):
    """在 BGR 图像上画 3D 立方体 12 棱。'--' 虚线手动分段。"""
    if corners_2d.shape[0] == 0:
        return img
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    is_dash = (linestyle == "--")
    for (i, j) in edges:
        p1 = corners_2d[i].astype(int)
        p2 = corners_2d[j].astype(int)
        if is_dash:
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


def compose_one_frame(frame_id, points, image, calib,
                      gt_boxes_lidar, gt_names,
                      pred_boxes_lidar, pred_names, pred_scores,
                      output_path, score_thresh=0.1):
    """合成单帧: BEV(左) + 相机(右) PNG。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img_bgr = image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    fig, (ax_bev, ax_img) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={"width_ratios": [1.4, 1.8]})

    if points is not None and points.shape[0] > 0:
        rcs = points[:, 3]
        ax_bev.scatter(points[:, 0], points[:, 1], c=rcs, cmap="viridis", s=4, alpha=0.6, zorder=1)

    n_gt, n_pred_filt = 0, 0
    if gt_boxes_lidar is not None and gt_boxes_lidar.shape[0] > 0:
        for i, name in enumerate(gt_names):
            color = CLASS_COLORS_GT.get(name, "#95a5a6")
            x, y, _, dx, dy, _, h = gt_boxes_lidar[i]
            _draw_box_bev(ax_bev, x, y, dx, dy, h, color, linestyle="-", linewidth=2.0, alpha=0.9)
            n_gt += 1

    if pred_boxes_lidar is not None and pred_boxes_lidar.shape[0] > 0:
        for i, (name, sc) in enumerate(zip(pred_names, pred_scores)):
            if sc < score_thresh:
                continue
            color = CLASS_COLORS_PRED.get(name, "#7f8c8d")
            x, y, _, dx, dy, _, h = pred_boxes_lidar[i]
            _draw_box_bev(ax_bev, x, y, dx, dy, h, color, linestyle="--", linewidth=1.6, alpha=0.8)
            n_pred_filt += 1

    ax_bev.set_xlim(BEV_XLIM); ax_bev.set_ylim(BEV_YLIM)
    ax_bev.set_aspect("equal")
    ax_bev.set_xlabel("X (forward) [m]"); ax_bev.set_ylabel("Y (left) [m]")
    ax_bev.set_title(
        f"BEV  ·  Color → Class  (Green=Car, Blue=Pedestrian, Red=Cyclist)\n"
        f"GT solid | Pred dashed  ·  GT: {n_gt}  ·  Pred (sc≥{score_thresh:.2f}): {n_pred_filt}",
        fontsize=10
    )
    ax_bev.grid(True, alpha=0.3)
    ax_bev.plot(0, 0, marker="^", color="white", markersize=10, markeredgecolor="black", zorder=5)

    if gt_boxes_lidar is not None and gt_boxes_lidar.shape[0] > 0:
        gt_corners3d = _lidar_boxes_to_corners_3d(gt_boxes_lidar[:, :7])
        gt_corners2d = _project_corners_to_image(gt_corners3d, calib)
        for i, name in enumerate(gt_names):
            color_rgb = CLASS_COLORS_GT.get(name, "#95a5a6")
            color_bgr = (int(color_rgb[5:7], 16), int(color_rgb[3:5], 16), int(color_rgb[1:3], 16))
            _draw_3d_cube_on_image(img_bgr, gt_corners2d[i], color_bgr, linestyle="-", line_thickness=2)

    ax_img.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax_img.set_title(f"Frame {frame_id}  ·  camera image (GT only)")
    ax_img.axis("off")

    legend_elements = [
        Line2D([0], [0], color="#2ecc71", lw=3, ls="-", label="GT  ·  Car  (green)"),
        Line2D([0], [0], color="#3498db", lw=3, ls="-", label="GT  ·  Pedestrian  (blue)"),
        Line2D([0], [0], color="#e74c3c", lw=3, ls="-", label="GT  ·  Cyclist  (red)"),
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


def iter_sample_ids_uniform(infos_pkl, n_samples, seed=42):
    """读 vod_infos_*.pkl, 等距取 n_samples 个 frame_id。"""
    with open(infos_pkl, "rb") as f:
        infos = pickle.load(f)
    ids = [str(info["point_cloud"]["lidar_idx"]) for info in infos]
    if n_samples >= len(ids):
        return ids
    idx = np.linspace(0, len(ids) - 1, n_samples).astype(int)
    idx = sorted({int(i) for i in idx})
    return [ids[i] for i in idx]


def load_frame_assets(dataroot, split, frame_id):
    """读一帧: radar 点云 + 相机图像 + calib + GT boxes (lidar)。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pcdet.utils import calibration_kitti

    root_split = Path(dataroot) / ("training" if split != "test" else "testing")

    pts = np.fromfile(str(root_split / "velodyne" / f"{frame_id}.bin"), dtype=np.float32).reshape(-1, 7)

    img_path = root_split / "image_2" / f"{frame_id}.jpg"
    if not img_path.exists():
        img_path = root_split / "image_2" / f"{frame_id}.png"
    img = cv2.imread(str(img_path))

    calib = calibration_kitti.Calibration(root_split / "calib" / f"{frame_id}.txt")

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
    with open(result_pkl, "rb") as f:
        det_annos = pickle.load(f)
    for d in det_annos:
        if str(d.get("frame_id")) == str(frame_id):
            return (np.asarray(d["boxes_lidar"], dtype=np.float64),
                    list(d["name"]),
                    np.asarray(d["score"], dtype=np.float64))
    return np.zeros((0, 7)), [], np.array([])


# ══════════════════════════════════════════════════════════════
#  results.json / results.csv
# ══════════════════════════════════════════════════════════════
def parse_results_from_text(result_str):
    """解析 test.py logger 打出的 result_str。"""
    # 形如:
    # Car AP@0.50, 0.25, 0.25:
    #   bbox AP:0.7000, 0.6500, 0.6000
    #   bev  AP:...
    #   3d   AP:...
    #   aos  AP:...
    # Car AP_R40@0.50, 0.25, 0.25:
    #   ...
    per_class = {}
    cur_cls, cur_tag = None, None
    for line in result_str.splitlines():
        m = re.match(r"(\w+)\s+AP(_R40)?@([\d\., ]+):", line)
        if m:
            cur_cls, r40 = m.group(1), bool(m.group(2))
            cur_tag = "AP_R40" if r40 else "AP"
            per_class.setdefault(cur_cls, {}).setdefault(cur_tag, {})
            continue
        m2 = re.match(r"\s*(bbox|bev|3d|aos)\s+AP:([\d\., -]+)", line)
        if m2 and cur_cls and cur_tag:
            vals = [float(x) for x in m2.group(2).replace(" ", "").split(",") if x]
            per_class[cur_cls][cur_tag][m2.group(1)] = {
                "easy": vals[0] if len(vals) > 0 else 0,
                "moderate": vals[1] if len(vals) > 1 else 0,
                "hard": vals[2] if len(vals) > 2 else 0,
            }
    return per_class


def write_results(eval_dir, output_dir):
    eval_dir = Path(eval_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 优先用 test.py 落盘的 results.json
    j = eval_dir / "results.json"
    per_class = {}
    ret_dict = {}
    summary = ""
    if j.exists():
        d = json.loads(j.read_text())
        per_class = d.get("per_class", {})
        ret_dict = d.get("ret_dict", {})
        summary = d.get("summary_str", "")
        if not per_class and summary:
            per_class = parse_results_from_text(summary)
    else:
        # fallback: 解析最新 log_eval_*.txt
        logs = sorted(eval_dir.glob("log_eval_*.txt"))
        if not logs:
            print(f"  [warn] no results.json or log_eval_*.txt in {eval_dir}")
            return None
        text = logs[-1].read_text(errors="ignore")
        per_class = parse_results_from_text(text)
        summary = text

    (output_dir / "results.json").write_text(
        json.dumps({"per_class": per_class, "ret_dict": ret_dict, "summary_str": summary},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")

    # CSV
    lines = ["class,task,difficulty,AP,AP_R40"]
    for cls, d in per_class.items():
        for task in ("bbox", "bev", "3d", "aos"):
            for diff in ("easy", "moderate", "hard"):
                ap = d.get("AP", {}).get(task, {}).get(diff)
                ap_r40 = d.get("AP_R40", {}).get(task, {}).get(diff)
                if ap is None and ap_r40 is None:
                    continue
                lines.append(f"{cls},{task},{diff},{ap if ap is not None else ''},{ap_r40 if ap_r40 is not None else ''}")
    (output_dir / "results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  results -> {output_dir / 'results.json'}")
    print(f"  results -> {output_dir / 'results.csv'}")
    return per_class


# ══════════════════════════════════════════════════════════════
#  loss 曲线
# ══════════════════════════════════════════════════════════════
def plot_tb_loss_curves(train_log_dir, output_dir):
    """从 TB events 画 rpn_loss / cls / loc / dir / total / lr 多子图。"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  [skip] tensorboard not installed, skip TB loss curves")
        return

    tb_dir = Path(train_log_dir) / "tensorboard"
    if not tb_dir.exists():
        print(f"  [skip] no tensorboard dir: {tb_dir}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    wanted = ["train/rpn_loss", "train/rpn_loss_cls", "train/rpn_loss_loc",
              "train/rpn_loss_dir", "train/loss", "meta_data/learning_rate"]
    available = [t for t in wanted if t in tags]
    if not available:
        print(f"  [warn] no target tags in TB. available: {tags}")
        return

    n = len(available)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, tag in enumerate(available):
        events = ea.Scalars(tag)
        xs = [e.step for e in events]
        ys = [e.value for e in events]
        axes[i].plot(xs, ys, lw=0.8, color="#1f77b4")
        axes[i].set_title(tag)
        axes[i].set_xlabel("step"); axes[i].grid(True, alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"TensorBoard scalars ({tb_dir.parent.name})")
    plt.tight_layout()
    out = output_dir / "tb_loss_curves.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  TB loss -> {out}")


def plot_log_loss_curve(train_log_dir, output_dir):
    """复用 visualize_loss.parse_log + visualize_loss 画 train.log 总 loss。"""
    log_dir = Path(train_log_dir) / "logs"
    if not log_dir.exists():
        print(f"  [skip] no logs dir: {log_dir}")
        return
    logs = sorted(log_dir.glob("train_*.log"))
    if not logs:
        print(f"  [skip] no train_*.log in {log_dir}")
        return
    log = logs[-1]
    steps, epoch_sorted = parse_log(log)
    if not steps and not epoch_sorted:
        print(f"  [skip] no loss lines parsed from {log}")
        return
    out = output_dir / "loss_curve.png"
    plot_loss_from_log(steps, epoch_sorted, out, title_suffix=f": {log.stem}")
    print(f"  log loss -> {out}")


# ══════════════════════════════════════════════════════════════
#  多帧可视化
# ══════════════════════════════════════════════════════════════
def visualize_n_frames(eval_dir, dataroot, output_dir, n_samples, score_thresh, seed=42):
    eval_dir = Path(eval_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    infos_pkl = Path(dataroot) / "vod_infos_val.pkl"
    if not infos_pkl.exists():
        print(f"  [error] missing {infos_pkl}")
        return

    frame_ids = iter_sample_ids_uniform(infos_pkl, n_samples, seed=seed)
    result_pkl = eval_dir / "result.pkl"

    n_ok = 0
    for fid in frame_ids:
        try:
            pts, img, calib, gt_lidar, gt_names = load_frame_assets(dataroot, "val", fid)
            pred_lidar, pred_names, pred_scores = lookup_predictions_for_frame(result_pkl, fid) \
                if result_pkl.exists() else (np.zeros((0, 7)), [], np.array([]))
            out = output_dir / f"frame_{fid}.png"
            compose_one_frame(fid, pts, img, calib, gt_lidar, gt_names,
                              pred_lidar, pred_names, pred_scores,
                              out, score_thresh=score_thresh)
            print(f"    frame {fid} -> {out}")
            n_ok += 1
        except Exception as e:
            print(f"    [skip] frame {fid}: {e}")
    print(f"  done: {n_ok}/{len(frame_ids)} frames")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_dir", required=True, help="output/.../eval/epoch_<N>/val/<eval_tag>/")
    p.add_argument("--dataroot", required=True, help="VoD radar_5frames 根目录")
    p.add_argument("--train_log_dir", default=None, help="含 logs/ 和 tensorboard/ 的目录 (默认 = eval_dir 的 ../..)")
    p.add_argument("--output_dir", default=None, help="可视化输出 (默认 = eval_dir/vis/)")
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--score_thresh", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="跑 results + loss + frames (默认)")
    g.add_argument("--results_only", action="store_true")
    g.add_argument("--loss_only", action="store_true")
    g.add_argument("--frames_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir) if args.output_dir else eval_dir / "vis"
    train_log_dir = Path(args.train_log_dir) if args.train_log_dir else eval_dir.parent.parent.parent

    only = sum(bool(x) for x in (args.results_only, args.loss_only, args.frames_only))
    if only == 0:
        args.all = True
    run_results = args.all or args.results_only
    run_loss = args.all or args.loss_only
    run_frames = args.all or args.frames_only

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=" * 60)
    print(f"VoD eval postprocess")
    print(f"  eval_dir:      {eval_dir}")
    print(f"  train_log_dir: {train_log_dir}")
    print(f"  output_dir:    {output_dir}")
    print(f"=" * 60)

    if run_results:
        print("\n[1/3] results.json / results.csv")
        write_results(eval_dir, output_dir)

    if run_loss:
        print("\n[2/3] loss curves")
        plot_log_loss_curve(train_log_dir, output_dir)
        plot_tb_loss_curves(train_log_dir, output_dir)

    if run_frames:
        print(f"\n[3/3] frame visualization (n={args.n_samples}, score_thresh={args.score_thresh})")
        visualize_n_frames(eval_dir, args.dataroot, output_dir,
                           args.n_samples, args.score_thresh, seed=args.seed)

    print("\nDONE.")


if __name__ == "__main__":
    main()

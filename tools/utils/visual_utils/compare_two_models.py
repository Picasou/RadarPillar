"""两组模型对比可视化 (支持 score 多阈值 sweep + train/val split +
                   可配置点云颜色 + 旋转 90° 的 BEV 坐标系 + 图内 legend block)。

布局:
    每个 frame 生成多张子图 (按 score 阈值划分):
        每个阈值 = 1 张 PNG, 3 列: [BEV model1 | BEV model2 | camera image (GT only)]
    文件命名: <split>_frame_<fid>_sc<NNN>.png   (split = train / val, 小写)

坐标系 (BEV):
    真实 lidar 数据 (x_fwd, y_left): 旋转 90° CCW 后画在屏幕坐标.
        屏幕坐标 (X', Y') = (-y_left, x_fwd).
    即: 屏幕"上方"对应车辆 +X 前进; "左方"对应车辆 +Y 左侧.
    Vehicle +X 范围 (0, 52)m, Vehicle +Y 范围 (-26, 26)m 映射到屏幕:
        屏幕 X' 轴 (水平): [-26, +26]  (左=车左, 右=车右)
        屏幕 Y' 轴 (垂直): [+52, 0]   (顶=车前, 底=车后, 上=正)

点云颜色 (--point_color_mode):
    uniform: 单一颜色 (默认 #AABF8E)
    rcs:     按 RCS (列 3) viridis
    doppler: 按 doppler 列 (--doppler_field, 默认 v_r) coolwarm

legend block:
    每个 BEV 子图内部的固定位置(右上角)画一个 legend block,
    列出 GT 类别色 + Pred model1 / model2 虚线样式. 不依赖外部共享 legend.

用法:
    python tools/utils/visual_utils/compare_two_models.py \
        --ckpt1_result <...>/model1/eval/.../result.pkl \
        --ckpt2_result <...>/model2/eval/.../result.pkl \
        --name1 "best_map52.56" --name2 "ckpt_epoch_100" \
        --dataroot data/VoD/view_of_delft_PUBLIC/radar_5frames \
        --output_dir output/radarpiller_compare \
        --n_samples 20 \
        --score_thresholds 0.1 0.3 \
        --split val \
        --point_color_mode doppler --doppler_field v_r
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

# tools/visualize_eval.py 同款: 把 tools/ 加进 sys.path, 让 `utils.visual_utils.xxx` 可解析
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.visual_utils.visualize_vod_eval import (
    CLASS_COLORS_GT, _box_corners_2d, _lidar_boxes_to_corners_3d,
    _project_corners_to_image, _draw_3d_cube_on_image,
    iter_sample_ids_uniform, load_frame_assets, lookup_predictions_for_frame,
    BEV_XLIM, BEV_YLIM,
)

# Pred 颜色 (与 GT 区分)
PRED1_COLOR = "#e74c3c"  # 红 - model1
PRED2_COLOR = "#8e44ad"  # 紫 - model2

# 点云统一颜色 (uniform 模式)
POINT_UNIFORM_COLOR = "#AABF8E"

# radar point .bin 列定义 (from pcdet vod_dataset): [x, y, z, rcs, v_r, v_r_comp, time]
# 列 4 = v_r (原始 doppler), 列 5 = v_r_comp (ego-motion compensated)
DOPPLER_FIELD_INDEX = {"v_r": 4, "v_r_comp": 5}


# ══════════════════════════════════════════════════════════════
#  坐标变换: lidar (x_fwd, y_left) -> 屏幕 BEV (X'=-y, Y'=x)
# ══════════════════════════════════════════════════════════════
# lidar 数据约定:
#   x: 前进 (forward), [0, 52]m
#   y: 左 (left),     [-26, 26]m
# 屏幕绘制约定 (用户要求):
#   上: X 正 (车前)        -> 屏幕 Y' 上
#   左: Y 正 (车左)        -> 屏幕 X' 左 (即 X' 小值靠屏幕左)
# 通过 90° CCW 旋转: (X'_screen, Y'_screen) = (-y, x)
def _lidar_to_screen(xy):
    """(N, 2) lidar frame -> (N, 2) screen frame after 90° CCW rotation."""
    if xy is None or xy.size == 0:
        return xy
    out = np.empty_like(xy, dtype=np.float64)
    out[:, 0] = -xy[:, 1]
    out[:, 1] = xy[:, 0]
    return out


def _lidar_box_corners_screen(cx, cy, dx, dy, heading):
    """lidar 框 4 角点 -> 屏幕坐标 (旋转 90° CCW)."""
    corners_lidar = _box_corners_2d(cx, cy, dx, dy, heading)  # (4, 2)
    return _lidar_to_screen(corners_lidar)


# ══════════════════════════════════════════════════════════════
#  点云着色 (已经在屏幕坐标里画)
# ══════════════════════════════════════════════════════════════
def _scatter_points(ax, points_lidar, mode, doppler_field):
    """在屏幕坐标画点云. points_lidar 为 (N, 7) lidar 帧数据.

    mode:
        uniform: 单一颜色
        rcs:     按 RCS (列 3) viridis
        doppler: 按 --doppler_field 列 coolwarm
    """
    if points_lidar is None or points_lidar.shape[0] == 0:
        return None
    xy_scr = _lidar_to_screen(points_lidar[:, :2])

    if mode == "uniform":
        sc = ax.scatter(xy_scr[:, 0], xy_scr[:, 1],
                        c=POINT_UNIFORM_COLOR, s=4, alpha=0.6, zorder=1)
        return None

    if mode == "rcs":
        rcs = points_lidar[:, 3]
        sc = ax.scatter(xy_scr[:, 0], xy_scr[:, 1],
                        c=rcs, cmap="viridis", s=4, alpha=0.6, zorder=1)
        return sc

    if mode == "doppler":
        col_idx = DOPPLER_FIELD_INDEX[doppler_field]
        dop = points_lidar[:, col_idx]
        # 颜色以 0 为中心对称 (蓝-近/红-远); vmin/vmax 自动用 95% 截尾
        v = float(np.percentile(np.abs(dop), 95)) or 1.0
        sc = ax.scatter(xy_scr[:, 0], xy_scr[:, 1],
                        c=dop, cmap="coolwarm", vmin=-v, vmax=v,
                        s=4, alpha=0.6, zorder=1)
        return sc

    raise ValueError(f"unknown point_color_mode: {mode}")


# ══════════════════════════════════════════════════════════════
#  BEV 框 (在屏幕坐标画, 已应用旋转)
# ══════════════════════════════════════════════════════════════
def _draw_box_bev(ax, cx, cy, dx, dy, h, color, linestyle, lw, alpha):
    """画 BEV 旋转框 + 朝向 (已旋转到屏幕坐标)."""
    corners = _lidar_box_corners_screen(cx, cy, dx, dy, h)  # (4, 2)
    poly = Polygon(corners, fill=False, edgecolor=color, linestyle=linestyle,
                   linewidth=lw, alpha=alpha)
    ax.add_patch(poly)
    # 朝向箭头: 从中心指 +X (lidar 前向). 在屏幕 = 中心指 Y' 上.
    # 屏幕 Y 翻转后, +Y' 上 = +y_screen = lidar +X.
    cx_scr, cy_scr = _lidar_to_screen(np.array([[cx, cy]]))[0]
    half_len = dx * 0.5
    ax.plot([cx_scr, cx_scr], [cy_scr, cy_scr + half_len],
            color=color, linewidth=1.0, alpha=alpha)


# ══════════════════════════════════════════════════════════════
#  图内 legend block (在 BEV 子图固定位置画一个类别+样式说明)
# ══════════════════════════════════════════════════════════════
def _draw_legend_block(ax, name1, name2, split_tag, x_label="", y_label="",
                       gt_only=False):
    """在 ax 内的左上角画一个 legend block, 显示 GT 类别色 + Pred 模型虚线.

    gt_only=True 时不显示 Pred 行 (例如 model 没有预测的情况).
    """
    handles = [
        Line2D([0], [0], color="#2ecc71", lw=3, ls="-", label="GT · Car"),
        Line2D([0], [0], color="#3498db", lw=3, ls="-", label="GT · Pedestrian"),
        Line2D([0], [0], color="#e74c3c", lw=3, ls="-", label="GT · Cyclist"),
    ]
    if not gt_only:
        handles += [
            Line2D([0], [0], color=PRED1_COLOR, lw=3, ls="--", label=f"Pred · {name1}"),
            Line2D([0], [0], color=PRED2_COLOR, lw=3, ls="--", label=f"Pred · {name2}"),
        ]
    leg = ax.legend(handles=handles, loc="upper left",
                    fontsize=9, framealpha=0.95,
                    edgecolor="gray", facecolor="white")
    leg.set_zorder(20)


# ══════════════════════════════════════════════════════════════
#  单栏 BEV 绘制
# ══════════════════════════════════════════════════════════════
def _draw_one_bev(ax, points_lidar, gt_boxes_lidar, gt_names,
                  pred_boxes_lidar, pred_names, pred_scores,
                  pred_color, score_thresh, title,
                  point_color_mode, doppler_field,
                  name1, name2, split_tag):
    """在一张 ax 上画: 点云 + GT(实线按类别) + 单一模型的 Pred(虚线 彩色, score ≥ 阈值).

    ax 坐标系: 已经由 _setup_bev_axes() 设为 "屏幕 X' = -y_lidar, 屏幕 Y' = x_lidar",
    Y 轴已 invert, 0 在底、52 在顶.

    在绘制层面, _scatter_points 和 _draw_box_bev 都已自动旋转数据到屏幕坐标.
    """
    _scatter_points(ax, points_lidar, point_color_mode, doppler_field)

    n_gt, n_pred = 0, 0
    if gt_boxes_lidar is not None and gt_boxes_lidar.shape[0] > 0:
        for i, name in enumerate(gt_names):
            color = CLASS_COLORS_GT.get(name, "#95a5a6")
            x, y, _, dx, dy, _, h = gt_boxes_lidar[i]
            _draw_box_bev(ax, x, y, dx, dy, h, color, "-", 2.0, 0.9)
            n_gt += 1

    if pred_boxes_lidar is not None and pred_boxes_lidar.shape[0] > 0:
        for i, (name, sc) in enumerate(zip(pred_names, pred_scores)):
            if sc < score_thresh:
                continue
            x, y, _, dx, dy, _, h = pred_boxes_lidar[i]
            _draw_box_bev(ax, x, y, dx, dy, h, pred_color, "--", 1.6, 0.95)
            n_pred += 1

    ax.set_title(f"{title}\nGT: {n_gt} · Pred: {n_pred} (sc≥{score_thresh:.2f})",
                 fontsize=10)

    # 自车位置: 一个朝上的小三角形 (X+ = 屏幕 Y 上)
    ax.plot(0, 0, marker="^", color="white", markersize=10,
            markeredgecolor="black", zorder=5)

    # 图内 legend block
    _draw_legend_block(ax, name1, name2, split_tag)


# ══════════════════════════════════════════════════════════════
#  BEV 坐标系设置 (屏幕 BEV: X' 横 = -y_lidar, Y' 纵 = x_lidar)
# ══════════════════════════════════════════════════════════════
def _setup_bev_axes(ax, xlim_y=(-26, 26), xlim_x=(-26, 52), tick_step=10):
    """配置 BEV 子图坐标系.

    lidar 数据 (cx, cy) 在经过 _lidar_to_screen 后:
        屏幕 X' 范围 = -y_lidar 范围  -> [-26, +26]
        屏幕 Y' 范围 =  x_lidar 范围  -> 默认 [-26, 52] (车后 26m + 车前 52m,
                                              以便 BACK 视觉上有区域)

    视觉方向:
        screen TOP  -> 车前 (lidar +X)
        screen BOTTOM -> 车后 (lidar -X)
        screen LEFT  -> 车左 (lidar +Y)
        screen RIGHT -> 车右 (lidar -Y)
    """
    ax.set_aspect("equal")
    # 屏幕 X' (= lidar -y):  -Y_lidar 在左 (+Y_lidar 车左), +Y_lidar 在右 (-Y_lidar 车右)
    ax.set_xlim(xlim_y[0], xlim_y[1])
    # 屏幕 Y' (= lidar x):   大值在顶 (车前), 小值在底 (车后)
    ax.set_ylim(xlim_x[0], xlim_x[1])

    xt = np.arange(xlim_y[0], xlim_y[1] + 1, tick_step)
    yt = np.arange(xlim_x[0], xlim_x[1] + 1, tick_step)
    ax.set_xticks(xt)
    ax.set_yticks(yt)

    # 屏幕轴标签: 用户语义
    ax.set_xlabel("X_screen = -Y_lidar  (left = +Y_lidar, vehicle LEFT)  ->")
    ax.set_ylabel("Y_screen =  X_lidar  (top = +X_lidar, vehicle FRONT)")

    ax.grid(True, alpha=0.3, which="major")


def _add_orientation_labels(ax):
    """在 ax 内部 4 个角附近加 4 个方位标签 (FRONT/BACK/LEFT/RIGHT).

    注意:
      - FRONT (车前, lidar +X) 在屏幕 Y' 顶 (大值方向)
      - BACK  (车后, lidar -X) 在屏幕 Y' 底
      - LEFT  (车左, lidar +Y) 在屏幕 X' 左 (因为 X' = -Y)
      - RIGHT (车右, lidar -Y) 在屏幕 X' 右
    """
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()  # (已经是 inverted: y_lo = 屏幕底, y_hi = 屏幕顶)
    pad_x = (x_hi - x_lo) * 0.02
    pad_y = (y_hi - y_lo) * 0.02

    corner_pad = dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="gray", alpha=0.7)

    # FRONT: 屏幕顶部正中 (车前)
    ax.text(0, y_hi - pad_y, "↑ FRONT  (+X_lidar, vehicle forward)",
            color="black", fontsize=8, ha="center", va="top",
            bbox=corner_pad, zorder=15)
    # BACK: 屏幕底部正中
    ax.text(0, y_lo + pad_y, "↓ BACK  (-X_lidar, vehicle rear)",
            color="black", fontsize=8, ha="center", va="bottom",
            bbox=corner_pad, zorder=15)
    # LEFT: 屏幕左中 (车左)
    ax.text(x_lo + pad_x, 0, "LEFT  (+Y_lidar, vehicle left) ←",
            color="black", fontsize=8, ha="left", va="center",
            bbox=corner_pad, zorder=15)
    # RIGHT: 屏幕右中
    ax.text(x_hi - pad_x, 0, "RIGHT  (-Y_lidar, vehicle right) →",
            color="black", fontsize=8, ha="right", va="center",
            bbox=corner_pad, zorder=15)


# ══════════════════════════════════════════════════════════════
#  合成三栏图
# ══════════════════════════════════════════════════════════════
def compose_compare(frame_id, points, image, calib,
                    gt_boxes_lidar, gt_names,
                    pred1_boxes, pred1_names, pred1_scores,
                    pred2_boxes, pred2_names, pred2_scores,
                    output_path, name1, name2, score_thresh, split_tag="",
                    point_color_mode="uniform", doppler_field="v_r"):
    """三栏布局: 左 BEV(model1 Pred) | 中 BEV(model2 Pred) | 右 camera image(GT only).

    共享 GT: 左+中两个 BEV 都画同一组 GT 实线框 + 同一组 radar 点云.
    Pred 颜色不同: model1=红虚线, model2=紫虚线,均按 score_thresh 过滤.
    每个 BEV 子图内自带 legend block (类别色 + 模型样式), 不依赖共享 legend.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img_bgr = image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    fig, (ax1, ax2, ax_img) = plt.subplots(
        1, 3, figsize=(30, 10),
        gridspec_kw={"width_ratios": [1.4, 1.4, 1.8]}
    )

    _setup_bev_axes(ax1)
    _setup_bev_axes(ax2)
    _add_orientation_labels(ax1)
    _add_orientation_labels(ax2)

    _draw_one_bev(ax1, points, gt_boxes_lidar, gt_names,
                  pred1_boxes, pred1_names, pred1_scores,
                  PRED1_COLOR, score_thresh,
                  f"BEV · {name1} {split_tag}".strip(),
                  point_color_mode, doppler_field,
                  name1, name2, split_tag)
    _draw_one_bev(ax2, points, gt_boxes_lidar, gt_names,
                  pred2_boxes, pred2_names, pred2_scores,
                  PRED2_COLOR, score_thresh,
                  f"BEV · {name2} {split_tag}".strip(),
                  point_color_mode, doppler_field,
                  name1, name2, split_tag)

    # 如点云颜色随标量 (rcs/doppler), 给两栏分别加 colorbar
    sc1 = _last_scatter_artist(ax1)
    sc2 = _last_scatter_artist(ax2)
    for sc, ax in [(sc1, ax1), (sc2, ax2)]:
        if sc is not None:
            cbar_label = "RCS (dB)" if point_color_mode == "rcs" else f"{doppler_field} (m/s)"
            cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label(cbar_label, fontsize=8)
            cb.ax.tick_params(labelsize=7)

    # GT 3D cube 在右侧图像上 (共享参考)
    if gt_boxes_lidar is not None and gt_boxes_lidar.shape[0] > 0:
        gt_corners3d = _lidar_boxes_to_corners_3d(gt_boxes_lidar[:, :7])
        gt_corners2d = _project_corners_to_image(gt_corners3d, calib)
        for i, name in enumerate(gt_names):
            color_rgb = CLASS_COLORS_GT.get(name, "#95a5a6")
            color_bgr = (int(color_rgb[5:7], 16),
                         int(color_rgb[3:5], 16),
                         int(color_rgb[1:3], 16))
            _draw_3d_cube_on_image(img_bgr, gt_corners2d[i], color_bgr,
                                   linestyle="-", line_thickness=2)

    ax_img.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax_img.set_title(f"Frame {frame_id} {split_tag} · camera image (GT only) · sc≥{score_thresh:.2f}".strip())
    ax_img.axis("off")

    # 注意: legend block 已内嵌到每个 BEV 子图内 (左上角), 不再需要外部 fig.legend.

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _last_scatter_artist(ax):
    """从 ax.collections 取最近一个 scatter 句柄 (用于 colorbar)."""
    if ax.collections:
        return ax.collections[-1]
    return None


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt1_result", required=True)
    p.add_argument("--ckpt2_result", required=True)
    p.add_argument("--name1", default="model1")
    p.add_argument("--name2", default="model2")
    p.add_argument("--dataroot", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--score_thresholds", type=float, nargs="+",
                   default=[0.1, 0.3, 0.5, 0.7],
                   help="每个 frame 对每个阈值各出 1 张 PNG, e.g. 0.1 0.3 0.5 0.7")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="val", choices=["train", "val"],
                   help="数据 split, 用于加载 GT/点云与选择 infos pkl")

    # 新加的 3 个配置项
    p.add_argument("--point_color_mode", default="uniform",
                   choices=["uniform", "rcs", "doppler"],
                   help="点云着色模式")
    p.add_argument("--doppler_field", default="v_r",
                   choices=["v_r", "v_r_comp"],
                   help="doppler 模式使用的列名 (v_r=raw doppler / v_r_comp=ego-motion compensated)")

    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    infos_pkl = Path(args.dataroot) / f"vod_infos_{args.split}.pkl"
    if not infos_pkl.exists():
        print(f"[error] missing {infos_pkl}")
        return

    frame_ids = iter_sample_ids_uniform(infos_pkl, args.n_samples, seed=args.seed)
    print(f"sampled {len(frame_ids)} frame_ids from {args.split}: {frame_ids}")
    print(f"score_thresholds: {args.score_thresholds}")
    print(f"point_color_mode: {args.point_color_mode}"
          + (f" (field={args.doppler_field})" if args.point_color_mode == "doppler" else ""))

    # 标题中 [TRAIN] / [VAL] 带括号, 文件名中 train/val 不带括号
    split_tag = f"[{args.split.upper()}]"
    split_name = args.split.lower()

    # 预加载两份 result.pkl, 避免每帧重读
    import pickle
    with open(args.ckpt1_result, "rb") as f:
        dets1 = pickle.load(f)
    with open(args.ckpt2_result, "rb") as f:
        dets2 = pickle.load(f)
    by_fid1 = {str(d["frame_id"]): d for d in dets1}
    by_fid2 = {str(d["frame_id"]): d for d in dets2}

    total = 0
    for fid in frame_ids:
        try:
            pts, img, calib, gt_lidar, gt_names = load_frame_assets(
                args.dataroot, args.split, fid)

            d1 = by_fid1.get(str(fid))
            d2 = by_fid2.get(str(fid))
            p1b = np.asarray(d1["boxes_lidar"], dtype=np.float64) if d1 else np.zeros((0, 7))
            p1n = list(d1["name"]) if d1 else []
            p1s = np.asarray(d1["score"], dtype=np.float64) if d1 else np.array([])
            p2b = np.asarray(d2["boxes_lidar"], dtype=np.float64) if d2 else np.zeros((0, 7))
            p2n = list(d2["name"]) if d2 else []
            p2s = np.asarray(d2["score"], dtype=np.float64) if d2 else np.array([])

            for sc in args.score_thresholds:
                sc_tag = f"sc{int(round(sc * 100)):03d}"
                out = output_dir / f"{split_name}_frame_{fid}_{sc_tag}.png"
                compose_compare(fid, pts, img, calib,
                                gt_lidar, gt_names,
                                p1b, p1n, p1s,
                                p2b, p2n, p2s,
                                out, args.name1, args.name2, sc,
                                split_tag=split_tag,
                                point_color_mode=args.point_color_mode,
                                doppler_field=args.doppler_field)
                print(f"  frame {fid} {split_tag} sc≥{sc:.2f} -> {out}")
                total += 1
        except Exception as e:
            print(f"  [skip] frame {fid}: {e}")

    print(f"\nDONE: {total} PNGs (={len(frame_ids)} frames × {len(args.score_thresholds)} thresholds, split={args.split}) -> {output_dir}")


if __name__ == "__main__":
    main()

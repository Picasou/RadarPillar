"""
Anchor analysis toolkit for RadarPillar.

Subcommands:
  verify   — Quantify anchor-GT IoU overlap (axis-aligned BEV)
  scatter  — Plot GT sizes vs configured anchors scatter chart
  dist     — Plot per-class GT size distribution histogram

Usage:
  python -m tools.utils.visual_utils.anchor_analysis verify
  python -m tools.utils.visual_utils.anchor_analysis scatter
  python -m tools.utils.visual_utils.anchor_analysis dist --class-name Cyclist
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# ── Shared constants ──────────────────────────────────────────────────
CYCLIST_NAMES = {
    'bicycle', 'rider', 'Cyclist', 'moped_scooter',
    'motor', 'ride_other', 'ride_uncertain',
}

# (class_name → original_name) mapping for grouping
CLASS_NAME_MAPPING = {
    'bicycle': 'Cyclist', 'rider': 'Cyclist', 'motor': 'Cyclist',
    'moped_scooter': 'Cyclist', 'ride_other': 'Cyclist',
    'ride_uncertain': 'Cyclist',
}

DEFAULT_INFO_PATH = Path(
    'data/VoD/view_of_delft_PUBLIC/radar_5frames/vod_infos_train.pkl'
)


# ── Data loading helpers ──────────────────────────────────────────────
def _load_infos(info_path: Path):
    if not info_path.exists():
        print(f'Error: {info_path} not found.')
        sys.exit(1)
    with open(info_path, 'rb') as f:
        return pickle.load(f)


def _extract_gt_boxes(infos, class_names=None):
    """Extract GT boxes from info dicts.

    Returns list of (class_name, dx, dy, dz, heading) tuples.
    If *class_names* is None, return all classes.
    """
    results = []
    for info in infos:
        annos = info.get('annos', None)
        if annos is None:
            continue
        names = annos.get('name', [])
        gt_boxes = annos.get('gt_boxes_lidar', None)
        if gt_boxes is None:
            continue
        for i, name in enumerate(names):
            if i >= len(gt_boxes):
                continue
            mapped = CLASS_NAME_MAPPING.get(name, name)
            if class_names and mapped not in class_names:
                continue
            box = gt_boxes[i]  # [x, y, z, dx, dy, dz, heading]
            results.append((mapped, box[3], box[4], box[5], box[6]))
    return results


# ══════════════════════════════════════════════════════════════════════
# verify — IoU-based anchor verification
# ══════════════════════════════════════════════════════════════════════
CURRENT_ANCHORS = [[0.85, 0.60, 1.20], [1.95, 0.80, 1.60]]
PROPOSED_ANCHORS = [[0.82, 0.76, 1.54], [1.89, 0.68, 1.38]]
ANCHOR_ROTATIONS = [0, np.pi / 2]
SMALL_LARGE_THRESH = 1.2  # meters


def _axis_aligned_bev_iou(gt_dx, gt_dy, gt_heading, anchor_l, anchor_w):
    h = np.abs(gt_heading) % np.pi
    if h <= np.pi / 4 or h >= 3 * np.pi / 4:
        aligned_l, aligned_w = gt_dx, gt_dy
    else:
        aligned_l, aligned_w = gt_dy, gt_dx
    intersection = min(aligned_l, anchor_l) * min(aligned_w, anchor_w)
    union = aligned_l * aligned_w + anchor_l * anchor_w - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _max_iou_for_gt(gt_dx, gt_dy, gt_heading, anchors):
    max_iou = 0.0
    for anchor in anchors:
        al, aw, _ah = anchor
        for rot in ANCHOR_ROTATIONS:
            a_l, a_w = (al, aw) if rot == 0 else (aw, al)
            iou = _axis_aligned_bev_iou(gt_dx, gt_dy, gt_heading, a_l, a_w)
            if iou > max_iou:
                max_iou = iou
    return max_iou


def _print_iou_stats(ious, label):
    ious = np.asarray(ious)
    n = len(ious)
    if n == 0:
        print(f"  [{label}] No samples.")
        return
    pct_ge_05 = np.mean(ious >= 0.5) * 100
    pct_ge_035 = np.mean(ious >= 0.35) * 100
    print(f"  [{label}]  N={n}")
    print(f"    IoU >= 0.50 : {pct_ge_05:6.2f}%")
    print(f"    IoU >= 0.35 : {pct_ge_035:6.2f}%")
    print(f"    Mean IoU    : {np.mean(ious):.4f}")
    print(f"    Median IoU  : {np.median(ious):.4f}")
    print(f"    Percentiles : p10={np.percentile(ious,10):.4f}  "
          f"p25={np.percentile(ious,25):.4f}  p50={np.percentile(ious,50):.4f}  "
          f"p75={np.percentile(ious,75):.4f}  p90={np.percentile(ious,90):.4f}")


def cmd_verify(args):
    """Verify anchor quality by computing axis-aligned BEV IoU against GT."""
    infos = _load_infos(args.info_path)
    boxes = _extract_gt_boxes(infos, class_names=['Cyclist'])

    print("=" * 70)
    print("Anchor Verification for Cyclist Class — VoD Radar Dataset")
    print("=" * 70)
    print(f"\nTotal cyclist GT boxes: {len(boxes)}")

    if not boxes:
        print("No cyclist boxes found. Exiting.")
        return

    dxs = np.array([b[1] for b in boxes])
    dys = np.array([b[2] for b in boxes])
    dzs = np.array([b[3] for b in boxes])
    headings = np.array([b[4] for b in boxes])

    print(f"\nGT Box Size Statistics (dx, dy, dz):")
    for label, arr in [('dx (length)', dxs), ('dy (width)', dys), ('dz (height)', dzs)]:
        print(f"  {label}: mean={np.mean(arr):.3f}  std={np.std(arr):.3f}  "
              f"min={np.min(arr):.3f}  max={np.max(arr):.3f}  median={np.median(arr):.3f}")

    small_mask = dxs < SMALL_LARGE_THRESH
    large_mask = dxs >= SMALL_LARGE_THRESH
    print(f"\nSmall cyclists (dx < {SMALL_LARGE_THRESH}m): {small_mask.sum()}")
    print(f"Large cyclists (dx >= {SMALL_LARGE_THRESH}m): {large_mask.sum()}")

    current_ious = np.array([_max_iou_for_gt(b[1], b[2], b[4], CURRENT_ANCHORS) for b in boxes])
    proposed_ious = np.array([_max_iou_for_gt(b[1], b[2], b[4], PROPOSED_ANCHORS) for b in boxes])

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"\n--- Current Anchors: {CURRENT_ANCHORS} ---")
    _print_iou_stats(current_ious, "Current - All")
    print(f"\n--- Proposed Anchors: {PROPOSED_ANCHORS} ---")
    _print_iou_stats(proposed_ious, "Proposed - All")

    for mask, tag in [(small_mask, f"SMALL (dx < {SMALL_LARGE_THRESH}m)"),
                      (large_mask, f"LARGE (dx >= {SMALL_LARGE_THRESH}m)")]:
        print("\n" + "=" * 70)
        print(f"{tag}")
        print("=" * 70)
        print("\n--- Current Anchors ---")
        _print_iou_stats(current_ious[mask], "Current")
        print("\n--- Proposed Anchors ---")
        _print_iou_stats(proposed_ious[mask], "Proposed")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    delta_mean = np.mean(proposed_ious) - np.mean(current_ious)
    delta_median = np.median(proposed_ious) - np.median(current_ious)
    delta_pct50 = (np.mean(proposed_ious >= 0.5) - np.mean(current_ious >= 0.5)) * 100
    delta_pct35 = (np.mean(proposed_ious >= 0.35) - np.mean(current_ious >= 0.35)) * 100
    print(f"  Mean IoU change    : {delta_mean:+.4f}  ({'better' if delta_mean > 0 else 'worse'})")
    print(f"  Median IoU change  : {delta_median:+.4f}  ({'better' if delta_median > 0 else 'worse'})")
    print(f"  IoU>=0.50 change   : {delta_pct50:+.2f}%")
    print(f"  IoU>=0.35 change   : {delta_pct35:+.2f}%")
    improved = np.sum(proposed_ious > current_ious)
    degraded = np.sum(proposed_ious < current_ious)
    same = np.sum(proposed_ious == current_ious)
    print(f"\n  Per-box: {improved} improved, {degraded} degraded, {same} unchanged")

    # Worst cases
    for label, ious in [("Proposed Anchors", proposed_ious), ("Current Anchors", current_ious)]:
        print(f"\n--- Worst 10 IoU values ({label}) ---")
        worst = np.argsort(ious)[:10]
        for idx in worst:
            _, dx, dy, dz, heading = boxes[idx]
            print(f"  IoU={ious[idx]:.4f}  dx={dx:.3f} dy={dy:.3f} dz={dz:.3f} heading={heading:.3f}")


# ══════════════════════════════════════════════════════════════════════
# scatter — GT sizes vs anchor scatter plot
# ══════════════════════════════════════════════════════════════════════
def cmd_scatter(args):
    """Plot GT sizes vs configured anchors as a scatter chart."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import yaml

    infos = _load_infos(args.info_path)
    boxes = _extract_gt_boxes(infos, class_names=['Car', 'Pedestrian', 'Cyclist'])

    # Load anchor config
    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f'Error: {config_path} not found.')
        sys.exit(1)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    anchor_configs = config['MODEL']['DENSE_HEAD']['ANCHOR_GENERATOR_CONFIG']

    # Group GT sizes by class
    gt_sizes = {'Car': [], 'Pedestrian': [], 'Cyclist': []}
    for cls, dx, dy, _dz, _heading in boxes:
        if cls in gt_sizes:
            gt_sizes[cls].append((dx, dy))

    plt.figure(figsize=(10, 8))
    colors = {'Car': 'green', 'Pedestrian': 'red', 'Cyclist': 'blue'}

    for cls, sizes in gt_sizes.items():
        if not sizes:
            continue
        sizes = np.array(sizes)
        plt.scatter(sizes[:, 0], sizes[:, 1], s=5, alpha=0.3,
                    label=f'{cls} GT', color=colors[cls])
        for anchor in anchor_configs:
            if anchor['class_name'] == cls:
                a_l = anchor['anchor_sizes'][0][0]
                a_w = anchor['anchor_sizes'][0][1]
                plt.scatter(a_l, a_w, s=200, marker='X', edgecolors='black',
                            label=f'{cls} Anchor ({a_l}x{a_w})', color=colors[cls])

    plt.xlabel('Length (L / dx) [m]')
    plt.ylabel('Width (W / dy) [m]')
    plt.title('GT Sizes vs Configured Anchors')
    plt.legend()
    plt.grid(True)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f'Saved anchor scatter plot to {save_path}')


# ══════════════════════════════════════════════════════════════════════
# dist — GT size distribution histogram
# ══════════════════════════════════════════════════════════════════════
def cmd_dist(args):
    """Plot per-class GT size distribution histogram."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    infos = _load_infos(args.info_path)
    class_name = args.class_name
    boxes = _extract_gt_boxes(infos, class_names=[class_name])

    dim_idx = {'length': 1, 'width': 2, 'height': 3}[args.dim]
    values = [b[dim_idx] for b in boxes]

    if not values:
        print(f'No {class_name} found in the info file.')
        return

    values = np.array(values)
    avg = np.mean(values)
    median = np.median(values)

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.axvline(avg, color='red', linestyle='dashed', linewidth=2,
                label=f'Mean: {avg:.2f}m')
    plt.axvline(median, color='green', linestyle='dashed', linewidth=2,
                label=f'Median: {median:.2f}m')

    dim_label = args.dim.capitalize()
    plt.xlabel(f'{dim_label} ({args.dim[0]}{["x","y","z"][dim_idx-1]}) [m]')
    plt.ylabel('Frequency')
    plt.title(f'{class_name} {dim_label} Distribution')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f'Saved {class_name} {dim_label.lower()} distribution plot to {save_path}')
    print(f'Total {class_name}: {len(values)}')
    print(f'Min: {values.min():.2f}m  Max: {values.max():.2f}m  Avg: {avg:.2f}m')


# ══════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Anchor analysis toolkit for RadarPillar',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # -- verify --
    p_verify = sub.add_parser('verify', help='Verify anchor-GT IoU overlap')
    p_verify.add_argument('--info_path', type=str, default=str(DEFAULT_INFO_PATH))

    # -- scatter --
    p_scatter = sub.add_parser('scatter', help='Plot GT sizes vs anchors scatter')
    p_scatter.add_argument('--info_path', type=str, default=str(DEFAULT_INFO_PATH))
    p_scatter.add_argument('--config_path', type=str,
                           default='tools/cfgs/models_configs/vod_models/vod_radarpillar.yaml')
    p_scatter.add_argument('--save_path', type=str, default='tools/anchor_verification.png')

    # -- dist --
    p_dist = sub.add_parser('dist', help='Plot per-class GT size distribution')
    p_dist.add_argument('--info_path', type=str, default=str(DEFAULT_INFO_PATH))
    p_dist.add_argument('--class_name', type=str, default='Cyclist',
                        choices=['Car', 'Pedestrian', 'Cyclist'])
    p_dist.add_argument('--dim', type=str, default='length',
                        choices=['length', 'width', 'height'])
    p_dist.add_argument('--save_path', type=str, default='tools/anchor_dist.png')

    args = parser.parse_args()

    cmd_map = {
        'verify': cmd_verify,
        'scatter': cmd_scatter,
        'dist': cmd_dist,
    }
    cmd_map[args.command](args)


if __name__ == '__main__':
    main()

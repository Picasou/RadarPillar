#!/usr/bin/env bash
# 速度量测链重构验证 (窗口/α-β 状态寄存航迹自身): 备份基线 0200/0201 → 重放 full pipeline → GT 对拍 (基线 vs 重构) 指标.
# 指标: 主动态 GT 目标的 coverage / IDSW / 断链段数 / 滞后 median / 速度比 median.
# 用法: bash tools/scripts/test/verify_tracker_velstate.sh [仅指标对比 SKIP_REPLAY=1]
set -euo pipefail
cd "$(dirname "$0")/../../.."   # 工程根

# conda 自探测（env=base 本机唯一环境）
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _c in "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        [ -f "$_c/etc/profile.d/conda.sh" ] && { source "$_c/etc/profile.d/conda.sh"; break; }
    done
fi
conda activate base

LOG_DIR=".tmp/velstate"
mkdir -p "$LOG_DIR"
BASELINE="$LOG_DIR/baseline"
CFGs=(
    "tracker/cfg/cfg_replay_trained_msr_input5_pp32_ch2d_trunk.yaml"
    "tracker/cfg/cfg_replay_gen_msr_input5_pp32_ch2d_trunk.yaml"
)

# ---- 1. 备份基线 bin (重放会 overlap=1 覆盖) ----
if [ "${SKIP_REPLAY:-0}" != "1" ]; then
    python - "$BASELINE" "${CFGs[@]}" <<'PYEOF'
import os, shutil, sys, yaml
baseline = sys.argv[1]
for cfg_path in sys.argv[2:]:
    paths = yaml.safe_load(open(cfg_path, encoding='utf-8'))['DATA']['paths']
    for p in paths:
        dst = os.path.join(baseline, os.path.basename(p.rstrip('/')))
        src_dir = os.path.join(p, 'radar.default')
        if not os.path.exists(os.path.join(src_dir, '0201.00000.bin')):
            print(f'[backup] 无基线 bin, 跳过: {p}')
            continue
        os.makedirs(dst, exist_ok=True)
        for f in ('0200.00000.bin', '0201.00000.bin'):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst, f))
        print(f'[backup] {os.path.basename(p.rstrip("/"))}')
PYEOF

    # ---- 2. 重放 (trained 3 条 + gen 8 条) ----
    for cfg in "${CFGs[@]}"; do
        echo "=== replay: $cfg ==="
        python -u -m tracker.tracker --cfg "$cfg"
    done
fi

# ---- 3. GT 对拍: 基线 vs 重构 ----
python - "$BASELINE" "${CFGs[@]}" <<'PYEOF'
import os, sys, numpy as np, yaml
sys.path.insert(0, '.')
from tracker.utils.rw_struct import struct_read, Raw_TrkHead, Raw_Trk
from tracker.loader import _GT_HEAD_DTYPE, _GT_REC_DTYPE

GATE_M = 3.0

def load_trk(bin_dir):
    heads = struct_read(os.path.join(bin_dir, '0200.00000.bin'), Raw_TrkHead)
    recs = struct_read(os.path.join(bin_dir, '0201.00000.bin'), Raw_Trk)
    frames, off = [], 0
    for h in heads:
        n = min(h.trk_num, len(recs) - off)
        frames.append([(recs[off + j].id, recs[off + j].x_m / 100.0, recs[off + j].y_m / 100.0,
                        recs[off + j].vx_mps / 100.0, recs[off + j].vy_mps / 100.0)
                       for j in range(max(n, 0))])
        off += h.trk_num
    return frames

def load_gt(seq_dir):
    gt_dir = os.path.join(seq_dir, 'gt.default')
    fh, fr = os.path.join(gt_dir, 'gt_radar_1200.00000.bin'), os.path.join(gt_dir, 'gt_radar_1201.00000.bin')
    if not (os.path.exists(fh) and os.path.exists(fr)):
        return None
    heads = np.frombuffer(open(fh, 'rb').read(), dtype=_GT_HEAD_DTYPE)
    data = open(fr, 'rb').read()
    rec_n = sum(int(h['trk_num']) for h in heads)
    if rec_n == 0:
        return None
    stride = len(data) / rec_n
    assert stride in (27.0, 28.0), f'GT 布局异常 {fr} ({stride:.2f}B/rec)'
    if int(stride) == 28:
        data = np.ascontiguousarray(np.frombuffer(data, np.uint8).reshape(rec_n, 28)[:, :27]).tobytes()
    recs = np.frombuffer(data, dtype=_GT_REC_DTYPE)
    frames, off = [], 0
    for h in heads:
        n = min(int(h['trk_num']), len(recs) - off)
        frames.append({int(recs[off + j]['id']): (recs[off + j]['x'] * 0.01, recs[off + j]['y'] * 0.01,
                                                   recs[off + j]['vx'] * 0.01, recs[off + j]['vy'] * 0.01)
                       for j in range(max(n, 0))})
        off += int(h['trk_num'])
    return frames

def metrics(trk_frames, gt_frames, gid):
    ids, lags, ratios = [], [], []
    for tf, gf in zip(trk_frames, gt_frames):
        if gid not in gf:
            continue
        gx, gy, gvx, gvy = gf[gid]
        cand = [(t, np.hypot(t[1] - gx, t[2] - gy)) for t in tf
                if np.hypot(t[1] - gx, t[2] - gy) <= GATE_M]
        if not cand:
            ids.append(None)
            continue
        t = min(cand, key=lambda c: c[1])[0]
        ids.append(t[0])
        gv = np.hypot(gvx, gvy)
        if gv > 1.0:
            u = np.array([gvx, gvy]) / gv
            lags.append(float(np.dot([t[1] - gx, t[2] - gy], u)))
            ratios.append(float(np.hypot(t[3], t[4]) / gv))
    matched = [i for i in ids if i is not None]
    cov = len(matched) / max(len(ids), 1)
    idsw = sum(1 for a, b in zip(ids, ids[1:]) if a is not None and b is not None and a != b)
    segs, prev = 0, None
    for i in ids:
        if i is not None and i != prev:
            segs += 1
        prev = i
    lag_med = float(np.median(lags)) if lags else float('nan')
    ratio_med = float(np.median(ratios)) if ratios else float('nan')
    return dict(cov=cov, idsw=idsw, segs=segs, lag=lag_med, ratio=ratio_med)

def main_target_id(gt_frames):
    path = {}
    prev = {}
    for gf in gt_frames:
        for gid, (x, y, _, _) in gf.items():
            if gid in prev:
                path[gid] = path.get(gid, 0.0) + np.hypot(x - prev[gid][0], y - prev[gid][1])
            prev[gid] = (x, y)
    return max(path, key=path.get) if path else None

baseline = sys.argv[1]
sum_base = dict(cov=0, idsw=0, segs=0, lag=[], ratio=[])
sum_new = dict(cov=0, idsw=0, segs=0, lag=[], ratio=[])
n_seq = 0
for cfg_path in sys.argv[2:]:
    for seq in yaml.safe_load(open(cfg_path, encoding='utf-8'))['DATA']['paths']:
        name = os.path.basename(seq.rstrip('/'))
        gt_frames = load_gt(seq)
        if gt_frames is None:
            print(f'[skip] {name}: 无 GT')
            continue
        gid = main_target_id(gt_frames)
        if gid is None:
            print(f'[skip] {name}: 无动态 GT')
            continue
        base_dir = os.path.join(baseline, name)
        if not os.path.exists(os.path.join(base_dir, '0201.00000.bin')):
            print(f'[skip] {name}: 无基线备份')
            continue
        m0 = metrics(load_trk(base_dir), gt_frames, gid)
        m1 = metrics(load_trk(os.path.join(seq, 'radar.default')), gt_frames, gid)
        print(f'--- {name} (主目标 gt_id={gid}) ---')
        print('  基线: coverage={cov:.2f} IDSW={idsw} 段数={segs} 滞后med={lag:+.2f}m 速度比={ratio:.2f}'.format(**m0))
        print('  重构: coverage={cov:.2f} IDSW={idsw} 段数={segs} 滞后med={lag:+.2f}m 速度比={ratio:.2f}'.format(**m1))
        for dst, m in ((sum_base, m0), (sum_new, m1)):
            for k in ('cov', 'idsw', 'segs'):
                dst[k] += m[k]
            for k in ('lag', 'ratio'):
                dst[k] += ([] if m[k] != m[k] else [m[k]])
        n_seq += 1
if n_seq:
    print('=== 汇总 ({} 序列) ==='.format(n_seq))
    for tag, s in (('基线', sum_base), ('重构', sum_new)):
        print('  {}: IDSW={} 段数={} 滞后med={:+.2f}m 速度比={:.2f}'.format(
            tag, s['idsw'], s['segs'],
            float(np.median(s['lag'])) if s['lag'] else float('nan'),
            float(np.median(s['ratio'])) if s['ratio'] else float('nan')))
PYEOF

#!/usr/bin/env python3
"""Scan Claude Code transcripts, aggregate token usage, render an interactive HTML dashboard.

Data source : ~/.claude/projects/*/*.jsonl
Aggregation : 30-minute CST buckets per model; granularity / metric / date filtering is
              done client-side from the embedded data.

This script pre-computes the DEFAULT view (full range, 1h, total) and inlines a static
snapshot into the HTML — so opening the file shows the chart even if JS is slow to boot.
JS then takes over for interactive updates.

Usage:
    python3 collect_cost.py
    python3 collect_cost.py --out /path/to/token_cost.html
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

CST = timezone(timedelta(hours=8))

MODEL_COLORS = {
    "glm-5.2": "#4EA8DE",
    "MiniMax-M3": "#F0883E",
    "k3": "#A371F7",
    "<synthetic>": "#6E7681",
}


def scan_projects(root: str):
    for path in sorted(glob.glob(os.path.join(root, "*", "*.jsonl"))):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = obj.get("message") or {}
                    usage = msg.get("usage")
                    ts = obj.get("timestamp")
                    if not usage or not ts:
                        continue
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(CST)
                    except ValueError:
                        continue
                    yield dt, msg.get("model") or "unknown", usage
        except OSError:
            continue


def bucket_30min_cst(dt: datetime) -> str:
    mi = 0 if dt.minute < 30 else 30
    return dt.replace(minute=mi, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M")


def aggregate(records):
    raw: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    keys: set[str] = set()
    for dt, model, usage in records:
        bk = bucket_30min_cst(dt)
        keys.add(bk)
        v = raw[model][bk]
        v[0] += int(usage.get("input_tokens") or 0)
        v[1] += int(usage.get("cache_creation_input_tokens") or 0)
        v[2] += int(usage.get("cache_read_input_tokens") or 0)
        v[3] += int(usage.get("output_tokens") or 0)

    seed = list(MODEL_COLORS.keys())
    models = sorted(raw.keys(), key=lambda m: (seed.index(m) if m in seed else 99, m))
    by_model = {m: {"buckets": dict(raw[m])} for m in models}
    rng = {"min": min(keys), "max": max(keys)} if keys else {"min": "", "max": ""}
    return by_model, models, rng


def totals(records):
    t = {"input": 0, "cache_creation": 0, "cache_read": 0, "output": 0}
    for _, _, u in records:
        t["input"] += int(u.get("input_tokens") or 0)
        t["cache_creation"] += int(u.get("cache_creation_input_tokens") or 0)
        t["cache_read"] += int(u.get("cache_read_input_tokens") or 0)
        t["output"] += int(u.get("output_tokens") or 0)
    t["total"] = t["input"] + t["cache_creation"] + t["cache_read"] + t["output"]
    return t


def fmt(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n/1e6:.2f}M"
    if n >= 1_000:
        return f"{n/1e3:.1f}k"
    return str(n)


# ============================================================================
# Pre-computation of the DEFAULT view (rendered into HTML as static fallback
# so opening the file shows the chart immediately).
# ============================================================================

def parse_parts(bk):
    return {"Y": int(bk[0:4]), "Mo": int(bk[5:7]), "D": int(bk[8:10]),
            "h": int(bk[11:13]), "mi": int(bk[14:16])}


def floor_key(bk, bucket):
    p = parse_parts(bk)
    mi = 0
    h = p["h"]
    if bucket == 30:
        mi = 0 if p["mi"] < 30 else 30
    elif bucket == 60:
        pass
    elif bucket == 240:
        h = (p["h"] // 4) * 4
    elif bucket == 1440:
        h = 0
    return f"{p['Y']:04d}-{p['Mo']:02d}-{p['D']:02d}T{h:02d}:{mi:02d}"


def bucket_label(bk, bucket):
    p = parse_parts(bk)
    md = f"{p['Mo']:02d}-{p['D']:02d}"
    if bucket >= 1440:
        return md
    return f"{md} {h:02d}:{mi:02d}".format(h=p['h'], mi=parse_parts(bk)['mi'] if bucket < 60 else (0 if bucket > 30 else (0 if parse_parts(bk)['mi'] < 30 else 30)))


def bucket_label_clean(bk, bucket):
    p = parse_parts(bk)
    md = f"{p['Mo']:02d}-{p['D']:02d}"
    if bucket >= 1440:
        return md
    if bucket >= 60:
        return f"{md} {p['h']:02d}:00"
    return f"{md} {p['h']:02d}:{p['mi']:02d}"


def metric_of(slot, metric):
    if metric == "total":
        return slot[0] + slot[1] + slot[2] + slot[3]
    idx = {"input": 0, "cache_creation": 1, "cache_read": 2, "output": 3}[metric]
    return slot[idx]


def compute_default_view(by_model, models, rng, default_state):
    start, end, bucket, metric = default_state
    per_model: dict[str, dict[str, list[int]]] = {m: {} for m in models}
    bucket_exists: set[str] = set()
    for m in models:
        for bk, v in by_model[m]["buckets"].items():
            if bk[:10] < start or bk[:10] > end:
                continue
            fp = floor_key(bk, bucket)
            bucket_exists.add(fp)
            cur = per_model[m].get(fp) or [0, 0, 0, 0]
            for i in range(4):
                cur[i] += v[i]
            per_model[m][fp] = cur

    bk_list = sorted(bucket_exists)
    labels = [bucket_label_clean(bk, bucket) for bk in bk_list]

    datasets = []
    for m in models:
        data = []
        for bk in bk_list:
            s = per_model[m].get(bk)
            data.append(metric_of(s, metric) if s else 0)
        if any(data):
            datasets.append({
                "model": m,
                "color": MODEL_COLORS.get(m, "#6E7681" if m != "unknown" else "#444C56"),
                "data": data,
            })

    # KPI totals
    kpis = {"input": 0, "cache_creation": 0, "cache_read": 0, "output": 0, "total": 0}
    for m in models:
        for fp, s in per_model[m].items():
            for i, k in enumerate(["input", "cache_creation", "cache_read", "output"]):
                kpis[k] += s[i]
    kpis["total"] = kpis["input"] + kpis["cache_creation"] + kpis["cache_read"] + kpis["output"]

    # Per-model totals under current metric
    per_model_tot = []
    for m in models:
        t = 0
        for fp, s in per_model[m].items():
            t += metric_of(s, metric)
        per_model_tot.append({"model": m, "t": t})
    grand = sum(e["t"] for e in per_model_tot)

    # Peak day (across all data, not range-filtered, so it shows a meaningful top day)
    day_tot: dict[str, int] = defaultdict(int)
    for m in models:
        for bk, v in by_model[m]["buckets"].items():
            day_tot[bk[:10]] += v[0] + v[1] + v[2] + v[3]
    sorted_days = sorted(day_tot.items(), key=lambda x: -x[1])
    peak = sorted_days[0] if sorted_days else (None, 0)
    top5 = sorted_days[:5]

    return {
        "labels": labels,
        "datasets": datasets,
        "kpis": kpis,
        "per_model_tot": per_model_tot,
        "grand": grand,
        "peak": peak,
        "top5": top5,
        "all_days": day_tot,
    }


# ============================================================================


def print_summary(records):
    """ASCII output: totals + per-model share + daily bar chart (per-model chars)."""
    t = totals(records)

    # per-model totals
    by_model: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for dt, model, usage in records:
        v = by_model[model]
        v[0] += int(usage.get("input_tokens") or 0)
        v[1] += int(usage.get("cache_creation_input_tokens") or 0)
        v[2] += int(usage.get("cache_read_input_tokens") or 0)
        v[3] += int(usage.get("output_tokens") or 0)

    # order: known palette first, then alpha
    seed = list(MODEL_COLORS.keys())
    models_ordered = sorted(by_model.keys(), key=lambda m: (seed.index(m) if m in seed else 99, m))
    grand = t["total"] or 1

    # terminal width
    try:
        cols = min(120, max(80, os.get_terminal_size().columns))
    except OSError:
        cols = 100

    # ---------- header ----------
    print("┌" + "─" * (cols - 2) + "┐")
    bar = " ▌" * 8
    title = "TOKEN COST · SUMMARY"
    print(f"│{title:^{cols-2}}│")
    print(f"│{bar:^{cols-2}}│")

    # ---------- grand totals (4 lines) ----------
    rows = [
        ("Total",         t["total"]),
        ("Input",         t["input"]),
        ("Cache Read",    t["cache_read"]),
        ("Cache Creation",t["cache_creation"]),
        ("Output",        t["output"]),
    ]
    for label, val in rows:
        pct = val / grand * 100 if grand else 0
        line = f"  {label:<16} {fmt(val):>10}   {pct:5.1f}%"
        print(f"│{line:<{cols-2}}│")

    print("├" + "─" * (cols - 2) + "┤")

    # ---------- per-model ----------
    print(f"│{'PER MODEL':^{cols-2}}│")
    # glyph per model (terminal-safe, distinctive)
    glyphs = {"glm-5.2":"█", "MiniMax-M3":"▓", "k3":"▒", "<synthetic>":"░", "unknown":"·"}
    extra_glyphs = ["◆","■","▲","●","★","◇","▼","◯"]
    glyph_iter = iter(extra_glyphs)
    model_glyph = {}
    for m in models_ordered:
        if m in glyphs: model_glyph[m] = glyphs[m]
        else: model_glyph[m] = next(glyph_iter, "·")

    for m in models_ordered:
        v = by_model[m]
        m_tot = sum(v)
        pct = m_tot / grand * 100 if grand else 0
        bar_width = max(0, int(pct / 100 * 30))
        bar_str = model_glyph[m] * bar_width
        line = f"  {model_glyph[m]} {m:<14} {fmt(m_tot):>10}  {pct:5.1f}%  {bar_str}"
        print(f"│{line:<{cols-2}}│")

    print("├" + "─" * (cols - 2) + "┤")

    # ---------- daily bar chart (last 14 days) ----------
    print(f"│{'DAILY · LAST 14 DAYS (per-model stack)':^{cols-2}}│")
    print(f"│{' ' + '─' * (cols-4) + ' ':^{cols-2}}│")

    # build per-model per-day totals
    per_day_per_model: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0,0,0,0]))
    for dt, model, usage in records:
        ymd = dt.strftime("%Y-%m-%d")
        v = per_day_per_model[ymd][model]
        v[0] += int(usage.get("input_tokens") or 0)
        v[1] += int(usage.get("cache_creation_input_tokens") or 0)
        v[2] += int(usage.get("cache_read_input_tokens") or 0)
        v[3] += int(usage.get("output_tokens") or 0)

    # pick last 14 days with activity
    all_days = sorted(per_day_per_model.keys())
    last14 = all_days[-14:] if len(all_days) > 14 else all_days
    day_max = max(
        sum(per_day_per_model[d][m]) for d in last14 for m in per_day_per_model[d]
    ) if last14 else 1

    plot_width = 40
    legend_str = "  " + "  ".join(f"{model_glyph[m]} {m}" for m in models_ordered)
    print(f"│{legend_str:<{cols-2}}│")

    for d in last14:
        per_m = per_day_per_model[d]
        total = sum(sum(per_m[m]) for m in per_m)
        # build stack: per-model blocks proportional to share, each glyph = its model
        # use width based on total, but composed of glyphs in order
        # simpler: vertical-stack of segments inside one bar
        # Since terminal is char-grid, render per-model as one char per "unit", proportional.
        bar_chars = []
        for m in models_ordered:
            mt = sum(per_m.get(m, [0,0,0,0]))
            if mt <= 0: continue
            n = max(1, round(mt / day_max * plot_width))
            bar_chars.append((model_glyph[m] * n, mt))
        bar = "".join(c for c, _ in bar_chars)
        # right-aligned total
        line = f"  {d[5:]} │ {bar:<{plot_width}}  {fmt(total):>8}"
        print(f"│{line:<{cols-2}}│")

    print("└" + "─" * (cols - 2) + "┘")

    # ---------- top-5 days ----------
    day_totals = [(d, sum(sum(v) for v in per_day_per_model[d].values())) for d in all_days]
    day_totals.sort(key=lambda x: -x[1])
    print()
    print(f"Top 5 days by total tokens:")
    for d, v in day_totals[:5]:
        print(f"  {d}  {fmt(v):>10}")


# ============================================================================


def build_fallback_chart(view) -> str:
    """Pure HTML/CSS stacked bar chart for the default view (used before Chart.js boots)."""
    labels = view["labels"]
    datasets = view["datasets"]  # [{model, color, data: [n..]}]
    if not labels or not datasets:
        return '<div style="padding:60px 0;text-align:center;color:var(--muted)">no data</div>'

    totals_per_col = [sum(d["data"][i] for d in datasets) for i in range(len(labels))]
    ymax = max(totals_per_col) or 1

    y_ticks = [fmt(int(ymax * f / 4)) for f in range(5)]  # 0, 25%, 50%, 75%, 100%

    cols_html = []
    for i, lbl in enumerate(labels):
        segments = []
        for d in datasets:
            v = d["data"][i]
            if v <= 0:
                continue
            h = max(0.6, v / ymax * 100)  # % of plot height
            segments.append(
                f'<div class="bar-seg" style="height:{h:.2f}%;background:{d["color"]}" '
                f'title="{html.escape(d["model"])} · {fmt(v)}"></div>'
            )
        step = max(1, len(labels) // 12)
        show_label = (i % step == 0) or i == len(labels) - 1
        lbl_text = html.escape(lbl) if show_label else ""
        cols_html.append(
            f'<div class="bar-col">'
            f'<div class="bar-stack">{"".join(segments)}</div>'
            f'<span class="label">{lbl_text}</span>'
            f'</div>'
        )

    yaxis_html = "".join(f'<span>{html.escape(t)}</span>' for t in y_ticks)
    plot_html = f'<div class="plot">{"".join(cols_html)}</div>'

    # x-axis: 5 evenly-spaced labels
    step = max(1, len(labels) // 4)
    xaxis_html = (
        f'<span>{html.escape(labels[0])}</span>'
        f'<span>{html.escape(labels[len(labels)//4])}</span>'
        f'<span>{html.escape(labels[len(labels)//2])}</span>'
        f'<span>{html.escape(labels[(3*len(labels))//4])}</span>'
        f'<span>{html.escape(labels[-1])}</span>'
    )

    return (
        f'<div class="yaxis">{yaxis_html}</div>'
        f'{plot_html}'
        f'<div class="xaxis">{xaxis_html}</div>'
    )


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Token Cost · Telemetry</title>
<style>
:root{
  --bg:#0B0E13; --panel:#11151C; --panel-2:#161B24;
  --border:#1F2630; --border-strong:#2A313C;
  --text:#D8DEE9; --muted:#6B7785; --dim:#3D4754;
  --accent:#4EA8DE; --accent-dim:#1E4E78;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  background:
    radial-gradient(circle at 1px 1px, #1a2030 1px, transparent 0) 0 0/24px 24px,
    var(--bg);
  color:var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue",
               "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
  font-size:14px; line-height:1.55;
  -webkit-font-smoothing:antialiased;
  letter-spacing:0.005em;
}
.mono{font-family: "SF Mono", "JetBrains Mono", "Cascadia Code", "Cascadia Mono",
       Menlo, Consolas, "Roboto Mono", ui-monospace, monospace;
       font-variant-numeric:tabular-nums; letter-spacing:0;}
.title{font-weight:650;letter-spacing:-0.02em}

.wrap{max-width:1280px;margin:0 auto;padding:28px 32px 60px}
.grid{display:grid;grid-template-columns:300px 1fr;gap:36px;align-items:start}
@media (max-width:960px){.grid{grid-template-columns:1fr;gap:24px}}

.rail{position:sticky;top:24px}
.eyebrow{
  font-family: ui-monospace, monospace; font-size:11px;
  color:var(--accent);letter-spacing:0.18em;font-weight:600;
  display:flex;align-items:center;gap:8px;
}
.eyebrow::before{content:"";display:inline-block;width:8px;height:8px;
  background:var(--accent);border-radius:50%;
  box-shadow:0 0 14px var(--accent);}
.title{font-size:28px;font-weight:650;margin:10px 0 6px;letter-spacing:-0.025em;line-height:1.15}
.title em{font-style:normal;color:var(--accent);font-weight:600}
.meta{font-size:12px;color:var(--muted);margin-bottom:2px}
.meta.mono{font-size:11px;color:var(--dim)}

hr{border:none;border-top:1px solid var(--border);margin:22px 0}

.kpi-stack{display:flex;flex-direction:column}
.kpi{
  padding:11px 0 11px 14px;border-bottom:1px solid var(--border);
  cursor:pointer;transition:background .12s;position:relative;
}
.kpi:first-child{padding-top:4px}
.kpi:last-child{border-bottom:none}
.kpi::before{
  content:"";position:absolute;left:0;top:50%;transform:translateY(-50%);
  width:2px;height:0;background:var(--accent);transition:height .15s;
}
.kpi.active::before{height:62%}
.kpi.active .val{color:var(--accent)}
.kpi:hover{background:var(--panel-2)}
.kpi .lbl{
  font-family: ui-monospace, monospace; font-size:10px;
  color:var(--muted);letter-spacing:0.12em;text-transform:uppercase;font-weight:500;
}
.kpi .val{font-size:22px;font-weight:600;margin-top:3px;transition:color .15s}

.peak .lbl{
  font-family: ui-monospace, monospace; font-size:10px;
  color:var(--muted);letter-spacing:0.12em;text-transform:uppercase;font-weight:500;
}
.peak .val{font-size:18px;font-weight:600;margin-top:4px}
.peak-bars{display:flex;gap:3px;margin-top:12px;height:36px;align-items:flex-end}
.peak-bars .pb{
  flex:1;background:var(--panel-2);border-top:2px solid var(--dim);
  min-height:2px;transition:all .15s;
}
.peak-bars .pb.top{background:var(--accent-dim);border-top-color:var(--accent)}
.peak-bars .pb.zero{opacity:.35}

main{min-width:0}
.controls{
  display:grid;grid-template-columns:1fr 1fr 1.3fr 1.5fr auto;gap:14px;
  background:var(--panel);border:1px solid var(--border);
  border-radius:6px;padding:14px 16px;margin-bottom:16px;
  align-items:end;
}
@media (max-width:760px){.controls{grid-template-columns:1fr 1fr}}
.field{display:flex;flex-direction:column;gap:5px}
.field label{
  font-family: ui-monospace, monospace; font-size:10px;
  color:var(--muted);letter-spacing:0.12em;text-transform:uppercase;font-weight:500;
}
.field input,.field select{
  background:var(--bg);color:var(--text);border:1px solid var(--border);
  border-radius:4px;padding:7px 9px;font:inherit;font-size:13px;
  font-family: ui-monospace, monospace;
}
.field input:focus,.field select:focus{outline:none;border-color:var(--accent)}
.presets{display:flex;gap:6px}
.presets button{
  font-family: ui-monospace, monospace; font-size:11px;font-weight:500;
  background:transparent;color:var(--muted);
  border:1px solid var(--border);border-radius:4px;
  padding:7px 11px;cursor:pointer;transition:all .15s;letter-spacing:0.04em;
}
.presets button:hover{color:var(--text);border-color:var(--border-strong)}
.presets button.active{color:var(--accent);border-color:var(--accent);background:var(--accent-dim)}

section{
  background:var(--panel);border:1px solid var(--border);
  border-radius:6px;padding:18px 20px;margin-bottom:14px;
}

/* ---- static fallback bar chart (visible until Chart.js takes over) ---- */
.fallback-chart{
  position:relative;height:460px;
  display:grid;grid-template-columns:64px 1fr;grid-template-rows:1fr 22px;
  border:1px solid var(--border);border-radius:6px;overflow:hidden;
  font-family:"SF Mono","JetBrains Mono",Menlo,monospace;z-index:1;
}
.fallback-chart .yaxis{
  grid-column:1;grid-row:1;
  display:flex;flex-direction:column-reverse;justify-content:space-between;
  padding:6px 8px;background:var(--panel-2);
  font-size:10px;color:var(--muted);text-align:right;
  border-right:1px solid var(--border);
}
.fallback-chart .plot{
  grid-column:2;grid-row:1;
  display:flex;align-items:flex-end;gap:2px;padding:8px 6px;min-height:0;
}
.fallback-chart .bar-col{
  flex:1 1 0;min-width:0;height:100%;
  display:flex;flex-direction:column;justify-content:flex-end;
  position:relative;
}
.fallback-chart .bar-stack{
  width:100%;display:flex;flex-direction:column-reverse;
  position:relative;
}
.fallback-chart .bar-seg{
  width:100%;display:block;min-height:1px;
  transition:opacity .15s;
}
.fallback-chart .bar-seg:hover{opacity:.78}
.fallback-chart .bar-col .label{
  position:absolute;left:50%;bottom:-18px;transform:translateX(-50%);
  font-size:10px;color:var(--muted);white-space:nowrap;
}
.fallback-chart .xaxis{
  grid-column:2;grid-row:2;
  display:flex;justify-content:space-between;align-items:center;
  padding:0 6px;border-top:1px solid var(--border);
  font-size:10px;color:var(--muted);
}
.fallback-chart.hidden{display:none}
canvas#chart{display:none;max-width:100%}
.chart-wrap{position:relative;height:460px}
canvas#chart.live{display:block;height:100%!important;width:100%!important}
.section-head{display:flex;justify-content:space-between;align-items:baseline;
  margin-bottom:14px;flex-wrap:wrap;gap:6px}
.section-title{
  font-family: ui-monospace, monospace; font-size:11px;
  color:var(--accent);letter-spacing:0.16em;text-transform:uppercase;font-weight:600;
}
.section-sub{font-size:12px;color:var(--muted);font-family: ui-monospace, monospace}

.legend{display:flex;flex-wrap:wrap;gap:14px 22px}
.legend-item{display:flex;align-items:center;gap:8px;font-size:13px;cursor:pointer;
  user-select:none;transition:opacity .15s}
.legend-item.muted{opacity:.35}
.legend-item .sw{width:12px;height:12px;border-radius:2px}
.legend-item .lbl-text{color:var(--text)}
.legend-item .pc{color:var(--muted);font-size:12px;font-family: ui-monospace, monospace}

table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:10px 0;border-bottom:1px solid var(--border)}
th{
  font-family: ui-monospace, monospace; font-size:10px;
  color:var(--muted);letter-spacing:0.12em;text-transform:uppercase;font-weight:500;
}
td{font-family: ui-monospace, monospace}
td .name{color:var(--text);font-family:inherit;font-size:13px}
td .share{color:var(--muted);font-size:12px}
td .bar-cell{width:38%;padding-left:14px}
td .bar-track{height:6px;background:var(--panel-2);border-radius:3px;overflow:hidden}
td .bar{height:100%;background:var(--accent);border-radius:3px;min-width:2px}

.empty{
  padding:48px 0;text-align:center;color:var(--muted);font-size:13px;
  font-family: ui-monospace, monospace;letter-spacing:0.04em;
}

footer{
  margin-top:28px;font-size:11px;color:var(--muted);
  display:flex;gap:18px;flex-wrap:wrap;align-items:center;
}
footer .tag{
  font-family: ui-monospace, monospace; letter-spacing:0.1em;
  color:var(--dim);font-size:10px;text-transform:uppercase;
}
footer .val{font-family: ui-monospace, monospace; color:var(--text);font-size:11px}

@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<div class="wrap">
  <div class="grid">
    <aside class="rail">
      <div class="eyebrow">TOKEN COST</div>
      <h1 class="title">Token Cost <em>Telemetry</em></h1>
      <div class="meta">__SUBTITLE__</div>
      <div class="meta mono">__GEN_TS__</div>

      <hr>

      <div class="kpi-stack" id="kpis">
        <div class="kpi active" data-metric="total"><div class="lbl">Total</div><div class="val mono" id="k-total">__K_TOTAL__</div></div>
        <div class="kpi" data-metric="input"><div class="lbl">Input</div><div class="val mono" id="k-input">__K_INPUT__</div></div>
        <div class="kpi" data-metric="cache_read"><div class="lbl">Cache Read</div><div class="val mono" id="k-cr">__K_CR__</div></div>
        <div class="kpi" data-metric="cache_creation"><div class="lbl">Cache Creation</div><div class="val mono" id="k-cc">__K_CC__</div></div>
        <div class="kpi" data-metric="output"><div class="lbl">Output</div><div class="val mono" id="k-output">__K_OUTPUT__</div></div>
      </div>

      <hr>

      <div class="peak">
        <div class="lbl">Peak day</div>
        <div class="val mono" id="peak-day">__PEAK_DAY__</div>
        <div class="peak-bars" id="peak-bars">__PEAK_BARS__</div>
        <div class="meta" style="margin-top:10px">last 21 days · top 5 highlighted</div>
      </div>
    </aside>

    <main>
      <div class="controls">
        <div class="field"><label for="start">Start</label><input type="date" id="start" value="__START__" min="__START__" max="__END__"></div>
        <div class="field"><label for="end">End</label><input type="date" id="end" value="__END__" min="__START__" max="__END__"></div>
        <div class="field"><label for="bucket">Granularity</label>
          <select id="bucket">
            <option value="30">30 min</option>
            <option value="60" selected>1 hour</option>
            <option value="240">4 hours</option>
            <option value="1440">1 day</option>
          </select>
        </div>
        <div class="field"><label for="metric">Metric</label>
          <select id="metric">
            <option value="total" selected>Total</option>
            <option value="input">Input</option>
            <option value="cache_creation">Cache Creation</option>
            <option value="cache_read">Cache Read</option>
            <option value="output">Output</option>
          </select>
        </div>
        <div class="presets" id="presets">
          <button data-preset="24">24h</button>
          <button data-preset="168">7d</button>
          <button data-preset="all">All</button>
        </div>
      </div>

      <section>
        <div class="section-head">
          <span class="section-title">Stacked by model</span>
          <span class="section-sub" id="ctitle">Total · stacked by model · __START__ → __END__</span>
        </div>
        <div class="fallback-chart" id="fallback">__FALLBACK_CHART__</div>
        <div class="chart-wrap"><canvas id="chart"></canvas></div>
      </section>

      <section>
        <div class="section-head">
          <span class="section-title">Legend · click to toggle</span>
          <span class="section-sub" id="lsub">__NM__ models · Σ __GRAND__</span>
        </div>
        <div class="legend" id="legend">__LEGEND__</div>
      </section>

      <section>
        <div class="section-head">
          <span class="section-title">Per-model breakdown</span>
          <span class="section-sub" id="tsub">Σ __GRAND__</span>
        </div>
        <table id="mtable">
          <thead><tr><th>Model</th><th>Tokens</th><th>Share</th><th></th></tr></thead>
          <tbody>__TABLE__</tbody>
        </table>
      </section>
    </main>
  </div>

  <footer>
    <span class="tag">SCOPE</span><span class="val">~/.claude/projects</span>
    <span class="tag">UNITS</span><span class="val">tokens</span>
    <span class="tag">TZ</span><span class="val">Asia/Shanghai</span>
    <span class="tag">GENERATED</span><span class="val" id="gen-ts">__GEN_TS__</span>
  </footer>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script src="chart.umd.min.js"></script>
<script>
'use strict';
const DATA = JSON.parse(document.getElementById('payload').textContent);
const METRIC_LABEL = {total:'Total',input:'Input',cache_creation:'Cache Creation',cache_read:'Cache Read',output:'Output'};
const SLOT = {input:0,cache_creation:1,cache_read:2,output:3};

function fmt(n){
  if(n>=1e9) return (n/1e9).toFixed(2)+'B';
  if(n>=1e6) return (n/1e6).toFixed(2)+'M';
  if(n>=1e3) return (n/1e3).toFixed(1)+'k';
  return String(n);
}
function colorOf(m){
  if(DATA.colors[m]) return DATA.colors[m];
  return m==='unknown' ? '#444C56' : '#6E7681';
}
function parseParts(bk){
  return {Y:+bk.slice(0,4), Mo:+bk.slice(5,7), D:+bk.slice(8,10),
          h:+bk.slice(11,13), mi:+bk.slice(14,16)};
}
function floorParts(p, bucket){
  let {Y,Mo,D,h,mi}=p;
  if(bucket===30) mi = mi<30?0:30;
  else if(bucket===60) mi=0;
  else if(bucket===240) { mi=0; h=Math.floor(h/4)*4; }
  else if(bucket===1440){ mi=0; h=0; }
  const pp=n=>String(n).padStart(2,'0');
  return `${Y}-${pp(Mo)}-${pp(D)}T${pp(h)}:${pp(mi)}`;
}
function bucketLabel(bk, bucket){
  const p=parseParts(bk); const pp=n=>String(n).padStart(2,'0');
  const md=`${pp(p.Mo)}-${pp(p.D)}`;
  if(bucket>=1440) return md;
  if(bucket>=60) return `${md} ${pp(p.h)}:00`;
  return `${md} ${pp(p.h)}:${pp(p.mi)}`;
}
function inRange(bk,startYmd,endYmd){
  const d=bk.slice(0,10);
  return d>=startYmd && d<=endYmd;
}
function escapeHtml(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}

const state = {
  start: DATA.range.min.slice(0,10),
  end:   DATA.range.max.slice(0,10),
  bucket: 60,
  metric: 'total',
  hidden: new Set(),
};

function aggregateAll(){
  const metric=state.metric, start=state.start, end=state.end, bucket=state.bucket;
  const perModel={}; const bucketExists={};
  DATA.models.forEach(m=>{
    perModel[m]={};
    const buckets=DATA.by_model[m].buckets;
    for(const bk in buckets){
      if(!inRange(bk,start,end)) continue;
      const v=buckets[bk];
      const fp=floorParts(parseParts(bk),bucket);
      bucketExists[fp]=1;
      const cur=perModel[m][fp]||[0,0,0,0];
      cur[0]+=v[0];cur[1]+=v[1];cur[2]+=v[2];cur[3]+=v[3];
      perModel[m][fp]=cur;
    }
  });
  const bkList=Object.keys(bucketExists).sort();
  function valOf(slot){
    if(metric==='total') return slot[0]+slot[1]+slot[2]+slot[3];
    return slot[SLOT[metric]] || 0;
  }
  const kpis={total:0,input:0,cache_creation:0,cache_read:0,output:0};
  DATA.models.forEach(m=>{
    for(const fp in perModel[m]){
      const s=perModel[m][fp];
      kpis.input+=s[0]; kpis.cache_creation+=s[1];
      kpis.cache_read+=s[2]; kpis.output+=s[3];
    }
  });
  kpis.total=kpis.input+kpis.cache_creation+kpis.cache_read+kpis.output;
  const perModelTot=DATA.models.map(m=>{
    let t=0; for(const fp in perModel[m]) t+=valOf(perModel[m][fp]);
    return {m, t};
  });
  return {bkList, perModel, kpis, perModelTot, valOf};
}

let chart=null;
const canvasEl = document.getElementById('chart');
const fallbackEl = document.getElementById('fallback');

function renderChart(agg){
  const labels=agg.bkList.map(bk=>bucketLabel(bk,state.bucket));
  const datasets=DATA.models
    .filter(m=>!state.hidden.has(m))
    .map(m=>({
      label:m,
      data:agg.bkList.map(bk=>{const s=agg.perModel[m][bk]; return s?agg.valOf(s):0;}),
      backgroundColor:colorOf(m),
      borderColor:colorOf(m),
      borderWidth:0,
      stack:'tokens',
      borderRadius:1,
      maxBarThickness: state.bucket<240 ? 18 : 32,
    }));

  // 1. If Chart.js is unavailable, render fallback (instant, never fails).
  if(typeof Chart === 'undefined'){
    renderFallback(agg, labels, datasets);
    fallbackEl.classList.remove('hidden');
    canvasEl.classList.remove('live');
    document.getElementById('ctitle').textContent =
      `${METRIC_LABEL[state.metric]} · ${agg.bkList.length} buckets · ${state.start} → ${state.end}`;
    return;
  }
  try{
    if(chart){chart.destroy();chart=null;}
    if(!labels.length){
      canvasEl.classList.remove('live');
      return;
    }
    canvasEl.classList.add('live');
    // Y axis ceiling = actual max of the *currently filtered* stacked column,
    // NOT all-time max — so 7-15 alone won't be visually crushed by 7-19's spike.
    const colTotals = labels.map((_, i) => datasets.reduce((a, d) => a + (d.data[i]||0), 0));
    const dataMax = colTotals.length ? Math.max(...colTotals) : 1;
    chart=new Chart(canvasEl,{
    type:'bar',
    data:{labels,datasets},
    options:{
      responsive:true,maintainAspectRatio:false,
      animation:{duration:200},
      interaction:{mode:'index',intersect:false},
      hover:{mode:'index',intersect:false,animationDuration:80},
      plugins:{
        legend:{display:false},
        tooltip:{
          enabled:true,
          backgroundColor:'#11151C',borderColor:'#4EA8DE',borderWidth:1,
          titleColor:'#4EA8DE',bodyColor:'#D8DEE9',footerColor:'#6B7785',
          padding:12,cornerRadius:6,boxPadding:6,boxWidth:10,boxHeight:10,
          titleFont:{family:'monospace',size:12,weight:'bold'},
          bodyFont:{family:'monospace',size:11},
          footerFont:{family:'monospace',size:10},
          displayColors:true,
          callbacks:{
            title:(items)=>`${items[0].label}`,
            label:(c)=>{
              const v = c.parsed.y;
              const grand = items.reduce((a,it)=>a+it.parsed.y,0);
              const pct = grand ? (v/grand*100).toFixed(1) : '0';
              return ` ${c.dataset.label}  ·  ${fmt(v)}  (${pct}%)`;
            },
            footer:(items)=>{
              const sum=items.reduce((a,it)=>a+it.parsed.y,0);
              return `Σ ${fmt(sum)}`;
            }
          }
        }
      },
      scales:{
        x:{stacked:true,ticks:{color:'#6B7785',font:{family:'monospace',size:10},
           maxRotation:45,minRotation:0,autoSkip:true,maxTicksLimit:18,autoSkipPadding:18},
           grid:{display:false},border:{color:'#1F2630'}},
        y:{stacked:true,beginAtZero:true,max:dataMax,
           ticks:{color:'#6B7785',font:{family:'monospace',size:10},
           callback:v=>fmt(v),maxTicksLimit:8},grid:{color:'#161B24'},border:{display:false}}
      },
      onHover:(evt, els)=>{
        evt.native.target.style.cursor = els.length ? 'crosshair' : 'default';
      }
    }
  });
  }catch(err){
    console.error('[token_cost] Chart.js failed, falling back:', err);
    renderFallback(agg, labels, datasets);
    fallbackEl.classList.remove('hidden');
    canvasEl.classList.remove('live');
    return;
  }
  document.getElementById('ctitle').textContent =
    `${METRIC_LABEL[state.metric]} · ${agg.bkList.length} buckets · ${state.start} → ${state.end}`;
}

function renderFallback(agg, labels, datasets){
  // Mirrors the Python-side grid layout (#fallback contains .yaxis + .plot + .xaxis).
  const wrap = fallbackEl;
  if(!labels.length || !datasets.length){
    wrap.innerHTML = '<div style="grid-column:1/3;grid-row:1/3;display:flex;align-items:center;justify-content:center;color:var(--muted);font-family:monospace">no data in selected range</div>';
    return;
  }
  const colTotals = labels.map((_, i) => datasets.reduce((a, d) => a + d.data[i], 0));
  const ymax = Math.max(...colTotals) || 1;
  const yticks = [0, .25, .5, .75, 1].map(f => fmt(f * ymax));

  const cols = labels.map((lbl, i) => {
    const segs = datasets.filter(d => d.data[i] > 0).map(d => {
      const h = Math.max(0.6, d.data[i] / ymax * 100);
      return `<div class="bar-seg" style="height:${h.toFixed(2)}%;background:${d.color}" title="${escapeHtml(d.label)} · ${fmt(d.data[i])}"></div>`;
    }).join('');
    const step = Math.max(1, Math.floor(labels.length / 12));
    const showLabel = (i % step === 0) || i === labels.length - 1;
    return `<div class="bar-col"><div class="bar-stack">${segs}</div>` +
           `<span class="label">${showLabel ? escapeHtml(lbl) : ''}</span></div>`;
  }).join('');

  const xsteps = [0, Math.floor(labels.length/4), Math.floor(labels.length/2),
                  Math.floor(labels.length*3/4), labels.length-1];
  const xaxis = xsteps.map(i => `<span>${escapeHtml(labels[i])}</span>`).join('');

  wrap.innerHTML =
    `<div class="yaxis">${yticks.map(t=>`<span>${escapeHtml(t)}</span>`).join('')}</div>` +
    `<div class="plot">${cols}</div>` +
    `<div class="xaxis">${xaxis}</div>`;
}

function renderKpis(agg){
  document.getElementById('k-total').textContent  = fmt(agg.kpis.total);
  document.getElementById('k-input').textContent  = fmt(agg.kpis.input);
  document.getElementById('k-cr').textContent     = fmt(agg.kpis.cache_read);
  document.getElementById('k-cc').textContent     = fmt(agg.kpis.cache_creation);
  document.getElementById('k-output').textContent = fmt(agg.kpis.output);
  document.querySelectorAll('.kpi').forEach(b=>{
    b.classList.toggle('active', b.dataset.metric===state.metric);
    b.style.cursor = 'pointer';
    b.onclick = ()=>{
      if(state.metric !== b.dataset.metric){
        state.metric = b.dataset.metric;
        render();
      }
    };
  });
}

function renderLegend(agg){
  const grand=agg.perModelTot.reduce((a,e)=>a+e.t,0);
  const el=document.getElementById('legend');
  el.innerHTML = agg.perModelTot.map(e=>{
    const pc=grand?Math.round(e.t/grand*100):0;
    const muted=state.hidden.has(e.m)?' muted':'';
    return `<span class="legend-item${muted}" data-model="${escapeHtml(e.m)}" title="${escapeHtml(e.m)} · ${fmt(e.t)}">
      <span class="sw" style="background:${colorOf(e.m)}"></span>
      <span class="lbl-text">${escapeHtml(e.m)}</span>
      <span class="pc">${fmt(e.t)} · ${pc}%</span>
    </span>`;
  }).join('');
  el.querySelectorAll('.legend-item').forEach(it=>{
    it.addEventListener('click',()=>{
      const m=it.dataset.model;
      if(state.hidden.has(m)) state.hidden.delete(m);
      else state.hidden.add(m);
      render();
    });
  });
  document.getElementById('lsub').textContent =
    `${agg.perModelTot.length} models · Σ ${fmt(grand)}`;
}

function renderTable(agg){
  const grand=agg.perModelTot.reduce((a,e)=>a+e.t,0);
  const tbody=document.querySelector('#mtable tbody');
  tbody.innerHTML = agg.perModelTot
    .sort((a,b)=>b.t-a.t)
    .map(e=>{
      const pc=grand?(e.t/grand*100):0;
      const barW=Math.max(2, Math.round(pc));
      return `<tr>
        <td><span class="name">${escapeHtml(e.m)}</span></td>
        <td>${fmt(e.t)}</td>
        <td class="share">${pc.toFixed(1)}%</td>
        <td class="bar-cell"><div class="bar-track"><div class="bar" style="width:${barW}%"></div></div></td>
      </tr>`;
    }).join('');
  document.getElementById('tsub').textContent = `Σ ${fmt(grand)}`;
}

function syncPreset(){
  document.querySelectorAll('.presets button').forEach(b=>{
    const p=b.dataset.preset;
    let active=false;
    if(p==='all'){
      active = (state.start===DATA.range.min.slice(0,10) && state.end===DATA.range.max.slice(0,10));
    } else {
      const hours=parseInt(p,10);
      const e=new Date(state.end+'T23:59:59+08:00');
      const s=new Date(e.getTime()-hours*3600*1000);
      const pp=n=>String(n).padStart(2,'0');
      active = (`${s.getFullYear()}-${pp(s.getMonth()+1)}-${pp(s.getDate())}`===state.start);
    }
    b.classList.toggle('active',active);
  });
}

function render(){
  syncPreset();
  const agg=aggregateAll();
  renderKpis(agg);
  renderChart(agg);
  renderLegend(agg);
  renderTable(agg);
}

document.getElementById('start').addEventListener('change',e=>{
  state.start=e.target.value;
  if(state.start>state.end) state.end=state.start;
  render();
});
document.getElementById('end').addEventListener('change',e=>{
  state.end=e.target.value;
  if(state.end<state.start) state.start=state.end;
  render();
});
document.getElementById('bucket').addEventListener('change',e=>{
  state.bucket=parseInt(e.target.value,10);
  render();
});
document.getElementById('metric').addEventListener('change',e=>{
  state.metric=e.target.value;
  render();
});
document.querySelectorAll('.kpi').forEach(b=>{
  b.addEventListener('click',()=>{
    state.metric=b.dataset.metric;
    document.getElementById('metric').value=state.metric;
    render();
  });
});
document.querySelectorAll('.presets button').forEach(b=>{
  b.addEventListener('click',()=>{
    const p=b.dataset.preset;
    if(p==='all'){
      state.start=DATA.range.min.slice(0,10);
      state.end=DATA.range.max.slice(0,10);
    } else {
      const hours=parseInt(p,10);
      const e=new Date(DATA.range.max+'T00:00:00+08:00');
      const s=new Date(e.getTime()-(hours-24)*3600*1000);
      const pp=n=>String(n).padStart(2,'0');
      state.start=`${s.getFullYear()}-${pp(s.getMonth()+1)}-${pp(s.getDate())}`;
      state.end=DATA.range.max.slice(0,10);
    }
    document.getElementById('start').value=state.start;
    document.getElementById('end').value=state.end;
    render();
  });
});

// First paint.
render();

// Hide fallback on boot if Chart.js is up (it's the layered overlay default).
if(typeof Chart !== 'undefined'){
  fallbackEl.classList.add('hidden');
  canvasEl.classList.add('live');
}
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Aggregate Claude Code token usage into an HTML dashboard.")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("summary", help="print ASCII summary to stdout (no HTML)")
    p_html = sub.add_parser("html", help="render the interactive HTML dashboard")
    p_html.add_argument("--root", default=os.path.expanduser("~/.claude/projects"))
    p_html.add_argument("--out", help="output HTML path (default: ./token_cost.html next to this script)")
    # legacy: bare invocation (no subcommand) renders HTML using the default root
    ap.add_argument("--root", default=os.path.expanduser("~/.claude/projects"))
    ap.add_argument("--out", help="output HTML path")
    args = ap.parse_args()

    if args.cmd == "summary":
        records = list(scan_projects(args.root))
        if not records:
            print(f"[err] no usage records found under {args.root}")
            return 1
        print_summary(records)
        return 0

    records = list(scan_projects(args.root))
    if not records:
        raise SystemExit(f"No usage records found under {args.root}")

    by_model, models, rng = aggregate(records)
    t = totals(records)

    payload = {
        "models": models,
        "by_model": by_model,
        "totals": t,
        "range": rng,
        "colors": dict(MODEL_COLORS),
        "generated_at_cst": datetime.now(CST).strftime("%Y-%m-%d %H:%M CST"),
    }

    # Default view
    default_state = (rng["min"][:10], rng["max"][:10], 60, "total")
    view = compute_default_view(by_model, models, rng, default_state)

    # Build HTML pieces
    legend_html = "".join(
        f'<span class="legend-item" data-model="{html.escape(e["model"])}" '
        f'title="{html.escape(e["model"])} · {fmt(e["t"])}">'
        f'<span class="sw" style="background:{html.escape(MODEL_COLORS.get(e["model"], "#444C56" if e["model"]=="unknown" else "#6E7681"))}"></span>'
        f'<span class="lbl-text">{html.escape(e["model"])}</span>'
        f'<span class="pc">{fmt(e["t"])} · {round(e["t"]/view["grand"]*100)}%</span>'
        f'</span>'
        for e in view["per_model_tot"] if e["t"] > 0
    )

    table_html = "".join(
        f'<tr>'
        f'<td><span class="name">{html.escape(e["model"])}</span></td>'
        f'<td>{fmt(e["t"])}</td>'
        f'<td class="share">{(e["t"]/view["grand"]*100):.1f}%</td>'
        f'<td class="bar-cell"><div class="bar-track"><div class="bar" style="width:{max(2, round(e["t"]/view["grand"]*100))}%"></div></div></td>'
        f'</tr>'
        for e in sorted(view["per_model_tot"], key=lambda x: -x["t"]) if e["t"] > 0
    )

    # Peak bars: last 21 days, top 5 highlighted
    all_days = sorted(view["all_days"].items())
    tail = all_days[-21:] if len(all_days) > 21 else all_days
    top5_keys = {p[0] for p in view["top5"]}
    if tail:
        m = max(d[1] for d in tail)
        bars = "".join(
            f'<div class="pb{" top" if d[0] in top5_keys else ""}" '
            f'style="height:{max(2, round(d[1]/m*36))}px" title="{d[0]} · {fmt(d[1])}"></div>'
            for d in tail
        )
    else:
        bars = ""
    peak_day = view["peak"][0] if view["peak"][0] else "—"

    # Fallback static bar chart (shown before Chart.js boots)
    fallback_chart_html = build_fallback_chart(view)

    subtitle = f"{len(models)} models · Σ {fmt(t['total'])} token"
    if rng["min"] and rng["max"]:
        subtitle += f" · {rng['min'][:10]} → {rng['max'][:10]}"

    doc = HTML_TEMPLATE
    doc = doc.replace("__SUBTITLE__", html.escape(subtitle))
    doc = doc.replace("__GEN_TS__", html.escape(payload["generated_at_cst"]))
    doc = doc.replace("__START__", rng["min"][:10])
    doc = doc.replace("__END__", rng["max"][:10])
    doc = doc.replace("__K_TOTAL__", fmt(view["kpis"]["total"]))
    doc = doc.replace("__K_INPUT__", fmt(view["kpis"]["input"]))
    doc = doc.replace("__K_CR__", fmt(view["kpis"]["cache_read"]))
    doc = doc.replace("__K_CC__", fmt(view["kpis"]["cache_creation"]))
    doc = doc.replace("__K_OUTPUT__", fmt(view["kpis"]["output"]))
    doc = doc.replace("__PEAK_DAY__", peak_day)
    doc = doc.replace("__PEAK_BARS__", bars)
    doc = doc.replace("__NM__", str(len(models)))
    doc = doc.replace("__GRAND__", fmt(view["grand"]))
    doc = doc.replace("__LEGEND__", legend_html)
    doc = doc.replace("__TABLE__", table_html)
    doc = doc.replace("__FALLBACK_CHART__", fallback_chart_html)
    doc = doc.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))

    out = args.out
    if not out:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "token_cost.html")

    with open(out, "w", encoding="utf-8") as fh:
        fh.write(doc)

    print(f"[ok] records={len(records)} models={models}")
    print(f"[ok] totals: {t}")
    print(f"[ok] written: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
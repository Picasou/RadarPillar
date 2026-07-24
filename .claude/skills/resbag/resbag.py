#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""resbag：训练/评估结果落袋归档 skill（详见 .claude/skills/resbag/SKILL.md）。

子命令:
  make  硬复制 → 算 params/flops → 读 results.json → 写 index.yaml + model_store.yaml
  list  glob <dataset>/*/model_store.yaml 聚合成跨实验总览（默认 stdout）
  show  打印单实验 model_store.yaml（--folder）

设计 spec：docs/superpowers/specs/2026-07-24-resbag-skill-design.md
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── 路径常量（__file__ 自保，不依赖外部 ROOT）──────────────────────────
SKILL_DIR = Path(__file__).resolve().parent
# resbag.py 位于 .claude/skills/resbag/resbag.py → ROOT = .claude/skills 的二级父目录
ROOT = SKILL_DIR.parents[2]
# 自保：subprocess 或 import pcdet 时找得到根
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CST = timezone(timedelta(hours=8))


# ── 工具 ─────────────────────────────────────────────────────────
def _log(msg, level="INFO"):
    print(f"[resbag][{level}] {msg}", flush=True)


def _safe(fn, default=None, label=""):
    """try 兜底：build/thop 类硬错误不让 resbag 崩。"""
    try:
        return fn()
    except Exception as e:
        _log(f"{label} 失败：{type(e).__name__}: {str(e)[:120]}", "WARN")
        return default


@contextlib.contextmanager
def _file_lock(path: Path, timeout_s: float = 30.0):
    """fcntl.flock LOCK_EX 阻塞；每 0.5s 轮询一次，超时上限 timeout_s。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = path.open("w")
    deadline = time.time() + timeout_s
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() > deadline:
                    raise TimeoutError(f"等 {path} 锁超过 {timeout_s}s")
                time.sleep(0.5)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def _store_lock_path(dataset: str, name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", f"{dataset}_{name}")
    return Path(f"/tmp/resbag_store_{safe}.lock")


def _atomic_write_yaml(path: Path, data: dict):
    """temp-file + fsync + os.replace 原子写 yaml（保字段顺序）。"""
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            import yaml  # 局部导入，避免顶层硬依赖
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True,
                           default_flow_style=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _now_iso() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d")


def _load_yaml(path: Path):
    import yaml
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ── 核心：make ───────────────────────────────────────────────────
def cmd_make(args):
    output_root = Path(args.output_root).resolve()
    if not output_root.exists():
        _log(f"--output_root 不存在：{output_root}", "ERROR")
        sys.exit(1)
    dataset = args.dataset
    tag = args.tag
    name = output_root.name  # e.g. 2026072222_radarnext_mdfen_0722_paper
    cfg_path = Path(args.cfg_file).resolve() if args.cfg_file else None
    model = args.model or "unknown"
    batch_size = int(args.batch_size) if args.batch_size else 1
    note = args.note or ""

    # 锁：dataset+name 维度串行化
    lock = _store_lock_path(dataset, name)
    try:
        with _file_lock(lock, timeout_s=30):
            _make_locked(output_root, dataset, tag, name, cfg_path,
                         model, batch_size, note)
    except TimeoutError as e:
        _log(f"{e}", "ERROR")
        sys.exit(2)


def _make_locked(output_root, dataset, tag, name, cfg_path, model,
                 batch_size, note):
    resbag_dir = output_root / "resbag"
    store_path = output_root / "model_store.yaml"
    ckpt_dir = output_root / "ckpt"

    # ── 1. 收集产物路径 + 探测 blocked 信号 ─────────────────────
    has_best = (output_root / "best.pth").exists()
    has_partial = (output_root / "FINISHED_PARTIAL").exists()

    # best_epoch 提取：**直接读 best.pth.epoch**（最稳，OpenPCDet train.py 保存时写入）
    best_epoch = None
    best_pth = output_root / "best.pth"
    if has_best:
        best_epoch = _read_best_epoch_from_pth(best_pth)
    # fallback：best_epoch 拿不到时，用 eval mAP max 的 ep
    if best_epoch is None and (output_root / "eval").exists():
        best_epoch = _best_epoch_from_eval_max_mAP(output_root / "eval", [])
    # 末 epoch = ckpts 最大号（last.pth 源）
    ckpts = []
    if ckpt_dir.exists():
        for cp in ckpt_dir.glob("checkpoint_epoch_*.pth"):
            mm = re.search(r"checkpoint_epoch_(\d+)", cp.name)
            if mm:
                ckpts.append((int(mm.group(1)), cp))
    last_epoch = max((ep for ep, _ in ckpts), default=None)

    # ── 2. 硬复制（核心，失败即退出，不产半成品 index.yaml）────
    resbag_dir.mkdir(parents=True, exist_ok=True)

    # cfg.yaml
    if cfg_path and cfg_path.exists():
        shutil.copy2(cfg_path, resbag_dir / "cfg.yaml")

    # train.sh（model-train gen 在 experiments/SH/ 下，命名 train_*<model>*.sh）
    sh_candidates = sorted((ROOT / "experiments" / "SH").glob(
        f"train_*{model}*.sh"))
    # 排除 eval 壳
    sh_candidates = [p for p in sh_candidates if "eval" not in p.name]
    if sh_candidates:
        shutil.copy2(sh_candidates[0], resbag_dir / "train.sh")
    else:
        _log(f"train.sh 源缺失：experiments/SH/train_*{model}*.sh", "WARN")

    # best.pth
    if has_best:
        shutil.copy2(best_pth, resbag_dir / "best.pth")

    # last.pth = ckpt 最大 epoch
    if last_epoch is not None:
        shutil.copy2(ckpt_dir / f"checkpoint_epoch_{last_epoch}.pth",
                     resbag_dir / "last.pth")

    # train.log（最新一份 log_train_*.txt）
    logs = sorted(output_root.glob("log_train_*.txt"),
                  key=lambda p: p.stat().st_mtime)
    if logs:
        shutil.copy2(logs[-1], resbag_dir / "train.log")

    # eval_results.json（best_epoch 对应，rglob + 容忍损坏 symlink）
    eval_results_src = None
    if best_epoch is not None and (output_root / "eval").exists():
        for rj in _safe_rglob(output_root / "eval", "results.json"):
            if f"epoch_{best_epoch}" in str(rj):
                eval_results_src = rj
                break
    if eval_results_src:
        shutil.copy2(eval_results_src, resbag_dir / "eval_results.json")

    # asset/ 整树硬复制（保留原原子目录名）
    src_asset = output_root / "asset"
    dst_asset = resbag_dir / "asset"
    if src_asset.exists():
        if dst_asset.exists():
            shutil.rmtree(dst_asset)
        shutil.copytree(src_asset, dst_asset)

    # ── 3. 算 params / flops（独立实现，try 兜底）─────────────
    params_m, trainable_m, flops_g = _compute_params_flops(
        cfg_path, batch_size
    )

    # ── 4. 读 results.json 三类 R40 + 算 mAP ───────────────────
    map_r40 = _extract_map(eval_results_src)

    # ── 5. 收集 commit / seed / optimizer ───────────────────────
    commit = _git_head_short(ROOT)
    seed, optimizer_str = _extract_seed_opt(cfg_path)

    # ── 6. 写 README.md 骨架（LLM 后续填主观段）────────────────
    _write_readme_skeleton(resbag_dir / "README.md",
                            name=name, dataset=dataset, tag=tag,
                            model=model, best_epoch=best_epoch,
                            last_epoch=last_epoch,
                            map_r40=map_r40,
                            params_m=params_m, flops_g=flops_g,
                            commit=commit,
                            has_best=has_best, has_partial=has_partial)

    # ── 7. 组装 index.yaml + model_store.yaml 并写盘 ─────────────
    store_data = {
        "version": 1,
        "folder": name,
        "tag": tag,
        "dataset": dataset,
        "map_r40": map_r40,
        "map_r11": {"car": None, "pedestrian": None, "cyclist": None, "mean": None},
        "params_m": params_m,
        "trainable_m": trainable_m,
        "flops_g": flops_g,
        "metric_caliber": {
            "map": "moderate",
            "recall": "R40",
            "iou": {"car": 0.5, "ped_cyc": 0.25},
            "filter": "EAA",
        },
        "seed": seed,
        "optimizer": optimizer_str,
        "commit": commit,
        "ts": _now_iso(),
        "status": _status(has_best, has_partial,
                           bool(eval_results_src), last_epoch),
        "note": note or (f"best.pth=ep{best_epoch}" if best_epoch else
                       "best.pth missing"),
    }
    if store_data["status"] != "done":
        _log(f"status={store_data['status']}（详见 index.yaml）", "WARN")

    _atomic_write_yaml(resbag_dir / "index.yaml", store_data)
    _atomic_write_yaml(store_path, store_data)
    _log(f"已落袋：{resbag_dir}")
    _log(f"已写总览：{store_path}")


# ── 算 params / flops（独立实现，不依赖 train_pipeline）─────────────
def _compute_params_flops(cfg_path: Path | None, batch_size: int):
    """return (params_m, trainable_m, flops_g)。失败 → (None, None, None)。"""
    if cfg_path is None or not cfg_path.exists():
        return None, None, None
    try:
        import torch
        import numpy as np
        from easydict import EasyDict
        from pcdet.config import cfg_from_yaml_file
        from pcdet.models import build_network
        local_cfg = EasyDict()
        cfg_from_yaml_file(str(cfg_path), local_cfg)
        pcr = local_cfg.DATA_CONFIG.POINT_CLOUD_RANGE
        vs = np.array(local_cfg.DATA_CONFIG.DATA_PROCESSOR[2]["VOXEL_SIZE"])
        gs = np.array([
            int((pcr[3] - pcr[0]) / vs[0]),
            int((pcr[4] - pcr[1]) / vs[1]),
            int((pcr[5] - pcr[2]) / vs[2]),
        ], np.int32)
        class _DS: pass
        ds = _DS()
        ds.class_names = list(local_cfg.CLASS_NAMES)
        ds.point_feature_encoder = _DS()
        ds.point_feature_encoder.num_point_features = 9
        ds.grid_size = gs
        ds.voxel_size = vs
        ds.point_cloud_range = list(pcr)
        net = build_network(model_cfg=local_cfg.MODEL,
                            num_class=len(ds.class_names), dataset=ds)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(dev).eval()
        n_params = sum(p.numel() for p in net.parameters())
        n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
        params_m = round(n_params / 1e6, 4)
        trainable_m = round(n_train / 1e6, 4)
        # thop
        try:
            import thop
            bs = max(1, batch_size)
            M, NPTS = 1000, 32
            NF = ds.point_feature_encoder.num_point_features
            voxels = torch.randn(M, NPTS, NF, device=dev)
            coords = torch.stack([
                torch.randint(0, bs, (M,)),
                torch.zeros(M, dtype=torch.int32),
                torch.randint(0, int(gs[1]), (M,)).to(torch.int32),
                torch.randint(0, int(gs[0]), (M,)).to(torch.int32),
            ], dim=1).to(dev).to(torch.int32)
            bd = {
                "voxels": voxels,
                "voxel_coords": coords,
                "voxel_num_points": torch.full((M,), NPTS, device=dev,
                                              dtype=torch.int32),
                "batch_size": bs,
            }
            macs, _ = thop.profile(net, inputs=(bd,), verbose=False)
            flops_g = round(2 * macs / 1e9, 3)
        except Exception as e:
            _log(f"thop 失败：{type(e).__name__}", "WARN")
            flops_g = None
        return params_m, trainable_m, flops_g
    except Exception as e:
        _log(f"build 网络失败：{type(e).__name__}: {str(e)[:120]}", "WARN")
        return None, None, None


def _read_best_epoch_from_pth(best_pth: Path) -> int | None:
    """从 best.pth（OpenPCDet train.py 标准产物）读 epoch 字段。"""
    try:
        import torch
        ckpt = torch.load(str(best_pth), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            ep = ckpt.get("epoch")
            return int(ep) if ep is not None else None
    except Exception:
        pass
    return None


def _extract_map(eval_results_src: Path | None) -> dict:
    keys = {"car": "Car_3d/moderate_R40",
            "pedestrian": "Pedestrian_3d/moderate_R40",
            "cyclist": "Cyclist_3d/moderate_R40"}
    out = {k: None for k in keys}
    if eval_results_src is None or not eval_results_src.exists():
        return {**out, "mean": None}
    try:
        import json
        d = json.loads(eval_results_src.read_text(encoding="utf-8"))
        ret = d.get("ret_dict", {})
        for k, full_key in keys.items():
            if full_key in ret:
                out[k] = round(float(ret[full_key]), 2)
        valid = [v for v in out.values() if v is not None]
        out["mean"] = round(sum(valid) / 3, 2) if len(valid) == 3 else None
    except Exception as e:
        _log(f"读 results.json 失败：{type(e).__name__}: {e}", "WARN")
    return {**out, "mean": out["mean"]}


def _best_epoch_from_eval_max_mAP(eval_root: Path, candidate_eps: list) -> int | None:
    """byte-match 多解时，扫描所有 results.json 找 pickbest 口径（moderate_R40 mean/cyclist）的 max。"""
    import json
    import re as _re
    if not eval_root.exists():
        return None
    candidates = set(candidate_eps)
    best = (None, -1.0)
    for rj in _safe_rglob(eval_root, "results.json"):
        m = _re.search(r"epoch_(\d+)", str(rj))
        if not m:
            continue
        ep = int(m.group(1))
        if candidate_eps and ep not in candidates:
            continue
        try:
            d = json.loads(rj.read_text(encoding="utf-8", errors="ignore"))
            ret = d.get("ret_dict", {})
            car = ret.get("Car_3d/moderate_R40")
            ped = ret.get("Pedestrian_3d/moderate_R40")
            cyc = ret.get("Cyclist_3d/moderate_R40")
            valid = [v for v in (car, ped, cyc) if v is not None]
            if len(valid) != 3:
                continue
            score = sum(valid) / 3
            if score > best[1]:
                best = (ep, score)
        except Exception:
            continue
    return best[0]


def _safe_rglob(root: Path, name: str) -> list:
    """rglob 容忍损坏 symlink：用 os.walk(followlinks=True) + try/except 隔离。
    rglob 默认 followlinks=False，遇损坏 symlink 直接不进入，丢失整支目录树。
    """
    hits = []
    try:
        for dirpath, dirnames, filenames in os.walk(str(root), followlinks=True):
            for f in filenames:
                if f == name:
                    hits.append(Path(dirpath) / f)
    except Exception as e:
        _log(f"_safe_rglob walk {root} 失败：{type(e).__name__}", "WARN")
    return hits


def _git_head_short(cwd: Path) -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, cwd=str(cwd),
                           timeout=5)
        if r.returncode == 0:
            return r.stdout.strip() or None
    except Exception:
        pass
    return None


def _extract_seed_opt(cfg_path: Path | None):
    if cfg_path is None or not cfg_path.exists():
        return None, None
    try:
        import yaml
        d = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        opt = (d.get("OPTIMIZATION") or {})
        seed = opt.get("FIX_RANDOM_SEED", None)
        seed_repr = f"{seed}" if seed is not None else "False"
        opt_repr = (f"{opt.get('OPTIMIZER', '?')} | LR={opt.get('LR', '?')} | "
                    f"WD={opt.get('WEIGHT_DECAY', '?')} | "
                    f"decay={[int(x) for x in opt.get('DECAY_STEP_LIST', [])]}")
        return seed_repr, opt_repr
    except Exception:
        return None, None


def _status(has_best, has_partial, has_eval, last_epoch):
    """make 永远只能产 done | blocked（spec §3.2）。"""
    if has_partial:
        return "blocked"  # 训练崩溃
    if not has_best or not has_eval or last_epoch is None:
        return "blocked"
    return "done"


def _write_readme_skeleton(path: Path, **ctx):
    """LLM 后续填主观段（结论/已知偏差/复现指引）。骨架含基本字段。"""
    lines = [
        f"# {ctx['name']} 训练报告（resbag skeleton，LLM 填充主观段）",
        "",
        "> **文档定位**：<一句话定位>",
        f"> **数据来源**：`output/train_log/{ctx['dataset']}/{ctx['name']}/`",
        f"> **评估口径**：moderate_R40（VoD EAA）",
        "",
        "## 摘要",
        "",
        f"- 模型 / tag：`{ctx['model']}` / `{ctx['tag']}`",
        f"- best epoch：{ctx.get('best_epoch')}",
        f"- 末 epoch：{ctx.get('last_epoch')}",
        f"- map_r40：{ctx.get('map_r40')}",
        f"- 参数量 / 计算量：{ctx.get('params_m')} M / {ctx.get('flops_g')} GFLOPs",
        f"- commit：{ctx.get('commit')}",
        "",
        "## 结论（LLM 填）",
        "",
        "<best ckpt 当前复测 mAP，与对照的 gap 归因>",
        "",
        "## 已知偏差（LLM 填）",
        "",
        "<结构/数据/评估口径偏差>",
        "",
        "## 复现指引（LLM 填）",
        "",
        f"```bash",
        f"python .claude/skills/resbag/resbag.py make \\",
        f"  --output_root output/train_log/{ctx['dataset']}/{ctx['name']} \\",
        f"  --dataset {ctx['dataset']} --tag {ctx['tag']} --model {ctx['model']} \\",
        f"  --cfg_file <path> --batch_size <N>",
        f"```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ── 跨实验聚合：list ─────────────────────────────────────────────
def cmd_list(args):
    """glob <dataset>/*/model_store.yaml，stdout 输出或写另起文件。"""
    if args.dataset:
        root = ROOT / "output" / "train_log" / args.dataset
    else:
        root = ROOT / "output" / "train_log"
    if not root.exists():
        _log(f"数据集目录不存在：{root}", "WARN")
        return
    entries = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            ms = d / "model_store.yaml"
            if ms.exists():
                entries.append(_load_yaml(ms))
    # 聚合
    summary = {
        "version": 1,
        "dataset": args.dataset,
        "n_experiments": len(entries),
        "experiments": entries,
    }
    import yaml
    out_str = yaml.safe_dump(summary, sort_keys=False, allow_unicode=True)
    if args.o:
        out_path = Path(args.o)
        # 严禁覆盖 model_store.yaml——做路径防御
        if out_path.name == "model_store.yaml":
            _log(f"禁止覆盖 model_store.yaml（list -o 视为另起派生文件）", "ERROR")
            sys.exit(2)
        _atomic_write_yaml(out_path, summary)
        _log(f"已写：{out_path}")
    else:
        print(out_str)


def cmd_show(args):
    if not args.folder:
        _log("show 需 --folder", "ERROR")
        sys.exit(2)
    # 在所有 dataset 下找 folder
    root = ROOT / "output" / "train_log"
    for ds in root.iterdir():
        if not ds.is_dir():
            continue
        ms = ds / args.folder / "model_store.yaml"
        if ms.exists():
            print(ms.read_text(encoding="utf-8"))
            return
    _log(f"未找到 {args.folder}", "ERROR")
    sys.exit(3)


# ── main ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(prog="resbag",
                                 description="训练结果落袋归档")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("make", help="落袋归档（主入口）")
    pm.add_argument("--output_root", required=True)
    pm.add_argument("--dataset", required=True)
    pm.add_argument("--tag", required=True)
    pm.add_argument("--model", required=True,
                    help="模型名（决定 train.sh 源 tools/scripts/train_<model>.sh）")
    pm.add_argument("--cfg_file", required=True)
    pm.add_argument("--batch_size", required=True, type=int)
    pm.add_argument("--note", default="")
    pm.set_defaults(func=cmd_make)

    pl = sub.add_parser("list", help="跨实验总览（聚合）")
    pl.add_argument("--dataset", default=None,
                    help="指定数据集；不指定则扫 output/train_log/*/")
    pl.add_argument("-o", default=None,
                    help="写到另起文件（如 vod/_index.yaml），**禁覆盖 model_store.yaml**")
    pl.set_defaults(func=cmd_list)

    ps = sub.add_parser("show", help="单实验总览")
    ps.add_argument("--folder", required=True)
    ps.set_defaults(func=cmd_show)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
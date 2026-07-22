#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练流水线确定性脚本（把可解析/可模板化的任务从大模型手里拿走）。

子命令：
  gen        按入参从 templates/template.sh 模板渲染 tools/scripts/train_<模型>.sh
  make_shell 造 train_<模型>.sh + 配套 eval_<模型>.sh（落到 tools/scripts/）
  preflight  启动前自检（模型落地/数据集/OUTPUT_ROOT 风格/batch 显存）
  brief      解析训练日志，按固定模板输出 10min 简报（供 crontab 调用）
  pickbest   对末 20 epoch 的 val 结果按 metric 挑 best.pth
  record     聚合 .tmp/ + metric，生成单次实验记录 md
  autofinish 训练结束后自动串跑 val→pickbest→record 收尾链（供训练机 cron 触发，不依赖会话）

设计原则：纯确定性，不调用大模型。LLM 只负责解析用户自然语言入参后调本脚本。
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

CST = timezone(timedelta(hours=8))  # UTC+8

# 工程根：本脚本在 .claude/skills/model-train/ 下，根在三级之上
ROOT = Path(__file__).resolve().parents[3]
# SCRIPTS_DIR = 本 skill 的脚本目录（仅 skill 资产 + templates/ 模板）
SCRIPTS_DIR = Path(__file__).resolve().parent
# SHELLS_DIR = 训练/评测壳产物落点（与 CLAUDE.md「训练/测试脚本放 tools/scripts/」对齐）
SHELLS_DIR = ROOT / 'tools' / 'scripts'
# 模板：去模型语义命名；落到 templates/ 子目录避免被误当执行入口
TEMPLATE = SCRIPTS_DIR / 'templates' / 'template.sh'
TEMPLATE_EVAL = SCRIPTS_DIR / 'templates' / 'template_eval.sh'

# 模板里的 CFG_FILE 默认值——造壳时全局替换这一行（train + eval 两个模板用同一默认）
TEMPLATE_DEFAULT_CFG = 'tools/cfgs/model/vod_models/radarpillar/vod_radarpillar.yaml'


def _validate_model_name(model: str) -> None:
    """模型名用于 train_<model>.sh 文件名，限制为小写字母/数字/下划线/短横线。"""
    if not re.match(r'^[a-z0-9_-]+$', model):
        sys.exit(
            f'[make_shell] 模型名非法 "{model}"：仅允许 [a-z0-9_-]，'
            f'避免路径/正则/换行注入。例：radarpillar / mdfen / point-pillar-v2'
        )


def _shell_path(model: str) -> Path:
    return SHELLS_DIR / f'train_{model}.sh'


def _eval_shell_path(model: str) -> Path:
    return SHELLS_DIR / f'eval_{model}.sh'


# ════════════════════════════════════════════════════════════════
#  make_shell: 造一份 train_<模型>.sh + 配套 eval_<模型>.sh（壳内部改 cfg 默认值）
# ════════════════════════════════════════════════════════════════
def _render_template(template_path: Path, out_path: Path, cfg_file: str, label: str) -> None:
    """通用：读模板 → 替换 CFG_FILE 默认值行 → 写 out_path；模板不存在/regex 不匹配则 sys.exit。"""
    if not template_path.exists():
        sys.exit(f'[make_shell] {label}模板不存在: {template_path}')

    text = template_path.read_text(encoding='utf-8')
    cfg_escaped = re.escape(TEMPLATE_DEFAULT_CFG)
    new_text, n = re.subn(
        rf'CFG_FILE="{cfg_escaped}"',
        f'CFG_FILE="{cfg_file}"',
        text,
    )
    if n == 0:
        sys.exit(
            f'[make_shell] {label}模板里没找到默认 CFG_FILE 行（{TEMPLATE_DEFAULT_CFG}），'
            f'模板结构可能已变更，请手动改 {out_path.name}'
        )

    out_path.write_text(new_text, encoding='utf-8')
    try:
        out_path.chmod(0o755)
    except OSError:
        pass  # Windows WSL 下 chmod 可能无效，不阻塞


def cmd_make_shell(args):
    """从 templates/template.sh 模板造一份 train_<模型>.sh，并同步造 eval_<模型>.sh。

    两份壳都改 CFG_FILE 默认值行（硬编码指向 radarpillar cfg）替换成用户给的 cfg。
    已存在 → 默认跳过；--force 强制覆盖（同时覆盖 train + eval 两份）。
    P2-13 修复: --force 时先 cp 到 .tmp/<日期>/<slug>/backup/ 备份。
    """
    _validate_model_name(args.model)
    train_out = _shell_path(args.model)
    eval_out = _eval_shell_path(args.model)

    if train_out.exists() and not args.force:
        print(f'[make_shell] 已存在 {train_out.relative_to(ROOT)}（未覆盖）')
        print(f'[make_shell] 提示：壳已存在应走 `gen` 改顶部变量；只有想换 cfg 默认值才需要 --force')
        return

    # P2-13 修复: --force 强制覆盖前先备份到 .tmp/<日期>/<slug>/backup/
    if args.force:
        from datetime import timezone, timedelta
        from pathlib import Path as _P
        CST = timezone(timedelta(hours=8))
        today = datetime.now(CST).strftime('%Y-%m-%d')
        # 用模型名做 slug(模型训练任务的天然 slug)
        slug = f'train-{args.model}'
        backup_dir = ROOT / '.tmp' / today / slug / 'backup'
        backup_dir.mkdir(parents=True, exist_ok=True)
        import shutil as _shutil
        if train_out.exists():
            ts = datetime.now(CST).strftime('%Y%m%d%H%M%S')
            bk = backup_dir / f'train_{args.model}_{ts}.sh.bak'
            _shutil.copy2(train_out, bk)
            print(f'[make_shell] 已备份 -> {bk.relative_to(ROOT)}')
        if eval_out.exists():
            ts = datetime.now(CST).strftime('%Y%m%d%H%M%S')
            bk = backup_dir / f'eval_{args.model}_{ts}.sh.bak'
            _shutil.copy2(eval_out, bk)
            print(f'[make_shell] 已备份 -> {bk.relative_to(ROOT)}')

    SHELLS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) train 壳
    _render_template(TEMPLATE, train_out, args.cfg_file, label='train')
    print(f'[make_shell] train 壳已生成 {train_out.relative_to(ROOT)}')

    # 2) eval 壳（同步造，避免 autofinish 时找不到 eval_<model>.sh）
    _render_template(TEMPLATE_EVAL, eval_out, args.cfg_file, label='eval')
    print(f'[make_shell] eval  壳已生成 {eval_out.relative_to(ROOT)}')

    print(f'[make_shell] CFG_FILE 默认值 → {args.cfg_file}')
    print(f'[make_shell] 下一步: 调 `gen --model {args.model} ...` 渲染顶部变量')


# ════════════════════════════════════════════════════════════════
#  gen: 从模板渲染 train_<模型>.sh
# ════════════════════════════════════════════════════════════════
def cmd_gen(args):
    if not TEMPLATE.exists():
        sys.exit(f'[gen] 模板不存在: {TEMPLATE}')

    _validate_model_name(args.model)
    out_sh = _shell_path(args.model)

    # P1-9 修复: USER_CUSTOMIZED 标记 — 防止 gen 静默覆盖用户手调过的壳
    if out_sh.exists():
        existing = out_sh.read_text(encoding='utf-8')
        if re.search(r'#\s*USER_CUSTOMIZED', existing):
            sys.exit(
                f'[gen] {out_sh.name} 含 USER_CUSTOMIZED 标记,'
                f'说明用户手调过该壳,gen 拒绝覆盖。\n'
                f'[gen] 如需强制覆盖,请先手动删除文件中含 `# USER_CUSTOMIZED` 的行,'
                f'或调 `make_shell --force`。'
            )

    # 壳缺 + auto_make_shell（默认 True） → 自动调 make_shell 造壳
    if not out_sh.exists():
        if args.auto_make_shell:
            print(f'[gen] 壳 {out_sh.name} 不存在，自动调 make_shell 造壳')
            cmd_make_shell(argparse.Namespace(
                model=args.model,
                cfg_file=args.cfg_file,
                dataset=args.dataset,
                force=False,
            ))
            if not out_sh.exists():
                sys.exit(f'[gen] 自动造壳失败，{out_sh.name} 仍未生成')
        else:
            sys.exit(
                f'[gen] 壳 {out_sh.name} 不存在；调 make_shell 造壳，'
                f'或加 --auto_make_shell 让 gen 自动造'
            )

    timestamp = datetime.now(CST).strftime('%Y%m%d%H')
    output_root = f'output/train_log/{args.dataset}/{timestamp}_{args.model}_{args.tag}'

    text = out_sh.read_text(encoding='utf-8')

    # 精确替换模板里的「必改」段（键值成对，避免误伤注释）
    def repl(key, value):
        nonlocal text
        # 匹配 KEY="..." 或 KEY=值（含已注释行也改，保证生效）
        text = re.sub(rf'(^{key}=)([^\n]*)',
                      lambda m: f'{m.group(1)}{value}',
                      text, count=1, flags=re.M)

    repl('CFG_FILE', f'"{args.cfg_file}"')
    repl('BATCH_SIZE', str(args.batch_size))
    repl('WORKERS', str(args.workers))
    repl('EPOCHS', str(args.epochs))
    repl('GPU', str(args.gpu))
    repl('EXTRA_TAG', f'"{args.tag}"')
    repl('OUTPUT_ROOT', f'"{output_root}"')

    # 可视化写在 .sh 里（对齐 temp.md 第 6 点）：训练本身不 viz，留 eval 阶段开关注释指引
    if args.visualize:
        text += f'\n# [可视化] 训练后 eval 请设 RUN_VIZ=True，详见 tools/scripts/eval_{args.model}.sh\n'

    out_sh.write_text(text, encoding='utf-8')
    print(f'[gen] 已生成 {out_sh.relative_to(ROOT)}')
    print(f'[gen] OUTPUT_ROOT = {output_root}')
    # 返回结构化结果供 LLM 落 .tmp/
    print(f'[gen] OUTPUT_ROOT_PATH={output_root}')


# ════════════════════════════════════════════════════════════════
#  preflight: 启动前自检（护城河）
# ════════════════════════════════════════════════════════════════
def cmd_preflight(args):
    fails = []

    # (a) cfg 定位 + 存在：若未给 cfg_file 但给了 --model，自动在 cfgs/ 下找
    cfg_file = args.cfg_file
    if not cfg_file and args.model:
        # 在 tools/cfgs/ 下模糊匹配模型名（不依赖大模型 glob）
        cfgs_dir = ROOT / 'tools' / 'cfgs'
        matches = list(cfgs_dir.rglob(f'*{args.model.lower()}*.yaml')) + \
                  list(cfgs_dir.rglob(f'*{args.model}.yaml'))
        matches = [m for m in matches if 'vod' in str(m).lower() or 'dataset' not in str(m).lower()]
        if len(matches) == 1:
            cfg_file = str(matches[0].relative_to(ROOT)).replace('\\', '/')
            print(f'[preflight] 自动定位 cfg: {cfg_file}')
        elif len(matches) > 1:
            fails.append(f'模型 {args.model} 匹配到多个 cfg，需显式指定 --cfg_file: {[str(m.relative_to(ROOT)) for m in matches[:5]]}')
        else:
            fails.append(f'模型 {args.model} 未在 tools/cfgs/ 找到对应 cfg（模型可能未落地）')

    if not cfg_file:
        fails.append('缺少 --cfg_file，且 --model 无法自动定位 cfg')

    cfg_path = ROOT / cfg_file if cfg_file else None
    if cfg_path and not cfg_path.exists():
        fails.append(f'cfg 不存在: {cfg_file}')

    # (a') 模型 detector 类是否注册（cfg 里 NAME 指向的 detector 在 pcdet 里能找到）
    if cfg_path is not None and cfg_path.exists():
        cfg_text = cfg_path.read_text(encoding='utf-8', errors='ignore')
        m = re.search(r'^\s*NAME:\s*(\S+)', cfg_text, re.M)
        if m:
            det_name = m.group(1)
            # 在 pcdet/models/detectors 下找同名文件
            det_dir = ROOT / 'pcdet' / 'models' / 'detectors'
            det_files = list(det_dir.glob('*.py')) if det_dir.exists() else []
            registered = any(det_name.lower() in f.stem.lower() for f in det_files)
            if not registered:
                fails.append(f'detector 类 {det_name} 未在 {det_dir} 找到对应文件，模型可能未落地')

    # (b) 数据集根目录：从 cfg 读 DATA_PATH（dataset_configs 里）或直接查 data/
    data_candidates = list((ROOT / 'data').glob(f'{args.dataset}*')) + \
                      list((ROOT / 'data').glob(f'{args.dataset.upper()}*'))
    if not data_candidates and (ROOT / 'data').exists():
        # 宽松：data 目录存在即算通过，具体路径由 train.py 自行校验
        pass
    elif not (ROOT / 'data').exists():
        fails.append('data/ 目录不存在，数据集无法定位')

    # P1-7 修复: 数据集 symlink 可达性校验 — 防止 dangling symlink 让 preflight 通过但训练崩
    for dc in data_candidates[:3]:
        if dc.is_symlink():
            try:
                target = dc.resolve(strict=True)  # strict=True: 不可达则 raise
                print(f'[preflight] symlink {dc.name} -> {target} OK')
            except (OSError, RuntimeError) as e:
                fails.append(f'data/{dc.name} symlink 不可达: {e};请检查挂载点')

    # P1-7 修复: 显存可行性 — 真实查 nvidia-smi 而非纯提示
    # P1-7 hotfix: 单样本估 1GB 太保守(实际 0.5GB 即可),改为保守 0.8GB/sample,且 bs<=8 不 fail
    try:
        import subprocess
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,memory.total',
                            '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            lines = r.stdout.strip().splitlines()
            if lines:
                free_mb, total_mb = (int(x) for x in lines[0].split(',')) if ',' in lines[0] else (0, 0)
                # 保守估 0.8GB/sample (VoD RadarPillar 实测峰值)
                est_mb = int(args.batch_size * 0.8 * 1024)
                # 阈值放宽到 0.9(留 10% 给其他开销),且仅当 bs>16 才 fail(小 bs 一般 OK)
                if args.batch_size > 16 and est_mb > free_mb * 0.9:
                    fails.append(
                        f'显存不足:batch={args.batch_size} 估需 {est_mb}MB,'
                        f'GPU 空闲 {free_mb}MB (total {total_mb}MB)。'
                        f'请降 batch_size 或换 GPU')
                else:
                    print(f'[preflight] 显存 OK: batch={args.batch_size} 估需 {est_mb}MB ≤ 空闲 {free_mb}MB')
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f'[preflight] [INFO] nvidia-smi 不可用,跳过显存校验')

    # P1-7 修复: 磁盘空间校验 — 防 No space left on device 中途崩
    # P1-7 hotfix: preflight 无 --epochs,默认 80
    try:
        import shutil as _shutil
        free_gb = _shutil.disk_usage(str(ROOT)).free / (1024 ** 3)
        epochs = getattr(args, 'epochs', 80)
        # 估: epochs × ckpt_size(默认 ~300MB) × 1.5(含 tensorboard/eval/log)
        ckpt_size_gb = 0.3
        need_gb = epochs * ckpt_size_gb * 1.5
        if free_gb < need_gb:
            fails.append(
                f'磁盘不足:可写空间 {free_gb:.1f}GB,估需 {need_gb:.1f}GB'
                f'(epochs={epochs} × ckpt + tensorboard)')
        else:
            print(f'[preflight] 磁盘 OK: 可用 {free_gb:.1f}GB ≥ 估需 {need_gb:.1f}GB')
    except Exception as e:
        print(f'[preflight] [INFO] 磁盘校验失败: {e}')

    # (b') 训练壳存在性：缺 → 提示 gen 会自动调 make_shell 造（非阻塞）
    if args.model:
        try:
            _validate_model_name(args.model)
            shell_path = _shell_path(args.model)
            if not shell_path.exists():
                print(f'[preflight] [INFO] 训练壳 {shell_path.relative_to(ROOT)} 不存在')
                print(f'[preflight]       gen 子命令会自动调 make_shell 复制 templates/template.sh + 同步造 eval 壳')
            else:
                print(f'[preflight] 训练壳: {shell_path.relative_to(ROOT)}（已存在，gen 仅渲染顶部变量）')
        except SystemExit:
            pass  # 模型名非法不阻塞 preflight，留给 make_shell/gen 报错

    # (b'') cfg 路径与数据集命名一致性：仅提示，不 fail
    if cfg_file and args.dataset:
        if args.dataset.lower() not in cfg_file.lower():
            print(f'[preflight] [INFO] cfg 路径 {cfg_file} 字符串里不含数据集名 {args.dataset}，确认数据集一致？')

    # (c) OUTPUT_ROOT 命名风格：列既有实验目录
    log_root = ROOT / 'output' / 'train_log' / args.dataset
    existing = sorted([p.name for p in log_root.glob('*') if p.is_dir()]) if log_root.exists() else []
    print(f'[preflight] 既有实验目录（命名风格参考）:')
    for e in existing[-5:]:
        print(f'  {e}')
    if not existing:
        print('  （无，将创建首个）')

    # (d) batch 显存可行性：查既有日志里同模型 batch 先例（宽松提示）
    print(f'[preflight] batch={args.batch_size} 显存可行性：请确认 GPU 显存 >= batch × 单样本占用')

    if fails:
        print('\n[preflight] [FAIL] 自检失败：')
        for f in fails:
            print(f'  - {f}')
        sys.exit(1)
    print('\n[preflight] [OK] 自检通过')


# ════════════════════════════════════════════════════════════════
#  brief: 解析日志输出 10min 简报
# ════════════════════════════════════════════════════════════════
def cmd_brief(args):
    log = Path(args.log)
    if not log.exists():
        sys.exit(f'[brief] 日志不存在: {log}')

    text = log.read_text(encoding='utf-8', errors='ignore')
    lines = text.splitlines()

    # 最后一个 epoch 标记 + 最后一条 loss/lr
    # 训练日志 tqdm 输出形式: "epochs:  33%|███▎  | 33/80 [..]" → 用 (.*) 通配进度条字符
    epoch_re = re.compile(r'epochs?:\s+\d+%\|.{0,40}?\|\s*(\d+)/(\d+)\s*\[', re.I)
    loss_re = re.compile(r'loss[=: ]+([0-9.]+)', re.I)
    lr_re = re.compile(r'lr[=: ]+([0-9.eE-]+)', re.I)
    # NaN/inf：只在 loss 数值语境里判断，避免被 INFO / nanometer / infinity 这种普通词误触
    nan_re = re.compile(r'loss[=: ]+(nan|inf)\b|nan ?loss|got ?nan', re.I)
    oom_re = re.compile(r'out of memory|cuda error|oom', re.I)

    cur_ep, total_ep = None, None
    last_loss, last_lr = None, None
    # 训练日志里 `epoch X/Y` 和 progress bar 用 \r 同行，splitlines 拆完只剩一截
    # 所以先用 splitlines 试一次；如果匹配不到，再按 \r 拆原文兜底（按行 split）
    for line in lines:
        if m := epoch_re.search(line):
            cur_ep, total_ep = int(m.group(1)), int(m.group(2))
        if m := loss_re.search(line):
            last_loss = m.group(1)
        if m := lr_re.search(line):
            last_lr = m.group(1)
    if cur_ep is None:
        # 兜底：按 \r 拆（progress bar 形式），逐行再试一次
        for chunk in text.split('\r'):
            if m := epoch_re.search(chunk):
                cur_ep, total_ep = int(m.group(1)), int(m.group(2))
                break

    has_nan = bool(nan_re.search(text))
    has_oom = bool(oom_re.search(text))

    # ETA：用最后 ckpt 文件的 mtime 估算（粗略）
    eta_str = 'N/A'
    if cur_ep and total_ep and cur_ep > 0:
        # 简单线性：需要单 epoch 耗时——从 ckpt 目录的文件 mtime 差推断
        ckpt_dir = Path(args.output_root) / 'ckpt' if args.output_root else None
        if ckpt_dir and ckpt_dir.exists():
            cks = sorted(ckpt_dir.glob('checkpoint_epoch_*.pth'),
                         key=lambda p: p.stat().st_mtime)
            if len(cks) >= 2:
                dt = cks[-1].stat().st_mtime - cks[-2].stat().st_mtime
                # P2-11 修复: dt 异常(<=0 或 > 单 epoch 阈值)视为不可信
                # 单 epoch 训练通常 30s ~ 30min,>30min 视为 ckpt 写盘卡顿或 NTP 回拨
                if dt <= 0:
                    eta_str = 'N/A (ckpt mtime 异常)'
                elif dt > 1800:
                    eta_str = f'N/A (ckpt 间隔 {dt/60:.0f}min 过大,可能是 ckpt 写盘卡顿)'
                else:
                    remaining = (total_ep - cur_ep) * dt
                    done_at = datetime.now(CST) + timedelta(seconds=remaining)
                    h, m = int(remaining // 3600), int(remaining % 3600 // 60)
                    eta_str = f'{h}h{m}m (≈完成 {done_at.strftime("%m-%d %H:%M")} CST)'
            elif len(cks) == 1:
                # P2-11 修复: 单 ckpt 时按 EPOCHS × 平均 epoch 估算
                # 没历史 epoch 耗时数据,只能标注 N/A 并提示"早期 epoch 等待"
                eta_str = 'N/A (仅 1 ckpt,等待下一个 epoch 后可推断)'

    last_ckpt = 'N/A'
    ckpt_dir = Path(args.output_root) / 'ckpt' if args.output_root else None
    if ckpt_dir and ckpt_dir.exists():
        cks = sorted(ckpt_dir.glob('checkpoint_epoch_*.pth'),
                     key=lambda p: p.stat().st_mtime)
        if cks:
            last_ckpt = cks[-1].name

    now = datetime.now(CST).strftime('%Y-%m-%d %H:%M')
    progress = f'{cur_ep}/{total_ep}' if cur_ep else 'N/A'
    pct = f'（{cur_ep/total_ep*100:.1f}% 进度）' if cur_ep and total_ep else ''

    print(f'[{now} CST] {args.model} ep{progress} | loss={last_loss or "N/A"} lr={last_lr or "N/A"} | ETA={eta_str} | ckpt={last_ckpt}')
    if pct:
        print(f'ep{progress}{pct}，lr {last_lr or "N/A"}，loss={last_loss or "N/A"}')
    print(f'ETA {eta_str}')
    print(f'{"[!] NaN 检测到" if has_nan else "无 NaN"}，{"[!] OOM 检测到" if has_oom else "无 OOM"}，'
          f'{"自愈触发" if (has_nan or has_oom) else "无自愈触发"}')

    # ============ P-partial 修复: NaN/OOM 真实自愈 ============
    # 1) 写 .tmp/<日期>/<slug>/BLOCKED.json 阻塞段(LLM 启动会话时 read-latest 可见)
    # 2) 若 ckpt 已写到当前 epoch 的上一个 → 写 marker + 触发降 batch 重启 shell
    if has_nan or has_oom:
        _self_heal(
            output_root=Path(args.output_root) if args.output_root else None,
            model=args.model,
            log=log,
            cur_ep=cur_ep,
            reason='nan' if has_nan else 'oom',
            last_ckpt=last_ckpt,
        )


def _self_heal(output_root: Path, model: str, log: Path,
               cur_ep: int | None, reason: str, last_ckpt: str):
    """NaN/OOM 自愈:
       a) 写 .tmp/<日期>/<slug>/BLOCKED.json(下次 init_lt_task / show-latest 可见)
       b) 写 .tmp/<date>/<slug>/self_heal.json(机器读触发链,记录降 batch/续 ckpt 决策)
       c) 若 train.py 进程已死 → 尝试用 ckpt+halved batch 重启(产物 shell 自愈脚本)
    """
    today = datetime.now(CST).strftime('%Y-%m-%d')
    # 用 log 的 basename 作 slug(去 .log + path 前缀),保证唯一
    slug = f'train-{model}-self_heal'
    tmp_dir = ROOT / '.tmp' / today / slug
    tmp_dir.mkdir(parents=True, exist_ok=True)

    blocked = tmp_dir / 'BLOCKED.json'
    blocked.write_text(json.dumps({
        'model': model,
        'reason': reason,
        'last_epoch': cur_ep,
        'last_ckpt': last_ckpt,
        'log': str(log),
        'time': datetime.now(CST).strftime('%Y-%m-%d %H:%M:%S CST'),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[self-heal] 阻塞段写入 {blocked.relative_to(ROOT)}')

    # b) 决策:是否自动降 batch 重启
    # 默认:首次自愈静默观察,二次自愈触发降 batch,三次 escalate 升人工
    history = tmp_dir / 'self_heal.json'
    cnt = 0
    if history.exists():
        try:
            cnt = json.loads(history.read_text(encoding='utf-8')).get('count', 0)
        except Exception:
            cnt = 0
    new_cnt = cnt + 1
    history.write_text(json.dumps({
        'count': new_cnt,
        'reason': reason,
        'cur_ep': cur_ep,
        'last_ckpt': last_ckpt,
        'time': datetime.now(CST).strftime('%Y-%m-%d %H:%M:%S CST'),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[self-heal] 自愈计数 = {new_cnt} (3 次后升级人工)')

    if new_cnt >= 3:
        # 三次自愈仍失败 → 阻塞等用户决策
        print(f'[self-heal] [ESCALATE] 3 次自愈未恢复,需人工介入:')
        print(f'[self-heal]   1) tail -f {log} 查最新错误')
        print(f'[self-heal]   2) 检查数据/超参是否合理')
        print(f'[self-heal]   3) 调 cleanup 或 删 {blocked} 后用 resume 重跑')
        return

    # c) 写"降 batch + 续 ckpt 重启"脚本(供用户手跑,不自动)
    # 用户决策:"自行 debug" — 不自动 pkill/sed/cron 触发
    # 脚本写在 .tmp/.../reduce_batch_and_resume.sh,用户自己看自己跑
    if output_root and last_ckpt != 'N/A':
        ckpt_path = output_root / 'ckpt' / last_ckpt
        sh = ROOT / 'tools' / 'scripts' / f'train_{model}.sh'
        new_bs_str = None
        if sh.exists():
            try:
                t = sh.read_text(encoding='utf-8')
                m = re.search(r'^BATCH_SIZE=(\d+)', t, re.M)
                if m:
                    orig = int(m.group(1))
                    halved = max(2, orig // 2)
                    new_bs_str = str(halved)
            except Exception:
                pass

        heal_sh = tmp_dir / 'reduce_batch_and_resume.sh'
        lines = [
            '#!/bin/bash',
            f'# 自愈脚本 — 由 train_pipeline.py brief 在 {datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S CST")} 写入',
            f'# 原因: {reason}, 当前 epoch={cur_ep}, last_ckpt={last_ckpt}',
            f'# 自愈计数: {new_cnt}/3',
            f'# !!! 用户决策:"自行 debug" — 此脚本【不会】自动跑,需要你手动审查+执行',
            '',
            f'cd {ROOT}',
            '',
            '# 1) 备份当前 shell',
            f'cp tools/scripts/train_{model}.sh tools/scripts/train_{model}.sh.bak.{datetime.now(CST).strftime("%Y%m%d%H%M%S")}',
            '',
            f'# 2) 改 BATCH_SIZE={new_bs_str or "??"} (原 batch / 2,下限 2)',
            f'sed -i "s/^BATCH_SIZE=.*/BATCH_SIZE={new_bs_str or "4"}/" tools/scripts/train_{model}.sh',
            '',
            '# 3) 杀掉旧 train.py 进程(若还活着)',
            f'pkill -f "tools/train.py" || true',
            '',
            f'# 4) 续 ckpt 重启(用 {last_ckpt})',
            f'CKPT="{ckpt_path}" bash tools/scripts/train_{model}.sh',
            '',
            f'echo "[self-heal] 已重启,新 batch={new_bs_str or "??"},续 ckpt={ckpt_path}"',
            f'echo "[self-heal] 监控: tail -f {log}"',
        ]
        heal_sh.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        try:
            heal_sh.chmod(0o755)
        except OSError:
            pass
        print(f'[self-heal] 重启脚本已写(需手动跑):{heal_sh.relative_to(ROOT)}')
        print(f'[self-heal]   审查后执行: bash {heal_sh}')
        print(f'[self-heal]   删警报 marker: rm {blocked}')


# ════════════════════════════════════════════════════════════════
#  pickbest: 末 20 epoch val 结果挑 best.pth
# ════════════════════════════════════════════════════════════════
def cmd_pickbest(args):
    output_root = Path(args.output_root)
    # val 结果通常在 output_root/eval/epoch_*/val/*/result.pkl 或 .json
    # 但 tools/test.py 的 --eval_all 模式会把 result 写到 eval/eval_all_default/...
    # （少传 --output_root 时，cfg.EXP_GROUP_PATH 含绝对路径前缀会让路径错位到
    #   output/<abs_cfg_path>/...）。两路都扫一下，兼容历史产物。
    best_epoch, best_metric = None, -1.0
    eval_root = output_root / 'eval'
    if not eval_root.exists():
        sys.exit(f'[pickbest] eval 目录不存在: {eval_root}（val 是否已跑？）')

    # 1) 标准路径：output_root/eval/epoch_<N>/val/<tag>/
    epoch_dirs = sorted(eval_root.glob('epoch_*'))
    # 2) eval_all_default 路径：可能直接在 eval/ 下，也可能错位到 output/<abs_path>/...
    epoch_dirs += sorted(eval_root.glob('eval_all_*/*/*/epoch_*'))
    # 3) 兜底：搜索 ROOT 之外/内的错位路径（cfg.EXP_GROUP_PATH 含 abs 前缀的产物）
    stray = ROOT / 'output' / 'home' / 'admin' / 'projects' / 'RadarPillar' / 'tools' / 'cfgs' / 'model' / 'vod_models' / 'radarnext' / 'vod_radarnext_mdfen' / args.tag if hasattr(args, 'tag') else None
    if stray is None:
        # 用 output_root 末段（rn_mdfen_0717_paper）推断
        stray_tag = output_root.name
        stray = ROOT / 'output' / 'home' / 'admin' / 'projects' / 'RadarPillar' / 'tools' / 'cfgs' / 'model' / 'vod_models' / 'radarnext' / 'vod_radarnext_mdfen' / stray_tag
    if stray.exists():
        epoch_dirs += sorted((stray / 'eval' / 'eval_all_default' / 'default').glob('epoch_*'))

    # 遍历各 epoch 的结果，提取 metric（兼容 result.pkl 文本里的 AP 数字）
    for ep_dir in epoch_dirs:
        m = re.search(r'epoch_(\d+)', ep_dir.name)
        if not m:
            continue
        ep = int(m.group(1))
        if ep < args.start_epoch:
            continue
        # 找结果文件
        result_files = list(ep_dir.rglob('result*')) + list(ep_dir.rglob('*.json'))
        metric = -1.0
        for rf in result_files:
            try:
                content = rf.read_text(encoding='utf-8', errors='ignore')
                # 口径：VoD EAA Car 3D AP moderate_R40（与 RPiN §0.1 锁定的 R40 moderate 一致）
                # 优先锚定 ret_dict 字段 'Car_3d/moderate_R40'（OpenPCDet eval.py:832 存该 key）
                mod_match = re.search(r'Car_3d/moderate_R40[^0-9-]*([0-9.]+)', content)
                if mod_match:
                    metric = max(metric, float(mod_match.group(1)))
                    continue
                # 回退：若结构化字段缺失，匹配 "3d   AP" 后第二列（moderate），而非取 max over 所有难度
                # eval.py 文本格式：3d   AP:{easy},{moderate},{hard}
                m = re.search(r'3d\s+AP:[0-9.]+,\s*([0-9.]+),', content)
                if m:
                    metric = max(metric, float(m.group(1)))
            except Exception:
                continue
        if metric > best_metric:
            best_metric, best_epoch = metric, ep

    if best_epoch is None:
        # P1-6 修复: 无 metric 时 fallback 保留最新 ckpt 作 best.pth
        ckpt_dir = output_root / 'ckpt'
        cks = sorted(ckpt_dir.glob('checkpoint_epoch_*.pth'),
                     key=lambda p: int(re.search(r'(\d+)', p.stem).group(1)))
        if not cks:
            sys.exit(f'[pickbest] 无 metric 且无 ckpt,失败')
        latest_ck = cks[-1]
        best_epoch = int(re.search(r'(\d+)', latest_ck.stem).group(1))
        print(f'[pickbest] [WARN] 未在 >= epoch {args.start_epoch} 找到 val metric')
        print(f'[pickbest] [WARN] fallback 最新 ckpt = epoch {best_epoch} (metric 缺失)')
        # 仍走下面的复制路径

    # 复制 best ckpt
    best_ckpt = output_root / 'ckpt' / f'checkpoint_epoch_{best_epoch}.pth'
    if not best_ckpt.exists():
        sys.exit(f'[pickbest] best ckpt 不存在: {best_ckpt}')
    target = output_root / 'best.pth'
    import shutil
    shutil.copy2(best_ckpt, target)
    print(f'[pickbest] best = epoch {best_epoch} (metric={best_metric:.4f})')
    try:
        print(f'[pickbest] 已复制 -> {target.relative_to(ROOT)}')
    except ValueError:
        # output_root 在 ROOT 之外（如绝对路径），fallback 显示绝对路径
        print(f'[pickbest] 已复制 -> {target}')


# ════════════════════════════════════════════════════════════════
#  record: 生成实验记录 md
# ════════════════════════════════════════════════════════════════
def cmd_record(args):
    output_root = Path(args.output_root)
    tmp = Path(args.tmp_file) if args.tmp_file else None
    timestamp = datetime.now(CST).strftime('%Y%m%d%H')

    # 参考 experiments/RPiN.md 风格：标题 + 配置表 + 结果
    lines = [
        f'# {args.model} 训练记录 ({timestamp}_{args.tag})',
        '',
        f'- 生成时间：{datetime.now(CST).strftime("%Y-%m-%d %H:%M")} CST',
        f'- 数据集：{args.dataset}',
        f'- OUTPUT_ROOT：`{output_root}`',
        '',
        '## 配置',
        f'| 项 | 值 |',
        f'|---|---|',
        f'| cfg | `{args.cfg_file}` |',
        f'| batch / workers / epochs | {args.batch_size} / {args.workers} / {args.epochs} |',
        f'| GPU | {args.gpu} |',
        f'| 备注 | {args.tag} |',
        '',
        '## 结果',
    ]

    # best.pth 信息
    best = output_root / 'best.pth'
    best_epoch_str = ''
    if best.exists():
        try:
            best_rel = best.relative_to(ROOT)
        except ValueError:
            best_rel = best
        lines.append(f'- best.pth：`{best_rel}`（见 pickbest 输出 epoch）')
    else:
        lines.append('- best.pth：未找到（pickbest 是否已跑？）')

    # 末 20 epoch val 结果聚合（口径：Car 3D AP moderate_R40，与 pickbest 一致）
    # 让达标判定有结构化数据源，而非依赖肉眼读日志
    eval_root = output_root / 'eval'
    ap_rows = []
    if eval_root.exists():
        # 兼容多路径：单 ckpt (eval/epoch_N/...) + eval_all_default（eval/eval_all_default/...）
        # + cfg.EXP_GROUP_PATH 含 abs 前缀导致错位的 output/<abs>/eval/eval_all_default/...
        ep_dirs = sorted(eval_root.glob('epoch_*'))
        ep_dirs += sorted(eval_root.glob('eval_all_*/*/*/epoch_*'))
        stray = ROOT / 'output' / 'home' / 'admin' / 'projects' / 'RadarPillar' / 'tools' / 'cfgs' / 'model' / 'vod_models' / 'radarnext' / 'vod_radarnext_mdfen' / output_root.name
        if stray.exists():
            ep_dirs += sorted((stray / 'eval' / 'eval_all_default' / 'default').glob('epoch_*'))
        for ep_dir in ep_dirs:
            m = re.search(r'epoch_(\d+)', ep_dir.name)
            if not m:
                continue
            ep = int(m.group(1))
            result_files = list(ep_dir.rglob('result*')) + list(ep_dir.rglob('*.json'))
            ap_val = None
            for rf in result_files:
                try:
                    content = rf.read_text(encoding='utf-8', errors='ignore')
                    mod_match = re.search(r'Car_3d/moderate_R40[^0-9-]*([0-9.]+)', content)
                    if mod_match:
                        ap_val = float(mod_match.group(1))
                        break
                    mt = re.search(r'3d\s+AP:[0-9.]+,\s*([0-9.]+),', content)
                    if mt:
                        ap_val = float(mt.group(1))
                        break
                except Exception:
                    continue
            if ap_val is not None:
                ap_rows.append((ep, ap_val))
    if ap_rows:
        lines.append('')
        lines.append('## 末段 val 结果（Car 3D AP moderate_R40，VoD EAA）')
        lines.append('| epoch | Car 3D AP |')
        lines.append('|---|---|')
        best_ap = max(r[1] for r in ap_rows)
        for ep, ap_val in ap_rows:
            mark = ' **(best)**' if ap_val == best_ap else ''
            lines.append(f'| {ep} | {ap_val:.4f}{mark} |')
        best_ep = [r[0] for r in ap_rows if r[1] == best_ap][0]
        best_epoch_str = f'（best = epoch {best_ep}, AP {best_ap:.4f}）'
        lines.append(f'- 口径说明：moderate_R40，IoU Car=0.5/Ped-Cyc=0.25，EAA（不做 Driving Corridor 过滤）{best_epoch_str}')

    # 简报历史(从 brief.log / .tmp/<slug>.md 取)
    # P2-12 修复: 同时支持 .tmp/<date>/<slug>/<slug>.md 机读段(由 init 写入)
    brief_log = output_root / 'brief.log'
    tmp_md = ROOT / '.tmp' / datetime.now(CST).strftime('%Y-%m-%d') / args.tag / f'{args.tag}.md'
    # P2-12: --tmp_file 优先;否则尝试 brief.log;再否则尝试 tmp_md
    sources = []
    if tmp and tmp.exists():
        sources.append((tmp, 'brief.log (autofinish 传入)'))
    if brief_log.exists() and (not tmp or brief_log != tmp):
        sources.append((brief_log, 'OUTPUT_ROOT/brief.log'))
    if tmp_md.exists():
        sources.append((tmp_md, '.tmp/<日期>/<slug>/<slug>.md 机读段'))

    if sources:
        lines.append('')
        lines.append('## 简报历史摘要')
        for src, label in sources:
            lines.append(f'### 来源: {label}')
            lines.append('```')
            lines.append(src.read_text(encoding='utf-8', errors='ignore').strip()[:3000])
            lines.append('```')

    out = ROOT / 'experiments' / f'{timestamp}_{args.model}_{args.tag}.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'[record] 已生成 {out.relative_to(ROOT)}')


# autofinish: 训练结束后自动串跑 val→pickbest→record（供训练机 cron 触发）
# 解决 GREEN 复验残留：收尾链不再依赖会话内 LLM poll
# P0-2 修复: subprocess.run 全部用 sys.executable 替代字面量 'python'
# P0-4 修复: 加 timeout + lockfile 防并发 + 0 长度 epoch 防御
def cmd_autofinish(args):
    import subprocess
    import fcntl
    output_root = Path(args.output_root)
    epochs = args.epochs

    # P0-4 修复: epochs < 20 时 start_epoch 不能为负
    start_epoch = max(0, epochs - 20)

    # P0-4 修复: lockfile 防并发 cron 触发
    lock = Path(f'/tmp/autofinish_{args.model}.lock')
    try:
        fd = lock.open('w')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, FileNotFoundError, OSError):
        print(f'[autofinish] 另一个实例在跑（lock={lock}）')
        return

    try:
        return _autofinish_locked(args, output_root, epochs, start_epoch)
    finally:
        try:
            fd.close()
        except Exception:
            pass
        lock.unlink(missing_ok=True)


def _autofinish_locked(args, output_root: Path, epochs: int, start_epoch: int):
    """autofinish 主体 — 加锁后跑。三步都加 timeout + 用 sys.executable。"""
    import subprocess

    # 判定训练是否结束：目标 ckpt 已生成
    target_ckpt = output_root / 'ckpt' / f'checkpoint_epoch_{epochs}.pth'
    if not target_ckpt.exists():
        print(f'[autofinish] 训练未结束（缺 {target_ckpt.name}），跳过')
        return

    print(f'[autofinish] 训练已结束，开始收尾链')

    # 1) 末 20 epoch val：复用 tools/scripts/eval_<模型>.sh 的 all 模式
    eval_sh = _eval_shell_path(args.model)
    if not eval_sh.exists():
        sys.exit(
            f'[autofinish] eval 壳不存在: {eval_sh}；'
            f'先用 make_shell --model {args.model} 造一份'
        )
    # env 覆盖：eval_<模型>.sh 用 :="${VAR:=default}" 读(env 能覆盖默认)
    # P0-5 修复: eval 模板已改成 :="${OUTPUT_ROOT:=default}",env 注入可生效
    cfg_file = args.cfg_file
    eval_env = os.environ.copy()
    eval_env.update({
        'EVAL_MODE': 'all',
        'START_EPOCH': str(start_epoch),
        'CKPT_DIR': f'{output_root}/ckpt',
        'OUTPUT_ROOT': str(output_root),
        'CFG_FILE': cfg_file,
        'EXTRA_TAG': args.tag,
        'BATCH_SIZE': str(args.batch_size),
        'WORKERS': str(args.workers),
        'GPU': str(args.gpu),
    })
    print(f'[autofinish] 1/3 末 20 epoch val (start_epoch={start_epoch})')
    try:
        r = subprocess.run(['bash', str(eval_sh)],
                           env=eval_env, cwd=ROOT, timeout=7200)
    except subprocess.TimeoutExpired:
        sys.exit(f'[autofinish] val 超时（2h）')
    if r.returncode != 0:
        sys.exit(f'[autofinish] val 失败（退出码 {r.returncode}）')

    # 2) pickbest
    print('[autofinish] 2/3 pickbest')
    try:
        r = subprocess.run([sys.executable, str(__file__), 'pickbest',
                            '--output_root', str(output_root),
                            '--start_epoch', str(start_epoch)],
                           cwd=ROOT, timeout=600)
    except subprocess.TimeoutExpired:
        sys.exit(f'[autofinish] pickbest 超时（10min）')
    if r.returncode != 0:
        sys.exit(f'[autofinish] pickbest 失败（退出码 {r.returncode}）')

    # 3) record
    print('[autofinish] 3/3 record')
    brief_log = output_root / 'brief.log'
    rec_args = [sys.executable, str(__file__), 'record',
                '--model', args.model, '--dataset', args.dataset,
                '--cfg_file', args.cfg_file, '--batch_size', str(args.batch_size),
                '--workers', str(args.workers), '--epochs', str(epochs),
                '--gpu', str(args.gpu), '--tag', args.tag,
                '--output_root', str(output_root)]
    if brief_log.exists():
        rec_args += ['--tmp_file', str(brief_log)]
    try:
        r = subprocess.run(rec_args, cwd=ROOT, timeout=300)
    except subprocess.TimeoutExpired:
        sys.exit(f'[autofinish] record 超时（5min）')
    if r.returncode != 0:
        sys.exit(f'[autofinish] record 失败（退出码 {r.returncode}）')

    print('[autofinish] [OK] 收尾链完成（val->pickbest->record）')


# ════════════════════════════════════════════════════════════════
#  register_cron / unregister_cron — P0-3 修复
#  取代 SKILL.md 里 LLM 手抄的 bash 一行;支持去重、注入、写入校验
# ════════════════════════════════════════════════════════════════
def _cron_read() -> str:
    """读 crontab,空时返回空串(不 raise)。"""
    import subprocess
    r = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
    if r.returncode != 0:
        return ''
    return r.stdout


def _cron_write(text: str) -> None:
    """写 crontab,失败 raise。"""
    import subprocess
    r = subprocess.run(['crontab', '-'], input=text, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f'[cron] crontab 写入失败: {r.stderr.strip()}')


def cmd_register_cron(args):
    """注册一条 cron 行,带 SKILL_ID 去重 + 写入校验。

    用法:
      train_pipeline.py register_cron \\
        --id <unique_id> --schedule "*/10 * * * *" \\
        --subcmd brief --log <LOG> \\
        --output_root <OUTPUT_ROOT> [--model <模型>]

    所有 args. 后会被追加到子命令(例如 --model --output_root --log)。
    """
    cron_id = f'skill_id={args.id}'  # 去重 key
    extra = ' '.join(f'--{k} {v}' for k, v in (args.subcmd_arg or []))
    line = (
        f'{args.schedule} cd {ROOT} && '
        f'{sys.executable} {__file__} {args.subcmd} '
        f'--model {args.model} --output_root {args.output_root} '
        f'--log {args.log} {extra} '
        f'>> {args.log}.brief.out 2>&1 '
        f'# {cron_id}'
    )

    # 1) 去重:crontab -l 删旧 SKILL_ID 行
    existing = _cron_read()
    new_lines = [l for l in existing.splitlines() if cron_id not in l]

    # 2) 追加新行
    new_lines.append(line)
    new_cron = '\n'.join(new_lines) + '\n'

    # 3) 写
    _cron_write(new_cron)

    # 4) 写后校验
    verify = _cron_read()
    if cron_id not in verify:
        sys.exit(f'[register_cron] [FAIL] 写入后 grep 校验失败,行未生效')
    print(f'[register_cron] [OK] id={args.id}')
    print(f'[register_cron]   schedule: {args.schedule}')
    print(f'[register_cron]   subcmd:   {args.subcmd} --model {args.model}')


def cmd_unregister_cron(args):
    """删除所有含 SKILL_ID 的 cron 行。"""
    cron_id = f'skill_id={args.id}'
    existing = _cron_read()
    old_lines = existing.splitlines()
    new_lines = [l for l in old_lines if cron_id not in l]
    if len(new_lines) == len(old_lines):
        print(f'[unregister_cron] 未找到 id={args.id},无需删除')
        return
    removed = len(old_lines) - len(new_lines)
    _cron_write('\n'.join(new_lines) + '\n')
    print(f'[unregister_cron] [OK] 删除了 {removed} 条 id={args.id} 的 cron')


def cmd_list_cron(args):
    """列所有含 SKILL_ID 标记的 cron 行(诊断用)。"""
    cron_id = f'skill_id={args.id}' if args.id else 'skill_id='
    existing = _cron_read()
    lines = [l for l in existing.splitlines() if cron_id in l]
    if not lines:
        print(f'[list_cron] 无 id={args.id} 的 cron 行')
        return
    print(f'[list_cron] 共 {len(lines)} 条 id={args.id}:')
    for l in lines:
        print(f'  {l}')


# ════════════════════════════════════════════════════════════════
#  cleanup — P2-14 修复
#  任务完成后清理:.tmp/ 整棵子目录 + crontab 两条 + 列旧 ckpt 供授权删除
# ════════════════════════════════════════════════════════════════
def cmd_cleanup(args):
    """清理任务残留:
       1) 删 .tmp/<日期>/<slug>/ 整棵子目录
       2) 删 crontab 含 skill_id=<slug> 的两条(brief + autofinish)
       3) 列 output/<dataset>/<...>/ckpt/ 中 epoch < best_epoch-5 的旧 ckpt(供用户授权删除)
    """
    from datetime import timezone, timedelta
    import shutil as _shutil
    CST = timezone(timedelta(hours=8))
    today = datetime.now(CST).strftime('%Y-%m-%d')
    slug = args.slug

    # 1) 删 .tmp/<日期>/<slug>/
    task_dir = ROOT / '.tmp' / today / slug
    if task_dir.exists():
        if args.dry_run:
            print(f'[cleanup] DRY-RUN: 准备删除 {task_dir.relative_to(ROOT)}')
        else:
            _shutil.rmtree(task_dir)
            print(f'[cleanup] [OK] 已删 {task_dir.relative_to(ROOT)}')
    else:
        print(f'[cleanup] .tmp/<今日>/{slug} 不存在,跳过')

    # 2) 删 crontab 两条
    if args.cleanup_cron:
        existing = _cron_read()
        old_lines = existing.splitlines()
        new_lines = [l for l in old_lines if f'skill_id={slug}' not in l]
        removed = len(old_lines) - len(new_lines)
        if removed > 0:
            if args.dry_run:
                print(f'[cleanup] DRY-RUN: 准备删除 {removed} 条 skill_id={slug} 的 cron')
            else:
                _cron_write('\n'.join(new_lines) + '\n')
                print(f'[cleanup] [OK] 已删 {removed} 条 skill_id={slug} 的 cron')

    # 3) 列旧 ckpt
    if args.output_root:
        ckpt_dir = Path(args.output_root) / 'ckpt'
        if ckpt_dir.exists():
            cks = sorted(ckpt_dir.glob('checkpoint_epoch_*.pth'),
                         key=lambda p: int(re.search(r'(\d+)', p.stem).group(1)))
            best = Path(args.output_root) / 'best.pth'
            best_epoch = None
            if best.exists():
                m = re.search(r'checkpoint_epoch_(\d+)', best.name + '_')  # best.name 是 best.pth
                # best.pth 是拷贝,不带 epoch。改从 log/record 推断
                # 简化:取最新 ckpt 数为 max_epoch
            if cks:
                max_epoch = int(re.search(r'(\d+)', cks[-1].stem).group(1))
                threshold = max(0, max_epoch - 5)  # 保留最后 5 个 + best
                old = [c for c in cks
                       if int(re.search(r'(\d+)', c.stem).group(1)) < threshold]
                print(f'[cleanup] output_root={args.output_root}')
                print(f'[cleanup]   保留 epoch >= {threshold} + best.pth')
                print(f'[cleanup]   建议删除的旧 ckpt {len(old)} 个:')
                total_size = 0
                for c in old:
                    sz = c.stat().st_size / (1024 ** 3)
                    total_size += sz
                    if args.dry_run:
                        print(f'    [DRY-RUN] {c.name} ({sz:.2f}GB)')
                print(f'[cleanup]   共 {total_size:.1f}GB(实际删除用 `rm <path>` 单独执行)')
                if not args.dry_run:
                    print(f'[cleanup]   注:本子命令不直接 rm 旧 ckpt(防误删),请确认后手工 rm')

    print(f'[cleanup] 完成')


# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_make = sub.add_parser('make_shell', help='造一份 train_<模型>.sh + 配套 eval_<模型>.sh（从 templates/template.sh / template_eval.sh 复制 + 改 CFG_FILE 默认值）')
    p_make.add_argument('--model', required=True, help='模型名（小写字母/数字/下划线/短横线）')
    p_make.add_argument('--dataset', required=True, help='数据集名（仅用于一致性提示）')
    p_make.add_argument('--cfg_file', required=True, help='目标 cfg 路径（替换模板里的 CFG_FILE 默认值）')
    p_make.add_argument('--force', action='store_true', help='已存在时强制覆盖')
    p_make.set_defaults(func=cmd_make_shell)

    p_gen = sub.add_parser('gen', help='从模板渲染 tools/scripts/train_<模型>.sh（壳缺时自动调 make_shell）')
    p_gen.add_argument('--model', required=True)
    p_gen.add_argument('--dataset', required=True)
    p_gen.add_argument('--cfg_file', required=True)
    p_gen.add_argument('--batch_size', type=int, default=16)
    p_gen.add_argument('--workers', type=int, default=2)
    p_gen.add_argument('--epochs', type=int, default=80)
    p_gen.add_argument('--gpu', type=int, default=0)
    p_gen.add_argument('--tag', required=True, help='OUTPUT_ROOT 备注')
    p_gen.add_argument('--visualize', action='store_true')
    p_gen.add_argument('--no_auto_make_shell', dest='auto_make_shell', action='store_false',
                       help='禁用自动造壳；壳缺则 fail')
    p_gen.set_defaults(auto_make_shell=True)
    p_gen.set_defaults(func=cmd_gen)

    p_pre = sub.add_parser('preflight', help='启动前自检')
    p_pre.add_argument('--cfg_file', help='cfg 路径（未给时用 --model 自动定位）')
    p_pre.add_argument('--model', help='模型名（用于自动定位 cfg，未给 --cfg_file 时必需）')
    p_pre.add_argument('--dataset', required=True)
    p_pre.add_argument('--batch_size', type=int, default=16)
    p_pre.set_defaults(func=cmd_preflight)

    p_brief = sub.add_parser('brief', help='10min 简报（crontab 调用）')
    p_brief.add_argument('--model', required=True)
    p_brief.add_argument('--log', required=True, help='训练日志路径')
    p_brief.add_argument('--output_root', help='OUTPUT_ROOT（用于 ckpt/ETA 推断）')
    p_brief.set_defaults(func=cmd_brief)

    p_pb = sub.add_parser('pickbest', help='末 20 epoch val 挑 best.pth')
    p_pb.add_argument('--output_root', required=True)
    p_pb.add_argument('--start_epoch', type=int, required=True, help='= EPOCHS - 20')
    p_pb.set_defaults(func=cmd_pickbest)

    p_rec = sub.add_parser('record', help='生成实验记录 md')
    p_rec.add_argument('--model', required=True)
    p_rec.add_argument('--dataset', required=True)
    p_rec.add_argument('--cfg_file', required=True)
    p_rec.add_argument('--batch_size', type=int, default=16)
    p_rec.add_argument('--workers', type=int, default=2)
    p_rec.add_argument('--epochs', type=int, default=80)
    p_rec.add_argument('--gpu', type=int, default=0)
    p_rec.add_argument('--tag', required=True)
    p_rec.add_argument('--output_root', required=True)
    p_rec.add_argument('--tmp_file', help='简报历史 .tmp 文件')
    p_rec.set_defaults(func=cmd_record)

    p_af = sub.add_parser('autofinish', help='训练结束后自动 val→pickbest→record（调 tools/scripts/eval_<模型>.sh，训练机 cron 触发）')
    p_af.add_argument('--model', required=True)
    p_af.add_argument('--dataset', required=True)
    p_af.add_argument('--cfg_file', required=True)
    p_af.add_argument('--batch_size', type=int, default=16)
    p_af.add_argument('--workers', type=int, default=2)
    p_af.add_argument('--epochs', type=int, default=80)
    p_af.add_argument('--gpu', type=int, default=0)
    p_af.add_argument('--tag', required=True)
    p_af.add_argument('--output_root', required=True)
    p_af.add_argument('--log', help='训练日志路径（可选，用于判定结束）')
    p_af.set_defaults(func=cmd_autofinish)

    # P0-3 修复: 新增 register_cron / unregister_cron / list_cron 子命令
    p_rc = sub.add_parser('register_cron', help='注册一条带 SKILL_ID 的 cron（自动去重 + 写后校验）')
    p_rc.add_argument('--id', required=True, help='唯一 ID(与 --slug 一致)')
    p_rc.add_argument('--schedule', required=True, help='cron 表达式，如 "*/10 * * * *"')
    p_rc.add_argument('--subcmd', required=True, choices=['brief', 'autofinish'])
    p_rc.add_argument('--model', required=True)
    p_rc.add_argument('--output_root', required=True)
    p_rc.add_argument('--log', required=True, help='训练日志路径')
    p_rc.add_argument('--subcmd_arg', action='append', default=[],
                      help='额外参数注入,格式 --subcmd_arg key=value (可多次)')
    p_rc.set_defaults(func=cmd_register_cron)

    p_urc = sub.add_parser('unregister_cron', help='删除所有含 SKILL_ID 的 cron 行')
    p_urc.add_argument('--id', required=True)
    p_urc.set_defaults(func=cmd_unregister_cron)

    p_lc = sub.add_parser('list_cron', help='列所有含 SKILL_ID 的 cron 行(诊断)')
    p_lc.add_argument('--id', required=True)
    p_lc.set_defaults(func=cmd_list_cron)

    # (P-partial 修复已收敛到 brief 内的 BLOCKED.json 警报 + reduce_batch_and_resume.sh 脚本)
    # 不注册自动自愈 cron/pkill —— 用户决策:"自行 debug",只报告不自动重启

    # P2-14 修复: 新增 cleanup 子命令
    p_cl = sub.add_parser('cleanup', help='任务完成后清理:.tmp + crontab + 列旧 ckpt')
    p_cl.add_argument('--slug', required=True, help='任务 slug(与 register_cron --id 一致)')
    p_cl.add_argument('--output_root', help='OUTPUT_ROOT(列旧 ckpt 用)')
    p_cl.add_argument('--cleanup_cron', action='store_true', help='同时清理 cron 行')
    p_cl.add_argument('--dry_run', action='store_true', help='只看不动')
    p_cl.set_defaults(func=cmd_cleanup)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

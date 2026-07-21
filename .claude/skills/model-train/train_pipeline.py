#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练流水线确定性脚本（把可解析/可模板化的任务从大模型手里拿走）。

子命令：
  gen        按入参从 train_radarpillar.sh 模板渲染 train_<模型>.sh
  preflight  启动前自检（模型落地/数据集/OUTPUT_ROOT 风格/batch 显存）
  brief      解析训练日志，按固定模板输出 10min 简报（供 crontab 调用）
  pickbest   对末 20 epoch 的 val 结果按 metric 挑 best.pth
  record     聚合 .tmp/ + metric，生成单次实验记录 md
  autofinish 训练结束后自动串跑 val→pickbest→record 收尾链（供训练机 cron 触发，不依赖会话）

设计原则：纯确定性，不调用大模型。LLM 只负责解析用户自然语言入参后调本脚本。
"""
import argparse
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

CST = timezone(timedelta(hours=8))  # UTC+8

# 工程根：本脚本在 .claude/skills/model-train/ 下，根在三级之上
ROOT = Path(__file__).resolve().parents[3]
# SCRIPTS_DIR = 本 skill 的脚本目录（脚本同目录），模板/生成脚本都放这里
SCRIPTS_DIR = Path(__file__).resolve().parent
TEMPLATE = SCRIPTS_DIR / 'train_radarpillar.sh'

# 模板里的 CFG_FILE 默认值——造壳时全局替换这一行
TEMPLATE_DEFAULT_CFG = 'tools/cfgs/model/vod_models/radarpillar/vod_radarpillar.yaml'


def _validate_model_name(model: str) -> None:
    """模型名用于 train_<model>.sh 文件名，限制为小写字母/数字/下划线/短横线。"""
    if not re.match(r'^[a-z0-9_-]+$', model):
        sys.exit(
            f'[make_shell] 模型名非法 "{model}"：仅允许 [a-z0-9_-]，'
            f'避免路径/正则/换行注入。例：radarpillar / mdfen / point-pillar-v2'
        )


def _shell_path(model: str) -> Path:
    return SCRIPTS_DIR / f'train_{model}.sh'


# ════════════════════════════════════════════════════════════════
#  make_shell: 造一份 train_<模型>.sh（壳内部改 cfg 默认值）
# ════════════════════════════════════════════════════════════════
def cmd_make_shell(args):
    """从 train_radarpillar.sh 模板造一份 train_<模型>.sh。

    模板复制 + 把 CFG_FILE 默认值行（硬编码指向 radarpillar cfg）替换成用户给的 cfg。
    已存在 → 默认跳过；--force 强制覆盖。
    """
    _validate_model_name(args.model)
    out = _shell_path(args.model)

    if out.exists() and not args.force:
        print(f'[make_shell] 已存在 {out.relative_to(ROOT)}（未覆盖）')
        print(f'[make_shell] 提示：壳已存在应走 `gen` 改顶部变量；只有想换 cfg 默认值才需要 --force')
        return

    if not TEMPLATE.exists():
        sys.exit(f'[make_shell] 模板不存在: {TEMPLATE}')

    text = TEMPLATE.read_text(encoding='utf-8')

    # 替换模板里的 CFG_FILE 默认值行（全局，避免漏改）
    cfg_escaped = re.escape(TEMPLATE_DEFAULT_CFG)
    new_text, n = re.subn(rf'CFG_FILE="{cfg_escaped}"', f'CFG_FILE="{args.cfg_file}"', text)
    if n == 0:
        sys.exit(
            f'[make_shell] 模板里没找到默认 CFG_FILE 行（{TEMPLATE_DEFAULT_CFG}），'
            f'模板结构可能已变更，请手动改 train_<模型>.sh'
        )

    out.write_text(new_text, encoding='utf-8')
    try:
        out.chmod(0o755)
    except OSError:
        pass  # Windows WSL 下 chmod 可能无效，不阻塞

    print(f'[make_shell] 已生成 {out.relative_to(ROOT)}')
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
        text += f'\n# [可视化] 训练后 eval 请设 RUN_VIZ=True，详见 eval_radarpillar.sh\n'

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
    if cfg_path.exists():
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

    # (b') 训练壳存在性：缺 → 提示 gen 会自动调 make_shell 造（非阻塞）
    if args.model:
        try:
            _validate_model_name(args.model)
            shell_path = _shell_path(args.model)
            if not shell_path.exists():
                print(f'[preflight] [INFO] 训练壳 {shell_path.relative_to(ROOT)} 不存在')
                print(f'[preflight]       gen 子命令会自动调 make_shell 复制模板并改 CFG_FILE 默认值')
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
    epoch_re = re.compile(r'epoch\s+(\d+)/(\d+)', re.I)
    loss_re = re.compile(r'loss[=: ]+([0-9.]+)', re.I)
    lr_re = re.compile(r'lr[=: ]+([0-9.eE-]+)', re.I)
    nan_re = re.compile(r'\bnan\b|nanloss|inf', re.I)
    oom_re = re.compile(r'out of memory|cuda error|oom', re.I)

    cur_ep, total_ep = None, None
    last_loss, last_lr = None, None
    for line in lines:
        if m := epoch_re.search(line):
            cur_ep, total_ep = int(m.group(1)), int(m.group(2))
        if m := loss_re.search(line):
            last_loss = m.group(1)
        if m := lr_re.search(line):
            last_lr = m.group(1)

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
                remaining = (total_ep - cur_ep) * dt
                done_at = datetime.now(CST) + timedelta(seconds=remaining)
                h, m = int(remaining // 3600), int(remaining % 3600 // 60)
                eta_str = f'{h}h{m}m (≈完成 {done_at.strftime("%m-%d %H:%M")} CST)'

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


# ════════════════════════════════════════════════════════════════
#  pickbest: 末 20 epoch val 结果挑 best.pth
# ════════════════════════════════════════════════════════════════
def cmd_pickbest(args):
    output_root = Path(args.output_root)
    # val 结果通常在 output_root/eval/epoch_*/val/*/result.pkl 或 .json
    # OpenPCDet 把每类 AP 写进 result，主指标取 Car AP (R40) 或首个 class
    best_epoch, best_metric = None, -1.0
    eval_root = output_root / 'eval'
    if not eval_root.exists():
        sys.exit(f'[pickbest] eval 目录不存在: {eval_root}（val 是否已跑？）')

    # 遍历各 epoch 的结果，提取 metric（兼容 result.pkl 文本里的 AP 数字）
    for ep_dir in sorted(eval_root.glob('epoch_*')):
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
        sys.exit(f'[pickbest] 未在 >= epoch {args.start_epoch} 找到任何 val 结果')

    # 复制 best ckpt
    best_ckpt = output_root / 'ckpt' / f'checkpoint_epoch_{best_epoch}.pth'
    if not best_ckpt.exists():
        sys.exit(f'[pickbest] best ckpt 不存在: {best_ckpt}')
    target = output_root / 'best.pth'
    import shutil
    shutil.copy2(best_ckpt, target)
    print(f'[pickbest] best = epoch {best_epoch} (metric={best_metric:.4f})')
    print(f'[pickbest] 已复制 -> {target.relative_to(ROOT)}')


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
        lines.append(f'- best.pth：`{best.relative_to(ROOT)}`（见 pickbest 输出 epoch）')
    else:
        lines.append('- best.pth：未找到（pickbest 是否已跑？）')

    # 末 20 epoch val 结果聚合（口径：Car 3D AP moderate_R40，与 pickbest 一致）
    # 让达标判定有结构化数据源，而非依赖肉眼读日志
    eval_root = output_root / 'eval'
    ap_rows = []
    if eval_root.exists():
        for ep_dir in sorted(eval_root.glob('epoch_*')):
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

    # 简报历史（从 .tmp/ 摘要）
    if tmp and tmp.exists():
        lines.append('')
        lines.append('## 简报历史摘要')
        lines.append('```')
        lines.append(tmp.read_text(encoding='utf-8', errors='ignore').strip())
        lines.append('```')

    out = ROOT / 'experiments' / f'{timestamp}_{args.model}_{args.tag}.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'[record] 已生成 {out.relative_to(ROOT)}')


# autofinish: 训练结束后自动串跑 val→pickbest→record（供训练机 cron 触发）
# 解决 GREEN 复验残留：收尾链不再依赖会话内 LLM poll
def cmd_autofinish(args):
    import subprocess
    output_root = Path(args.output_root)
    epochs = args.epochs
    start_epoch = epochs - 20

    # 判定训练是否结束：目标 ckpt 已生成
    target_ckpt = output_root / 'ckpt' / f'checkpoint_epoch_{epochs}.pth'
    if not target_ckpt.exists():
        print(f'[autofinish] 训练未结束（缺 {target_ckpt.name}），跳过')
        return

    print(f'[autofinish] 训练已结束，开始收尾链')

    # 1) 末 20 epoch val：复用 eval_radarpillar.sh 的 all 模式
    eval_sh = SCRIPTS_DIR / 'eval_radarpillar.sh'
    eval_env = f'EVAL_MODE=all START_EPOCH={start_epoch} CKPT_DIR={output_root}/ckpt OUTPUT_ROOT={output_root}'
    print(f'[autofinish] 1/3 末 20 epoch val')
    r = subprocess.run(f'cd {ROOT} && {eval_env} bash {eval_sh}', shell=True)
    if r.returncode != 0:
        sys.exit(f'[autofinish] val 失败（退出码 {r.returncode}）')

    # 2) pickbest
    print('[autofinish] 2/3 pickbest')
    r = subprocess.run(['python', str(__file__), 'pickbest',
                        '--output_root', str(output_root), '--start_epoch', str(start_epoch)])
    if r.returncode != 0:
        sys.exit(f'[autofinish] pickbest 失败（退出码 {r.returncode}）')

    # 3) record
    print('[autofinish] 3/3 record')
    brief_log = output_root / 'brief.log'
    rec_args = ['python', str(__file__), 'record',
                '--model', args.model, '--dataset', args.dataset,
                '--cfg_file', args.cfg_file, '--batch_size', str(args.batch_size),
                '--workers', str(args.workers), '--epochs', str(epochs),
                '--gpu', str(args.gpu), '--tag', args.tag,
                '--output_root', str(output_root)]
    if brief_log.exists():
        rec_args += ['--tmp_file', str(brief_log)]
    r = subprocess.run(rec_args)
    if r.returncode != 0:
        sys.exit(f'[autofinish] record 失败（退出码 {r.returncode}）')

    print('[autofinish] [OK] 收尾链完成（val->pickbest->record）')


# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_make = sub.add_parser('make_shell', help='造一份 train_<模型>.sh（从 train_radarpillar.sh 复制 + 改 CFG_FILE 默认值）')
    p_make.add_argument('--model', required=True, help='模型名（小写字母/数字/下划线/短横线）')
    p_make.add_argument('--dataset', required=True, help='数据集名（仅用于一致性提示）')
    p_make.add_argument('--cfg_file', required=True, help='目标 cfg 路径（替换模板里的 CFG_FILE 默认值）')
    p_make.add_argument('--force', action='store_true', help='已存在时强制覆盖')
    p_make.set_defaults(func=cmd_make_shell)

    p_gen = sub.add_parser('gen', help='从模板渲染 train_<模型>.sh（壳缺时自动调 make_shell）')
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

    p_af = sub.add_parser('autofinish', help='训练结束后自动 val→pickbest→record（训练机 cron 触发）')
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

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

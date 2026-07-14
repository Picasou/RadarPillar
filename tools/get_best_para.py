import argparse
import csv
import datetime
import json
import os
import time
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from utils.train_utils.optimization import build_optimizer


def parse_config():
    # 与 train.py 保持一致的命令行参数；额外追加 sweep 相关参数
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='whether to use wandb')
    parser.add_argument('--skip_eval', action='store_true', default=False, help='skip the post-training evaluation')

    # sweep 相关参数
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=None,
                        help='batch sizes to sweep (默认围绕 --batch_size / cfg 取 ±2 倍)')
    parser.add_argument('--worker_list', type=int, nargs='+', default=None,
                        help='num_workers 候选列表 (默认 [0, 2, 4, 8, 16, cpu_count])')
    parser.add_argument('--warmup_iters', type=int, default=10,
                        help='每组配置在计时前运行的 warmup 迭代数')
    parser.add_argument('--timed_iters', type=int, default=50,
                        help='每组配置实际计时的迭代数 (超过则取 epoch 长度)')
    parser.add_argument('--allow_oom', action='store_true', default=False,
                        help='CUDA OOM 时是否继续 sweep 后续配置')
    parser.add_argument('--seed', type=int, default=666, help='随机种子')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def resolve_sweep_ranges(args, cfg):
    """确定 batch_sizes 与 worker_list 的候选集合。"""
    base_bs = args.batch_size if args.batch_size is not None else cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    if args.batch_sizes is None or len(args.batch_sizes) == 0:
        # 默认: base_bs 附近的 2 的幂
        batch_sizes = sorted(set([
            max(1, base_bs // 2),
            base_bs,
            base_bs * 2,
            base_bs * 4,
        ]))
    else:
        batch_sizes = sorted(set(args.batch_sizes))

    if args.worker_list is None or len(args.worker_list) == 0:
        cpu_count = os.cpu_count() or 4
        candidates = [0, 2, 4, 8, 16, cpu_count, args.workers]
        worker_list = sorted(set(w for w in candidates if w >= 0))
    else:
        worker_list = sorted(set(args.worker_list))

    return batch_sizes, worker_list


def _is_oom(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return 'out of memory' in msg or 'cuda oom' in msg


def _free_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_config(model, train_loader, optimizer, model_func, grad_norm_clip,
                     batch_size, num_workers, warmup_iters, timed_iters, logger):
    """对单组 (batch_size, num_workers) 运行 warmup + 计时, 返回吞吐统计或错误信息。"""
    _free_cuda()
    model.train()

    grad_clip = grad_norm_clip  # 避免重复 getattr

    def _train_step(batch):
        optimizer.zero_grad()
        ret = model_func(model, batch)
        loss = ret.loss
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    # warmup: 让 DataLoader workers 启动 + cudnn benchmark 完成首次 autotune
    try:
        it = iter(train_loader)
        for _ in range(min(warmup_iters, len(train_loader))):
            batch = next(it)
            _train_step(batch)
    except StopIteration:
        pass
    except Exception as e:
        _free_cuda()
        if _is_oom(e):
            return None, f'OOM during warmup: {e}'
        raise

    # 计时
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    actual_iters = min(timed_iters, len(train_loader))
    if actual_iters <= 0:
        return None, 'Empty dataloader'

    try:
        it = iter(train_loader)
        t_start = time.perf_counter()
        for _ in range(actual_iters):
            batch = next(it)
            _train_step(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()
    except StopIteration:
        return None, 'Dataloader exhausted before timed_iters completed'
    except Exception as e:
        _free_cuda()
        if _is_oom(e):
            return None, f'OOM during timing: {e}'
        raise

    elapsed = t_end - t_start
    iters_per_sec = actual_iters / elapsed
    return {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'iters': actual_iters,
        'elapsed_s': elapsed,
        'iters_per_sec': iters_per_sec,
        'samples_per_sec': actual_iters * batch_size / elapsed,
        'avg_iter_ms': elapsed * 1000.0 / actual_iters,
    }, None


def main():
    args, cfg = parse_config()

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / ('log_get_best_para_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start get_best_para**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    if args.fix_random_seed or cfg.OPTIMIZATION.get('FIX_RANDOM_SEED', True):
        common_utils.set_random_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    batch_sizes, worker_list = resolve_sweep_ranges(args, cfg)
    logger.info(f'Sweep ranges: batch_sizes={batch_sizes}, worker_list={worker_list}')

    grad_norm_clip = getattr(cfg.OPTIMIZATION, 'GRAD_NORM_CLIP', None)

    # 先用一组 (bs, nw) 拿到 train_set 以构建模型
    seed_bs = batch_sizes[0]
    seed_nw = worker_list[0]
    logger.info(f'Building initial dataloader (bs={seed_bs}, nw={seed_nw}) to obtain dataset for model...')
    train_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=seed_bs,
        dist=False, workers=seed_nw,
        logger=logger, training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=1,
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model_func = model_fn_decorator()

    # sweep
    results = []
    best = None
    oom_seen = False
    total = len(batch_sizes) * len(worker_list)
    idx = 0
    for batch_size, num_workers in product(batch_sizes, worker_list):
        idx += 1
        logger.info(f'[{idx}/{total}] Try bs={batch_size}, nw={num_workers}')
        try:
            _, train_loader, _ = build_dataloader(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                batch_size=batch_size,
                dist=False, workers=num_workers,
                logger=logger, training=True,
                merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
                total_epochs=1,
            )
        except Exception as e:
            logger.info(f'  Failed to build dataloader: {e}')
            continue

        optimizer = build_optimizer(model, cfg.OPTIMIZATION)
        stats, err = benchmark_config(
            model, train_loader, optimizer, model_func, grad_norm_clip,
            batch_size, num_workers, args.warmup_iters, args.timed_iters, logger,
        )
        if err is not None:
            logger.info(f'  {err}')
            if 'OOM' in err:
                oom_seen = True
                if not args.allow_oom:
                    logger.info('  Stop sweep on first OOM (use --allow_oom to continue).')
                    # 释放当前 loader 占用的资源
                    del train_loader, optimizer
                    _free_cuda()
                    break
            del train_loader, optimizer
            _free_cuda()
            continue

        logger.info(
            f'  iters/s={stats["iters_per_sec"]:.3f}  '
            f'samples/s={stats["samples_per_sec"]:.1f}  '
            f'avg_iter={stats["avg_iter_ms"]:.1f}ms  '
            f'iters={stats["iters"]}'
        )
        results.append(stats)

        if best is None or stats['samples_per_sec'] > best['samples_per_sec']:
            best = stats

        # 释放本组 dataloader / optimizer, 避免 workers 累积
        del train_loader, optimizer
        _free_cuda()

    # 汇总
    summary_path = output_dir / 'get_best_para_summary.json'
    csv_path = output_dir / 'get_best_para_summary.csv'

    sorted_results = sorted(results, key=lambda x: -x['samples_per_sec'])

    logger.info('**********************Results (sorted by samples/s desc)**********************')
    logger.info(f"{'bs':>4} {'nw':>4} {'iters/s':>10} {'samples/s':>12} {'avg_iter_ms':>14}")
    for r in sorted_results:
        logger.info(
            f"{r['batch_size']:>4} {r['num_workers']:>4} "
            f"{r['iters_per_sec']:>10.3f} {r['samples_per_sec']:>12.1f} {r['avg_iter_ms']:>14.1f}"
        )

    if best is not None:
        logger.info('**********************Best**********************')
        logger.info(
            f"batch_size={best['batch_size']}, num_workers={best['num_workers']} "
            f"({best['samples_per_sec']:.1f} samples/s, {best['iters_per_sec']:.3f} iters/s)"
        )
        logger.info('Recommended command-line flags for train.py:')
        logger.info(f'  --batch_size {best["batch_size"]} --workers {best["num_workers"]}')
    else:
        logger.warning('No successful configurations completed.')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'batch_size', 'num_workers', 'iters', 'elapsed_s',
            'iters_per_sec', 'samples_per_sec', 'avg_iter_ms',
        ])
        writer.writeheader()
        for r in sorted_results:
            writer.writerow(r)

    with open(summary_path, 'w') as f:
        json.dump({
            'cfg_file': args.cfg_file,
            'extra_tag': args.extra_tag,
            'sweep': {
                'batch_sizes': batch_sizes,
                'worker_list': worker_list,
                'warmup_iters': args.warmup_iters,
                'timed_iters': args.timed_iters,
            },
            'best': best,
            'results': sorted_results,
            'oom_seen': oom_seen,
        }, f, indent=2)

    logger.info(f'Summary CSV  -> {csv_path}')
    logger.info(f'Summary JSON -> {summary_path}')
    logger.info('**********************End get_best_para**********************')


if __name__ == '__main__':
    main()
# python tools/get_best_para.py --cfg_file tools/cfgs/model/vod_models/vod_radarpillar.yaml
#  --batch_sizes 4 8 16 32 --worker_list 4 8 12 16 --timed_iters 100 --warmup_iters 20
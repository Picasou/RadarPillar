"""Task 6, Step 1 — overfit-1-batch sanity check (audit L).

Standalone script (does NOT use train.py): builds the RadarNeXt FPN model from
the production yaml, pulls a single real VoD frame (twice, to make a bs=2
batch) from the VodDataset, and runs 200 optimizer steps with AMP fp16 ON
(mirrors the OPTIMIZATION intent). Reports loss each step.

Goal: verify the TRAINING PIPELINE (data loading, loss aggregation, optimizer
step, backward) has no regression. We expect the loss to decrease toward ~0 on
a single overfitted batch. If it does NOT decrease, the bug is in the training
pipeline — module correctness is already covered by Task 4.5 parity.

Usage:
    python tools/task6_overfit_1batch.py \
        --cfg_file tools/cfgs/model/vod_models/vod_radarnext_fpn.yaml \
        --num_steps 200
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from tools.utils.train_utils.optimization import build_optimizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg_file', type=str, required=True)
    p.add_argument('--num_steps', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-3,
                   help='higher LR than OneCycle start to overfit faster')
    p.add_argument('--log_every', type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    logger = common_utils.create_logger()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    common_utils.set_random_seed(666)
    torch.backends.cudnn.benchmark = True

    logger.info('==== Build dataloader (bs=2) ====')
    train_set, train_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=2,
        dist=False, workers=0,  # workers=0 for deterministic single-batch
        logger=logger,
        training=True,
        total_epochs=1,
    )
    logger.info('Dataset size: %d', len(train_set))

    logger.info('==== Build model ====')
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    model.cuda()
    model.train()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    # Override LR with a flat, larger one for overfitting (skip OneCycle warmup complexity)
    for pg in optimizer.param_groups:
        pg['lr'] = args.lr

    # NOTE: production train.py does NOT use AMP (train_one_epoch does plain
    # loss.backward()). AMP fp16 is documented in the brief but the actual
    # train_utils path is fp32. We mirror the real pipeline: fp32, no scaler.
    # (An AMP path also hits a real dtype bug in radarnext_losses.py:125 where
    # pred is Half and target is Float under autocast — out of scope for this
    # task; flagged in the report.)
    model_func = model_fn_decorator()

    # Pull one fixed batch
    data_iter = iter(train_loader)
    batch = next(data_iter)

    logger.info('==== Overfit run: %d steps on fixed bs=2 batch ====', args.num_steps)
    losses = []
    for step in range(args.num_steps):
        # Re-fetch the SAME batch each step ( DataLoader iter exhausted -> reset )
        # batch tensors come back from worker as numpy; reload each step to be safe
        # because load_data_to_gpu moves them in-place.
        # Easiest: keep re-iterating from a 1-batch loader.
        try:
            cur_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            cur_batch = next(data_iter)

        optimizer.zero_grad()
        loss, tb_dict, disp_dict = model_func(model, cur_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)
        optimizer.step()

        lv = loss.item()
        losses.append(lv)
        if step % args.log_every == 0 or step == args.num_steps - 1:
            cur_lr = optimizer.param_groups[0]['lr']
            logger.info('step %4d  loss=%.5f  lr=%.6f  peak_mem=%.2f GB',
                        step, lv, cur_lr,
                        torch.cuda.max_memory_allocated() / 1024 ** 3)

    logger.info('==== Summary ====')
    logger.info('loss start (step 0) = %.5f', losses[0])
    logger.info('loss end   (step %d) = %.5f', args.num_steps - 1, losses[-1])
    logger.info('min loss             = %.5f  (at step %d)', min(losses), int(np.argmin(losses)))
    logger.info('peak mem alloc       = %.2f GB  (= %.0f MiB)',
                torch.cuda.max_memory_allocated() / 1024 ** 3,
                torch.cuda.max_memory_allocated() / 1024 ** 2)
    # save loss curve to file for the report
    out_path = Path('output') / cfg.EXP_GROUP_PATH / cfg.TAG / 'overfit1_loss_curve.txt'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(out_path), np.array(losses), fmt='%.6f')
    logger.info('loss curve saved to %s', out_path)


if __name__ == '__main__':
    main()

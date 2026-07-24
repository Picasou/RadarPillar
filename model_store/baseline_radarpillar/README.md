# baseline_radarpillar

RadarPillar 阶段 0 锚点（baseline val 锚点 0-3）。

## 1. 基本信息

| 项 | 值 |
|---|---|
| tag | `stage0_base_rpillar` |
| 主计划章节 | §5（阶段 0）/ §5.3 Task 0-3 |
| 落袋路径 | `/home/dministrator1/RadarPillar/model_store/baseline_radarpillar` |
| 训练时间 | 2026-07-10 |
| 训练机 env | `angle`（CUDA_VISIBLE_DEVICES=0） |
| 来源产物 | `output/train_log/vod/radarpillar_base/` |

## 2. 模型架构

| 组件 | 名称 | 关键配置 |
|---|---|---|
| model | `PointPillar` | — |
| 3D backbone | `PillarAttention` | ATTN_CHANNELS=32, NUM_HEADS=1, FFN_CHANNELS=32, USE_LAYER_NORM=True |
| VFE | `PillarVFE` | NUM_FILTERS=[32], USE_VELOCITY_DECOMPOSITION=True |
| 2D backbone | `BaseBEVBackbone` | LAYER_NUMS=[3,5,5], STRIDES=[2,2,2], NUM_FILTERS=[32,32,32] |
| head | `AnchorHeadSingle` | class_agnostic=False, USE_DIRECTION_CLASSIFIER=True, NUM_DIR_BINS=2 |

## 3. 训练参数

| 项 | 值 |
|---|---|
| optimizer | `adam_onecycle` |
| lr | 0.003 |
| weight_decay | 0.01 |
| BATCH_SIZE_PER_GPU | 16 |
| WORKERS | 2 |
| EPOCHS | 80 |
| eval_interval | 1 |
| early_stop | R40 moderate, patience=30, start_epoch=10 |
| early_stop.metric_weights | [0.2, 0.3, 0.5] |
| early_stop.mode | max |

## 4. mAP（最终评估 / epoch 56，early_stop 触发）

### 3D R11+R40 moderate

| 类别 | R11_3D | R40_3D |
|---|---|---|
| Car | 39.25 | 36.18 |
| Pedestrian | 42.37 | 40.76 |
| Cyclist | 68.66 | 69.22 |
| **Mean** | **50.09** | **48.72** |

### BEV AP（参考）

| 类别 | R11_BEV | R40_BEV |
|---|---|---|
| Car | 50.06 | 49.18 |
| Pedestrian | 51.84 | 50.96 |
| Cyclist | 72.29 | 73.19 |
| **Mean** | **58.06** | **57.78** |

> 文件名 `radarpillar_vod_best_map52.56.pth` 中的 52.56 = 3 类 BEV-AP_r40 加权均值（52.56 ≈ 49.18×0.2 + 50.96×0.3 + 73.19×0.5），不是 3D-AP 均值。

## 5. 性能（参数量 / 计算量）

| 项 | 值 |
|---|---|
| 参数量（M，trainable） | **N/A** — 未跑 `tools/scripts/param_check.py`，可手动补 |
| 计算量（G） | **N/A** — 未跑 thop，可手动补 |

> 阶段 0 base 后期补跑：写入本表即可。

## 6. 评估（val 协议）

| 项 | 值 |
|---|---|
| 协议 | VoD 5-frame EAA |
| IoU 阈值 | Car=0.5 / Ped-Cyc=0.25 |
| 召回口径 | R11 + R40 moderate（双记录） |
| 评估脚本 | `pcdet/datasets/kitti/kitti_object_eval_python/eval.py:751-757` |

## 7. seed / commit

| 项 | 值 |
|---|---|
| seed | **N/A** — `OPTIMIZATION.FIX_RANDOM_SEED = False`（用 torch + numpy 默认随机） |
| commit | **N/A** — 训练日（2026-07-10）早于现 commit（2026-07-24），未保留快照 |

## 8. 已知偏差 / 局限

- **无训练过程日志**：原 `radarpillar_base` 早期版本未保留 per-step train.log，现有 `train.log` 是末次 eval 日志（`eval_base_train_20260710-182000.log`），仅含 eval 段
- **无 per-epoch val 序列**：仅 epoch 56 一次 eval 被 early_stop 选中，未保留前序 epoch 数据（曲线 csv 不存）
- **seed 不固定 / commit 不锁定**：无法 bit-exact 复现，只能「同分布 + 同 cfg + 同 ckpt（best.pth）」eval 复现

## 9. 复现指引

> 完整流程看 `train.sh`；下面是手动版。

### 9.1 仅 val（推荐用于 baseline 对账）

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate angle
cd /home/dministrator1/RadarPillar

CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 python -u tools/test.py \
    --cfg_file model_store/baseline_radarpillar/cfg.yaml \
    --batch_size 4 --workers 2 \
    --ckpt model_store/baseline_radarpillar/best.pth \
    --extra_tag baseline_radarpillar_val \
    --output_root output/val/baseline_radarpillar
```

### 9.2 从零训（80 epoch + early_stop）

```bash
bash model_store/baseline_radarpillar/train.sh
```

### 9.3 续训（自愈场景）

```bash
CUDA_VISIBLE_DEVICES=0 python -u tools/train.py \
    --cfg_file model_store/baseline_radarpillar/cfg.yaml \
    --batch_size 16 --workers 2 \
    --ckpt model_store/baseline_radarpillar/last.pth \
    --start_epoch <from_epoch> \
    --extra_tag baseline_radarpillar_resume
```

## 10. asset/

`asset/` 子目录用于存 loss 曲线、mAP 曲线、PR 曲线、bbox 可视化等图。当前 baseline 缺早期日志，无图可放，留空目录供后续补：
- `asset/loss_curve.png` — train loss 随 epoch 变化
- `asset/val_map_r40.png` — R40 moderate mAP 随 epoch 变化（eval 序列绘出）
- `asset/pred_visual/` — val 集预测可视化（按帧 / 按类）

补图命令示例（thop + matplotlib）：

```bash
python -c "
import torch, sys
sys.path.insert(0, '.')
from pcdet.config import cfg, cfg_from_yaml_file
cfg_from_yaml_file('model_store/baseline_radarpillar/cfg.yaml', cfg)
from pcdet.models import build_network
m = build_network(cfg.MODEL, dataset=None).cuda().eval()
n = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f'trainable params: {n/1e6:.2f} M')
"
```

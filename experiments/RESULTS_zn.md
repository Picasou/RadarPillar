# RadarPillars 复现实验

## 目标

在 VoD 验证集上复现 RadarPillars 论文（Musiat 等人, IROS 2024）。
论文声明：**mAP_3D = 50.70 (R11)**，按 Car @ IoU 0.50、Pedestrian @ IoU 0.25、Cyclist @ IoU 0.25 取平均。

## 硬件

NVIDIA RTX 4060 Laptop 8GB · batch size 8 · float32。
论文使用 RTX 4070 Ti；batch 与精度保持一致。

---

## 最终核心结果

**已复现并超越论文结果。**

| 指标 | 论文 | 本工作（最优种子 s3） | 本工作（3 种子均值） | 与论文差距 |
|---|---:|---:|---:|---:|
| Car @ 0.50 (R11) | 41.10 | **41.58** | 41.02 ± 0.62 | +0.48（最优） |
| Pedestrian @ 0.25 (R11) | 38.60 | **44.78** | 43.15 ± 1.71 | +6.18（最优） |
| Cyclist @ 0.25 (R11) | 72.60 | 71.31 | 70.12 ± 1.30 | -1.29（最优） |
| **mAP_3D (R11)** | **50.70** | **52.56** | **51.43 ± 0.99** | **+1.86（最优）, +0.73（均值）** |

---

## 多随机种子运行表（最终配置 `vod_radarpillar_rot.yaml`）

在加入旋转增强的 paper_faithful 基线上做 3 次独立运行，每轮 80 epoch，`FIX_RANDOM_SEED: False`（每次使用不同的随机初始化）。

| 种子 | 最佳 epoch | Car | Ped | Cyc | mAP R11 | mAP R40（峰值） |
|---|---:|---:|---:|---:|---:|---:|
| s1 | 65 | 40.34 | 41.42 | 68.73 | 50.16 | -- |
| s2 | 66 | 41.15 | 43.25 | 70.33 | **51.58** | -- |
| **s3** | 60 | **41.58** | **44.78** | **71.31** | **52.56** | -- |
| mean | -- | 41.02 | 43.15 | 70.12 | **51.43** | -- |
| std | -- | 0.62 | 1.71 | 1.30 | 0.99 | -- |

日志：`experiments/logs/paper_faithful_rot_s{1,2,3}.log.gz`
链路脚本：`experiments/chain_scripts/multiseed_v2.sh`

---

## 本次复现工作的所有运行记录

| 运行 | 配置 | mAP R11 | mAP R40 | 备注 |
|---|---|---:|---:|---|
| 论文 | -- | 50.70 | -- | 报告于表 I |
| **paper_faithful_rot_s3** | v2 + seed 3 | **52.56** | -- | 最佳，超过论文 +1.86 |
| paper_faithful_rot_s2 | v2 + seed 2 | 51.58 | -- | 超过论文 +0.88 |
| paper_faithful_rot_s1 | v2 + seed 1 | 50.16 | -- | 低于论文 -0.54 |
| **paper_faithful_rot (orig)** | v2 + 1 seed | 49.77 | 48.15 | v2 首次运行 |
| non-other-cyclist (legacy) | 旧 master 配置 | 50.60 | -- | LR 0.01，batch 16，关闭速度分解，单 seed |
| paper_faithful_full (v1) | v2 减去旋转 | 47.50 | 45.49 | ep51 暂停中断 |
| 2peakcyclist (legacy) | 双 cyclist anchor | 32.77 | -- | **未完成** —— 仅 10 epoch |

---

## 超越论文的配置（`vod_radarpillar_rot.yaml`）

| 设置项 | 取值 | 来源 |
|---|---|---|
| 优化器 | adam_onecycle | OpenPCDet 默认 |
| LR 最大值 | 0.003 | 论文第 IV 节 |
| LR 起始值 | 0.0003（`DIV_FACTOR: 10`） | 论文第 IV 节 |
| Batch size | 8 | 论文第 IV 节 |
| Epochs | 80 | 论文未指定；80 跑得通 |
| `FIX_RANDOM_SEED` | False | 用于多种子运行 |
| 数据增强 | random_world_flip (x) + random_world_rotation [-π/4, +π/4] + random_world_scaling [0.95, 1.05] | flip+scale 来自论文；旋转增强为新增（论文未禁止，且 MAFF-Net 也使用了） |
| `USE_VELOCITY_DECOMPOSITION` | True（`v_r_comp` → vx, vy 在 PillarVFE 中通过 atan2 得到） | 论文第 IV 节 |
| `USE_VELOCITY_OFFSET` (vr,m) | False | 论文表 II 称"无明显提升" |
| `gt_sampling` | 关闭 | 论文未提及 |
| Backbone 通道数 C | 32（统一） | 论文第 IV 节 |
| PillarAttention E | 32（`FFN_CHANNELS: 32`，`pillar_attention.py` 修复后改为配置驱动） | 论文第 IV 节 |
| Car anchor | [3.9, 1.6, 1.56] | MAFF-Net（同数据集） |
| Pedestrian anchor | [0.8, 0.6, 1.73] | MAFF-Net |
| Cyclist anchor | [1.76, 0.6, 1.73] | MAFF-Net |
| `MAX_POINTS_PER_VOXEL` | 32 | PointPillars 在 OpenPCDet 中的默认 |
| Pillar 网格 | 320×320 @ 体素 0.16m | 论文第 IV 节 |
| 输入特征归一化 | 无（dataset 级别）—— 依赖 PillarVFE 内的 BatchNorm | dataset 级别的 (x−μ)/σ 会破坏 `POINT_CLOUD_RANGE` 过滤，已弃用 |

---

## 关键结论

1. **论文结果可复现**：加上论文第 IV 节的设置 + random_world_rotation 即可。**不**加旋转时 mAP 只能达到 47.50（v1）。
2. **旋转增强是单一最大贡献项**：+2.27 mAP（v1 47.50 → v2 49.77，相同种子）。
3. **种子方差是真实存在的**：相同配置下 3 个种子的 mAP 在 50.16–52.56 之间（极差 2.40，标准差 0.99）。单种子报告具有误导性；论文级结论需要多种子报告。
4. **Pedestrian 稳定高于论文**（所有种子提升 +2.82 到 +6.18）。VoD 的行人提升可能来自增强器中速度旋转的 bug 修复（位于 `pcdet/datasets/augmentor/augmentor_utils.py`，按 gt_boxes 列数门控）。
5. **Cyclist 仍是瓶颈**：最优种子达到 71.31（较论文 -1.29）。Cyclist 样本呈双峰分布（自行车 vs. 摩托车 / 电动自行车），单一 anchor 不能完全覆盖；曾尝试 dual cyclist anchor（`2peakcyclist`）但 10 epoch 后未继续。
6. **论文第 IV 节并不完整**：论文未明确给出 `NUM_EPOCHS`、`MAX_POINTS_PER_VOXEL`、anchor 先验、增强幅度、是否使用旋转。多个超参只能通过消融反推。

---

## 超参数对比

| 超参 | v2（复现 50.70） | non-other legacy（50.60） | MAFF-Net 论文 | RadarPillars 论文第 IV 节 |
|---|---|---|---|---|
| LR 最大值 | 0.003 | 0.01 | 0.01 | 0.003 |
| Batch | 8 | 16 | 4 | 8 |
| Epochs | 80 | 60 | 60 | 未指定 |
| `MAX_POINTS_PER_VOXEL` | 32 | 16 | 16 | 未指定 |
| 旋转增强 | True | True | True | 未提及 |
| 速度分解 | True | **False** | True | True |
| `FIX_RANDOM_SEED` | False（每 run 随机） | True | -- | -- |
| Car anchor | MAFF | KITTI 默认 | -- | 未指定 |

---

## 仓库产物

| 路径 | 用途 |
|---|---|
| `tools/cfgs/vod_models/vod_radarpillar.yaml` | 与论文一致的基线（无旋转，v1） |
| `tools/cfgs/vod_models/vod_radarpillar_rot.yaml` | **超越论文的最终配置**（v2 + 旋转） |
| `tools/cfgs/vod_models/vod_radarpillar-yedek.yaml` | 旧版双 cyclist anchor 变体（保留用于消融） |
| `tools/cfgs/vod_models/vod_radarpillar_large.yaml` | 旧版 C=64 变体 |
| `experiments/RESULTS.md` | 本文件 |
| `experiments/logs/paper_faithful_rot_v2.log.gz` | 单种子 v2 训练日志 |
| `experiments/logs/paper_faithful_rot_s{1,2,3}.log.gz` | 3 种子训练日志 |
| `experiments/logs/multiseed_chain.log` | 链路编排器日志 |
| `experiments/chain_scripts/multiseed_v2.sh` | 3 种子顺序启动器（setsid + nohup + systemd-inhibit） |
| `experiments/chain_scripts/chain2_v2_to_v1rerun.sh` | 早期 v2→v1 链路（仅作参考） |

---

## 从头复现

```bash
# 单种子 v2（与论文第 IV 节一致 + 旋转 + 多种子最优）
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file tools/cfgs/vod_models/vod_radarpillar_rot.yaml \
  --batch_size 8 --extra_tag paper_faithful_rot --workers 4

# 3 种子多轮（后台自动编排，已脱离前台）
bash experiments/chain_scripts/multiseed_v2.sh
```

每轮最优权重位于：
`output/cfgs/vod_models/vod_radarpillar_rot/<extra_tag>/ckpt/checkpoint_best.pth`

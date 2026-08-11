# Head 阶段实验：CenterHead / 2DHead 变体构建

日期：2026-08-04
项目：RadarPillar

## 目标

为 b1 / b8 / n3 / n6 / n7 五个实验分别构造 CenterHead 与 2DHead 检测头变体，跑出对照实验。

## 变体矩阵

| 源 | Center 变体 | 2D 变体 | 总计 |
|---|---|---|---|
| b1 | head_b1_center | head_b1_2d | 2 |
| b8 | head_b8_center | head_b8_2d | 2 |
| n3 | head_n3_center | head_n3_2d | 2 |
| n6 | head_n6_center | head_n6_2d | 2 |
| n7 | (跳过) | head_n7_2d | 1 |
| **合计** | **4** | **5** | **9** |

## 关键设计决策

### D1. 命名空间完全分离
- YAML：`head_{b1,b8,n3,n6,n7}_{center,2d}.yaml`
- SH：`train_rpillar_head_{...}_{...}.sh`
- EXTRA_TAG：`head_{...}_{...}`
- 与源 `b1/b8/n3/n6/n7` 命名空间解耦

### D2. 2D head 用新类
新增 `RadarNeXtCenterHead2DNoZ`，继承 `RadarNeXtCenterHead2D`：
- `COMMON_HEADS`：`{reg:(2,2), dim:(2,2), rot:(2,2)}`，无 height
- `CODE_WEIGHTS`：`[1.0] * 6`
- `BBOX_CODE_SIZE`：5
- 训练 loss：6-cat L1（offset_xy, log_lw, sin_cos_h），无 z 监督
- 评估：fill z via `ANCHOR_BOTTOM_HEIGHTS`（兼容 box 7 维）

### D3. Center head 沿用
`RadarNeXtCenterHead` 已存在，cfg 与 `head_center.yaml` 对齐：
- 8-cat（reg2 + height1 + dim3 + rot2）
- 3D 评估

### D4. 评估口径
- Center 变体：`early_stop` 用 `*_3d/moderate_R40`
- 2D 变体：`early_stop` 用 `*_bev/moderate_R40`
- `POST_PROCESSING.EVAL_METRIC` 均为 `vod`（同时出 3d 和 bev key；2D 变体 3d key 退化，仅看 bev）

### D5. 源 yaml 派生
每份新 yaml = 源 yaml 全文 + 覆盖 `MODEL.DENSE_HEAD` 段 + 调整 `early_stop.metrics`。
其余段（DATA_CONFIG / VFE / BACKBONE_3D / MAP_TO_BEV / BACKBONE_2D / OPTIMIZATION）按源保留。

### D6. batch size 对齐源
- b1 / b8：bs=8（源即 8）
- n3 / n6 / n7：bs=8（源即 8）

## 改动清单

| # | 文件 | 操作 |
|---|---|---|
| 1 | `pcdet/models/dense_heads/radarnext_center_head_2d_noz.py` | 新建（~120 行） |
| 2 | `pcdet/models/dense_heads/__init__.py` | 注册新类（1 行） |
| 3 | `experiments/YAML/head_b1_center.yaml` | 新建 |
| 4 | `experiments/YAML/head_b1_2d.yaml` | 新建 |
| 5 | `experiments/YAML/head_b8_center.yaml` | 新建 |
| 6 | `experiments/YAML/head_b8_2d.yaml` | 新建 |
| 7 | `experiments/YAML/head_n3_center.yaml` | 新建 |
| 8 | `experiments/YAML/head_n3_2d.yaml` | 新建 |
| 9 | `experiments/YAML/head_n6_center.yaml` | 新建 |
| 10 | `experiments/YAML/head_n6_2d.yaml` | 新建 |
| 11 | `experiments/YAML/head_n7_2d.yaml` | 新建 |
| 12 | `experiments/SH/train_rpillar_head_b1_center.sh` | 新建 |
| 13 | `experiments/SH/train_rpillar_head_b1_2d.sh` | 新建 |
| 14 | `experiments/SH/train_rpillar_head_b8_center.sh` | 新建 |
| 15 | `experiments/SH/train_rpillar_head_b8_2d.sh` | 新建 |
| 16 | `experiments/SH/train_rpillar_head_n3_center.sh` | 新建 |
| 17 | `experiments/SH/train_rpillar_head_n3_2d.sh` | 新建 |
| 18 | `experiments/SH/train_rpillar_head_n6_center.sh` | 新建 |
| 19 | `experiments/SH/train_rpillar_head_n6_2d.sh` | 新建 |
| 20 | `experiments/SH/train_rpillar_head_n7_2d.sh` | 新建 |

合计：1 head + 1 注册 + 9 yaml + 9 sh = 20 文件。

## 风险

| 风险 | 缓解 |
|---|---|
| 6-cat reg/dim 头与 cfg 维度不匹配 | 训练 1 epoch 验 loss 不为 NaN |
| 2D 头 predict 缺 z 字段 | predict override 按类填 anchor_bottom_heights |
| 命名与 head_center.yaml 撞名 | cfg 文件名唯一、EXTRA_TAG 唯一 |
| 旧 head_2d.yaml 是否仍能跑 | 未改原 `RadarNeXtCenterHead2D`，回归 sanity |

## 不做

- 不改 `RadarNeXtCenterHead` / `RadarNeXtCenterHead2D` 原类
- 不改现有 b1/b8/n3/n6/n7 yaml / sh
- 不跑训练（仅构造入口，由用户后续触发）
- 不做代码层验证脚本（除注册/导入 sanity）

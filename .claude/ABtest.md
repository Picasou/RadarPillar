# RPiN 阶段1 — a4 失败机制对症方案档位

- 落档时间：2026-07-28
- 作者：Claude（人工审核：iHoward）
- 维护：方案 A / B 全过程各自独立落盘，本文档为唯一真相源

---

## 0. 失败背景

a4（PillarAttentionRes）在 D13 协议下劣于 a0 共 3.04 pp（Car 3D AP R40：34.65 vs 37.69）。
mAP/Cyc 三轴同时落后，**唯一假设根因** = 残差加和两端分布量级错配。

a4 当前 forward：

```python
block_out = identity + block_out   # identity = raw PillarVFE 输出（未归一化）
                                   # block_out = TransformerBlock 输出（block 内部已经过 norm1/norm2）
```

关键事实（严格成立）：
- `identity` = raw PillarVFE (n,32) 输出，方差不可控
- `block_out` 在 TransformerBlock 内部被 `norm1`、`norm2` 压回均 0 方 1

`+` 操作在量级错配的两端做加和 → PillarVFE 输出的几何特征被 TransformerBlock 输出压制。

---

## 1. 方案总览

| 项 | 方案 A（LN-Post 残差归一化） | 方案 B（零初始化 ffn） |
|---|---|---|
| 改动文件 | `pillar_attention_resA.py`（forward） | `pillar_attention_resB.py`（`__init__`） |
| 改动行数 | 2 行 | 2 行 |
| 新增 params | **0**（复用现有 LN） | **0** |
| FLOPs 增量 | **0**（LN 是轻量归一化） | 0 |
| 解决机制 | skip/attn 量级错配（全程生效） | epoch 0 attn 路径污染（仅 epoch 0） |
| 与 a4 失败假设对应 | **直接对应** | 间接 |
| cfg | `experiments/YAML/a4_lnpost.yaml` | `experiments/YAML/a4_rezero.yaml` |
| train 壳 | `experiments/SH/train_rpillar_a4_lnpost.sh` | `experiments/SH/train_rpillar_a4_rezero.sh` |
| eval 壳 | `tools/scripts/eval/eval_rpillar_a4_lnpost.sh` | `tools/scripts/eval/eval_rpillar_a4_rezero.sh` |
| model slug | `rpillar_a4_lnpost` | `rpillar_a4_rezero` |
| OUTPUT_ROOT | `output/train_log/vod/2026072814_rpillar_a4_lnpost_lnpost/` | `output/train_log/vod/2026072814_rpillar_a4_rezero_rezero/` |
| 启动序列 | **先** | 后（待方案 A 实测后启动） |

---

## 2. 执行规则（硬约束）

### 2.1 不复用规则

每方案的 cfg / shell / OUTPUT_ROOT **完全独立落盘**，禁止任何一方复用另一方资产：

| 文件类型 | 方案 A | 方案 B | a4 旧 |
|---|---|---|---|
| cfg | `a4_lnpost.yaml` | `a4_rezero.yaml` | `a4.yaml` |
| train shell | `train_rpillar_a4_lnpost.sh` | `train_rpillar_a4_rezero.sh` | `train_rpillar_a4.sh` |
| eval shell | `eval_rpillar_a4_lnpost.sh` | `eval_rpillar_a4_rezero.sh` | `eval_rpillar_a4.sh` |
| 模型代码 | 含 LN-Post 改动 | 含零初始化改动 | 无改动 |
| OUTPUT_ROOT | `..._lnpost_lnpost/` | `..._rezero_rezero/` | `..._a4_a4/` |

### 2.2 串行规则

- 两方案**串行执行**（方案 A 先跑，拿到实测结果再决定方案 B）
- 单 GPU（GPU=0）、bs=16、80 epoch、CPU eval 协议与 a4 一致
- 跑方案 B 前必须先撤掉方案 A 的代码改动，否则两方案同时生效（污染实测对照）

### 2.3 协议一致性

- D13：VoD 5f EAA，IoU Car=0.5 / Ped·Cyc=0.25，3D AP moderate，**Car R40 主判**
- 末 20 epoch 必 eval（autofinish 兜底）
- pickbest 按 Car R40 max
- CPU eval 单协议，禁止 GPU eval 切换

---

## 3. 方案 A：LN-Post 残差归一化

### 3.1 改动（2 行）—— **已落地**

文件：`pcdet/models/backbones_3d/pillar_attention_res.py`
位置：`PillarAttentionRes.forward` 第 49-50 行

原代码：

```python
if self.res_shortcut and identity.shape == block_out.shape:
    block_out = identity + block_out
```

改后（实际已写到磁盘）：

```python
if self.res_shortcut and identity.shape == block_out.shape:
    # ResNet 短接：B1 + TransformerBlock(B1)
    # 方案 A（LN-Post 残差归一化）：复用父类 norm1/norm2 归一化 identity，
    # 使 skip 与 block_out 量级对齐（避免 raw PillarVFE 被 attn 输出压制）。
    identity_normed = self.norm2(self.norm1(identity))
    block_out = identity_normed + block_out
```

### 3.2 严格保证

- **0 新增 params**：复用 `self.norm1`、`self.norm2`（父类 PillarAttention 已存在的 LayerNorm，**已注册在 state_dict 中**，不需另行构造）
- **0 FLOPs 显著增量**：LayerNorm 是 O(N·C) 轻量归一化，相比 backbone 卷积可忽略
- 残差加和前，skip 与 attention 输出都被压回均 0 方 1 的同一分布
- epoch 0 时 `identity_normed` 量级与 `block_out`（attn 输出）量级一致——避免 raw PillarVFE 在 epoch 0 就被压制

### 3.3 不保证

- 不保证 a4_lnpost 性能等同 a0（a0 是结构上**无 attention**，方案 A 是**有 attention 但有归一化 skip**）
- 不保证 attention 学到有用信号（只给了 attention 一个与 skip 同分布的起点）
- 不保证收敛更稳（只保证前向/反向首步不会因为量级错配压垮 PillarVFE）

### 3.4 当前训练状态

| 项 | 值 |
|---|---|
| 启动时间 | 2026-07-28 14:05 CST |
| 训练 PID | 2064779（嵌套 train.py） |
| 真实日志 | `output/train_log/vod/2026072814_rpillar_a4_lnpost_lnpost/logs/train_20260728-140646.log` |
| 当前进度 | ep0/it89，loss=2.2，lr=0.0003 |
| 322 it × 80 ep @ 3.65 it/s ≈ 76 min |
| cron | brief 10min + autofinish 7 \* \* \* \* |
| 训练输出目录 | `output/train_log/vod/2026072814_rpillar_a4_lnpost_lnpost/` |

---

## 4. 方案 B：零初始化 ffn

### 4.1 改动（2 行）—— **待方案 A 实测后落地**

文件：`pcdet/models/backbones_3d/pillar_attention_res.py`
位置：`PillarAttentionRes.__init__` 末尾（在 `super().__init__() → self.short_cut` 之后）

新增 2 行：

```python
import torch.nn as nn
nn.init.zeros_(self.ffn[2].weight)
nn.init.zeros_(self.ffn[2].bias)
```

`ffn[2]` 解释：父类 `PillarAttention.__init__` 定义 `self.ffn = nn.Sequential(Linear, GELU, Linear)`，`ffn[2]` 是第二个 Linear 层，即 FFN 输出投影回 `attn_channels` 的那一层。

### 4.2 严格保证

- epoch 0 时 `ffn` 输出严格为 0（`Linear(W=0, b=0) → 0`，前向公式 `y = Wx + b = 0·x + 0 = 0`）
- attention block 输出**第一帧**严格等于其内部 `norm1` 后的恒等结构（`x → attn → x + attn_out → norm1 → ... → norm1(x + ffn_out)`，ffn_out=0 时 block_out = `norm1(x + attn_out)`）
- 给 PillarVFE 一个**仅 epoch 0**不被 attention 路径污染的起点

### 4.3 不保证

- **不保证**后续 epoch attention 能从 0 学到有用信号——一旦 ffn 学到非零 W（通常 1~2 epoch 后开始），机制即失效
- 与方案 A 不同：**方案 A 全程生效**；**方案 B 仅 epoch 0 生效**
- 不保证 a4_rezero 性能等同 a0（只补了一帧的干净起点）

### 4.4 与方案 A 的关键差异

| 维度 | 方案 A | 方案 B |
|---|---|---|
| 生效时长 | 80 epoch 全部 | 仅 epoch 0（之后 ffn 学到非零 W 即失效） |
| 改动本质 | skip 路径归一化（对症量级错配） | ffn 起点压制（对症 epoch 0 污染） |
| 假设验证场景 | 假设"残差两端量级错配"是 **整个训练期** 主导机制 | 假设"epoch 0 被压制"是 **关键期** 主因 |
| 工程含义 | 加 LN 使 skip 输出稳定 | 用零初始化让 ffn 不抢戏 |

---

## 5. 方案 A vs a4 裁决表（待实测填）

方案 A 跑完后，按 Car 3D AP R40 moderate 主判（D13）：

| 方案 A vs a4 (34.65) | 处置 |
|---|---|
| ≥ 35.65（+1pp 明确赢） | 跑方案 B 做完整 ablation 表 |
| 33.65 - 35.65（±1pp 平手区间） | 跑方案 B；成本严格相等（都 0 params 增量）→ 由 mAP R40 裁决 |
| < 33.65（-1pp 明确输） | A 归档，**不跑**方案 B |

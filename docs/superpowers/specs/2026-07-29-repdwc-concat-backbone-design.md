# RepDWC concat backbone 重构

## 问题
b5-b8 用 `RepDWCNoneBackbone`（取首层直出），与 b1-b4 的 `BaseBEVBackbone`（3 层 concat）混了 3 个变量：block 类型 / 有无 deblock+concat / 输出通道。不构成公平对照。

## 目标
b5-b8（+n4）改为「RepBlock 多尺度 → ConvTranspose deblock → concat」，与 b1-b4 **只差 block 类型**。保持类名 `RepDWCNoneBackbone`、文件 `repdwc_none.py` 不变，只改内部实现。`RepDWCBackbone` 零改动。

## 改动项
1. **`pcdet/models/backbones_2d/repdwc_none.py`**：`RepDWCNoneBackbone` 不再取 `outs[0]` 直出，改为用 `UPSAMPLE_STRIDES/NUM_UPSAMPLE_FILTERS` 构 ConvTranspose deblock（与 `BaseBEVBackbone` 同构），把三层多尺度上采样到 160×160 后 concat。`num_bev_features` 由 `OUT_CHANNELS[0]` 改为 `sum(NUM_UPSAMPLE_FILTERS)`。`NUM_UPSAMPLE_FILTERS` 数量须等于 `NUM_OUTPUTS`（assert）。

2. **YAML `experiments/YAML/`**：b5/b6/b7/b8/n4 的 `BACKBONE_2D` 加 `UPSAMPLE_STRIDES:[1,2,4]` + `NUM_UPSAMPLE_FILTERS`（取值同各档 OUT_CHANNELS）。新增 b9。全部 `BATCH_SIZE_PER_GPU:16`。

   | tag | block | OUT_CHANNELS | NUM_UPSAMPLE_FILTERS | concat→ | 对照 |
   |-----|-------|--------------|----------------------|---------|------|
   | b5 | RepBlock | [32,32,32] | [32,32,32] | 96 | b1 |
   | b6 | RepBlock | [32,64,128] | [32,64,128] | 224 | b2 |
   | b7 | RepBlock | [64,128,256] | [64,128,256] | 448 | b3 |
   | b8 | RepBlock | [64,64,64] | [64,64,64] | 192 | b4 |
   | b9(新) | RepBlock | [16,16,16] | [16,16,16] | 48 | b1 同拓扑更小 |
   | n4 | RepBlock | [64,128,256] | [64,128,256] | 448 | b3 |

3. **测试 `tests/rpin/test_rpin_modules.py`**：`test_repdwcnone_outs0_design` 改为断言 concat 行为——n4 的 `num_bev_features=448`、输出 `(448,160,160)`（替代旧的"首层 64ch@160×160"）。

4. **训练脚本 `experiments/SH/`**：b9 无对应脚本，按模板 `.claude/skills/model-train/templates/train.sh.template` 生成 `train_rpillar_b9.sh`（与 b5-b8 同构，仅 CFG_FILE/EXTRA_TAG 指 b9）。b5-b8、n4 脚本已存在，复用。

## 验收
- 改写后的测试通过。
- b5-b9、n4 各用对应 `experiments/SH/train_rpillar_*.sh` **跑 1-epoch 训练确认无报错**：`EPOCHS=1 RUN_MODE=foreground bash experiments/SH/train_rpillar_b5.sh`（其余同理）。foreground 模式日志直出，确认正常起训、无 shape/断言/配置错误。
- b5-b9、n4 跑 `stage_stats` 采样 Params/FLOPs，确认 b5↔b1 / b6↔b2 / b7↔b3 / b8↔b4 仅 block 类型不同。

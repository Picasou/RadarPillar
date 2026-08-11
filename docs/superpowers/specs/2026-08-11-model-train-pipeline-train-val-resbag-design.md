# model-train: 每模型 train+val+resbag 训练链 + 全 pipeline 结论文档

## 背景与目标

model-train skill 经"收编通用 pipeline"重构后，`pipeline.sh` 退化成只跑单条命令 + 查 marker 的通用壳，**丢失了重构前 `train_pipeline.py` 的 train→val→pickbest→record 收尾链**；`run_all_repdwc.sh` 传的 `RUN_PICKBEST/RUN_RESBAG` 接口对不上，整个链路断裂。

**目标**：恢复并固化"每个模型 = train + val + resbag"的训练链，并在整批跑完后自动产出跨实验结论文档。一个模型必须 train+val+resbag 三件齐全才算完成。

## 架构（两层）

```
任务清单 (每行一个模型 + resbag 参数)
        │  generate_workflow.sh
        ▼
workflow_<task>.sh  ── 串行循环每个模型 ──┐
                                         │
        ┌──── pipeline.sh (per-model 训练链) ────┐
        │  ① train    → best.pth 候选 ckpt        │
        │  ② val      → 末 N epoch eval results   │
        │  ③ pickbest → 选 best.pth               │
        │  ④ record   → 单实验记录 md             │
        │  ⑤ resbag   → 落袋 + model_store.yaml   │ ← 成功标志
        └─────────────────────────────────────────┘
        │  全模型跑完
        ▼
结论文档 (跨实验, 套报告模板 + resbag list 聚合)
```

## per-model 训练链（pipeline.sh）

pipeline.sh **纯做训练链**，不再保留通用 RUN_CMD 模式（这 skill 本就是 model-train）。链入口统一走 `make_shell` 生成的模型壳（train/eval），不直裸调 train.py：

| 段 | 职责 | 成功判据 | 复用来源 |
|---|---|---|---|
| ⓪ make_shell | 按 model/cfg 生成 `train_<model>.sh` + `eval_<model>.sh` | 两壳存在（幂等） | `train_pipeline.py make_shell` |
| ① train | 跑 train 壳 | 目标 epoch ckpt 生成（crash 则取末 N ckpt 部分收尾） | 重构前 autofinish 判活逻辑 |
| ② val | 跑 eval 壳 all 模式，末 N epoch 批量 eval（GPU） | results.json 齐全 | 重构前 autofinish 1/3 |
| ③ pickbest | 按 metric 选 best.pth | best.pth 落盘 | `train_pipeline.py pickbest` |
| ④ record | 聚合 metric + thop，写单实验记录 md | 记录 md 生成 | `train_pipeline.py record` |
| ⑤ resbag | 硬复制产物 + 算 params/flops + 写 model_store.yaml | **model_store.yaml 存在 = 模型成功** | `resbag.py make` |

**marker 统一**：⑤ 产出的 `model_store.yaml` 即成功标志。workflow 起点跳过、done_notifier 终点校验、断点续跑，全用它。

## 全 pipeline 结论文档

所有模型跑完后，聚合一层**跨实验结论文档**（区别于 ④ 的单实验记录）：

- 数据源：`resbag list` 跨实验聚合（所有 model_store.yaml）+ 各模型单实验记录
- 结构：核心结果对比表 / 哪个最优 / gap 归因 / 产物清单（参照 `实验报告模板.md` 章节骨架）
- 落点：workflow 产物目录下，与各模型 OUTPUT_ROOT 同级

## 任务 spec 扩展

现格式 `tag|cmd|marker` 不含 resbag 参数。扩展为带训练链参数：

- 每行模型携带：dataset / model / cfg_file / batch_size（resbag + val 必需）
- `output_root` 由 marker 路径目录自动推（marker = model_store.yaml）
- 缺参数 = 非法任务（因为"必须有"）

具体字段分隔符留实施阶段定，spec 只约束语义。

## 失败语义

- **各段独立 retry**：train 失败重训；train 成功但 val/pickbest/resbag 失败只重试该段，**不重训**（省 GPU）
- **resbag 失败不阻断队列**：单模型最终失败标 FAIL，workflow 继续下一个，done_notifier 末尾报 partial
- **crash 兜底**：训练进程死 + ckpt 停更 >2h → 取末 N ckpt 走部分收尾（沿用重构前 autofinish 逻辑），产物标 `FINISHED_PARTIAL`
- resbag 自身失败仍不抛（`check=False`），但缺 model_store.yaml → 该模型不算成功

## 复用资产（从 git `535e969` 捞回）

- `train_pipeline.py`：autofinish / pickbest / record / make_shell（含 lockfile、超时缩放、partial 收尾等硬化）
- `templates/template.sh` + `templates/template_eval.sh`：train/eval 壳模板
- 现有不动：`实验报告模板.md`、`resbag.py`、helpers/（tmux/watchdog/brief/done_notifier）

## 不做（out of scope）

- 不修 legacy `run_all_repdwc.sh`（b5-b9 已跑完、死代码，接口坏留 note 不动）
- 不动 4 类保护机制（tmux/watchdog/brief/done_notifier）与 H2-H5 硬化
- 不保证训练收敛到目标 AP（取决于 cfg/数据/seed）

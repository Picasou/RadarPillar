# RadarPillar 训练任务 Lessons Learned

> 持续累积的训练 / 评测 / 排查经验。每章独立成文，可在后续工作中追加新章节。

## 目录

- **Chapter 1**：[radarpilar 复现问题排查与 Cyclist AP 暴跌定位](#chapter-1)
- *Chapter 2：（占位）— 待补充*
- *Chapter 3：（占位）— 待补充*

---

# Chapter 1：radarpilar 复现问题排查与 Cyclist AP 暴跌定位

> 时间：2026-07-17 ~ 2026-07-19 · 状态：已完成

## 1.1 全模型对比表

| 顺序 | 模型 | 参数量 | 训练集 | 推理 GT | Car | Ped | Cyclist | mAP | 与论文差 |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 论文 1-frame | 0.27M | 3 类 | 3 类 | 36.0 | 35.5 | 66.4 | 46.0 | baseline |
| 2 | 论文 5-frame | 0.27M | 3 类 | 3 类 | **41.1** | **38.6** | **72.6** | **50.7** | +4.7 |
| 3 | 论文 PointPillars baseline | 4.84M | 3 类 | 3 类 | 30.2 | 25.6 | 62.8 | 39.5 | -6.5 |
| 4 | 0709（历史参考）| 0.18M | 3 类 | 3 类 | 37.92 | 35.60 | 66.95 | 46.82 | +0.82 |
| 5 | base（历史）| 0.18M | 3 类 | 3 类 | — | — | 69.22 | 48.72 | +2.72 |
| 6 | v1 (plan 基线) | 0.18M | 5 类 | 5 类 | 36.37 | 35.19 | 17.50 | 29.69 | -16.31 |
| 7 | v2 (修复尝试) | 0.18M | 5 类 | 5 类 | — | 34.14 | 15.89 | 28.01 | -17.99 |
| 8 | v3 (复现 0709) | 0.18M | 3 类 | 3 类 | 35.95 | 33.70 | 64.56 | 44.74 | -1.26 |

排序逻辑：① 论文 baseline（reference）→ ② 论文 ablation → ③ 历史参考（0709、base）→ ④ 本次 v1 → v2 → v3。

## 1.2 摘要

**问题**：v1 Cyclist 3D AP 17.50，远低于论文 66.4。  
**根因**：5-class 聚合把 bicycle、rider、moped_scooter、motor 合并进 Cyclist，验证集从 1434 GT 膨胀到 6886 GT，0.18M 模型扛不住任务膨胀。  
**修复**：CLASS_MAPPINGS 改回 3 类 `[Car, Pedestrian, Cyclist]` + 复现 0709 训练参数（bs=16, NMS=0.1, 80ep, no warmup）→ v3 = 64.56 / 44.74，**复现度 97.2%**。

## 1.3 如何找到真正的漏洞（4 步排除法）

| 步骤 | 验证 | 结果 | 结论 |
|---|---|---|---|
| ① 数据 | raw → 聚合 infos | pre 6685 / post 32338，+25,653 | 聚合生效但**不是 bug** |
| ② 评测 | cv2 vs shapely，10,650 对 | diff 3.5e-6，NMS 残留 0/47,843 | CPU eval 不是 bug |
| ③ 预测分布 | GT vs PRED 统计 | z 中心 -0.005 vs -0.686（偏低 0.7m，所有类一致）| 数据/评测不能解释 |
| ④ **复现实验（关键）** | 同一份 0709 ckpt predictions 分别用 3 类 / 5 类 GT 重测 | **3 类=66.95 ✅ vs 5 类=15.30** | **5-class 聚合是 100% 根因** |

**核心方法论**：**多假设并存时，构造反事实实验**（同一个 model 换 GT/换配置/换 model 重测，看哪个维度真正影响结果）——AP 数字本身无法定位问题，**对比性实验**才能锁定根因。

## 1.4 v3 修复路径

| 项 | plan 默认 | v3 | 理由 |
|---|---|---|---|
| CLASS_MAPPINGS | 5 类 | **3 类** `[Car, Pedestrian, Cyclist]` | **关键修复** |
| BATCH_SIZE | 8 | 16 | 与 0709 一致 |
| LR_WARMUP | True | False | 与 0709 一致 |
| NMS_THRESH | 0.15 | 0.1 | 与 0709 一致 |
| EPOCHS | 80 | 80 | 用户指定 |

## 1.5 论文 baseline 全表（3 类 GT）

来源：[RadarPillars 论文 arXiv:2408.05020](https://arxiv.org/html/2408.05020v1) Table I/IV/V。

**Table I**（frame 设定）

| 设定 | Car AP50 | Ped AP25 | **Cyclist AP25** | mAP |
|---|---:|---:|---:|---:|
| **1-frame baseline** | **36.0** | **35.5** | **66.4** | **46.0** |
| 3-frame | 40.2 | 39.2 | 71.8 | 50.4 |
| 5-frame | 41.1 | 38.6 | 72.6 | 50.7 |
| 1-frame corridor | 69.4 | 47.1 | 85.4 | 67.3 |
| 5-frame corridor | 70.5 | 52.3 | 87.9 | 70.5 |

*plan v4 错引 1-frame 当 5-frame（66.4 vs 72.6）是 v1 失败隐性原因之一。*

**Table IV**（PillarAttention E ablation）

| Dim | Car | Ped | **Cyc** | mAP |
|---|---:|---:|---:|---:|
| E=16 | 33.3 | 23.5 | 56.0 | 37.6 |
| **E=32 (base)** | 36.3 | 23.4 | **59.1** | 39.6 |
| **E=128（最优）** | **38.1** | **28.1** | **62.4** | **42.9** |

**Table V**（backbone scaling ablation）

| Channels | Params (M) | Car | Ped | **Cyc** | mAP |
|---|---:|---:|---:|---:|---:|
| (16,16,16) | 0.11 | 31.8 | 28.4 | 60.5 | 40.2 |
| **(32,32,32) (base)** | **0.26** | 33.4 | 30.4 | **62.3** | **42.0** |
| (64,64,64) | 0.79 | 36.3 | 28.6 | **63.0** | **42.6** |
| Baseline PointPillars (64,128,256) | 4.84 | 30.2 | 25.6 | 62.8 | 39.5 |

## 1.6 关键经验（7 条）

1. **排除法**：数据 → 评测 → 预测分布 → 模型，逐层独立证据排除
2. **看分布不看数字**：GT vs PRED 统计（z、置信度、IoU）能定位异常
3. **复现实验是终极验证**：多假设并存时，构造反事实实验（换 GT/换配置）才是真相
4. **5-class aggregation 不是无害的**：任务膨胀，0.18M 模型扛不住；论文靠 0.27M+ backbone + 5-frame
5. **plan 与参考实现必须对齐**：plan v4 错引 1-frame 当 5-frame 是 v1 失败隐性原因之一
6. **评分阈值暴露问题**：Cyclist 0.25 最宽松 + 物体最矮 + 模型最小 → 三重不利叠加
7. **训练参数小差异大影响**：bs 8→16、warmup、NMS 0.15→0.1，单项 ~9% AP

## 1.7 产出物路径

| 路径 | 说明 |
|---|---|
| `output/train_log/vod/202607171624_radarpiller_bs8/` | v1（5-class，17.50）|
| `output/train_log/vod/202607181848_radarpiller_bs8/` | v2（15.89）|
| `output/train_log/vod/202607191930_radarpiller_bs8/` | **v3（最终，64.56）**|
| `tools/test_cpu.py` | CPU eval 工具 |
| `.tmp/train_progress_rp_0716.md` | 完整进度文件 |
| `.tmp/infos_backup/vod_infos_*.pkl` | pre-aggregation 备份 |

## 1.8 一句话总结

**v1 Cyclist 17.5 vs 论文 66.4 不是模型问题，是评测口径不一致**：5-class GT（6886）评 0709 ckpt 直接掉到 15.3，复现 3-class GT（1434）就回到 66+。CLASS_MAPPINGS 改回 `[Car, Pedestrian, Cyclist]` + 复现 0709 训练参数 → v3 = 64.56，**复现度 97.2%**。

---

# Chapter 2：（占位）

> 状态：待补充

---

# Chapter 3：（占位）

> 状态：待补充
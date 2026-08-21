---
name: modellist
description: Use when 需要把模型信息写入或更新 experiments/<DATASET>/model.xlsx 汇总表（如"填模型表/更新 model.xlsx/模型信息落盘/落袋到表"，YAML 生成时落架构、训练结束出 eval 后落性能指标，或命名规则变更后同步表格）
---

# modellist —— 模型信息落盘 model.xlsx

## 概述

把模型架构（YAML）+ params/FLOPs + eval mAP 按命名规则写入
`experiments/<DATASET>/model.xlsx` 的 MODEL sheet。
命名规则权威版本同步维护于 `mynote/McRadarPillar.md`「模型表命名规则」。

## 两个落盘时机

```bash
# ① YAML 生成时 —— 落架构列 + FLOPs/PARAMs（结构属性，无需训练）
python .claude/skills/modellist/modellist.py --stage arch --models <新yaml名>

# ② 训练结束出 eval 后 —— 落 2D*/3D* mAP 性能列
python .claude/skills/modellist/modellist.py --stage perf

# 全量（默认 all）
python .claude/skills/modellist/modellist.py [--dry-run] [--models a b] [--run_dir <dir>]
```

数据源：yaml（架构）→ run 目录 `model_store.yaml`（params/FLOPs，缺则 resbag thop 现算 bs=4、固定 seed）→ `eval/**/results.json`（mAP，取最新）。

## 模板铁律（写入时必须遵守）

- **只写值，不改格式**：Times New Roman、居中、列宽、表头、合并单元格一律不碰
- **空列分隔列**（B/J/S 等）是分组设计，永不写入、永不删除
- 列位置从表头动态读取（禁止硬编码列号）；表头缺列时报错退出，让人工补模板
- 新增行自动复制模板数据行（第 2 行）样式
- 空值不覆盖已有内容（避免误清）

## 列与命名（quick reference）

- **MODEL_TAG** : yaml 名（如 `msr_radarpillar`）
- **INPUT** : used_feature_list `[x,y,z,dop_x_gnd,dop_y_gnd,rcs]`
- **VFE** : `PN[32]`（NUM_FILTERS）
- **3DBACKBONE** : `PAtt[head:1]`（PillarAttention，NUM_HEADS）
- **2DBACKBONE** : `PP[3,5,5]*[32,32,32]` / `RepDwc[3,5,5]*[64,64,64]`
- **NECK** : `UPSAMPLE[1,2,4]+CONCAT[96]`（内置 deblock 上采样+concat，C=各级上采样通道和）/ `FPN[C]` / `MDFEN[C]`
- **HEAD** : `anchorbased` / `anchorfree`（CenterHead 系）
- **NMS** : `NMS[gpu,0.1,mc]`（impl,iou[,mc=多类]）；mask 免 NMS 写 `MASK`
- **FLOPs / PARAMs** : `5.597G` / `0.187M`，thop bs=4 口径
- **2D*/3D* mAP** : 2D=bev、3D=3d，moderate R40；`mAP`=CYC/PER/CAR/TRUCK 均值
- **OB** : 障碍物（原始标签 type 7），未参与训练/评估 → 留空
- 无该模块写 `—`；无数据留空

## 常见坑

| 坑 | 处理 |
|---|---|
| baseline run 目录 tag 是短名（`*_msr` 而非 `*_msr_radarpillar`） | find_run_dir 规则 3 兜底；仍找不到用 `--run_dir` |
| `YAML/msr.yaml` 是指针文件（内容为文件名） | 自动跳过非 dict yaml |
| resbag `map_r40` 是 VoD 3 类口径（无 Truck、÷3） | 本 skill 直接读 results.json，4 类口径 |
| 空 run 目录（中断）无产物 | 匹配时按产物过滤 |
| 表在 Excel 中打开（`~$` 锁文件） | 写入后勿从 Excel 侧覆盖保存 |

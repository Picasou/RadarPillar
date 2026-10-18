# RadarPillar 模型训练与部署 Salon 分享

> **一句话定位**：在上一章自标注数据集（AUTOSIL → MSR v1/v2/v3）的基础上，本章讲两件事——**怎么训练出一个可部署的轻量化检测模型**（贪心消融选型 + 无人值守训练生产线 + 评估口径工程），以及**怎么把它安全地送上板**（S0-S5 对拍门禁 + INT8 量化梯子 + 板端 3.7× 提速）；终点是 AXU4EV（ZU5EV）板上 full worst 67.8ms 达标、对拍零回归。

---

## 0. 一页速览（30 秒版）

**承接上一章**：数据侧（AUTOSIL 标注软件、MSR 三代数据集构建）不再展开；本章从「拿到一个数据集」开始，到「板上跑通」结束。**链路定位**：Net 替代感知链路 `ADC→dets→objs→trackers` 中的 B 段（dets→objs），上游 A 与下游 C 保持现状。

**两条主线，四项核心成果**：

- **训练线 · 选型**：贪心消融五阶段（VoD 确立架构 → MSR 三代迁移验证），产出 0.19-0.82M 参数轻量模型族——0.27M 反超 1.64M RadarNeXt，0.68M 以 1/7 参数超 4.86M 标准 PointPillar +7.0 BEV mAP
- **训练线 · 生产线**：7 个自建 skill 把「跑一次实验」压缩成一行 tag 输入——63 次训练 run、60 个自包含落袋实验、1802 checkpoint、491 份 eval；无人值守过夜串跑，done_notifier 实测 6 天存活
- **部署线 · 质量门禁**：S0-S5 六步链 + S1-S4 四级数值对拍（allclose 到 1e-6 噪声地板）+ INT8 五档量化梯子——**parity 失败不出包**，精度风险显式红旗
- **部署线 · 板端交付**：8 个自描述部署包；front CPU 优化阶梯 B0→b1→b2→b3→NEON，full worst **252→67.8ms（3.7×）**，板上 50 帧 label 一致 **123/123=100%**、中心距 0.332m 不劣化

**可信度锚点**：一切自研改动站在数值对拍的复现 baseline 之上——RadarNeXt→OpenPCDet 移植 AP+参数量≈论文（参数判定 PASS ≤2%），VoD Cyclist 复现度 **97.2%**（64.56 vs 论文 66.4）。

---

## 1. 链路背景：本工程在感知链路中的位置

**为什么引入学习类方法**：规则链路维护成本高、复杂场景已进性能瓶颈；数据驱动信息挖掘更彻底（属性间/点云间联系）、版本迭代更快（少量 baseline → 少量 golden case 微调）。

**现有链路（非本工程，量产基线）**：

```
ADC --A--> dets --B--> objs --C--> trackers
```

- A/B/C 主流程均以 FFT / OPTISIC / PDA 等成熟算法为基础，再借助规则类语义适配不同车辆场景
- A1 当前瓶颈：**横穿 + 启停 + 鬼影 + 静止目标检测**——瓶颈集中在目标的准确检测阶段；若检测能取得高置信度，后续跟踪完全可以覆盖

**两条演进路线**：

- Net 替代【A+B】（S5 路线）：提取当前算法簇未发现的规律与信息，提高检测置信度——远期方向
- **Net 替代【B】（本工程 · 轻量化路线）**：保留原有 RSP 处理流程，充分利用位置+物理信息完成目标检测

**技术脉络：PointNet → PointPillars → RadarPillar**（每代只解决一件事）：

- **PointNet / PointNet++**：开创点云直接学习（逐点编码、邻域分层聚合），但逐点/逐球查询算力高，强项在语义分割——排除
- **VoxelNet / SECOND**：体素化 +（稀疏）3D 卷积，检测精度上来了，但 3D conv 的计算与内存仍重
- **PointPillars**：把点云收进柱体（pillar）、投影成 BEV 伪图像，退回 2D CNN——无 3D conv，算力骤降，点云检测第一次具备上边缘设备的可能【√】；但原生面向 lidar，参数量对边缘设备仍偏重
- **雷达特化**：**RadarPillar** 把 doppler/RCS 等雷达物理量做进 pillar 特征（轻量 0.19M）；**RadarNeXt** 用 RepDWC+MDFEN+CenterHead 做到雷达 SOTA（1.64M）

→ 本工程取两者之间：**RadarPillar 轻量底座 + 消融式吸收 RadarNeXt 的有效组件**（PillarAttention / CenterHead 系），一段式、参数少、计算量小、可部署。pipeline 与各段职责：

```
dets --> VFE --> Backbone --> Neck --> Head --> objs
        特征工程+     高维度抽象    改变感受野、    下判决书
        数据增强      与整合        整合邻域信息    （分类+回归）
```

**部署约束前置**：目标硬件 ZU5EV（CPU 仅 2 核可用）+ DPU、10-20 Hz——这个约束从第一天就在，反向决定了后续每一次选型（轻量 head、控通道、重参数化、INT8 可量化）。

**每个模块的选型怎么定——三个问题，三套机制**（不凭感觉，每问都有工程答案）：

- **X 模块为什么选这个结构？** → 贪心消融：固定已最优、单变量扫描、胜者固化进下一轮底座（§3.2）
- **为什么用这种维度的组合？** → 容量档公平对照：[16]/[32]/[64,64,64]/[64,128,256] 逐档实测 Params/FLOPs/mAP 横比，用实测账目而非论文数字定容量（§3.2）
- **为什么这个 bs / epoch / 优化器？** → 显存勘察定 bs 上限（8G 卡五模型 bs=8 峰值 2777MiB）+ OOM 自动回退兜底；100ep 已到收益平台（末 10ep 进入单 epoch 波动量级），pickbest 末 10 双口径取平台内最优而非赌最后一个；超参（bs/epoch/adam_onecycle/seed）与结构一样写进「固定项」冻结——**超参不确定，消融就不可比**（§3.2 / §4.1）

> 🖼 **图位 1｜链路位置图**（建议制作，开篇第一张图）
> - **内容**：横向链路 `ADC --A--> dets --B--> objs --C--> trackers`，B 段高亮为「本工程：Net 替代」，A/C 段标注 FFT/OPTISIC/PDA + 规则语义（量产现状）；链路上方浮标 A1 四大瓶颈（横穿/启停/鬼影/静止目标），下方标注两条演进路线（A+B 远期 / B 本工程）
> - **显示效果**：听众第一眼即明白工程边界——不重造整条链，只换检测这一段，且替换段自带部署约束

---

## 2. 全景：训练→部署运行 pipeline

```
┌─ 训练线 ─────────────────────────────────────────────────────────────┐
│ 数据集(上一章) --create_msr_infos--> pkl + gt_database                 │
│   --冒烟/显存勘察--> experiments/<DS>/{YAML,SH}（贪心消融矩阵）        │
│   --model-train skill--> tmux 串行: train --skip_eval → eval 末10ckpt │
│   → pickbest(max+median) → viz → resbag 落袋 → model.xlsx 落表        │
│   → 跨实验报告 → (下游) tracker 回灌验证 / test_cpu 交叉验证          │
└──────────────────────────────────────────────────────────────────────┘
┌─ 部署线 ─────────────────────────────────────────────────────────────┐
│ best.pth --model-deploy skill--> ① golden_dump 50帧基准               │
│   → ② ONNX 量化编译 xmodel → ③ C++ 编译 → ④ parity S1-S4 对拍门禁    │
│   → ⑤ INT8 梯子 AP 终判 → pack_store 落袋 → flash-sdcard 制卡        │
│   → 板上 50 帧验收(67.8ms 达标) + 板端 front 优化迭代                 │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.1 工程框架：基于 OpenPCDet 的二次开发

- **为什么是 OpenPCDet**：点云检测工具链的事实标准——注册式模块组装、KITTI/VoD 评测内核、spconv/iou3d_nms 基础设施现成，自研火力全部集中在雷达特化点
- **配置驱动组装**：DATASET / VFE / BACKBONE_3D / BACKBONE_2D / DENSE_HEAD 全是注册槽位，yaml 写 `NAME + params` 即插拔——自研模块（PillarAttention、CenterHead 系 ×11、RepDWC 系、PP 系 neck）与原生模块同等待遇，被 100+ 个 cfg 复用
- **统一数据管线**：`DatasetTemplate.prepare_data` 固定链路（类过滤→增强→特征选列→voxelize），新数据集只实现 `get_lidar/get_label/evaluation` 少数接口——MSR adapter 一套代码吃三代数据
- **继承扩展不改源**：`RadarPillarAnchorHeadSingle` 继承原生 AnchorHeadSingle 只覆写推理 mask、`MC_RadarPillars` 继承 PointPillar 零改动——上游升级不冲突
- **目录分层**：`pcdet`（模型库）/ `tools`（入口+cfgs）/ `experiments`（实验矩阵）/ `tracker`（仿真）/ `deployment`（部署）；`dataset.yaml × model.yaml` 分层组合，**一份 cfg 即一个可复现实验**，训练/跟踪/部署同仓库闭环、产物自描述落袋

---

## 3. 训练篇 I：模型设计与贪心消融

### 3.1 一段式管线与模块设计（创新点）

**输入端——PillarVFE 雷达特征工程**（原生 ~100 行扩展到 296 行）：径向 doppler→笛卡尔 vx/vy 分解、相对速度独立分解、逐点速度偏移、register_buffer 速度归一化、`FEATURE_ORDER` 声明式特征拼装（14 种命名特征任意排列 + 一致性 assert）——把雷达物理量做进特征编码，而不是当普通点云。

**PillarAttention（自研核心创新）**：pillar 级 self-attention 插在 VFE 之后、scatter 之前，为稀疏 pillar 补全局上下文。关键工程化：`key_padding_mask` 摆脱 per-sample for 循环、单次 batch attention、Uniform Scaling 统通道。**+6,464 参数（+3.6%）换 mAP_R40 +3.3**（45.43→48.71），经 a0-a4 + SE 候选对照组裁决胜出，是全工程引用最多的非原生模块（104 个 yaml）。一次失败（残差变体 Car AP 反降）→ 根因分析（skip 与 block 量级错配）→ LN-Post/ReZero 两路零成本修复——完整的算法迭代证据链。

**Backbone——RepDWC 重参数化**：MobileOne 多分支训练、推理融合单 conv（`reparameterize()` 完整实现，融合前后 round-trip diff 3.8e-6）——「可部署性」的代码级证据。专为公平对照自建 `RepDWCNoneBackbone`（deblock+concat），使 RepDWC 组与 standard 组**只差 block 类型一个变量**。

**Head——CenterHead 系消融矩阵（11 个文件）**：RadarNeXtCenterHead 忠实移植（含与 mmcv 差 18% 的 IoU kernel 逐字重移植 + parity 记录）；其上派生 2D/2DNoZ/ablZ/ablH/lossup、轻量三兄弟（Narrow/Merged/Trunk）与 vel head：

- **z+h 联合监督发现**（五臂受控实验）：BEV 52.59(联合) > 51.87(双去) > 50.52(只 z) > 49.71(只 h)，loss 加权对照证伪「loss 量级」假设——**增益来自联合监督的共享 trunk 特征，单独任一不涨反掉**；机制旁证：Car rotErr 0.253→0.136 减半。原创洞察：**「不考的科目（z/h 监督）能提升考的科目（BEV 检测）」**
- **Trunk 全共享头**：四头共享一层 trunk + 每头 1×1 final，参数 -36%、FLOPs -47%，精度反升 +0.44——「head 的空间感受野是精度承载，宽度不是」
- **vel head**：+130 参数（近零成本）加 vx/vy 回归，MSRv2 上 Ped BEV AP +3.98（28.32→32.30），直接对症 tracker 行人横穿 MISS；关键工程坑：禁增强避污染会掉 3.4 点，正确解法是 flip 镜像同步取反点级速度列（`NEG_POINT_COLS`，上游框架没有的雷达特有语义，配数值单测）

**推理期 pillar mask 免 NMS（自研）**：非空 pillar 之外的 anchor/heatmap logit 压 -1e9，NMS 候选 4096→1024，从源头砍板端 NMS 量；训练与匹配零改动。（对照实验该线弱于 CenterHead ~10pp，负结果如实落表。）

### 3.2 消融实验的贪心设计理念

**核心思想：按数据流 VFE→Backbone→Neck→Head 分维度贪心寻优，固定已最优、每轮只开一个维度，把实现次数从乘积型变成加法型。**（4 模块各 4 候选：贪心 16 次 vs 全组合 256 次）

落地的证据链（不是口号，是可审计的机制）：

1. **每份阶段报告显式声明「固定项」**，唯一变量单列；cfg 之间 diff 级单变量（a0↔a1 仅 7 行）
2. **next_stage_cfg 固化链**：`vod_radarpillar_a0.yaml → a0_b1.yaml → a1_b1_noneck.yaml`——文件名即贪心决策序列，胜者物理固化为下一轮底座
3. **度量端与超参端同样只动一个维度**：bs/epoch/adam_onecycle/seed 全部进「固定项」声明冻结（超参不确定则消融不可比）、统一 R40 moderate EAA 口径、pickbest 协议从 max 修正为 median（防 cherry-pick）

五阶段战报（VoD 公开集上确立架构，再迁移到自有数据）：

![VoD 复现阶段 Cyclist 分析图](salon_assets/vod_cyclist_dist.png)
*图：VoD 复现阶段的分析图之一（Cyclist 分布，`docs/visualizations/`）——复现锚点不是跑一次训练，而是把「为什么论文数字长这样」拆到分布层面*

- **stage0 锚点**：RadarPillar 0.184M mAP 48.71 > RadarNeXt-MDFEN 1.64M 45.46——「简单 pillar 底座容量已饱和」是整条贪心链的出发点
- **stage1 注意力**：A\* 改判勘误（原 Car 3D 主判选 a0，改 mAP 主判选 a1 PillarAttention）——**勘误链完整留档**，文档与实训 cfg 对齐
- **stage2 容量**：RepDWC 公平对照（eval bug 修复后整批重评，结论反转）——RepDWC 以 1/4 参数打平 standard，b8 全场最优
- **stage3 neck**：无 neck 直连双第一，MDFEN 参数 4.5× 全负增益——「论文里的重 neck 在本场景不适用」的实证
- **stage4 head + z/h**：head-swap 11 实验定 BEV 主力头，z/h 三件套定监督口径

迁移到 MSR v1（40 模型全指标落 `model.xlsx`）三条轴重演同一贪心法：**换头是最大精度杠杆**（anchor→center 3D +7.45，增益几乎全在 Ped +23.80）；**backbone 唯一有效方向是容量加宽**（pp64 +2.67，RepDwc 同宽全面真伤）；**trunk 瘦身成立**（-36%/-47% 反升 +0.44）。MSRv2 五模型套件 + 标准 PointPillar 基线收官对照。

> 🖼 **图位 2｜贪心消融固化链路图**（建议制作，本工程方法论的名片图）
> - **内容**：五个节点链 stage0 → stage1(A\*=PillarAttention) → stage2(C\*=b1) → stage3(neck\*=无) → stage4(head\*=CH2D)，节点间箭头标 `next_stage_cfg` 固化文件名（`a0 → a0_b1 → a1_b1_noneck`），每节点挂胜者 mAP/参数量；旁支灰色标淘汰项（SE/SEDWConv/RepDWC 档/MDFEN）与两次勘误标记（A\* 改判、eval bug 重评）
> - **显示效果**：一图讲清「乘积变加法」——主干只有 5 步，淘汰分支一目了然，且每步都有落盘 cfg 可回溯

> 🖼 **图位 3｜容量-精度散点图**（建议制作，回答「为什么这种维度组合」）
> - **内容**：横轴 Params（对数）、纵轴 BEV mAP，散点为 b1/b5/b8/b9 + MC 系 pp32/ppmix/pp64/radarNeXt/标准 PP，轻量最优点（CH2D-Trunk）星标；数据源 `stage2/repdwc_concat_fair_comparison.md` + `model.xlsx`
> - **显示效果**：>10.9 GFLOPs 三个模型挤在 48.2-49.1 平台、自研点在左上角——「堆算力无增益」的视觉证据

### 3.3 成绩单（口径见文末说明）

**MSRv2 五模型套件**（统一 pillar 底座、bs=8、100ep、voxel 1.0×0.5m、range x[0,200]/y[±20]、5 类含 StaticObject）：

| 模型 | 参数量 | GFLOPs | mAP(5类) | 结构差异 |
|---|---|---|---|---|
| RadarPillar（baseline） | **0.19M** | **5.8** | 47.9 | AnchorHeadSingle |
| **CH2D-Trunk（最优）** | 0.27M | 10.9 | **52.2** | CenterHead2D 共享 trunk |
| PPMix-CH2D | 0.68M | 20.3 | 49.1 | 通道 32/64/64 混配 |
| PP64-CH2D | 0.82M | 27.9 | 48.4 | 通道全 64 |
| RadarNeXt（SOTA 参照） | 1.64M | 35.5 | 48.2 | RepDWC+MDFEN 全家桶 |

- **CH2D-Trunk 仅 +42% 参数/+88% 计算量换 +4.3 mAP，同时反超计算量 3.2× 的 RadarNeXt**；>10.9 GFLOPs 的三个模型全落 48.2-49.1 平台——轻量化应沿「换 head、控通道」推进，而非堆 backbone
- **对标准 PointPillar 基线**（4 类口径）：自研 ppmix_ch2d_trunk 0.68M 达 55.13，标准 PP 4.86M/38.7G 仅 48.17——**1/7 参数、1/52 FLOPs，反超 +7.0pt**
- **MSR v1 全家榜**（4 类）：3D 最高 radarNeXt 91.82(1.64M)，自研 centerhead3d_trunk 91.05(0.27M)=**99.2% 精度、1/6.1 参数、1/3.2 FLOPs**；2D 最高 pp64_ch2d_trunk 92.93(0.82M) 反超 radarNeXt 92.10
- 部署导向三档选型定型：精度档（teacher 不部署）/ tracker 刚需档（centerhead3d_trunk 0.27M/1.405G）/ 极限资源档（repdwc32 双地板 0.060M/0.370G）——**部署锚定模型 msrv2_pp64_ch2d_trunk 即由此选出**

**同帧跨模型对比**（MSRv2 **testing 集** 帧 00001671——全 testing 集类多样性最高的帧，GT 6 框 = Car×2·Ped×1·Truck×2·Static×1；三模型各自独立渲染、天然对照，该帧预测 7/4/8 框仅示意，结论以 testing AP 总体口径为准。`visualize_msr.py --split testing` 生成，已归档 `salon_assets/`）。每张为三面板 `[Camera | BEV+GT | BEV+Pred]`：点云按 doppler_gnd 对地多普勒蓝↔灰↔红着色，GT 细实线无填充、pred 粗线+半透明类色填充（Car 黄/Ped 品红/Cyc 草绿/Truck 青绿），title 标各自目标数：

![RadarPillar 基线（AnchorHeadSingle）](salon_assets/bev_test_radarpillar_1671.png)
*RadarPillar 0.19M（anchor 头基线）*

![CH2D-Trunk（本工程轻量最优）](salon_assets/bev_test_ch2dtrunk_1671.png)
*CH2D-Trunk 0.27M（CenterHead2D 共享 trunk）*

![RadarNeXt（1.64M SOTA 参照）](salon_assets/bev_test_radarnext_1671.png)
*RadarNeXt 1.64M（RepDWC+MDFEN 全家桶）*

**读图要点**：同帧同 GT 下对比 Pred 面板——anchor 头单尺寸先验框不住行人尺度散布（总体 Ped 12.4），CH2D-Trunk 换头后行人召回显著改善（总体 Ped 28.1）且参数仅 +42%；而 6 倍参数的 RadarNeXt 在该口径下并未带来更多收益。批量看图可用 `visualize_two_models_msr.py`（同帧 `[GT | model1 | model2]` 三 BEV 面板，直接吃两个 ckpt 实时推理）。

---

## 4. 训练篇 II：从拿到数据集到训练完成的生产线

### 4.1 八步标准流程（新数据集 → 出指标，以 MSRv3 为例全程 ~2 天）

1. **数据落盘与校验**：五目录（IMAGES/LABELS/POINTS/PARAMS/IMAGESETS）+ struct.json 动态 schema（上一章定义的格式，`check_msr.py` 校验）
2. **infos 生成**：`create_msr_data.py` → `create_msr_infos`（PIL header 缓存把耗时大户 77% 压掉、多线程、单帧缺文件容忍不毁整批）→ 训练/val/testing pkl + `create_groundtruth_database`（gt_sampling 素材库）
3. **dataset yaml**：`CLASS_LABEL_MAP` 覆写即可换代（v3 类别语义重排零代码）、range/voxel/选列适配（MSR 200m 长距、1.0×0.5 单层大柱对极稀疏点云）
4. **冒烟与显存勘察**：mini pkl 160/24 帧 1-epoch 冒烟（不触原始 pkl）+ bs sweep（五模型 bs=8 峰值 2777MiB，8G 卡安全）
5. **实验 YAML/SH**：按规范仿照 `tools/scripts/` 新建对应脚本（conda 自探测、工程根定位、output_root 时间戳分目录防覆盖、`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 绕 tensorboardX 坑）；yaml 头注声明派生关系与唯一差异
6. **model-train skill 一条龙**（输入仅 tag 列表）：自动配对 SH+YAML → 模板渲染 `_full.sh` → tmux 串行（parent=/init 跨会话存活）→ `train --skip_eval` → eval 末 10 ckpt → **pickbest max+median 双落**（BEV mAP 主口径、Car_3d 兜底，"2DNoZ 系 3D AP 恒 0"的教训直接写进模板注释）→ viz → resbag → `.done` marker
7. **落袋与落表**：resbag 自包含硬复制（cfg/权重/日志/指标 + `model_store.yaml` 总账，原子写+文件锁）；modellist 分两时机写 `model.xlsx`（YAML 生成时落架构、eval 后落 mAP）；`make_conclusion.py` 聚合跨实验报告
8. **交叉验证与判读**：test.py(GPU) 与 test_cpu.py(纯 CPU) 双口径互证 + 三面板可视化逐帧核对

### 4.2 无人值守的韧性链（每条对应一个真实故障模式）

- **OOM**：bs>4 溢出即清 ckpt 以 bs=4 全量重训，记 BS_EFF
- **NaN/inf**：train log 检出即把 ckpt 目录隔离（`ckpt_quarantine_*`）再中止——防 auto-resume 回放毒权重秒级再炸
- **断电**：未传 ckpt 时按 mtime 从新到旧找第一个可加载的 ckpt 续训（断电常截断「正在写的最新 ckpt」）
- **进程**：tmux parent=/init 跨会话存活（**done_notifier 实测 6 天不死**）；watchdog 10min 自愈 driver；cron 10min 简报（ep/loss/ETA + 显存 >7G 告警）
- **幂等**：marker 跳过已跑 + 整链 retry N 次 + `bash -n` 语法检查 + 维护锁防人工手术与自愈打架

> 🖼 **图位 4｜无人值守运行时截图**（建议制作，展示「生产线」实感）
> - **内容**：两联截图——左：tmux 会话里 N 个模型串行的 `_full.sh` 输出（每个模型一段 train/eval/pickbest/resbag 流水）；右：cron brief 的 10min 简报样例（ep/loss/ETA/driver 健康/显存水位）
> - **显示效果**：一屏呈现「人不在，实验在跑、在汇报、在自愈」——夜间批跑 12 模型 22.7h 的现场感

结果：**60/63 run 完整落袋**，12 模型批跑 22.7h（单模型 92-117min），绝大多数训练在夜间/无人值守完成。

### 4.3 评估口径工程：度量正确性先于数字好看

工程里最贵的错误都是度量错误，本工程把它们全部变成了文档与单测：

- **eval 内核复用**：`get_msr_eval_result` 借 KITTI eval 内核（免 calib 伪相机轴变换，与 lidar 系严格等价），阈值矩阵按雷达特性重定义（大车 BEV 0.5 / 行人·二轮车·静态物 0.25、3D 全类 0.25）、R40 双口径并行
- **rotate_iou 修复史**：numba CUDA SIGSEGV → pcdet iou3d_nms drop-in 替换（GPU eval 提速）；替换引入 criterion=2 误返 IoU 致 **3D AP 全错（Car_3d=0）** → 改返 BEV 交集面积，修复后整批判重评——「3D≤BEV 合法性判据」自此写进纪律
- **双口径交叉验证**：test.py（GPU，三重 SIGSEGV patch）与 test_cpu.py（纯 CPU shapely/OpenCV 重建全链）互为仲裁；GPU eval 存疑时独立复核
- **pickbest 双落**：max（best.pth）防漏峰值 + median（best_median.pth）防单 epoch 尖峰，两份权重都随 resbag 归档

### 4.4 训练成果的下游验证：检测回灌跟踪闭环

训练出的模型不是停在 mAP——直接回灌 tracker 全链（αβ/KF/EKF/IMM 滤波 + KM 匹配 + 21 项 MOT 指标）做系统级验证，5 模型 × {trained 19 序列, 泛化 11 序列} × {检测回灌, 全 pipeline} 全自动跑完：

- 泛化 11 序列：**vel head 模型 MOTA 58.0% / FP 仅 1081（六模型最低）**，无 vel 同构模型 32.7%——+25.3pp，速度量测链的跟踪收益直接量化
- **反直觉发现**：radarNeXt 检测 mAP 最高但泛化域 MOTA -93.3%（FP 17695 灾难性虚警）——**「检测 mAP 高 ≠ 跟踪好」**，score 分布与召回-虚警权衡才是跟踪侧关键，这条结论直接指导了部署选型

**泛化 11 序列实测（vel 模型，`metrics_default.txt` 摘录，报告全文已归档 `salon_assets/`）**：

```text
DATASET (11 seqs)  match_dist 2.00m  dt 0.10s  (micro-pooling 全帧合并)
  Total: TP 7418  FP 1081  FN 3210  IDS 171 | MOTA 58.0%  MOTP 0.585m | IDF1 69.1% | HOTA 72.3% | AMOTA 28.4% | VDE -0.05s
  1(轿车): MOTA 64.2%  IDF1 75.2%  HOTA 77.7%      2(行人): MOTA 9.8%   ← 已知短板：横穿 MISS（见 §7.1 债务）
```

![行人长序列 MOTA/MOTP 演化曲线](salon_assets/mot_gen_vel_pedmove_curve.png)
*图：泛化集行人运动长序列（1450 帧）逐帧 MOTA/MOTP 演化曲线（evaluator 自动落盘）——单序列 MOTA 84.3%、轿车类 99.9%，长时序稳定跟踪*

![行人横穿序列（已知边界案例）](salon_assets/mot_gen_vel_pedcross_curve.png)
*图：行人横穿序列（220 帧）——横穿方位角超出训练分布（94% 训练行人 ±12° 内）+ 回波稀疏，检测端先裁导致 MOTA -8.0%。放这张是要说明：边界在哪有数据、有根因、有改进路线（type7 同款思路：扩训练分布/降模型内阈值）*

> 🖼 **图位 5｜跟踪过程动画**（建议制作，salon 现场最抓眼的内容）
> - **制作**：`tracker mode=2` 跑任一序列开 visualize 落 GIF/MP4（visualizer 已支持 960 宽 GIF / 1440 宽 MP4），或 `visualize_msr_seq.py --save mp4`（坐标轴与 doppler 色标全程固定防跳变）
> - **显示效果**：三面板逐帧动画——点云按对地多普勒着色、检测框半透明类色、航迹带 ID 拖尾；行人横穿、锥桶列队、掉头等序列名即场景说明，播放 10-20 秒胜过千字描述

### 4.5 性能工程制度化

CenterHead `get_targets_single` 逐目标 GPU 碎调用曾占训练 CPU 70%（GPU util 仅 25%、训练慢 2.4×、epoch 194s）。处理方式是完整闭环：py-spy record 定位（先测量再动手）→ numpy 向量化（批量子下卡/纯 numpy 循环/每 task 一次上卡）→ 300 随机含边界 case golden 对拍 ≤1ulp → 三度踩坑后把「新 head 覆写自查三步」写进 CLAUDE.md 强制规范（grep 覆写→复用 numpy 版→allclose 对拍，附全库覆写清单）。

---

## 5. 部署篇：从 ckpt 到板上 67.8ms

### 5.1 三段拆图与单一锚点

整模型不可导出（字典流/`.item()` 控制流/动态 pillar 数/MHA/LayerNorm/GELU/scatter/NMS 均 DPU 不支持），冻结为三段：

```
N×6 点特征 --C++ 前段(CPU)--> spatial[1,32,80,200] --DPU--> logits[1,11,40,100] --C++ 后段(CPU)--> 检测框
              pillarize+VFE+attention+scatter         backbone+head conv 栈         decode+NMS
```

- **attention true-N 动态**：真实非空 pillar p99=533/max=627，静态 pad 到 MAX_VOXELS=3096 计算量暴涨 ~60×——前段按真实 pillar 数计算，这是 C++ 前段存在的理由
- **单一锚点**：全工程唯一换模型入口 = `deployment/dep_common.py` 三行（CKPT/CFG/CKPT 路径），当前锚定 msrv2_pp64_ch2d_trunk
- **换模型分层适配**：L1 同构换权零 C++ 改动直进六步 / L2 局部改动从第 4 步续跑 / L3 结构重写停下确认——**50+ 训练产物收敛为 6 个结构家族**，对拍门禁是适配正确性的机器判定器

### 5.2 S0-S5 六步链与对拍门禁（model-deploy skill 自动化，全链 20-40 分钟）

①锚定 → ②golden_dump（testing 174 帧确定性抽 50，hook 抓四层基准，manifest 绑 ckpt sha256 + 确定性复查）→ ③make_xmodel（容器内量化编译 INT8 xmodel）→ ④build_cpp → ⑤对拍验证 → ⑥pack_store 落袋。

**S1-S4 四级对拍，实测数值**（0917 验收包 `parity_report.md`）：

- **S1** ONNX vs PyTorch：allclose(1e-5) 逐帧全过 + 20 组随机补测，worst_abs 5.5e-05，失败 0
- **S2** C++ 前段 vs PyTorch CPU 前段（同设备紧对拍剥离跨设备噪声）：worst_abs 2.2e-06（fp32 噪声地板）
- **S3** C++ 后段 decode：框数/label 全一致，score 差 5.1e-07、中心距 6.6e-07m
- **S4** 全链（本地点 bin 从 pillarize 跑到底）：框数/label 全一致，中心距 7.2e-04m，失败帧 0

**门禁规则（质量设计核心）**：任何 S1-S4 失败 = **不落袋不出包**；C++/权重/golden 任何改动后第 ⑤ 步整个重跑；板上验收再叠一层（板上 50 帧输出 vs PC golden，口径同 S3/S4）。数值口径逐位复刻：spconv z 丢弃/前 16 点、VFE padding 行参与 max、f_cluster 4-lane 求和序、exp 不 clamp——这些「无文档行为」全部对 golden 验证后才敢写进 C++。

### 5.3 INT8 量化梯子：自动爬档与诚实的天花板

五档梯子（`int8_ladder.py`）按「病→药→档位」设计：d0 golden50 校准 → d1 per-channel → d2 256 帧 → d3 1024 帧 → d4 混合精度 → d5 QAT；断点续跑且 ladder_state.json 绑锚点 sha256（换模型自动作废）。AP 终判门禁：同帧配对 MSR eval R40 BEV，worst Δ≥-1pt 严过即停 / ≥-2pt 宽过标黄 / 灰区自动扩 174 帧复判。

**终局（如实呈现）**：梯子爬到顶 **-8.43pt**（Car BEV moderate R40 63.19→54.76）；d2/d3/d4/d5 四档 ap.json 完全相同——calib 扩帧、mix_bit、QAT（loss 降 52% 仅 +0.34pt）全部无效。根因锁定 **per-tensor 量化粒度 × hm 头结构敏感**（量化误差 hm mean_abs 0.502），权重侧手段全排除；改善只剩 vai_q_onnx 落地后的 per-channel / 敏感层 FP16 两条路线。**红旗机制**：INT8 不达标但 S1-S4 过 = 照常落袋、报告标红风险——精度风险显式传递，不静默。

### 5.4 板端优化战记：full worst 252 → 67.8ms（3.7×）

**硬约束**：ZU5EV CPU 仅 2 核可用、1.2GHz、目标 full worst ≤70ms（DPU 45 + back 2.2 固定 → front 须从 205ms 压到 ≤23ms，约 9×）。

```
诊断先行（板上分段计时）: worst 帧 attention 194.9ms 占 front 95%（纯 C++ 标量 double）
优化阶梯(每档对拍零回归):
  B0   double 标量基线             full worst 252.0ms
  b1   float32 重写+BN折叠          119.6ms
  b2   +ffast-math                  114.9ms   (检查点1 ≤92 ✓)
  b3   +std::thread 双线程           87.8ms   (检查点2 ≤46 ✓)
  neon NEON 微核+fast_expf           67.76ms  (attention 40.3→20.3ms) ✓达标, mean 54.6ms
```

- **方法论**：先诊断后动手——四段计时锁定真凶是 attention 标量 double（非 ORT 调用、非 voxel 化），PC x86 同构验证归因后才开优化；每档设量化检查点、每档跑 50 帧 + 对拍，**优化不以数值回归为代价**
- **阶段3 撞墙改道的证据链**：PFN+attention 上 DPU 的全图 fp32 对拍 3.3e-3 可达，但 INT8 逐算子量化下 exp 平方链误差放大 64×、动态范围 473× 超量程（sim 发散）——用数据说话后拍板改道纯 CPU 路线
- **NEON 微核三板斧**：scores j16×c8 寄存器阻塞（c 升序 fma 保数值序）、AV 32 累加器驻 8 个 q 寄存器、fast_expf（2^r 六次多项式+阶码位运算，PC 实测 spatial 偏差 1.04e-6）；f_cluster 4-lane 求和序、LN 内部 double 等口径逐位保留
- **终验**：板上 50 帧 label 一致 **123/123=100%**、中心距 mean 0.332m 与 B0 基线不劣化、漏配/额外框清单完全一致；CMake 固化（BOARD 模式 FASTMATH/FRONT_THREADS 默认 ON，部署链裸 `-DBOARD=ON` 即达标档）

> 🖼 **图位 6｜优化阶梯瀑布/柱图**（建议制作，部署章的收尾图）
> - **内容**：横轴 B0/b1/b2/b3/neon 五档、双柱（front worst + full worst），数值 204.6/72.1/70.6/43.4/23.5 与 252.0/119.6/114.9/87.8/67.76，70ms 目标画横虚线；每档顶标注对拍零回归 ✓
> - **显示效果**：阶梯下探 + 达标线穿越的视觉冲击，「每一步都对着 golden 对拍走」的工程纪律随图带出

**板上检测框原始输出样例**（`deployment/board_results_20260928/00000336.txt`，首行框数，每行 `x y z dx dy dz rot score label`；已归档 `salon_assets/board_det_00000336.txt`）：

```text
2
27.000000 2.500000 -0.850000 1.648721 0.606531 0.000000 0.000000 0.880797 3
27.000000 2.000000 -0.850000 1.648721 0.606531 0.000000 0.000000 0.377541 3
```

这批 50 帧原始框文件 + `board_parity.py` 对拍脚本就是「板上验收」的全部证据链——不是截图，是可重放的数据。

### 5.5 上板工具链：版本铁律与两轮硬仗

- **版本铁律三代齐平**：Vivado=PetaLinux=**2024.1**、DPU IP DPUCZDX8G v4.x、编译容器 vitis-ai-pytorch-cpu 3.5.0.306、板端 VART 同代——判死先例（PetaLinux 2020.1 无 vai_q_onnx/ConvTranspose 编不过、混代 XSA 不兼容）都有记录
- **PetaLinux 幂等构建**：bitbake sstate 任务级增量 + 阶段标记；WSL 适配三招（线程内存钳制、github 网络自愈、时钟漂移防护）；rootfs/kernel/dtsi 配置固化（VART 库 + DPU 驱动现成模块 + cma=512M）——零驱动代码
- **两轮黑屏排障**：二分实验（裸机 BOOT 出字洗清启动链）→ 锁定 fsbl：petalinux 推导的 psu_init 与 XSA 内嵌版两套 PS 配置矛盾、DDR 时序 30+ 处不同、BOOT.BIN 分区字节替换必炸 ROM checksum → 终局以验证能亮的 BOOT.BIN + 自建 image.ub 上板验收（store 包内 md5 与板上卡逐字节同源）
- **SD 卡双分区制卡**：FAT32 启动三件按文件 cp（**严禁 dd 裸写**——毁 FAT32 元数据，审查项教训）+ EXT4 rootfs；flash-sdcard skill 强制证据确认（设备名/容量/可移动标志）防「应该就是这块卡」

**交付形态**：`deployment/store/` 8 个自描述部署包（898M），最新 20260928_1744 含 fat32/ext4 双分区全套上板物（BOOT.BIN/image.ub/rootfs/radar_pipeline_board/xmodel/weights/50 帧 points）+ report（golden_manifest/ladder_state/parity）——任何人拿到包可复现上板。

> 🖼 **图位 7｜store 包目录树**（建议制作，30 秒讲清交付物）
> - **制作**：`tree -L 2 deployment/store/20260928_1744_msrv2_pp64_ch2d_trunk` 截图，fat32/ext4/report 三块着色
> - **显示效果**：README/cpp/bin/weights/models/points/report 七件套一屏——「自描述」不是形容词，是目录结构

---

## 6. 工程方法论：AI 协作的 skill 化生产线

**理念：把「跑一次实验」从十几个手工步骤压缩成一行 tag 输入；可靠性设计进流程而不是靠人盯。** 7 个自建 skill 共 4125 行，覆盖训练→落袋→汇总→部署→烧卡全生命周期，历经 6 次大迭代（初版→对抗测试 14 项修复→两次架构收敛→持续硬化）：

- **model-train**：tag 列表 → 自动配对脚本+cfg → 模板渲染 → tmux 串行 → 跨实验报告；4 类保护 + H1-H8 硬化。**29 个实验脚本收敛为 1 份模板**（净删 1342 行）
- **resbag**：训练产物自包含落袋（cfg/权重/日志/指标 + `model_store.yaml` 总账）——60 个袋实证；数据源纪律「不反喂 LLM 写的记录、一律重读原始产物」，原子写+文件锁
- **modellist**：model.xlsx 两时机落表（YAML 生成时落架构、eval 后落 mAP），只写值不改格式
- **model-deploy**：六步链编排 + 对拍门禁 + 换模型分层适配——8 个自描述部署包；arch.json 无默认必填（防静默用错板）
- **flash-sdcard**：制卡烧写；破坏性操作强制证据确认
- **long-term-task-plan**：长周期任务宪章——0 介入三铁律、对抗审查门（5 个并行 agent 各从分支完备性/环境依赖/失败自愈/中断点/ETA 一个维度攻击 plan）、阶段存档恢复
- **get_token_cost**：AI 协作自身的成本度量（交互式 HTML）

**设计原则**：「能用脚本完成的，不依赖大模型」——确定性归档/调度/存档全脚本化，LLM 只做判断段；skill 间数据口径差异显式写进文档不靠记忆；自我攻击式质量文化（14 项对抗测试、resbag spec 4 维深审 24 findings）。

**人机分工**（cc-connect 桥接）：过滤工具调用/思考等中间噪声只推关键进度；长任务定时简报；人只做决策（选型/判读/定案），执行细节交给 agent。

**AI 协作成本也可视化**：[`salon_assets/token_cost.html`](salon_assets/token_cost.html)（单文件交互图，浏览器直接打开，源 `.claude/skills/get_token_cost/`）——
交互方式：柱状图按时间桶**堆叠各模型**的 token 用量，顶部可切换 Total/Input/CacheCreation/CacheRead/Output 五个指标，日期范围 + 30min/1h/4h/1day 颗粒度全前端过滤（无需重跑），hover 显示数值。
显示效果：一眼看出协作成本的时间分布——7 月高峰（移植+工具链+消融矩阵三线并进）与之后的收敛期对比，正是「工程能力沉淀为流程后成本下降」的直接证据。

---

## 7. 完成度与交付清单

### 7.1 分线完成度

- **训练线**：✅ 生产可用——三代 MSR 入库/贪心选型/60 run 落袋/model.xlsx 全指标；MSRv3 首批 2 模型已训完待定版落表。债务：vel 速度回归 MAE 未定量（eval 管线只算 AP）、MSRV2 model.xlsx 三行误贴待清理重落、vel 系列报告模板未填
- **部署线**：✅ 落板验收——8 个 store 包、板上 67.8ms 达标。债务：INT8 -8.43pt 待 per-channel 路线、DPU 实交付 B512（书面约定 B1600）、vel head 未接 tracker 量测链
- **下游跟踪线**：✅ 四族滤波/21 指标/回灌体系全实装（252 例测试）；债务：行人横穿泛化 MISS（训练 94% 行人 ±12° 方位角内 + 回波稀 + 模型内 0.2 阈值先裁，泛化类级 Ped MOTA 9.8%，改善首选举横穿样本入训/降阈值，同锥桶 type7 先例）、motion_status/isghost 恒 0 留给时序判定、test_filter 9 例测试桩待同步

### 7.2 交付产品清单（可指认、可复现）

- **代码**：pcdet 自研+移植 103 文件（head 家族 2919 行/注意力 349 行/2D 骨干 1668 行）、deployment C++17 全链 13 文件（三开关 CMake）、可视化体系 11 脚本 2967 行、tools 训练/评估/回灌脚本 24 个、skills 4125 行、tracker 6606 行
- **配置与实验矩阵**：101 YAML + 101 SH（VoD 46 + MC 46 + MSRV2 7 + MSRV3 2）、tracker cfg 64 份、dataset cfg 三代齐备
- **模型仓**：60 个 resbag 袋 + store 5 个精选包（含权重与全套指标）；3 份 model.xlsx
- **部署包**：8 个自描述 store 包（898M），最新含双分区全套上板物 + parity 报告
- **报告与基准**：17 份跨实验报告、五份 stage 阶段报告 + next_stage_cfg 固化链、DEPLOY_GUIDE/deploy_checklist、PTQ 教程、MOT 指标百科、golden 50 帧 + manifest、S1-S4 parity 实测、板上 50 帧输出 + 对拍脚本、13 份 MOT eval 报告

### 7.3 工作量总账

- 时间线：2026-05-19 → 09-28（4.4 个月），97 commits，+182k/-34k 行；三幕演进：复现锚定（7 月上旬）→ 自研迭代与消融矩阵（7 月中-8 月）→ 系统集成落板（8 月下-9 月），7 月为绝对高峰（57 commits）
- 训练：63 run / 1802 ckpt / 491 eval / 1840 张 viz（output/ 21G）
- 效率副产物：CenterHead 训练提速 2.4×、GPU eval 提速（numba → pcdet iou3d_nms）、板端 3.7×

---

## 8. 工程价值

1. **轻量化证据链完整闭环**：0.19-0.82M 参数模型族在三代数据上稳定对标/反超 1.64M RadarNeXt 与 4.86M 标准 PointPillar，且直接完成 INT8 上板——不是纸面 FLOPs 对比，是「精度≈重量级基线、参数低 1-2 个数量级、板上 67.8ms」的端到端证据
2. **每一次决策可审计**：贪心消融有固化链、每 run 有 resbag 总账、部署有对拍门禁、板上验收有原始框文件——从 cfg 到板端，任何数字都能在 5 分钟内指出处
3. **度量正确性先于数字好看**：rotate_iou 3D AP 全错修复、z/h 口径消融、pickbest 双口径、FLOPs bs 口径勘误（radarNeXt 旧值夸大 8×）——最贵的度量错误全部变成了文档与单测
4. **可复用的方法论资产**：贪心消融协议、无人值守韧性链、resbag 落袋、S1-S4 对拍门禁、PTQ 梯子、skill 化生产线——换数据集/换模型/换板卡均可直接复用
5. **AI 协作范式的实证**：7 skill + cc-connect 人机分工，单人 4.4 个月完成训练+跟踪+部署三线交付——工程能力沉淀为流程而非个人经验

---

## 9. 踩坑-修复精选（训练与部署侧）

- **3D AP 全错（Car_3d=0）**：rotate_iou_pcdet criterion=2 误返 IoU 致交集面积算错 → 改返 BEV 交集面积，修复后整批判重评
- **flip 污染速度通道**：镜像后 dop_y_gnd 与几何 y 方向矛盾 → NEG_POINT_COLS 同步取反 + 数值单测
- **CenterHead 训练慢 2.4×**：get_targets_single 逐目标 GPU 碎调用（CPU 70%）→ numpy 向量化 + 1ulp 对拍 + CLAUDE.md 制度化
- **黑屏两轮**：两套 PS 配置矛盾且 DDR 点不亮 → 二分实验锁定 fsbl、逐字节对齐重打 BOOT.BIN
- **板上 attention 40ms 不动**：瓶颈在访存不在算术（60MB 反复搬运摊不进 32KB L1）→ NEON 微核寄存器阻塞 + fast_expf
- **protobuf7 撞 tensorboardX**：训练脚本显式 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 绕过
- **eval 30 分钟死等**：eval_all 无待评 ckpt 时立即退出，不再死等

---

## 10. 下一步

- **INT8 精度攻坚**：vai_q_onnx 落地 → per-channel / 敏感层 FP16，冲 -8.43pt → ≤-2pt 门禁
- **vel head 接线 tracker**：box[7:9] 速度量测替换 vx=vy=0 占位，泛化域 MOTA 有望再上一档；补速度回归 MAE 评估
- **MSRv3 定版**：极稀疏场景（~30 点/帧）的 anchor/增强调参，Truck AP=0 攻坚
- **MSRV2 model.xlsx 误贴行清理重落**，vel 系列报告补填
- **部署链回填**：neon 终版对拍实测数值回填部署包 report；B512→B1600 协商

---

### 附：口径说明（引用数字前必读）

- **MSR v1（MC_DATASET，4 类）与 MSRv2（5 类含 StaticObject）是两代数据集**，mAP 不可跨代混比；§3.3 五模型表为 v2 五类口径（pickbest 末 10ep），对标准 PP 的 +7.0pt 为 v2 四类口径（`experiments/MSRV2_DATASET/model.xlsx`）
- VoD 指标均为 val R40 moderate EAA；FLOPs 有 bs=1 与 thop bs=4 两套口径（radarNeXt 单帧实为 4.5G，旧 35.5G 为 bs8 误记）
- MOT 的 v1 与 v2 MOTP 口径不同源，跨代只比 MOTA/HOTA 趋势
- 板上计时为 50 帧 worst 口径；INT8 -8.43pt 为 50 帧子集 R40 配对口径（与全量 eval 不同源）
- 主要出处：`experiments/*/stage*/`、`experiments/*/model.xlsx`、`output/train_log/*/_report_*.md`、`deployment/DEPLOY_GUIDE.md`、`deployment/store/*/report/`、`connect.md`、`mynote/*`、`/mnt/d/DataSet/.trained|.generalization/v2/*/mode2/eval.default/`
- **图表产物**：文中图片与数据均归档于 `salon_assets/`（同帧 BEV 三图为 testing 集帧 00001671 重渲——`visualize_msr.py --split testing --indices`，非训练集帧；MOT 曲线与 metrics 取自数据盘 eval.default、板上框输出取自 board_results_20260928、token_cost.html 取自 get_token_cost skill）；标「图位 N」的为建议制作位，框架与预期显示效果已给出

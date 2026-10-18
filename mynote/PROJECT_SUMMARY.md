# RadarPillar + Tracker 工程阶段性总结

> 覆盖范围：检测模型 / tracker 跟踪链路 / 部署上板 / 数据集与工具链，全主线。  
> 数据截至 2026-10-18（工作区含未提交的 MSRv3 eval/viz 修复）。  
> 纪律：全部数字取自仓库可追溯来源，来源随文标注；无仓库出处的数字一律不写，或明确标注"未落盘"。证据缺口集中列于附录 A。

---

## 0. 总述

**定位**：以 4D 毫米波雷达点云为输入的轻量化 3D 目标检测（RadarPillar）→ 跟踪全链（tracker/）→ FPGA 板端 INT8 部署的端到端工程。检测架构源自 RadarPillars 论文（arXiv:2408.05020），在自有 MC-A1 雷达数据（MSR 系列）上自适应改造。

**主线四阶段**（97 commits，2026-05-19 ~ 2026-09-28，约 4.4 个月；阶段划分依据 git log）：

1. 移植与复现保真（2026-07，VoD 公开集）：RadarNeXt / RadarPillars 双复现 + 数值对拍
2. 架构消融与选型（2026-07 ~ 08）：VoD stage0-4 贪心链（46 cfg）→ MSR v1 四维消融矩阵（39 模型 / 41 训练，一周内完成）
3. 跟踪全链与 SIL 回灌（2026-09）：MSRv2 五模型套件 + vel head + 双模式回灌体系
4. 部署上板（2026-09 ~ 10）：ONNX → INT8 → DPU → 板上验收 + Aries 新板迁移

**头条成果**（溯源见分节）：

- VoD 复现 3D mAP 52.56 vs 论文 50.70（+1.86），参数 0.1865M vs 论文 0.27M（-30.9%，差异集中在 backbone_2d，复现笔记对该差异标红区警示——结构与论文非逐层一致）——AP 口径移植保真达成（`README_zn.md` L14-21、`note/radarpillar复现结论.md`）
- 同任务下比 RadarNeXt 轻约 9 倍还更准：stage0 RadarPillar 0.184M / R40 48.71 超 RadarNeXt-MDFEN 1.64M / 45.46（`salon.md`、`experiments/PUBLIC_DATASET/stage0/`）
- MSR v1 最优 2D 92.93（pp64_ch2d_trunk，0.815M）超 RadarNeXt 复刻 92.10（1.64M），省 22% 计算量（`experiments/MC_DATASET/model.xlsx`）
- 3D 轻量档 centerhead3d_trunk：1/6.1 参数、1/3.2 计算量，保持 RadarNeXt 3D 精度 99.2%（91.05 vs 91.82）——tracker 刚需（z/h 输出）选型
- 资源地板 repdwc32：0.060M / 0.370G，全矩阵最小（`experiments/MC_DATASET/model.xlsx`）
- 部署对拍 S1-S4 全 PASS（worst 误差 5.1e-7 ~ 4.1e-4，逐段见 2.1）；板上 50 帧全链 67.8 ms/帧（目标 ≤70 ms）（`deployment/store/20260917_1814_*/report/parity_report.md`、`deployment/BOARD_DEPLOY.md`）
- 训练提效：CenterHead target 向量化后同头家族 epoch 197 s → 63 s（≈3.1×，run 日志首末时间戳推算 + CLAUDE.md）

**当前最大未决问题**：INT8 量化掉点天花板 -8.43 pt（门禁 ≤1~2 pt）未解决；横穿行人泛化缺口（泛化集 mAP 个位数）；MSRv3 eval 口径异常修复未收口。

---

## 一、模型构建

### 1.1 设想目的

- **工程命题**：车载 4D 雷达（MC-A1）点云 → 轻量检测 → tracker 跟踪 → ZU3EG 级 FPGA（双核 A53 + DPU B512）实时运行。资源约束决定了"参数 0.06M~0.9M 量级 + 单帧 GFLOPs 个位数"的选型空间（`mynote/ZU3EG_部署评估与方案.md`）。
- **设计文档即实验蓝图**：`mynote/McRadarPillar.md` 把架构拆成六段流水线，每段显式给出候选空间——VFE PN[32/64/128]、Attention head 数 1/2、backbone PP/RepDwc × 通道档、neck concat/FPN/MDFEN、head anchor/center、NMS 传统/pillar-mask。后续消融实验严格沿候选空间扫描，"设计文档→消融矩阵→模型表"一一对应。
- ispassable（可通行性）通道：MC-A1 独有字段设计（文档明确标注"无论文依据"），**设计未落地**（pcdet/ 下无实现）。

### 1.2 架构（权威配置）

权威配置 `experiments/PUBLIC_DATASET/YAML/a1.yaml`（VoD stage1 胜者）；MSR 系 7 个 YAML 位于 `experiments/MSRV2_DATASET/YAML/`。数据流（MSR 口径）：

```
6 列点特征 [x,y,z,dop_x_gnd,dop_y_gnd,rcs]
  → pillar 化（1.0×0.5 m pillar，16 点/pillar，range x[0,200] y[-20,20]）
  → PillarVFE（PointNet 风格 PN32，逐 pillar max）
  → PillarAttention（32 通道 1-head self-attention，gather 非空 pillar → scatter 回 BEV）
  → PointPillarScatter（BEV 伪图）
  → BaseBEVBackbone [3,5,5]×32（三级下采样）+ UPSAMPLE[1,2,4]+CONCAT[96]
  → Head（AnchorHeadSingle 或 RadarNeXtCenterHead 系）
  → 多类 NMS / pillar-mask 免 NMS
  → pred_dicts（7 维框 [x,y,z,l,w,h,rot]；vel 版 9 维 +[vx,vy]）
```

- **特征工程**：径向速度分解（`v_r_comp·cosφ / sinφ`）与对地多普勒列 dop_x_gnd/dop_y_gnd 是 MSR 系 6 列输入的核心（`pcdet/models/backbones_3d/vfe/pillar_vfe.py`、`tools/cfgs/dataset/msrv2_dataset.yaml`）。
- **head 变体文件群**（`pcdet/models/dense_heads/`）：radarnext_center_head.py（基类，8-cat z/h 联合监督）、_2d_noz（6-cat，z/h 双去）、_abl_z / _abl_h（7-cat，单加回 z 或 h）、_trunk/_merged/_narrow（瘦身族独立文件；noconv/k1 为 yaml 参数变体，无独立文件）、_2d_trunk_vel（8-cat+vel）。

### 1.3 创新性（如实三层分层）

**第一层：论文模块引入 + 工程化**（对外统一口径；2026-10-09 已纠正早期"自研"表述，纠正记录存于工程记忆，`salon.md` 中仍有纠正前旧表述）：

- PillarAttention：设计来自 RadarPillars 论文（`note/radarpillar复现结论.md`:32"与论文一致"）；工程化点 = key_padding_mask 摆脱逐样本 for 循环、单次 batch attention、FFN 通道配置化（`pcdet/models/backbones_3d/pillar_attention.py` "修改4"注释）。消融实证：+6,464 参数（+3.6%）换 mAP_R40 +3.3（45.43→48.71）（`salon.md`:103）
- RepDWC / MDFEN / RadarNeXtCenterHead：RadarNeXt 原版拓扑逐参数对齐移植（`msr_radarNeXt.yaml` 头注：对齐原工程 radarnext.py）
- 注意：这层的价值是"移植保真 + 工程化改造"，不 claim 架构发明权

**第二层：自研工程设计**（仓库可溯源的设计与实证）：

- **vel head**：+130 参数将对地 vx,vy 纳入 head 监督，MSRv2 Ped BEV 28.32→32.30（+3.98，ep100 口径；best 口径见 4.2）（`salon.md`:111、`pcdet/models/dense_heads/radarnext_center_head_2d_trunk_vel.py`）
- **head 瘦身族（trunk/merged/narrow）**：共享 trunk + 每头 1×1；ch3d_trunk 精度反升（2D +0.44 / 3D +0.35）同时参数 -36%、FLOPs -47%（`mynote/实验结论整理1.0.md` 表 1-2）
- **pillar-mask 免 NMS**：推理期非空 pillar 外 heatmap logit / anchor cls 压 -1e9，NMS_PRE_MAXSIZE 4096→1024，训练零改动（`radarpillar_center_head.py` / `radarpillar_anchor_head_single.py`）
- **对地速度分解特征链**（USE_GND_VELOCITY）与 MSR 伪 camera 轴 eval 变换（免 calib，IoU 与 lidar 系严格等价，`pcdet/datasets/msr/msr_utils.py`:181-193）

**第三层：方法论产出**（可复用的实验科学）：

- 四维消融矩阵体系 + 判定纪律（见 1.4）
- z/h 联合监督机制分析：五臂受控实验 + 捷径分析，结论"共享 backbone 下监督是特征的老师"（`mynote/LESSONS_LEARNED.md` Chapter 2）
- 口径纪律：FLOPs 统一 bs1、数据修正记录落盘（`mynote/实验结论整理1.0.md` 第四节）

### 1.4 消融实验体系（方式与结论）

**三层推进**：

1. **VoD stage0-4 贪心链**（46 cfg，`experiments/PUBLIC_DATASET/configs_summary.md` + stage*/stage_stats.csv）：stage0 三 baseline 锚点 → stage1 3D backbone 注意力（a0 直连/a1 PAtt/a2 SE/a3 SEDWConv/a4 PAttRes 变体）→ stage2 2D backbone → stage3 neck → stage4 head 十行对比。逐段贪心保留胜者。
2. **MSR v1 四维矩阵**（39 模型 / 41 训练，2026-08-18~08-24，`experiments/MC_DATASET/model.xlsx` 40 行 + `mynote/实验结论整理1.0.md`）：HEAD（换头 + 5 档瘦身）× BACKBONE（13 变体：PP 通道几何族 / RepDwc 族 / stride 族）× NECK（UPSAMPLE+CONCAT vs MDFEN）× VFE（PN32→PN64，7 对自配对）。
3. **MSRv2 五模型套件 + vel 迭代**（`experiments/MSRV2_DATASET/SH/` 7 脚本）：radarpillar / radarNeXt / pp64_ch2d_trunk / ppmix_ch2d_trunk / centerhead2d_trunk 五分位代表 + pointpillar 标准基线 + vel 头三次迭代。

**受控实验设计范本（z/h 五臂，`mynote/LESSONS_LEARNED.md` Ch.2）**：seed=42、80 epoch、末 10 ckpt 均值压 run 内噪声、lossup 臂精确复刻 loss 量级以隔离"量级/内容"变量、双阈值召回（rec@0.25/0.5）+ matchIoU/dimErr 三件套分离"检出/框精度"。

**判定纪律**：差距 <1 mAP 判"相当"；单 seed 结论一律标"方向性"；ch2d 家族无 z/h、3D AP 结构性恒 0，跨家族仅比 2D。

**核心结论**（`mynote/实验结论整理1.0.md`）：

- 有效杠杆排序：换 head（2D +6.83~+7.64，Ped +23.8~+26.3）> head 瘦身 trunk（零精度代价，FLOPs -47%）> backbone 容量加宽（pp64/ppmix 2D +2.0~+2.7）
- 无效/有害杠杆：RepDwc 同宽三档全面真伤（-2.5~-3.9）；MDFEN 精度亏参数反增；PN64 七对三涨四跌高方差"免费抽奖"；多头 PAtt 负向；stride 变体负收益
- 规律：精度主线 = Ped 小目标（13 变体摆幅 17.4），Car/Truck 97+ 饱和；"true-2D 更省算"不成立（去 z/h 仅省 ~12% 算力）
- z/h 结论（如实含置信度）：单独 z 有害（BEV 49.72 < 双去 51.87，场景共模捷径）、单独 h 无效（50.52，类别查表捷径）、z+h 联合有效（52.59，+0.72 mean / Ped best +2.22，其中真检出增益 ~0.9、~1.3 为阈值交互；置信度 ~75%，n=1 seed）
- 部署选型结论：第一步定"要不要 3D"（tracker 刚需 z/h 只有 ch3d 家族），第二步按算力档取该族最优——形成 B400/B800/B1600 三档 8 款选型表

**实验空缺（如实继承）**：ch3d_trunk × 宽 backbone 从未验证（3D 榜大概率未封顶）；RepDwc 宽度档未覆盖 RadarNeXt 真实宽度；关键对（trunk vs ch3d 等 <1 mAP 带）缺种子重复；MDFEN 对比捆绑 bs 与分辨率两变量未拆。

---

## 二、工程化现状

### 2.1 已落地效果（数字）

**部署链全通**（部署模型 msrv2_pp64_ch2d_trunk；链路定义 `deployment/deploy_checklist.md`）：

- golden 基准：testing 174 帧确定性抽样 50 帧，含 ckpt sha256 与复检标记（`deployment/simulator/golden/manifest.json`）
- S1 ONNX vs PyTorch：50 帧 allclose(1e-5) 全过，worst_abs 5.5e-5；S2 C++ 前段 worst 2.2e-6；S3 C++ 后段 worst score 5.1e-7；S4 C++ 全链 worst score 4.1e-4 / 中心距 7.2e-4 m，失败帧 0（`deployment/store/20260917_1814_*/report/parity_report.md`）
- INT8：vai 量化容器 vitis-ai-pytorch-cpu 3.5.0.306 → xmodel 编译目标 DPUCZDX8G B512（meta.json 实测口径）
- 板端：PetaLinux 2024.1 统一工具链五件镜像 + 交叉编译 radar_pipeline_board；AXU4EV 板 09-28 验收 50 帧、全链 67.8 ms/帧（≤70 ms 目标，余量约 3%；此数字当前仅存文档转述，一手对拍产物缺失，见附录 A）
- 产物：7 个 store 部署包（5 个板端模型包 165~172M + 1 个 Aries 系统卡 99M + 1 个 PC 仿真链包 9.0M），deployment/ 共 8158 文件 / 18G

**tracker SIL 回灌体系**（`tools/scripts/test/replay_tracker_msrv2_sil*.sh`）：

- mode1 检测回灌 63 序列（检测 Obj 原地覆盖 0201 对象流）、mode2 跟踪回灌 63 序列（RoadTest 23 + SceneTest 40，0200+0201 双流覆盖）
- 工程自洽：原对象流逐文件备份 0201bak、`.done` 幂等标记整体跳过、失败重试 2 次、以 0101 点云流定帧数避免自我污染

**训练与实验基建**：model-train / resbag / modellist 三 skill 链（tmux 串行、断点续跑、自动落袋）；65 个 model_store run + VoD stage 链约 29 次评测；跨实验聚合报告 14 份（`output/train_log/msr/_report_*.md`）

**工程效率**：CenterHead get_targets_single numpy 向量化，同头家族 epoch 197 s → 63~65 s（按 run 日志首末时间戳 / 80 epoch 推算：全程 4h23m → 1h25m；≈3.1×。CLAUDE.md 记载 2.4× 为相对 anchor 基线 81 s 口径）；板端 front 段 float32 重写 + NEON 微核 + 双线程（应对两核 CPU、无 libgomp）

### 2.2 特点与优势

- **证据链文化**：每步对拍有数字、golden 确定性复检、md5 三方校验、"S1-S4 过但 full-quant 失败 = 红旗落袋"的显式风险流通规则、变更必重编红线（固化于 `.claude/skills/model-deploy|flash-sdcard|build-sdcard/SKILL.md`）
- **轻量化硬优势**：全家族 0.060M~0.815M；MSRv2 上 0.19M 的 radarpillar 3D 三类均值 43.79 vs 4.861M 标准 PointPillars 43.34——同为三类口径、1/25 参数打平（见 4.2 口径警示）
- **端到端全栈自研**：bin 流格式（ctypes ×100 量化读写）→ 数据集 → 训练 → 部署 → 板上输出 → SIL 回灌，全链无外部黑盒
- **复现/排查纪律**：反事实实验定位根因（VoD Cyclist AP 17.50→64.56，复现度 97.2%）、预测分布对比、口径修正落盘

### 2.3 不确定与缺点（如实清单）

1. **INT8 掉点未解决**：梯子六档中 d0_50（50 帧校准）worst Δ -8.77 pt，d2/d3/d4/d5 四档稳定收敛于 worst Δ -8.43 pt（Car BEV R40）、mean -3.66 pt；calib 扩帧零增益、mixbit 无效、QAT 仅 +0.34 pt。根因判定 = per-tensor 量化粒度 × heatmap 头敏感；解法指望 vai_q_onnx per-channel 路线，但该路线当前是债（容器无 API 入口，探测笔记引用悬空）。全部现行 store 包按红旗规则带风险流通（`deployment/store/20260928_1744_*/models/quantize/`、`mynote/PTQ.md`）
2. **泛化缺口（横穿行人）**：训练集 94% 行人在 ±12° 方位角内 + 横穿回波稀 + 模型内 0.2 阈值先裁；泛化 8 序列 radarpillar mAP bev 1.7 / 3d 2.5、radarNeXt 6.8 / 7.9（`output/generalization_eval_report.md`）；泛化类级 Ped MOTA 9.8%（`salon.md`:355）
3. **MSRv3 未收口**：Truck 3D AP=0、Car/Vehicle 出现 3D > BEV 违反合法性判据（eval 口径修复进行中、未提交）；两 run status=blocked、xlsx 指标未回填（`output/train_log/msrv3/`）
4. **统计纪律债**：全消融矩阵单 seed；trunk vs ch3d（+0.44/+0.35）、ppmix vs pp64（-0.64）均在"相当带"内未做种子复验；PN64 方差来源未定（`mynote/实验结论整理1.0.md` 第三节）
5. **部署侧风险**：书面约定 B1600 但全部 xmodel 实编 B512（文档与产物不一致）；67.8 ms 一手数据缺失；Aries 新板阻塞在网卡物理数据路（ENET3 EMIO vs 板上 MIO 直连，软件侧已通）；两核 CPU 下 67.8 ms 余量仅 3%
6. **落盘卫生**：MSRv2 training 帧数未落盘（仅 testing=174 有三处出处）；`salon.md` 残留纠正前"自研"表述；头部汇总表 FLOPs 旧口径（radarNeXt 曾夸大 8 倍后勘误）

---

## 三、工程结构提炼

**六大分层**（数据 → 模型 → 实验 → 跟踪 → 部署 → 工具）：

- **数据层** `pcdet/datasets/`：msr_dataset.py（三代共用类，CLASS_LABEL_MAP 覆写机制）+ dataset yaml 10 份（`tools/cfgs/dataset/`）+ KITTI eval 内核改造（vod_eval 参数跳过图像过滤、MSR 段伪 camera 轴变换、get_msr_eval_result）
- **模型层** `pcdet/models/`：pillar_vfe / pillar_attention（+res/resA/resB 变体）/ pointpillar_scatter / base_bev_backbone / rep_dwc / mdfen_neck / dense_heads 变体文件群（1.2 节）
- **实验层** `experiments/<DATASET>/{YAML,SH,model.xlsx}` + `output/train_log/<ds>/`（run + resbag + model_store.yaml + 聚合报告）。模型表命名语法由 `mynote/McRadarPillar.md` 定义（PN[n]/PAtt[head:H]/PP[l]*[c]/RepDwc/UPSAMPLE+CONCAT[c]/anchorbased|anchorfree），配置↔表格↔文档三方对应
- **跟踪层** `tracker/`：12 模块 3927 行主代码 + 2883 行测试 + 67 份 cfg。KM（Kuhn-Munkres）全局最优关联（GNN 口径，欧氏/马氏 gap）；滤波四族统一路由（α-β / KF-CV / EKF-CTRV / IMM，状态恒 [x,y,vx,vy]）；双通道速度更新（差分 v_meas + α-β 平滑）；存在概率记分航迹管理（出生 55 / 增 10 / 减 15）；FrameLogger 零侵入逐帧归因（NEW/DROP/DEAD 全链路日志）
- **部署层** `deployment/`：converter（13 脚本：导出/量化/编译一键链 + INT8 梯子）+ simulator（golden + 对拍）+ board（cpp 源 26 文件 / petalinux / vivado / weights）+ store（7 部署包，自描述清单）
- **工具层** `tools/`：scripts 训测脚本族 + visual_utils（11 文件 2970 行，draw_box_bev 单入口、车规朝向、序列定轴）+ skills（model-train / model-deploy / resbag / modellist / build-sdcard / flash-sdcard）

**解耦机制**：Cfg 配置树单一来源（tracker 10 段配置）；dataset yaml 与 model yaml 正交；每实验 resbag 自包含归档 + model_store.yaml 汇总；部署包 store 自描述（report 含 golden/parity/board 三 manifest）。

**端到端数据流**：雷达 bin 流（0101 点云 / 0200+0201 对象流）→ dataset → 训练/eval → best.pth → resbag → 部署链（golden → ONNX → INT8 → xmodel）→ 板上 radar_pipeline → 0200/0201 航迹 bin 流（×100 定点量化）→ SIL 回灌下游。

---

## 四、工作量与结果

### 4.1 工作量统计

- **git**：97 commits（feat 41 / chore 16 / fix 15 / refactor 12 / docs 7 / data 4，另 2 条非规范格式），2026-05-19 ~ 09-28；15 条 fix 中热点 = eval/IoU 3 条 + 增广/特征列 3 条 + 脚本/环境 4 条
- **训练评测量**：65 个 model_store run（VoD 9 / MSR v1 41 / MSRv2 12 / MSRv3 2 / 快照 1）+ VoD stage 链约 29 次评测 ≈ **94 次训练/评测**，覆盖 4 套数据集
- **配置与脚本**：模型 YAML 101 份（PUBLIC 46 / MC 46 / MSRV2 7 / MSRV3 2）+ 训测 SH：experiments 下 107 份、含 tools/scripts 共 123 份 + dataset yaml 10 份
- **代码规模**：tracker 6810 行（主 3927 + 测试 2883）；可视化 11 文件 2970 行；部署链脚本约 29 个 + 每包 C++ 源 19 文件（src 18 + test 1）；deployment/ 8158 文件 18G
- **数据集入库**：VoD / MSR v1 / MSRv2 / MSRv3 四套（v3：training 1675 / val 716 / testing 4，每帧约 30 点极稀疏）

### 4.2 分数据集结果汇总（headline，口径随表标注）

**VoD（复现保真，3D mAP R11，5-frame）**：本工作最优种子 52.56（41.6/44.8/71.3）vs 论文 50.70（41.1/38.6/72.6），+1.86；3 种子均值 51.43±0.99；参数 0.18650M vs 论文 0.27M（`README_zn.md`）

**MSR v1（MC 消融矩阵，mAP@R40 moderate EAA，4 类均值）**：

| 代表模型 | 2D | 3D | 参数 | FLOPs | 定位 |
|---|---|---|---|---|---|
| msr_radarpillar（基线） | 83.14 | 83.25 | 0.187M | 0.738G | anchor 基线 |
| msr_pp64_ch2d_trunk | **92.93** | 结构性 0 | 0.815M | 3.524G | 2D 全矩阵最高 |
| msr_radarNeXt（复刻） | 92.10 | **91.82** | 1.640M | 4.502G | 3D 最高 / teacher |
| msr_centerhead3d_trunk | 90.41 | 91.05 | 0.270M | 1.405G | 3D 轻量档（tracker 选型） |
| msr_repdwc32 | 77.65 | 79.18 | **0.060M** | **0.370G** | 资源地板 |

（全表 40 行见 `experiments/MC_DATASET/model.xlsx`；Delta 矩阵与判定见 `mynote/实验结论整理1.0.md`）

**MSRv2（车载实测难度，moderate R40 per-class，resbag eval_results）**：五模型套件中 centerhead2d_trunk BEV 五类均值（含 Truck/Static）最高 52.21（Car 69.63 / Ped 28.05 / Cyc 53.94 / Truck 74.43 / Static 35.00，0.270M）；3D 三类均值（Car/Ped/Cyc）radarNeXt 47.47（1.64M）> radarpillar 43.79（0.19M）> pointpillar 43.34（4.861M），五类 3D 复算为 53.45 / 52.34 / 50.42（`output/train_log/msr/_report_msr_pointpillar.md` 与各 resbag）。vel head 三次迭代 Ped BEV 26.10 → 21.72（禁增强回退；注意 run1 为 msr 基座、跨数据集不可作单变量参照）→ 32.35（flip 速度列同步取反后；best 口径，无 vel 基线 28.05，净 +4.3）。**口径警示**：MSRv2 resbag 中 Car/Truck 出现 3D > BEV（如 radarNeXt Car 3D 82.93 vs BEV 67.39），与 MSRv3 异象同源、eval 管线待复核——上表 3D 数值均经该管线产出，引用须谨慎

**MSRv3（进行时）**：ep100 val 3D Car 35.28 / Cyclist 33.00 / Vehicle 55.61 / Truck 0；口径异常待修（`output/train_log/msrv3/`）

**部署（golden 50 帧门禁口径，eval_msr_boxes）**：FP32 Car_bev 63.19 / Ped 6.25 / Cyc 12.14 / Truck 46.80 / Static 7.92；INT8 全档 worst Δ -8.43 pt（`deployment/store/20260928_1744_*/models/quantize/ladder/`）

### 4.3 关键修复清单（择要，均有 commit 或文档）

- rotate_iou_pcdet criterion=2 误返 IoU 致 3D AP 全错（Car_3d=0）→ 改返 BEV 面积（commit 94dfc66）
- flip 增广镜像未同步取反点级速度列 → NEG_POINT_COLS 机制 + 数值测试（commit 3d2cb8b）
- CenterHead get_targets_single 逐目标碎 GPU 调用 → numpy 向量化 ≈3.1×（commit 6680815）
- MSRv2 testing infos 硬编码 has_label=False 致 AP 全空 → 参数化修复（msr_dataset.py:503 注释）
- VoD 5-class 聚合致 Cyclist AP 假性暴跌 17.50 → 3 类映射复现 64.56（LESSONS_LEARNED Ch.1）
- protobuf 4/5 与 tensorboardX 冲突 → 环境变量绕过（salon.md:390）
- numba WSL2+CUDA SIGSEGV → pcdet iou3d_nms GPU wrapper 替换（commit c117643）
- 锥桶（type7）零检出 → type7 入训 + StaticObject eval 口径（commit 861b670）

---

## 附录 A：证据缺口清单（如实）

| 缺口 | 现状 |
|---|---|
| 板上 67.8 ms 一手对拍产物 | 仅 `deployment/BOARD_DEPLOY.md` 文档转述；`board_results_20260928/` 目录与 `connect.md` 均不存在 |
| MSRv2 training 帧数 | 仓库无落盘（testing=174 有三处出处）；仅存工程记忆 |
| `mynote/nndct_api_探测_20260921.md` | 三处文档引用，文件本体在工作区与 git 历史均未找到（悬空引用） |
| RadarNeXt 论文原文数值对照表 | 未入仓，仅有复现值（stage0 报告内 tag 标记 0722_paper，非 git tag） |
| MSRv2 model.xlsx 回填 | 仅 4 行（radarNeXt / centerhead2d_trunk / vel 系等 8+ run 未回填，指标仅存各 run resbag） |
| MSRv3 全部指标 | xlsx 未回填、run status=blocked、eval 口径修复未提交 |
| `salon.md`:90,103 "自研核心创新" | 纠正前旧表述，与 2026-10-09 口径（论文引入+工程化）不一致 |

## 附录 B：主要来源索引

- 实验结论与消融：`mynote/实验结论整理1.0.md`（39 模型矩阵）、`experiments/MC_DATASET/model.xlsx`、`experiments/PUBLIC_DATASET/configs_summary.md` + stage*/stage_stats.csv
- 排查史：`mynote/LESSONS_LEARNED.md`（Ch.1 VoD 复现、Ch.2 z/h 根因）
- 架构设计：`mynote/McRadarPillar.md`；复现：`note/radarpillar复现结论.md`、`README_zn.md`
- 部署：`deployment/deploy_checklist.md`、`deployment/BOARD_DEPLOY.md`、`deployment/DEPLOY_GUIDE.md`、`mynote/PTQ.md`、`mynote/ZU3EG_部署评估与方案.md`、store 各包 report/
- 跟踪与回灌：`tracker/`（含 `doc/design/spec/` 三份设计文档）、`tools/scripts/test/replay_tracker_msrv2_sil*.sh`
- 训练产物：`output/train_log/{vod,msr,msrv3}/`（run + resbag + model_store.yaml + `_report_*.md`）、`output/trained_eval_report.md`、`output/generalization_eval_report.md`
- 汇报材料（注意 FLOPs 旧口径与"自研"旧表述）：`salon.md` 及同名 html

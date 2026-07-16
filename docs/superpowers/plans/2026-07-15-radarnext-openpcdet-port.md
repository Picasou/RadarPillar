# RadarNeXt → OpenPCDet 复现 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以 RadarPillar（OpenPCDet 工程）为 base，把 RadarNeXt（MMDetection3D 工程）按 OpenPCDet 风格忠实复现，实现 Rep-DWC backbone → FPN/MDFEN neck → CenterHead + dIoU，模型 build 成功、训练收敛、参数量与论文 Table II 吻合、VoD mAP 达到论文报告值。

**Architecture:** RadarNeXt 原生是 MMDet3D（registry + Python config + `Det3DDataSample`/`InstanceData` 数据契约）。移植目标是 OpenPCDet（YAML config + `Detector3DTemplate` 拓扑 + `batch_dict` 数据契约）。两者的模块边界一一对应：`voxel_encoder`↔`VFE`、`middle_encoder`↔`MAP_TO_BEV(PointPillarScatter)`、`backbone`↔`BACKBONE_2D`、`neck`↔`BACKBONE_2D` 内 FPN 段或新增 neck 段、`bbox_head`↔`DENSE_HEAD`。复现的核心是把 RadarNeXt 的"答案代码"逐模块翻译成 OpenPCDet 注册类，**数值/结构保持不变**，只改外壳与数据契约。

**Tech Stack:** PyTorch 2.4.1+cu124、OpenPCDet（spconv、pointnet2、iou3d_nms 等自研 CUDA ops）、CUDA 12.4（nvcc）、VoD KITTI-format 雷达点云数据集、Python 3.12.7（base conda 环境）。

---

## 全局约束（Global Constraints）

逐条抄自需求与研读结论，每个任务的验收都隐含遵守：

- **目标档位**：先交付 **FPN 版**（论文 Table II / Table I：**0.899M 参数 / VoD 47.98 mAP**），作为环境与移植正确性的验收锚点；再交付 **MDFEN 版**（1.580M / 50.48 mAP）作为进阶档。/goal 文本以 0.899M 为准。
- **环境（双环境策略，审计修正）**：
  - **FPN 档**：用 **base** conda 环境（Python 3.12.7 / torch 2.4.1+cu124）即可——FPN 是纯卷积，无 DCNv3 依赖，base 足够。补装 open3d + 编译 pcdet 算子。
  - **MDFEN 档**：**不接受跑不动**。base 的 Py3.12+cu124 对 DCNv3 CUDA 算子编译预期失败，故 **新建专用环境 `radarnext310`（Python 3.10 + torch 2.1 + cu121）**——这是社区验证 DCNv3 可编译最充分的栈。在该环境重新 `setup.py develop` 编译 pcdet ops + 编译 DCNv3。两个环境并存，互不污染。
- **GPU 限制**：RTX 3070 Ti **8GB** 显存。FPN bs=4 fp16 理论可行但无余量，**必须先 bs=1 冒烟测 `max_memory_allocated` 再线性推安全 bs**。MDFEN 档用 CUDA DCNv3 时显存可控，仍需 bs=1 冒烟反推；若用纯 pytorch DCNv3 兜底则显存更高，按反推 bs 执行。AMP（fp16）默认开启。**DCNv3 编译失败时按执行协议 §6 手段链穷尽尝试，不允许失败**（纯 pytorch 版是兜底保证）。
- **dIoU 损失（已澄清，勿重复造轮子）**：PillarNeXt 源码 `loss.py` 的 `IouRegLoss` 内部用的就是 `bbox3d_overlaps_diou`（3D IoU + `inter_diag/outer_diag` 中心距惩罚）——**这正是论文式（3）dIoU**。所以移植时：`with_reg_iou=True` 配 `IouRegLoss` 原样翻译即得 dIoU 损失；`with_iou=True` 配 `IouLoss` 是另一条独立的"IoU-score 回归辅助损失"（监督 head 预测的 iou 通道对齐真实 3D IoU），用 **aligned** 一对一 IoU（`boxes_aligned_iou3d_gpu`，OpenPCDet 无此函数需自实现或取对角线）。**不需要新增 diou3d_utils**，直接搬运 `bbox3d_overlaps_diou`。
- **代码风格**：严格模仿 OpenPCDet 现有风格——类继承 `VFETemplate`/`nn.Module`/`BaseBBoxCoder`、通过 `pcdet/models/**/__init__.py` 的 `__all__` 字典注册、配置用 YAML、forward 走 `batch_dict` 流水线、类带 `model_cfg` 参数。**不得**把 MMDet3D 的 registry/`MODELS.register_module()`/`Det3DDataSample` 直接搬进来。
- **数据契约**：VoD gt_boxes 为 `[x, y, z, l, w, h, r]` LiDAR 系，z 已抬到体积中心（见 `vod_dataset.py` 的 `loc_lidar[:, 2] += h/2`），与 RadarNeXt head 期望的 `gravity_center`（体积中心）语义一致——移植 `get_targets` 时**直接用 gt_boxes 即可**，无需再做中心转换。注意：RadarNeXt `predict` 末尾 `bboxes[:,2] -= bboxes[:,5]*0.5` 把 box 转回**底面中心**供评估——需核对 OpenPCDet VoD evaluator 的 z 语义决定保留与否（见 Task 4）。
- **点特征（9维陷阱，必改）**：VoD 雷达点原始 7 维 `[x,y,z,rcs,v_r,v_r_comp,time]`，RadarNeXt 原版 `Radar7PillarFeatureNet` 是**纯 7 维**。但 RadarPillar 的 `PillarVFE` 开 `USE_VELOCITY_DECOMPOSITION=True` 后会把 `v_r_comp` 再拆成 `vx,vy` → 实际 **9 维**，破坏 0.899M 参数量对账。**FPN/MDFEN 档 YAML 必须设 `USE_VELOCITY_DECOMPOSITION: False`**（保留 7 维，对应 `Radar7PillarFeatureNet`）。`use_elevation` 无独立开关，含 z 即 `USE_ABSOLUTE_XYZ: True`。
- **Voxel/Range/网格**：`voxel_size=[0.16, 0.16, 5]`（z 维须使 `round((2−(−3))/5)=1`，否则 `PointPillarScatter` 断言 nz==1 失败）、`point_cloud_range=[0,-25.6,-3,51.2,25.6,2]`、`grid_size=[320,320,1]`（dataset 自动算，不在 YAML 手填）。
- **参数量验收口径（必改）**：论文 Table II 的 **0.899M 是 reparameterize 后的推理态单分支参数量**。验收时必须 `build(训练态) → reparameterize_model() → 再 sum(numel)`；并在 TASK.md 同时记录训练态/推理态两套数值（训练态约 1.5~2x 推理态）。
- **类别顺序**：统一用 OpenPCDet VoD dataset 序 `[Car, Pedestrian, Cyclist]`；head 的 `tasks=[{num_class:3, class_names:[Car,Pedestrian,Cyclist]}]` 必须与 dataset `CLASS_NAMES` **完全相同**（head 内部 `gt_labels_3d` 直接 0/1/2 对应，rectifier 按 label 索引取值）。mmdet3d 的 `[Ped,Cyc,Car]` 仅作数值参考。
- **不做/约束**：commit 在 `feature/radarnext-port` 分支、**每完成一个 Task commit 一次**（master 保持干净；不 force push、不 commit 到 master）；不新增未确认函数签名；不动 tracker/；不降级 numba（base 已是 0.60.0+numpy1.26.4 正确组合）；装包前 `pip freeze > /tmp/base_before.txt` 留快照。

### 数据流契约（移植时所有模块遵守）

```
batch_dict:
  points            (N, 8)  [x,y,z,rcs,vr,vr_comp,time,batch_idx]   # PillarVFE 输入（7维+batch）
  voxel_features    (M, T, F) → PFN → (M, C=32)                      # VFE 输出（7维，关 decomposition）
  voxel_coords      (M, 4)   [b, z, y, x]
  → PointPillarScatter → (B, 32, 320, 320) BEV
  → RepDWC(stride 2×2×2) → [(B,64,160,160),(B,128,80,80),(B,256,40,40)]  # 多尺度
  → neck:
      FPN  (SecondFPN, upsample_strides=[0.5,1,2]) → (B,384,80,80)        # 三分支对齐到 80×80
      MDFEN (PAN+DCN+MultiMAPFusion, fusion_strides=[1,2]) → (B,384,160,160)  # 对齐到 160×160
  → DENSE_HEAD(CenterHead):
      SepHead(stride=2 for FPN: ConvTranspose 80→160 对齐 target；
              stride=1 for MDFEN: 已是 160×160 无需 deblock)
      → task 级 [hm, reg, height, dim, rot, iou] @ feature_map=160×160
      → target 的 feature_map_size = grid_size[:2]//out_size_factor = 320//2 = 160
```

> **分辨率自洽性（审计修正）**：FPN 档 neck 输出 80×80，但 head target 是 160×160，靠 `SepHead(stride=2)` 的 `ConvTranspose2d` deblock 把 80→160 弥合；MDFEN 档 neck 已输出 160×160，head `stride=1`。两档 head 的 `strides` 参数不同，YAML 必须分档配置，不可共用。

---

## 执行协议（Execution Protocol）

### 1. 执行期零询问·全程自主跑完（强需求，Lock Before Run）
**强需求：任务一旦启动，从 Task 0 一路自主执行到 Task 9，中途不停下来询问用户。** 所有技术选型一律采用 §2 锁定的默认值；遇到意外也自主决策到底，把决策与偏差记入 TASK.md。
- **遇到二选一**：采用锁定默认值，备选记 TASK.md。
- **不可逆操作**（删数据/覆盖唯一 ckpt/force push）：**默认禁止直接做**，一律改用安全默认自主执行，**不靠询问**——删除先移到 `*.trash`/备份（`ckpt → ckpt.bak`），覆盖先留快照；无安全等价物时跳过该步并记 TASK.md，继续后续。
- **硬阻塞**（如 op 编译失败）：按"主路径→备选1→备选2"顺序自主 fallback（Task 0/7 已列）；记 TASK.md 继续后续，不询问。**MDFEN 档 DCNv3 不允许失败**——见 §6 穷尽式手段链，必须试到 DCNv3 能用为止（纯 pytorch 版是兜底保证）。
- **唯一例外**：用户主动发消息介入时，响应即可。否则不中断。

### 2. 执行前已锁定决策（避免执行中询问）
- **neck 落地方式（审计修正）**：OpenPCDet `module_topology` 无独立 neck 槽位。决定 **新增一个 `SecondFPN`/`MDFENNeck` 类，作为唯一的 `BACKBONE_2D`**——即 RepDWC 与 FPN/MDFEN 串联封装进**同一个 backbone_2d 模块**（forward 内先 RepDWC 出多尺度，再过 neck 出单尺度），不新增拓扑槽位、不改 detector3d_template。文件结构以"单一 BACKBONE_2D 类"为准。
- **detector（审计修正）**：**不复用 PointPillar**——CenterHead 是 anchor-free，与 `Detector3DTemplate.post_processing` 的 anchor 契约（`batch_cls_preds/batch_box_preds`）不兼容。**新增 `CenterPoint` 风格 detector**（注册到 `detectors/__init__.py`），forward 训练期调 `dense_head.get_loss()`、测试期调 **head 自己的** `post_processing(batch_dict)`。
- **dIoU（审计澄清，最终方案）**：不新增 dIoU utils。损失两条线——① `with_iou=True` + `IouLoss`：IoU-score 回归辅助损失，监督 head 的 iou 通道，用 **aligned** IoU（搬运 `boxes_aligned_iou3d_gpu` 语义，OpenPCDet 无需自写可对 `boxes_iou3d_gpu` 取对角线），weight=`iou_weight`=1；② `with_reg_iou=True` + `IouRegLoss`：直接搬运源码的 `bbox3d_overlaps_diou`，**这就是 dIoU**，weight=`iou_reg_weight`=0.5。两条损失并存。
- **类别顺序**：统一用 OpenPCDet VoD dataset 序 `[Car, Pedestrian, Cyclist]`；head `tasks.class_names` 与 dataset 完全相同。mmdet3d `[Ped,Cyc,Car]` 仅作数值参考。
- **gt_sampling（审计修正）**：忠实复现 RadarNeXt（radar 点稀疏，不用 Copy-Paste）→ FPN/MDFEN yaml **显式 override 去掉 `gt_sampling`**，只保留 global scale + flip。`vod_dataset_radar.yaml` 默认开着 gt_sampling，必须在本档 yaml 里 override。
- **batch_size（审计修正）**：不直接上 bs=4。先 **bs=1 冒烟**测 `torch.cuda.max_memory_allocated()`，反推单样本占用 → 线性推安全 bs。FPN 目标 4（OOM→2）；MDFEN 目标 1（OOM→梯度累积等效 batch=2）。**MDFEN 不接受"跑不动就放弃"——OOM 先排查环境/DCNv3 是否真用 CUDA 版，不得轻易降级。**
- **双环境（审计修正）**：FPN 用 base(py3.12/cu124)；MDFEN 新建 `radarnext310`(py3.10/torch2.1/cu121) 以编译 DCNv3。两环境并存。
- **MDFEN 训练成本（注明原因，审计修正）**：MDFEN 含 PAN 双向 + MultiMAPFusion + 多个 DCNv3，结构重于 FPN；即便用 CUDA DCNv3，单 step 仍慢，80 epoch × 5139 帧预计 **显著长于 FPN 档**（FPN 估 6-15h，MDFEN 估 1.5-2x）。此为**结构本身的计算量**，非环境缺陷，属正常预期，Task 6/7 注明。
- **MDFEN 对抗审查（审计修正）**：MDFEN 移植完成后**必须 fan-out 多 agent 对抗审查**（源码忠实度 + OpenPCDet 适配 + DCNv3 数值正确性），与论文原实现逐项对比，**有差距则反复迭代**到无差距（见 Task 7 Step 7）。FPN 档复用 Task 0 阶段的同一套审计方法。
- **验证主路径（用户方法论，核心）**：**数值对拍（Parity Test）是移植正确性的主判据**，不是训练 mAP。每个核心模块（RepDWC/SecondFPN/MDFENNeck/CenterHead/loss/detector）与 RadarNeXt 原版**同输入同权重对拍**，输出 `allclose` 即 PASS。对拍全过 = 结构移植正确。**短训练（10-20ep）只作双保险**，看 loss 下降即可，不要求 mAP 精确对账。详见执行协议 §5 与 Task 4.5。
- **训练前置（审计修正）**：全量/短训练前必须先 **overfit-1-batch（1帧×200step，loss 应回降到≈0）**，验证训练管线无 bug。
- **AMP**：默认 fp16 开启省显存。
- **seed**：训练用 `--fix_random_seed`（OpenPCDet 既有 flag）保证可复现。
- **commit**：在 `feature/radarnext-port` 分支**每完成一个 Task commit 一次**（master 保持干净；不 force push、不 commit 到 master）；master 留作基线。

### 3. TASK.md 执行日志（实时状态镜像，非累积流水账）
新建 `docs/superpowers/plans/TASK.md`，**是"实时状态镜像"，随进度持续改写，始终保持反映此刻真实状态——而非罗列累积历史**。固定四段结构：
- **当前进度**：进行中 Task/Step、已完成列表、阻塞项——**始终是最新状态，过时项即删/改，不保留历史堆叠**
- **关键结果**：参数量/对拍结论/mAP 填**实测最新值**（`[待测]`→实测值，旧值覆盖，不并存）
- **决策与偏差**：每条 1 行——**仅留当前仍有效的决策**，已失效/被推翻的删除
- **环境**：GPU/batch/AMP/seed

**规则**：每完成一个 Task，**更新对应段落**而非追加（除非该条是新信息）；不贴大段输出；数值字段（参数量/对拍/loss/mAP）必填且取最新。

### 4. 上下文 80% 压缩-恢复协议（Context Compaction）
- 执行中持续监控上下文占用。**一旦超过 80%** 即触发：
  1. 把未完成 Task 的进度、待办 Step、文件改动清单、参数量、报错摘要写进 TASK.md「当前进度」与「决策与偏差」。
  2. 触发上下文摘要（context summarization）。
  3. 压缩完成后的**第一动作**：重新 Read `TASK.md` 恢复浓缩记忆，继续未完成 Step。
- TASK.md 是压缩后的**唯一可信进度来源**；"做到哪了"以 TASK.md 为准，**不依赖对话记忆**。

### 5. 数值对拍验证方法学（Parity Test）—— 正确性主验证路径（用户方法论）

**核心思想**：RadarNeXt 原工程已是 ground-truth 实现。OpenPCDet 移植版与原版**同输入、同权重 → 输出应逐元素一致**。这是移植正确性的金标准，**比"训 80ep 对 mAP"更精准、更快、定位更细**。

**两个前提（缺一不可，否则对比无意义）：**
1. **权重对齐**：从 RadarNeXt 原模型 `state_dict` 按"层名映射表"逐层拷贝到 OpenPCDet 模块。随机初始化的权重两版必然不同，必须显式加载对齐。层名映射表写在 `tests/parity/weight_map_*.json`。
2. **输入对齐**：用**固定 seed 的合成张量**（`torch.manual_seed(0)` 生成），同时喂给两版模块；或用同一份真实 VoD 帧转成两边各自输入格式。合成输入适合单模块对拍，真实输入适合端到端 detector 对拍。

**判定标准**：输出逐元素 `allclose(atol=1e-4, rtol=1e-3)`（fp32）或 `atol=1e-3`（fp16/含 DCNv3 grid_sample 浮点误差）。**不达标即定位到该模块**，下钻到子层复拍，直到找到错位层（参数名/形状/初始化/数据流/dtype 任一）。

**对拍点（核心模块 + 端到端，逐个建脚本）：**
| 对拍点 | OpenPCDet 模块 | RadarNeXt 对照 | 输入 |
|---|---|---|---|
| 1 | `RepDWCBackbone` | `radarnext/rep_dwc.RepDWC` | 合成 (B,32,320,320) |
| 2 | `SecondFPN` | `mmdet3d second_fpn.SECONDFPN` | RepDWC 的 3 尺度输出 |
| 3 | `RadarNeXtFPNBackbone`(端到端backbone) | RepDWC+SECONDFPN 串联 | 合成 BEV |
| 4 | `MDFENNeck` | `radarnext/MDFENNeck.MDFENNeck` | RepDWC 3 尺度 |
| 5 | `CenterHead`(forward) | `radarnext/radarnext_head.RadarNeXt_Head` | 合成特征图 + 合成 gt |
| 6 | loss | `pillarnext/loss` 各 Loss | head forward 输出 + 合成 gt |
| 7 | 端到端 detector | `radarnext.RadarNeXt` 整体 | 真实 VoD 1 帧（两端格式） |

**实现机制**：
- 每个对拍脚本：`tests/parity/test_parity_<module>.py`，负责 (a) 构造对齐输入；(b) 两版各加载对齐权重；(c) 各 forward；(d) `torch.allclose` 断言；(e) 失败时打印首处不一致的 index/数值/相对误差，便于下钻。
- 两版权重对齐脚本：`tests/parity/build_weight_map.py`——读 RadarNeXt 训练 ckpt（或随机初始化后）的 state_dict，产出层名映射 JSON。**注意 reparam 态**：原版有训练态/推理态两套 state_dict，对拍用**训练态**（推理态对拍作为 Task 9 的一部分）。

**对拍与训练的关系（用户决策）**：对拍是**主验证**（移植正确 = 对拍全过 = 满足 /goal 的结构正确性）；**短训练（10-20ep，看 loss 下降 + 粗略 mAP）是对拍通过后的双保险**，用于确认训练管线（数据/优化器/调度）无 regression。**不再要求 80ep 全量 + mAP 精确对账作为唯一判据**——对拍全过 + 短训 loss 正常下降即可视为复现成功；若短训 mAP 偏离论文，结合对拍结论判定是"移植 bug"（对拍应能定位）还是"训练超参差异"（batch/lr/epoch 缩放）。

### 6. MDFEN/DCNv3 不允许失败 · 穷尽式手段链（强需求）

**强需求：MDFEN 档 DCNv3 必须跑通，不允许失败、无 BLOCKED 退出口。** DCNv3 编译不成功就**一直尝试**到能用为止，按以下手段链顺序自主推进，每条失败自动走下一条，全程记 TASK.md：

1. **base 环境（py3.12/cu124）直接编译** InternImage `ops_dcnv3` —— 预期失败（Py3.12 移除 distutils），但先试，记录确切报错。
2. **新建 `radarnext310`（py3.10 + torch2.1 + cu121）编译** —— 社区验证最充分的栈，大概率成功。修官方 setup.py 的 distutils 兼容问题。
3. **钉死 InternImage 官方精确组合** —— py3.10 + torch2.1.0 + **cu118** + 匹配 gcc/cumm/spconv（按官方 README Issue 里被反复确认能编通的版本号），新建独立环境重试。
4. **纯 pytorch `DCNv3_pytorch`（grid_sample）兜底** —— **无需 CUDA 编译、纯 Python，任何环境必能跑**。这是 MDFEN"不允许失败"的**终极保证**：功能与 CUDA 版等价（用 Task 7 Step6 的 CUDA-vs-pytorch 对拍验证一致性），仅速度慢、显存高。到此 MDFEN 一定能对拍 + 短训跑通。
5. **对齐原工程** —— 若 pytorch 版也异常，回 `/home/admin/projects/RadarNeXt` 看它原本如何解决 DCNv3 依赖（原工程能跑通 MDFEN，必有可用路径），复刻其 `requirements`/编译方式。

**收敛保证**：手段链第 4 条（纯 pytorch）是兜底，**逻辑上保证 MDFEN 永不失败**——它可能慢，但一定能用、一定能对拍验证正确性。因此 MDFEN 档**不存在 BLOCKED 出口**，必须试到 DCNv3 能用。仅在手段 1-5 全部执行完毕仍卡住时，才视为需用户介入的极端情况（但这已被手段 4 排除）。

---

## 文件结构（File Structure）

移植新增/修改的文件，按职责单一、随变化聚类的原则划分。`★` 表示这是把 RadarNeXt "答案"翻译过来的核心文件，数值必须与原文逐字一致。

### 新增文件

| 文件 | 职责 |
|---|---|
| `pcdet/models/backbones_2d/mobileone_blocks.py` ★ | MobileOneBlock（训练多分支 + 推理 reparameterize）、SEBlock、`reparameterize_model()`。翻译自 `mobileone_blocks.py`。注意 `rbr_scale` 仅 `kernel_size>1` 存在，pointwise 1×1 分支只有 rbr_conv+rbr_skip。 |
| `pcdet/models/backbones_2d/rep_common.py` ★ | `ConvBNReLU`、`Transpose`（反卷积上采样，默认 kernel=stride=2）、`RepBlock`（n=1 时 `self.block is None`，只 fuse）。翻译自 `common.py`。 |
| `pcdet/models/backbones_2d/rep_dwc.py` ★ | `RepDWCBackbone`：多阶段 RepBlock（layer_strides=[2,2,2]），输出 3 尺度 list。翻译自 `rep_dwc.py`。 |
| `pcdet/models/backbones_2d/second_fpn.py` ★ | **新增 SecondFPN 类**（不可复用 BaseBEVBackbone）：照搬 mmdet3d `second_fpn.py` 的 deblock 构造，`upsample_strides=[0.5,1,2]` + `use_conv_for_no_stride=True`，三尺度对齐到 **80×80**，concat 成 (B,384,80,80)。 |
| `pcdet/models/backbones_2d/radarnext_backbone_fpn.py` ★ | **单一 BACKBONE_2D 模块**：封装 RepDWC + SecondFPN 串联（forward 内多尺度→融合），作为 BACKBONE_2D 槽位唯一模块（审计 A/G 修正）。 |
| `pcdet/models/backbones_2d/mdfen_neck.py` ★ | MDFENNeck（PAN + DeformLayer[use_ffn=False→裸 DCNv3] + MultiMAPFusion[fusion_strides=[1,2]→160×160]）。翻译自 `MDFENNeck.py` + `common.py`/`DeformFFN.py`。**MDFEN 档才需要**。 |
| `pcdet/ops/dcnv3/` ★ | DCNv3 算子：CUDA 版（InternImage 官方，Py3.12/cu124 编译预期失败）+ `DCNv3_pytorch`（grid_sample 纯 torch fallback）。**MDFEN 档才需要**。 |
| `pcdet/models/dense_heads/radarnext_center_head.py` ★ | CenterHead：`SepHead`（FPN stride=2 / MDFEN stride=1）+ `get_targets`（收 batch_dict tensor 契约）+ `get_loss` + head 自己的 `post_processing`。翻译自 `radarnext_head.py` 单尺度输入版。 |
| `pcdet/models/dense_heads/radarnext_losses.py` ★ | 搬运 `bbox3d_overlaps_diou`（=dIoU）+ `IouLoss`（aligned）+ `FastFocalLoss`/`RegLoss`。**不新增 dIoU utils**。 |
| `pcdet/models/detectors/centerpoint.py` ★ | **新增 CenterPoint 风格 detector**：forward 训练调 `dense_head.get_loss()`、测试调 head 的 `post_processing(batch_dict)`（审计 B 修正，不复用 PointPillar）。 |
| `tools/cfgs/model/vod_models/vod_radarnext_fpn.yaml` | FPN 档完整 config（含 `BACKBONE_3D` 留空、VFE 关 decomposition、去 gt_sampling、tasks/rectifier/code_weights[8]、CLASS_AGNOSTIC:False、SepHead stride=2）。 |
| `tools/cfgs/model/vod_models/vod_radarnext_mdfen.yaml` | MDFEN 档 config（neck 换 MDFENNeck、SepHead stride=1、num_repeats=[1,1,1,1]）。 |
| `tools/reparam_model.py` | build→`reparameterize_model`→保存推理态 ckpt + 统计推理态参数量（审计 E 修正，验收用）。 |
| `tools/scripts/create_vod_data.py` | 顶层 pkl 生成入口；class_names 硬编码 `[Car,Ped,Cyclist]`（dataset yaml 无此 key）；save_path 与 data_path 同指 `radar_5frames`。 |
| `tests/parity/build_weight_map.py` | 读 RadarNeXt state_dict，产出 OpenPCDet↔RadarNeXt 层名映射 JSON（训练态）。 |
| `tests/parity/test_parity_repdwc.py` | 对拍点1：RepDWCBackbone vs RepDWC。 |
| `tests/parity/test_parity_secondfpn.py` | 对拍点2：SecondFPN vs SECONDFPN。 |
| `tests/parity/test_parity_backbone_fpn.py` | 对拍点3：RadarNeXtFPNBackbone 端到端 backbone。 |
| `tests/parity/test_parity_mdfen.py` | 对拍点4：MDFENNeck（MDFEN 档）。 |
| `tests/parity/test_parity_centerhead.py` | 对拍点5：CenterHead forward。 |
| `tests/parity/test_parity_loss.py` | 对拍点6：各 Loss（含 dIoU/IouLoss）。 |
| `tests/parity/test_parity_detector.py` | 对拍点7：端到端 detector，真实 VoD 1 帧。 |
| `tests/test_radarnext_param_count.py` | 参数量验收：build 训练态→reparam→测推理态，断言 ≈0.899M/1.580M（±5%）。 |

### 修改文件

| 文件 | 改动 |
|---|---|
| `pcdet/models/backbones_2d/__init__.py` | 注册 `RadarNeXtFPNBackbone`（FPN 档 BACKBONE_2D）。 |
| `pcdet/models/dense_heads/__init__.py` | 注册 `RadarNeXtCenterHead`。 |
| `pcdet/models/detectors/__init__.py` + `build_detector` | **注册新增的 CenterPoint detector**（审计 B）。 |
| `pcdet/ops/__init__.py` / `setup.py` | 注册并编译 DCNv3 op（MDFEN 档）。 |
| `tools/cfgs/dataset/vod_dataset_radar.yaml` | 本档 yaml 里 **override 去掉 gt_sampling**（忠实复现 RadarNeXt）；确认 `DATA_PATH/INFO_PATH` 路径。 |

---

## 关键翻译对照表（移植时逐条核对，数值不可改）

| RadarNeXt (MMDet3D) | OpenPCDet 对应 | 注意 |
|---|---|---|
| `@MODELS.register_module()` + `MODELS.build()` | `__init__.py` 的 `__all__` dict + `build_xxx` 拓扑 | 删除所有 `register_module`/`registry` 引用 |
| `BaseModule` / `init_cfg` | `nn.Module` + 手动 `init_weights`（Kaiming） | RepDWC 原 `init_cfg=dict(type='Kaiming')` → OpenPCDet `weight_init` |
| `batch_inputs_dict['voxels']` | `batch_dict['voxel_features'/'voxel_coords']` | 整条 forward 改走 `batch_dict` |
| `batch_data_samples[i].gt_instances_3d` | `batch_dict['gt_boxes']` (B,N,7) + `gt_names` | `get_targets` 输入从 InstanceData 改成 tensor |
| `Radar7PillarFeatureNet` | `PillarVFE`（已支持 7 维） | 复用，配 `USE_VELOCITY_DECOMPOSITION` 等 |
| `PointPillarsScatter` | `PointPillarScatter`（MAP_TO_BEV） | 已有，复用 |
| `rep_dwc.RepDWC` | `RepDWCBackbone`（封装进 BACKBONE_2D） | 内部 RepBlock/MobileOneBlock 数值原样 |
| `SECONDFPN`（FPN neck） | **新增 `SecondFPN` 类**（不可复用 BaseBEVBackbone） | upsample_strides=[0.5,1,2]，三尺度→**80×80**→concat (B,384,80,80) |
| `MDFENNeck` | `MDFENNeck`（封装进 BACKBONE_2D） | MDFEN 档；channels_list=[64,128,256,128,64,128,256]；输出 **160×160** |
| `RadarNeXt` detector | **新增 `CenterPoint` 风格 detector**（不复用 PointPillar） | 测试期调 head 自己的 post_processing，不走 template anchor 逻辑 |
| `RadarNeXt_Head` | `RadarNeXtCenterHead`（DENSE_HEAD 槽） | 构造签名对齐 build_dense_head 8 参数；tasks 单任务3类；SepHead stride FPN=2/MDFEN=1；CLASS_AGNOSTIC:False |
| `IouRegLoss`(PillarNeXt) | 搬运 `bbox3d_overlaps_diou` | **源码本身就是 dIoU**（3D IoU − inter_diag/outer_diag），不新增 |
| `IouLoss`(PillarNeXt) | 搬运，用 **aligned** 一对一 IoU | OpenPCDet 无 aligned 版，取 `boxes_iou3d_gpu` 对角线 |
| `rotate_nms_pcdet`(PillarNeXt) | OpenPCDet `multi_classes_nms`/`rotate_iou_gpu` | NMS 阈值照搬：iou_thr=0.2, pre=1000, post=83 |
| `reparameterize_model` + `Rep_Checkpoint_Hook` | build→`reparameterize_model`→测推理态参数量 | **参数量验收必须走 reparam**（论文 0.899M 是推理态）；额外 init: BN weight uniform_ |

---

## Task 0: 环境修复与验证（前置阻塞）

**Files:**
- Modify: base conda 环境（`pip install` / `python setup.py develop`）
- Verify: `tools/train.py` 能 import 成功

**Interfaces:** 无（最底层前置）

**风险点**：8GB 显存 + cu124 + Python 3.12 组合编译自研 ops 可能有坑。DCNv3（InternImage 版）官方支持到 cu11x/cu12x，cu124 需试。

- [ ] **Step 1: 确认现状**（已知：torch 2.4.1+cu124 ✓、spconv 2.3.8 ✓、cuda avail ✓、numba 0.60.0 ✓、numpy 1.26.4 ✓）

Run: `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); import spconv,numba,numpy; print('spconv/numba/numpy ok')"`
Expected: `2.4.1+cu124 12.4 True ... ok`

- [ ] **Step 2: 装包前留回滚快照 + 补装缺失依赖（勿动 numba/numpy）**

Run:
```
pip freeze > /tmp/base_before.txt
pip install open3d scikit-image easydict pyyaml tqdm pccm tensorboardX
```
Expected: 装上；`import open3d`（若 vod pkl 生成路径不依赖则可跳过，先 `grep -rn "import open3d" pcdet/datasets/vod/` 确认必要性）。
> ⚠️ **不要钉 numba==0.58.0**——base 已是 0.60.0+numpy1.26.4 的正确组合，降级反而触发 ufunc 冲突。

- [ ] **Step 3: 编译 OpenPCDet 自研 C++/CUDA ops（设环境变量防 8GB OOM kill）**

Run:
```
export TORCH_CUDA_ARCH_LIST="8.6"   # 只编 3070Ti 的 sm_86，跳过其它 arch
export MAX_JOBS=2                   # 限制 nvcc 并行，防编译期 OOM
cd /home/admin/projects/RadarPillar && python setup.py develop 2>&1 | tee /tmp/pcdet_build.log
```
Expected: 编译 `iou3d_nms_cuda`/`pointnet2_stack`/`pointnet2_batch`/`roiaware_pool3d`/`roipoint_pool3d` 成功，`pcdet/ops/*/` 下出现 `.so`。
> 源码已是现代版（无 AT_CHECK/THC 残留，ABI=cxx11=False 与 torch2.4 一致），主要风险是编译期 OOM，靠两个环境变量规避。失败则 `grep "error:" /tmp/pcdet_build.log` 定位。

- [ ] **Step 4: 验证 pcdet 可 import + 可 build 空 detector**

Run: `python -c "from pcdet.models import build_network; from pcdet.config import cfg; print('OK')"`
Expected: `OK`（不再 `ImportError: iou3d_nms_cuda`）。

- [ ] **Step 5: 显存预算冒烟（先 bs=1）**

Run: 用现有 `vod_radarpillar.yaml`（或 kitti）跑 1 step，`bs=1`，记录 `torch.cuda.max_memory_allocated()`。
Expected: 不 OOM；记下单样本占用，供 Task 5/6 反推安全 batch_size。

**验收**：`python -c "import pcdet.models"` 成功，`nvidia-smi` 可见显存占用。

---

## Task 1: VoD 数据软链 + pkl 生成

**Files:**
- Create: `data/VoD/view_of_delft_PUBLIC` → 软链 `/mnt/d/DataSet/vod/extracted/view_of_delft_PUBLIC`
- Create: `tools/scripts/create_vod_data.py`
- Verify: `vod_infos_train.pkl` / `vod_infos_val.pkl` 生成

**Interfaces:** 产出供后续所有训练/评估使用的 pkl 与 gt_database。

**数据布局要求**（KITTI-format，`VodDataset` 期望）：
```
data/VoD/view_of_delft_PUBLIC/
  radar_5frames/           ← DATA_PATH 指向这里（YAML 里 radar_5frames）
    training/velodyne/*.bin
    training/label_2/*.txt
    training/calib/*.txt
    training/planes/*.txt  (可选)
    ImageSets/{train,val,test}.txt
```

- [ ] **Step 1: 建软链**（WSL 下 `/mnt/d` 是 D 盘）

Run: `ln -s /mnt/d/DataSet/vod/extracted/view_of_delft_PUBLIC data/VoD/view_of_delft_PUBLIC`
Expected: `ls data/VoD/view_of_delft_PUBLIC` 见 `radar_5frames`、`radar`、`lidar` 等。
> ⚠️ `/mnt/d`（Windows 盘）I/O 显著慢于 ext4。训练时长会受影响（估 6–15h 可能更长）。若磁盘允许，**优先把数据拷到 WSL ext4**（如 `~/projects/.../data`）而非软链，加速 I/O。软链作为最小可行方案。

- [ ] **Step 2: 核对 ImageSets 与目录结构**

Run: `ls data/VoD/view_of_delft_PUBLIC/radar_5frames/training/{velodyne,label_2,calib}` 与 `.../ImageSets/`
Expected: 三目录齐全；ImageSets 含 `train.txt`(5139)/`val.txt`(1296)/`train_val.txt`(6435)/`test.txt`(2247)。train+val 无重叠（实测交集 0），与 label_2(6435) 对齐。

- [ ] **Step 3: 写 `create_vod_data.py` 入口（注意路径约定）**

内容：`class_names=['Car','Pedestrian','Cyclist']`（硬编码，dataset yaml 无此 key）；`data_path` 与 `save_path` 都指 `data/VoD/view_of_delft_PUBLIC/radar_5frames`（与 YAML `INFO_PATH` 解析路径一致，否则训练 FileNotFoundError）；调 `VodDataset.create_vod_infos(dataset_cfg, class_names, data_path, save_path, workers=4)`。

- [ ] **Step 4: 生成 pkl**

Run: `python tools/scripts/create_vod_data.py`
Expected: 产出 `vod_infos_train.pkl`、`vod_infos_val.pkl`、`vod_infos_trainval.pkl`、`vod_infos_test.pkl`、`vod_dbinfos_train.pkl`、`gt_database/`。
> `vod_infos_test.pkl`（基于 test.txt 2247 帧无标签）评估不用，生成仅为对齐脚本。本档忠实复现关闭 gt_sampling，`gt_database` 即使生成也不启用，可在 TASK.md 记"未启用"。

- [ ] **Step 5: 冒烟验证 pkl 可被 VodDataset 加载**

Run: `python -c "from pcdet.datasets import build_dataloader; ..."` 或 `python tools/check_data_consistency.py`
Expected: 能 load 一条 info，`gt_boxes` shape=(N,7)，点云可读。

**验收**：pkl 文件存在、shape 正确、点云/标注对齐。

---

## Task 2: RepDWC backbone 移植（★ 核心翻译）

**Files:**
- Create: `pcdet/models/backbones_2d/mobileone_blocks.py`（MobileOneBlock/SEBlock/reparameterize_model）
- Create: `pcdet/models/backbones_2d/rep_common.py`（ConvBNReLU/Transpose）
- Create: `pcdet/models/backbones_2d/rep_dwc.py`（RepBlock + RepDWCBackbone）
- Modify: `pcdet/models/backbones_2d/__init__.py`（注册 RepDWCBackbone）
- Test: 单独 build RepDWCBackbone，喂 (B,64,320,320)，断言输出 3 尺度 + shape

**Interfaces:**
- Consumes: PointPillarScatter 输出 `(B, C_in=32, 320, 320)` BEV 特征（NUM_BEV_FEATURES=32，审计修正 M4——非 64）
- Produces: `class RepDWCBackbone(nn.Module)`，`__init__(model_cfg, input_channels=32)`，`forward` 返回 `list[Tensor]`：`[(B,64,160,160),(B,128,80,80),(B,256,40,40)]`（`out_channels=[64,128,256]`、`layer_strides=[2,2,2]`）

**翻译要点（数值不可改）：**
- `MobileOneBlock`：训练时 `rbr_conv`(num_conv_branches=1) + `rbr_scale`(kernel=1，**仅当 `kernel_size>1` 存在**) + `rbr_skip`(BN，仅 `in==out and stride==1`)，推理时 `reparameterize` 融合。**pointwise 1×1 分支无 rbr_scale**（审计 M11）。
- `RepBlock._make_stage`：`use_dwconv=True` 时 = depthwise MobileOneBlock(groups=in, kernel=3) + pointwise MobileOneBlock(kernel=1)。`n=1` 时 `self.block is None`，只有 1 个 fuse（审计 num_repeats）。
- `RepDWCBackbone`：`in_filters=[32, *out[:-1]]`，逐 stage 堆 RepBlock，返回最后 `num_outputs=3` 个。

- [ ] **Step 1: 移植 mobileone_blocks.py**（去 mmdet3d 依赖，纯 torch）
- [ ] **Step 2: 移植 rep_common.py**（ConvBNReLU、Transpose、RepBlock）
- [ ] **Step 3: 移植 rep_dwc.py**（RepDWCBackbone，forward 返回多尺度 list）
- [ ] **Step 4: 注册到 `__init__.py`**
- [ ] **Step 5: 参数量自检**（build RepDWCBackbone，喂 `(2,32,320,320)`，打印 3 个输出 shape 与参数量，记录用于 Task 8 对账）

Run: `python -c "构建 RepDWCBackbone(model_cfg, 32); 喂 (2,32,320,320); 打印 shapes + 参数量"`
Expected: shapes=[(2,64,160,160),(2,128,80,80),(2,256,40,40)]。

---

## Task 3: FPN neck 接入（审计修正：新增 SecondFPN，封装为单 BACKBONE_2D，无 DCNv3）

**Files:**
- Create: `pcdet/models/backbones_2d/second_fpn.py`（**新增 SecondFPN 类，不可复用 BaseBEVBackbone**）
- Create: `pcdet/models/backbones_2d/radarnext_backbone_fpn.py`（封装 RepDWC + SecondFPN 为单一 BACKBONE_2D）
- Modify: `__init__.py` 注册 `RadarNeXtFPNBackbone`

**Interfaces:**
- Consumes: RepDWC 的 3 尺度 list `[(B,64,160,160),(B,128,80,80),(B,256,40,40)]`
- Produces: 单尺度融合特征 `(B, 384, 80, 80)`（审计修正——**非 160×160**）

**关键（审计修正 A/G/C）：**
- OpenPCDet `module_topology` 无独立 neck 槽位 → 把 RepDWC + SecondFPN 串联封装成**一个 BACKBONE_2D 模块** `RadarNeXtFPNBackbone`，其 forward 内先 RepDWC 出多尺度，再过 SecondFPN 出单尺度，返回 `batch_dict['spatial_features_2d']`。
- `SecondFPN` 照搬 mmdet3d `second_fpn.py` 的 deblock 构造（**不可复用 BaseBEVBackbone**——其 blocks 段是 ZeroPad+Conv，与 RepDWC 不兼容）。`upsample_strides=[0.5,1,2]` + `use_conv_for_no_stride=True`：64ch@160→stride0.5→**80×80**；128ch@80→stride1→80×80；256ch@40→stride2(deconv)→80×80；三者 concat → **(B,384,80×80)**。

- [ ] **Step 1: 移植 SecondFPN**（照搬 `second_fpn.py` deblock 逻辑，去 mmdet3d）
- [ ] **Step 2: 封装 RadarNeXtFPNBackbone**（RepDWC + SecondFPN 串联，作为 BACKBONE_2D 唯一模块）
- [ ] **Step 3: shape 自检**：`(2,32,320,320)` → 三尺度 → **(B,384,80,80)**

---

## Task 4: CenterPoint detector + CenterHead + dIoU 移植（★ 核心翻译，最复杂；审计大改）

**Files:**
- Create: `pcdet/models/detectors/centerpoint.py` ★（**新增 detector**，审计 B）
- Create: `pcdet/models/dense_heads/radarnext_center_head.py` ★（SepHead + CenterHead + head 自己的 post_processing）
- Create: `pcdet/models/dense_heads/radarnext_losses.py` ★（搬运 `bbox3d_overlaps_diou` + `IouLoss`/`FastFocalLoss`/`RegLoss`，**不新增 dIoU utils**，审计 D）
- Modify: `detectors/__init__.py` + `build_detector`（注册 CenterPoint）、`dense_heads/__init__.py`（注册 head）
- Test: head 单独 forward + loss；detector 端到端 forward

**Interfaces:**
- Consumes: neck 输出单尺度特征；`batch_dict['gt_boxes']` (B,N,7) + `gt_names`
- Produces:
  - `class RadarNeXtCenterHead(nn.Module)`，构造签名**严格对齐 build_dense_head**：`__init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training=True)`（审计 M7）
  - `class CenterPoint(nn.Module)`：forward 训练调 `dense_head.get_loss()`、测试调 **head 自己的** `post_processing(batch_dict)`（不走 template anchor post_processing）

**翻译要点（逐条核对，数值不可改）：**
- `tasks=[{num_class:3, class_names:['Car','Pedestrian','Cyclist']}]`，**与 dataset CLASS_NAMES 完全相同**（head 内 `gt_labels_3d` 0/1/2 直接对应，rectifier 按 label 索引取值）。
- `common_heads={reg:(2,2), height:(1,2), dim:(3,2), rot:(2,2), iou:(1,2)}`；YAML 里 `(2,2)` 解析成 list `[2,2]`，代码用 `len()`/下标取，**勿依赖 tuple 类型**。
- `share_conv_channel=64`、`num_hm_conv=2`、`init_bias=-2.19`、`final_kernel=3`。
- `SepHead` 的 `stride` 参数（审计 C）：**FPN 档 stride=2**（ConvTranspose 把 80→160 对齐 target 的 feature_map=160）；**MDFEN 档 stride=1**（已 160×160 无需 deblock）。两档 YAML 分开配。
- `code_weights=[1.0]*8`（长度=8：reg2+height1+dim3+rot2，**非 7**，审计 #7）；`bbox_code_size` 是 7。
- 损失（审计 D，最终方案）：
  - `with_corner=True`、`corner_weight=1`（corner_hm 辅助头）
  - `with_iou=True` + `IouLoss`：IoU-score 辅助损失，**aligned** 一对一 IoU（OpenPCDet 无 aligned 版，对 `boxes_iou3d_gpu` 取对角线），`iou_weight=1`
  - `with_reg_iou=True` + `IouRegLoss`：搬运源码 `bbox3d_overlaps_diou`，**=dIoU**（3D IoU − inter_diag/outer_diag），`iou_reg_weight=0.5`
  - focal loss（`FastFocalLoss`）+ L1 reg（`RegLoss`）原样搬
- `get_targets`：`max_objs=500`、`dense_reg=1`、`gaussian_overlap=0.1`、`min_radius=2`、`out_size_factor=2`、`feature_map_size = grid_size[:2] // out_size_factor = 160`（从传入 grid_size 推，勿硬编码 320）。**收 batch_dict['gt_boxes']，不调 .gravity_center**（已是体积中心）。
- `predict`/`post_processing`：`rectifier=[[0.5,0.5,0.5]]`（顺序随 class_names）、NMS `nms_iou_threshold=0.2`/`pre=1000`/`post=83`/`score_threshold=0.1`。**z 语义**：源码 `bboxes[:,2]-=bboxes[:,5]*0.5` 把 box 转底面中心供评估——核对 OpenPCDet VoD evaluator z 语义后决定保留/删除（见全局约束）。用 OpenPCDet `multi_classes_nms`/`rotate_iou_gpu` 替换 `rotate_nms_pcdet`。
- **去掉** mmdet3d 的 `decouple_pred_processing`/`channels_list` 多尺度分支（FPN/MDFEN 已融合成单尺度）。
- detector init：Conv2d 用 Kaiming + **BN weight `uniform_`**（审计 #10，非常数 1）。

- [ ] **Step 1: 新增 CenterPoint detector**（注册，forward 训练 get_loss / 测试 head.post_processing）
- [ ] **Step 2: 移植 losses**（`bbox3d_overlaps_diou`=dIoU + aligned `IouLoss` + focal/reg，去 mmdet3d）
- [ ] **Step 3: 移植 SepHead**（FPN stride=2 / MDFEN stride=1 分档）
- [ ] **Step 4: 移植 get_targets / get_targets_single**（收 batch_dict tensor 契约，feature_map 从 grid_size 推）
- [ ] **Step 5: 移植 loss_by_feat**（focal + L1 reg + aligned IouLoss + dIoU(IouRegLoss) + corner）
- [ ] **Step 6: 移植 predict / post_processing**（解码 + rectifier + z语义核对 + OpenPCDet NMS）
- [ ] **Step 7: 注册 + 单元冒烟**：detector 喂假 batch_dict（随机 gt_boxes），跑 forward→loss，断言 loss 标量 finite

Run: `python -c "构造 CenterPoint + 假数据; loss=det.dense_head.get_loss(); print(loss, finite)"`
Expected: loss 正有限标量，各子损失键齐全。

---

## Task 4.5: 数值对拍框架（正确性主验证，用户方法论）★核心

> **定位**：这是移植正确性的**主验证路径**。RadarNeXt 原工程是 ground-truth，OpenPCDet 移植版与之**同输入、同权重 → 输出应逐元素一致**。对拍全过 = 结构移植正确（满足 /goal 的正确性维度）；短训练（Task 6/7）只作双保险。

**Files:**
- Create: `tests/parity/build_weight_map.py`（层名映射）
- Create: `tests/parity/test_parity_repdwc.py` / `secondfpn.py` / `backbone_fpn.py` / `mdfen.py` / `centerhead.py` / `loss.py` / `detector.py`（7 个对拍点）
- Create: `tests/parity/conftest.py`（共享：合成输入生成、权重加载、allclose 断言与失败下钻打印）

**两个前提（执行协议 §5）：**
1. **权重对齐**：`build_weight_map.py` 读 RadarNeXt 训练态 `state_dict` → 产出 `weight_map_<module>.json`（OpenPCDet 层名 → RadarNeXt 层名）。对拍脚本据此把原版权重 `load_state_dict` 进 OpenPCDet 模块（形状不符即报错，本身也是一种校验）。
2. **输入对齐**：`conftest.py` 用 `torch.manual_seed(0)` 生成固定合成张量，同一对象喂两版（端到端 detector 用真实 VoD 1 帧，转两版各自输入格式）。

**判定标准**：fp32 `allclose(atol=1e-4, rtol=1e-3)`；fp16/含 DCNv3 `atol=1e-3`。失败时打印首处不一致 index/数值/相对误差 → 下钻到子层复拍。

- [ ] **Step 1: 搭对拍脚手架**（`conftest.py`：合成输入生成器 + 权重加载器 + allclose 断言器 + 失败下钻打印）
- [ ] **Step 2: 写 `build_weight_map.py`**——逐模块产出层名映射 JSON（先 RepDWC，再逐步扩展到 FPN/MDFEN/head）。对每模块，比对两版 `state_dict().keys()`，建立一一映射；形状必须匹配，不匹配即结构翻译有误。
- [ ] **Step 3: 对拍点1 RepDWC**：合成 (B,32,320,320) → 两版各加载对齐权重 → forward → 对拍 3 尺度输出。
Expected: 三尺度 allclose PASS。FAIL → 下钻 RepBlock/MobileOneBlock（重点查 rbr_scale 条件、num_conv_branches、dwconv groups、BN eps）。
- [ ] **Step 4: 对拍点2 SecondFPN**：喂 RepDWC 的 3 尺度（对拍点1 已验过的输出）→ 对拍 80×80 输出。
Expected: PASS。FAIL → 查 upsample_strides=[0.5,1,2] 的 deconv/conv 选择、use_conv_for_no_stride、BN。
- [ ] **Step 5: 对拍点3 端到端 FPN backbone**：合成 BEV → RepDWC+SecondFPN 串联 → 对拍 (B,384,80,80)。
Expected: PASS（点1+2 过则应过，作为集成验证）。
- [ ] **Step 6: 对拍点5 CenterHead forward**：合成特征图 + 合成 gt_boxes → 两版 forward → 对拍各 task 输出（hm/reg/height/dim/rot/iou）。
Expected: PASS。FAIL → 查 SepHead（FPN stride=2）、share_conv_channel、common_heads 通道数、init_bias。
- [ ] **Step 7: 对拍点6 loss**：head 输出 + 合成 gt → 两版算 loss → 对拍各子损失（focal/reg/dIoU/IouLoss/corner）。
Expected: PASS。**重点**：验证 dIoU 实现等价（搬运的 `bbox3d_overlaps_diou` 与原版一致）、IouLoss 用 aligned IoU、code_weights 长 8。
- [ ] **Step 8: 对拍点4 MDFENNeck**（MDFEN 档，Task 7 后做）：合成 3 尺度 → 对拍 160×160 输出。
Expected: PASS（atol 放宽到 1e-3，含 DCNv3）。FAIL → 重点查 DCNv3（CUDA vs pytorch 版对拍）、former_deform2 位置、channels_list、num_repeats。
- [ ] **Step 9: 对拍点7 端到端 detector**：真实 VoD 1 帧 → 两版 detector 完整 forward（train 态取 loss、test 态取 pred）→ 对拍 loss（train）或 pred_dicts（test）。
Expected: PASS。FAIL → 综合前 6 点结论定位（detector 是顶层集成，单模块全过则此处应过，差异多半在数据契约/VFE/Scatter 转换）。

**验收（Task 4.5 DoD）**：对拍点 1/2/3/5/6/7 全 PASS = FPN 链路移植正确；点4 待 Task7。任一 FAIL → 下钻定位、修复、重拍，**不进 Task 6 训练**。结论写 TASK.md「关键结果·对拍」。

---

## Task 5: FPN 档 YAML 装配 + 参数量核验（走 reparam，审计 E）

**Files:**
- Create: `tools/cfgs/model/vod_models/vod_radarnext_fpn.yaml`
- Create: `tools/reparam_model.py`

- [ ] **Step 1: 写 YAML**（必须含审计修正项）
  - `MODEL.NAME=CenterPoint`（新增 detector，非 PointPillar）
  - `VFE.NAME=PillarVFE` + `USE_VELOCITY_DECOMPOSITION: False`（7维，审计 F）+ `USE_ABSOLUTE_XYZ: True` + `NUM_FILTERS: [32]`
  - `BACKBONE_3D: ~`（**留空**，审计 M5——勿保留 PillarAttention）
  - `MAP_TO_BEV.NAME=PointPillarScatter` + `NUM_BEV_FEATURES: 32`
  - `BACKBONE_2D.NAME=RadarNeXtFPNBackbone`（RepDWC+SecondFPN 封装）
  - `DENSE_HEAD.NAME=RadarNeXtCenterHead` + `CLASS_AGNOSTIC: False`（审计 M6）+ tasks/common_heads/code_weights[8]/rectifier/SepHead stride=2
  - `DATA_AUGMENTOR` override **去掉 gt_sampling**（审计 K），只留 random_world_flip[x] + random_world_scaling[0.95,1.05]
- [ ] **Step 2: build 冒烟**（先 bs=1）

Run: `python tools/train.py --cfg_file ...fpn.yaml --batch_size 1 --epochs 0 --eval_tag smoke`
Expected: detector 构建成功，print 各模块，进 dataloader 不崩。
- [ ] **Step 3: 参数量核验（走 reparam，审计 E）**

Run: build 训练态 → `reparameterize_model(model)` → `sum(p.numel())`，并在 TASK.md 同时记训练态/推理态。
Expected: **推理态 ≈0.899M**（容差 ±5%：0.854~0.944M）；训练态约 1.5~2x（属正常，论文报推理态）。
> 偏移排查：逐模块对账 RadarNeXt 原仓库 `sum(numel)`；常见坑——误带 PillarAttention、VFE 开了 decomposition（9维）、num_conv_branches 训练态多分支、share_conv_channel=64 写错、pointwise 误加 rbr_scale。

---

## Task 6: FPN 档短训练双保险（对拍通过后）

> **定位变更（用户方法论）**：移植正确性已由 **Task 4.5 数值对拍**保证（主验证）。本 Task 只跑**短训练（10-20ep）**作双保险——确认训练管线（数据/优化器/调度/AMP）无 regression，看 loss 正常下降即可。**不再要求 80ep 全量 + mAP 精确对账**。若短训 loss 异常或 mAP 大幅偏离，结合对拍结论判定是移植 bug（对拍应已捕获）还是训练超参差异。

**Files:** 无新增，跑训练

- [ ] **Step 1: overfit-1-batch（验证训练管线，审计 L）**

Run: `--batch_size 1` + 单帧跑 ~200 step。
Expected: loss 单调下降到≈0。**不降→排查训练管线 bug**（注意：模块正确性已由对拍保证，此处异常多在数据加载/loss 聚合/优化器 step）。

- [ ] **Step 2: 显存冒烟反推安全 batch_size（审计 M1）**

Run: `--batch_size 1` 跑 1 step 记 `max_memory_allocated()`，按 `(8GB×0.85)/单样本` 反推安全 bs。
Expected: 推出安全 bs（目标 4，OOM→2），记 TASK.md。

- [ ] **Step 3: 短训练 10-20 epoch**（安全 bs，AMP fp16，`--fix_random_seed`）

Run: `python tools/train.py --cfg_file ...fpn.yaml --batch_size <B> --epochs 15 --fix_random_seed`
Expected: loss 单调下降、无 NaN、lr schedule 正常、产 ckpt。预计 1-3h。

- [ ] **Step 4: 粗略评估 VoD mAP**

Run: `python tools/test.py --cfg_file ...fpn.yaml --ckpt <last>`
Expected: **粗略 mAP 接近 47.98**（短训未充分收敛，可低于论文，但应在一个合理量级，如 ≥40）。
> 判读：若 mAP 合理 → 训练管线 OK，FPN 复现成功；若 mAP 异常低 → 回 Task 4.5 对拍复检（对拍 PASS 则问题在超参缩放，记 TASK.md；对拍若实际有疏漏则修复）。
> **可选**：若用户后续要精确对标 47.98，再决定是否补全量 80ep（Task 6-extend，非默认）。

---

## Task 7（进阶档·必交付）: MDFEN neck + DCNv3 移植 + 对抗审查

> **定位变更（用户要求）**：MDFEN **不接受跑不动，DCNv3 不允许失败**。base(py3.12/cu124) 对 DCNv3 CUDA 编译失败是**环境问题**。按**执行协议 §6 穷尽式手段链**自主推进：base 试编 → 新建 `radarnext310`(py3.10+torch2.1+cu121) → 钉死官方精确组合 → **纯 pytorch `DCNv3_pytorch` 兜底（保证永不失败）** → 对齐原工程。**没有 BLOCKED 出口，必须试到 DCNv3 能用为止**，全程记 TASK.md。

**Files:**
- Create: `pcdet/ops/dcnv3/`（CUDA op 优先 + `DCNv3_pytorch` fallback 兜底）
- Create: `pcdet/models/backbones_2d/mdfen_neck.py`（DeformLayer + MDFENNeck + MultiMAPFusion）
- Modify: `setup.py` / `pcdet/ops/__init__.py`（编译注册）
- Create: `tools/cfgs/model/vod_models/vod_radarnext_mdfen.yaml`

**Interfaces:**
- Consumes: RepDWC 3 尺度 list
- Produces: 单尺度融合特征 **(B,384,160×160)**（MultiMAPFusion `fusion_strides=[1,2]`，审计 #13）

**翻译要点：**
- MDFEN config（`radarnext.py` 值）：`dcn_layer=False, former=True, latter=False, dcn_ids=[2], group=4, use_ffn=False, multi_fusion=True, num_repeats=[1,1,1,1]`。
- **`dcn_layer=False` 时 DCN 走 `former_deform2`**（PAN 自顶向下第3支路 concat 后、RepBlock 前），输入 channel=`channels_list[0]+channels_list[4]=128`。**`use_ffn=False` → DeformLayer 内部是裸 `DCNv3`（非 DeformFFN）**。
- `num_repeats=[1,1,1,1]` → n=1 时 `self.block is None`（审计 #12）。
- 输出分辨率 **160×160**（MultiMAPFusion），head SepHead stride=1。
- 去掉 mmdet3d registry；DCNv3 输入 channels-last，模块内 permute。

- [ ] **Step 1: 新建 `radarnext310` 环境**（Py3.10 + torch2.1 + cu121）
Run:
```
conda create -n radarnext310 python=3.10 -y
conda activate radarnext310
pip install torch==2.1.* torchvision --index-url https://download.pytorch.org/whl/cu121
pip install <fpn 档同款依赖：numpy/scipy/numba/spconv-cu120/...>
```
Expected: `torch.cuda.is_available() True`，spconv 可用。
- [ ] **Step 2: 编译 pcdet ops + DCNv3，按 §6 手段链穷尽尝试（不允许失败）**
按执行协议 §6 顺序自主推进，直到 `import DCNv3` 成功：
1. base 环境（py3.12/cu124）试编 InternImage `ops_dcnv3`（`TORCH_CUDA_ARCH_LIST=8.6`）—— 预期失败，记确切报错。
2. 新建 `radarnext310`(py3.10+torch2.1+cu121)，`python setup.py develop` + 编 `ops_dcnv3`。
3. 若仍失败 → 钉死官方精确组合（py3.10+torch2.1.0+cu118+匹配 gcc/cumm/spconv），新建独立环境重试，修 setup.py 的 distutils 兼容。
4. **若 CUDA 版三条全失败 → 启用纯 pytorch `DCNv3_pytorch`（grid_sample，无需编译，必能跑）**。用 Step6 的 CUDA-vs-pytorch 对拍确认等价后正式采用。
5. 若 pytorch 版也异常 → 回 `/home/admin/projects/RadarNeXt` 复刻其 DCNv3 依赖解决方式。
Expected: `pcdet/ops/*/` + DCNv3 `.so`（或 pytorch 版）就绪；`import DCNv3` 成功。**手段链终点保证 DCNv3 一定可用。**
- [ ] **Step 3: 移植 DCNv3_pytorch（无论 CUDA 是否成功都移植，供对拍验证 CUDA 版正确性 + 兜底）**
- [ ] **Step 4: 移植 DeformLayer + MultiMAPFusion + MDFENNeck**（PAN 双向 + former_deform2 DCN + multi_fusion）
- [ ] **Step 5: 注册；写 mdfen yaml（SepHead stride=1、num_repeats=[1,1,1,1]）；build 冒烟 + 显存探测**
Run: `python tools/train.py --cfg_file ...mdfen.yaml --batch_size 1 --epochs 0`
Expected: 参数量 ≈ **1.580M**（走 reparam，±5%）；显存探测不 OOM（OOM 排查是否真用了 CUDA DCNv3）。
- [ ] **Step 6: 对拍点4 MDFENNeck（主验证，Task 4.5 Step8）+ DCNv3 数值对拍**
Run: 合成 3 尺度 → OpenPCDet MDFENNeck vs RadarNeXt MDFENNeck（同输入同权重）→ 对拍 160×160 输出（atol=1e-3，含 DCNv3）。**另**：CUDA `DCNv3` vs `DCNv3_pytorch` 在相同输入下 forward 对拍，验证 CUDA 版正确。
Expected: PASS。FAIL → 下钻 DeformLayer/DCNv3/MultiMAPFusion/PAN 双向，定位错位层。
- [ ] **Step 7: overfit-1-batch + 显存反推 bs + 短训练(10-20ep) + 粗评**
Run: overfit-1-batch → 反推 bs（目标1，OOM→梯度累积）→ 15ep → `tools/test.py` 粗评。
Expected: loss 正常下降；粗略 mAP 接近 50.48（短训未充分收敛，可偏低但应在合理量级）。
> **训练成本注明**：MDFEN 结构重于 FPN，即便用 CUDA DCNv3，单 step 仍慢，短训 15ep 预计约 1-4h；若用户后续要精确对标 50.48 跑全量 80ep，预计约为 FPN 全量的 1.5-2x（FPN 估 6-15h → MDFEN 估 10-30h）。此为结构计算量所致，非环境缺陷。
- [ ] **Step 8: fan-out 对抗审查 MDFEN 移植正确性（对拍的补充，反复迭代）**
对拍是主验证；本步用对抗 agent 作**第二意见**：(a) 源码忠实度——MDFENNeck/DeformLayer/MultiMAPFusion 逐分支对照 RadarNeXt 原码（DCN 位置 former_deform2、channels_list、num_repeats、fusion_strides、PAN 双向数据流）；(b) OpenPCDet 适配——forward 契约、batch_dict 流、160×160 与 head target 对齐。对拍已覆盖的不再重复。发现差距 → 修复 → 重拍/重审，**直到无差距**。结论写 `experiments/radarnext_repro_report.md` 的 MDFEN 表。

**验收**：MDFEN 档对拍点4 PASS + 参数量≈1.580M（走reparam）+ 短训 loss 正常 + 粗评 mAP 合理。DCNv3 编译失败时按 §6 手段链穷尽尝试，**不允许失败**（纯 pytorch 版兜底）。

---

## Task 8: 终局验收（对拍为主轴 + 短训练双保险）

> **定位变更（用户方法论）**：移植正确性的**主判据是 Task 4.5 数值对拍全过**（结构正确性，精确、快速、可定位）。mAP 因短训练未充分收敛，**仅作粗略参考**，不再要求精确对账到论文值。本 Task 汇总对拍结论 + 参数量 + 短训练 loss/粗评，给出最终复现判定。

**Files:**
- Create: `experiments/radarnext_repro_report.md`（对账报告）
- 收集：Task 4.5 各对拍点结论、Task 5/7 参数量、Task 6/7 短训练 loss/粗评 mAP

**论文锚点（VoD 验证集，5-scans，抄自 Table I/II，作 mAP 量级参考）：**

| 档位 | 参数量 | mAP（论文，量级参考） |
|---|---|---|
| **FPN** | **0.899M** | 47.98 |
| **MDFEN** | **1.580M** | 50.48 |

（PAN 1.531M/48.15 不交付，备查；参数量阶梯 FPN0.899 < PAN1.531 < MDFEN1.580。）

**验收维度（分档独立判定）：**

| 维度 | FPN | MDFEN | 性质 |
|---|---|---|---|
| 数值对拍（Task4.5 点1/2/3/5/6/7） | 全 PASS | 点4 PASS | **主判据·硬指标** |
| 参数量（走reparam） | [0.854,0.944]M | [1.501,1.659]M | 硬指标 |
| 短训练 loss | 单调下降、无NaN | 同 | 双保险 |
| 粗评 mAP | 合理量级(≥40) | 合理量级(≥42) | 参考（短训未收敛可偏低） |
- mAP 受 batch_size/epoch/seed 影响，短训练未充分收敛会偏低，**仅作量级参考**，不作为硬判据。各类 AP 用作偏差定位。
- **FPS**：论文在 A4000 上测（FPN 83.57 / MDFEN 67.10），本机 3070Ti 不可比，**仅记录不判**。

- [ ] **Step 1: 数值对拍终审汇总（主判据）—— 分档独立判定**

Run: 汇总 Task 4.5 各对拍点结论（FPN: 点1/2/3/5/6/7；MDFEN: 点4），填对账表。
Expected: FPN 六点全 PASS + MDFEN 点4 PASS = **移植正确性确认（满足 /goal 的正确性维度）**。任一 FAIL → 已在 Task 4.5 下钻修复，此处仅汇总。

- [ ] **Step 2: 参数量终审（走 reparam，结构正确性硬指标）—— 分档独立判定**

Run: 分别 build 两档 → `reparameterize_model()` → `sum(p.numel())/1e6`（推理态），分档填账。TASK.md 记训练态/推理态两套。
Expected: FPN ∈ [0.854,0.944]M → PASS；MDFEN ∈ [1.501,1.659]M → PASS。
> 超容差排查（按档）：逐模块打印参数量，与 RadarNeXt 原仓库对应档 `sum(numel)` 对比。常见坑：误带 PillarAttention、VFE 开 decomposition(9维)、num_conv_branches 训练态多分支、share_conv_channel=64 写错、pointwise 误加 rbr_scale、未走 reparam、FPN/MDFEN 参数填反档。

- [ ] **Step 3: 短训练健康度 + 粗评 mAP（双保险·参考）**

Run: 读 Task 6/7 短训练日志（loss 单调下降/无NaN/lr 正常）+ 粗评 mAP（评估 `ret_dict` 取 `Car/Pedestrian/Cyclist_3d/moderate_R40` 算术均）。
Expected: loss 正常下降 + 粗评 mAP 在合理量级（FPN≥40 / MDFEN≥42，短训未收敛可偏低）。
> mAP 大幅偏低（如 <30）→ 回 Task 4.5 对拍复检（对拍 PASS 则问题在训练超参缩放：batch/lr/epoch/gt_sampling，记 TASK.md；对拍若有疏漏则修复重拍）。

- [ ] **Step 4: 训练健康度核查**

Run: 读训练日志，确认 loss 单调下降、无 NaN、lr schedule 正常、未早停在 plateau。
Expected: loss 曲线合理，best ckpt 来自收敛后段。

- [ ] **Step 5: 生成对账报告（分档）**

写入 `experiments/radarnext_repro_report.md`，**FPN 档与 MDFEN 档各一张独立对账表**，每表含：对拍点结论(主)、参数量实测值(推理态)、短训练 loss、粗评 mAP、与论文差值、判定（PASS/FAIL）、偏差排查结论、环境与超参。
Expected: 报告完整，两档每项有判定标记。

**验收（Definition of Done for Task 8，分档独立判定）：**
- **FPN 档**：数值对拍(点1/2/3/5/6/7)全 PASS + 参数量[0.854,0.944]M + 短训 loss 正常 → **复现成功（满足 /goal）**。粗评 mAP 仅参考。
- **MDFEN 档**：对拍点4 PASS + 参数量[1.501,1.659]M + 短训 loss 正常 → **进阶成功**。MDFEN **必交付**——DCNv3 按 §6 手段链穷尽尝试，**不允许失败**（纯 pytorch 版兜底保证），不中断询问。
- 任一档对拍 FAIL 或参数量超容差 → 回 Task 4.5/对应 Task 修复重拍；短训 mAP 偏低但对拍 PASS → 记为超参缩放差异，不判 FAIL。
- **/goal 达成线 = FPN 档（对拍全过 + 参数量达标 + 短训正常）**；MDFEN 同标准 = 完整复现。

---

## Task 9: 重参数化推理优化（可选）

**Files:**
- Create: `tools/reparam_model.py`（调 `reparameterize_model` 把训练 ckpt 转推理单分支图）
- Verify: 重参数化后参数量下降、mAP 不变、FPS 提升

- [ ] **Step 1: 实现 reparam 脚本**
- [ ] **Step 2: 转换 ckpt + 测速**（对比 train-time 多分支 vs reparam 单分支 FPS）
Expected: 参数量↓、mAP 持平、FPS↑（对齐论文"参数减少 71%、速度+9%"方向）。

---

## 验收清单（Definition of Done，分档判定·对拍为主轴）

对照 /goal 与需求逐条。**FPN 档与 MDFEN 档独立验收；数值对拍是主判据，短训练是双保险：**

1. ✓ VoD 数据软链建立，pkl 生成正确（Task 1）
2. ✓ 环境：base(FPN) + `radarnext310`(MDFEN) + pcdet ops/DCNv3 编译通过（Task 0/7）
3. ✓ Rep-DWC / FPN / MDFEN / CenterPoint+CenterHead+dIoU 按 OpenPCDet 风格实现（Task 2/3/4/7）
4. ✓ yaml + VoD pkl 齐全（需求 item 5）
5. ✓ **FPN 档**：数值对拍(点1/2/3/5/6/7)全过(Task 4.5) + 参数量≈0.899M(Task 5) + 短训 loss 正常(Task 6)
6. ✓ **MDFEN 档**：对拍点4过 + 参数量≈1.580M + 短训 loss 正常(Task 7) + 对抗审查无差距
7. ✓ **Task 8 终局分档对账**：两档各（对拍+参数量+短训）全绿

**/goal 达成线 = FPN 档（数值对拍全过 + 参数量达标 + 短训练正常）；完整复现 = FPN+MDFEN 双档达标。**

---

## Self-Review（计划自检，已含 3 份对抗审计修正）

**0. 审计已修正项索引（fan-out 三子智能体成果）：**
- A. neck 落地：无独立槽位 → 封装为单一 BACKBONE_2D（Task 3）
- B. 新增 CenterPoint detector，不复用 PointPillar（Task 4）
- C. FPN 输出 80×80 / MDFEN 160×160，SepHead stride 分档（Task 3/4）
- D. dIoU 即源码 `bbox3d_overlaps_diou`，不新增 utils（Task 4）
- E. 参数量验收走 reparam（Task 5/8）
- F. VFE 关 decomposition 保 7 维（Task 5）
- G. 新增 SecondFPN 类，不复用 BaseBEVBackbone（Task 3）
- H. 不降级 numba（Task 0）
- I. 编译设 `TORCH_CUDA_ARCH_LIST`/`MAX_JOBS`（Task 0）
- J. MDFEN 必交付：环境问题新建 radarnext310(Py3.10+torch2.1+cu121) 编译 DCNv3（Task 7）
- K. yaml override 去 gt_sampling（Task 5）
- L. overfit-1-batch 前置（Task 6）
- M. head 构造签名对齐 build_dense_head / CLASS_AGNOSTIC / BN uniform / rbr_scale 条件 / num_repeats / code_weights[8] / aligned IoU（散落各 Task）
- **N. 验证方法论升级（用户）：数值对拍(Parity)为主判据(Task 4.5)，短训练(10-20ep)为双保险(Task 6/7)，不再要求 80ep+mAP 精确对账**

**1. Spec coverage：** item1→Task1，item2→Task0，item3→Task2/3/4/7+对照表，item5→Task1/5/7，item6→Task4.5(对拍)+Task5(参数量)+Task6/7(短训)+Task8(汇总)，全部覆盖。

**2. 占位符扫描：** 无 TBD/TODO；所有决策已锁定（执行协议 §1/§2/§5），数值锚点为论文 Table I/II + 源码 + 对拍判定。

**3. 类型/命名一致性：** `RadarNeXtFPNBackbone`(Task3,Task5)、`CenterPoint`(Task4)、`RadarNeXtCenterHead`(Task4,Task5)、`MDFENNeck`(Task7)；`get_loss()`/`post_processing(batch_dict)`/`reparameterize_model()` 契约贯穿；对拍脚本命名 `test_parity_<module>.py` 一致。`out_channels=[64,128,256]`、`channels_list=[64,128,256,128,64,128,256]`、`code_weights[8]`、`num_repeats=[1,1,1,1]` 在定义与 YAML 两处一致。

**4. 风险已标注并给缓解：** 8GB 显存(bs 冒烟+AMP)、DCNv3 编译(新环境 cu121+对拍验证 CUDA 正确)、类别顺序(强制对齐)、gt_box 中心语义(predict 的 z 偏移核对)、base 装包回滚(freeze 快照)、I/O(/mnt/d 慢建议拷 ext4)、**对拍前提(权重逐层拷贝+合成输入，层名映射脚本)**。

---

## Execution Handoff

计划已保存至 `docs/superpowers/plans/2026-07-15-radarnext-openpcdet-port.md`。两种执行方式：

**1. Subagent-Driven（推荐）** — 每个 Task 派一个新 subagent，任务间我来 review，快速迭代。适合本计划这种"翻译+逐模块验收"的线性结构。

**2. Inline Execution** — 在当前会话里按 executing-plans 批量执行，带 checkpoint。

你选哪种？

# RadarPillar + Tracker 工程问答复盘（对抗式 Q&A）

> 形式：挑战者 agent 提问（22 问，由浅入深四层）× 工程作者 agent 逐问查证后如实作答，独立双 agent 生成。  
> 目的：温故知新、查缺补漏、探索改进可能性、稳定模型输出。  
> 如实纪律：回答显式分三类——(a) 有仓库证据的当时考量（带路径）；(b) 无记录、标注为"现在复盘/推测"；(c) 直接承认的缺口。不粉饰、不编造"其实考虑过"。  
> 生成日期：2026-10-18。事实底座：PROJECT_SUMMARY.md 及其来源索引。

---

## L1 背景与动机

**L1-1 这是一个纯雷达单模态工程：MSRv2 上 Ped BEV 只有 28~32，横穿泛化集 mAP 低到个位数——到底什么真实应用场景能容忍这个精度、又非雷达不可？摄像头这么便宜，为什么从第一天起就不考虑 radar+camera 后融合，而把自己逼进"纯雷达 3D 检测"这条最难走的路？**

答：先纠正一个前提——工程定位不是"找个能容忍 28 AP 的量产场景"，而是"在既有量产链路里替换 B 段（dets→objs）"：`salon.md` §1 写明现有量产基线就是 `ADC→dets→objs→trackers` 纯雷达链路（FFT/OPTISIC/PDA），对标物是规则链路而非理想精度，工程命题是可行性验证。所以"非雷达不可"不是我们论证出来的，是链路现实给定的。但我得承认：Ped BEV 28~32、泛化 mAP 1.7~2.5（`output/generalization_eval_report.md`）对任何量产口径都不可接受，我不粉饰。radar+camera 后融合我 grep 了全仓（mynote/、salon.md、tracker doc），零记录——第一天就没纳入候选。当时理由（现在复盘）是数据集侧 MSR 系 IMAGES 目录有图但标定不可用（calib 全 0 不可投影）、工程边界只圈了 B 段，但没留下书面的排除决策，这算记录缺口。

**L1-2 "ZU3EG 双核 A53 + DPU B512 + 参数 0.06~0.9M"这条资源包络，是整车 BOM 的真实约束还是自设的靶子？一块 Jetson 级 SoC 就能消掉整个 INT8/DPU 痛点——为坚持 FPGA 付出的 8.43pt 量化掉点和整条部署链工期，换回来的核心收益究竟是什么？**

答：是真实板卡约束，不是自设靶子。`mynote/ZU3EG_部署评估与方案.md` §3 写的是"功能 A 已占用后剩"——LUT 71%、BRAM 78% 已被既有功能吃掉，两核 A53 是系统占用后的剩余；这块板要同时承载既有功能，这个包络是硬件组给的现状。Jetson 级 SoC 能消掉 INT8/DPU 痛点——这个替代方案我没有任何评估记录，当时就没想，承认。现在复盘坚持 FPGA 的可辩理由只有：与既有功能 A 同板共存、确定性行为、量产链路形态延续；但"换 NPU 类 SoC 整体更省工程量"这个反问是成立的，我没答过卷。

**L1-3 为什么选 RadarPillars 论文作底座，而不是从 PointPillars 起步、或直接移植 RadarNeXt？"比 RadarNeXt 轻 9 倍还更准"是选型前就有的先验，还是复现后才撞上的惊喜？复盘来看，这篇论文的方法哪些部件在你们数据上真站住了，哪些是死重？**

答：是复现后的实测结论，不是先验。顺序是：7 月先做 RadarPillars 复现（`note/radarpillar复现结论.md`：参数 0.18650M vs 论文 0.27M，层数对齐），然后 VoD stage0 三 baseline 锚点对照里 RadarPillar 0.184M/R40 48.71 超 RadarNeXt-MDFEN 1.64M/45.46（`experiments/PUBLIC_DATASET/stage0/`）——"轻 9 倍还更准"就是这一对数字。另外质询前提要补一半：PointPillars 标准基线（MSRv2 上 4.861M/43.34）和 RadarNeXt 整模型复刻（3D 91.82）我们都跑了，它们是矩阵里的 baseline/上限参照，不是没做，只是没当底座。论文部件在 MSR 上真站住的：pillar 化 + PillarVFE 雷达特征工程、（VoD 上的）PillarAttention；死重：论文 (64,64,64) 宽 backbone（32 通道够用）、MDFEN（MSR 三档 0 正）。但 PAtt 在 MSR 自有数据上其实没有受控对照——这接 L2-2。

**L1-4 设计文档 McRadarPillar.md 里写了 MC-A1 独有的 ispassable 通道，还配了 BCE loss 公式、标注"无论文依据"，最终一行代码没实现——为什么蓝图里要放一个没想清楚的字段，又为什么悄悄放弃、连个决策记录都不留？**

答：属实，认。`mynote/McRadarPillar.md` 里确实配了 L6×BCE 公式并自注"无论文依据"，pcdet/ 和 tools/ 下 grep 零实现——数据结构里 GT 有这个字段，设计文档就照着字段全画满了。为什么放弃没记录：没有显式决策，真实状态是设计文档是全字段 wishlist，工程精力全部压在 head 主线（换头/瘦身/vel）上，ispassable 从进不了前三优先级，就一直躺着。"连决策记录都不留"这个批评我接——这违反了我自己定的证据链纪律，是纪律缺口，不是有意裁剪。

---

## L2 架构与选型

**L2-1 4D 雷达的核心卖点就是俯仰维，你们却在 pillar 化时把 z 塌进 BEV 伪图、再靠 z/h 联合监督在 head 侧辛苦追回垂直结构——这不就是自己制造问题再自己解决吗？稀疏 voxel 或让 z 在网内保留更深不是更顺理成章？做过任何 voxel 对照实验，还是根本没试就锁死了 pillar？**

答：先修一点措辞——点在进 pillar 前带 z（6 列输入含 z），PillarVFE 点特征有 z，塌的是 backbone 侧 2D 卷积没有垂直感受野，z/h 联合监督是在这个损失上补判别信息，不是先把 z 扔了再捡回来。但核心指控我认：voxel（3D sparse conv）对照实验没做过，全仓 grep spconv/VoxelBackBone 零结果；pillar-first 是部署先验（`mynote/ZU3EG_部署评估与方案.md` §1 直接以 2D CNN 可部署性排除 3D conv 路线）加论文底座路径依赖锁死的，不是数据上的对照裁决。"部署约束否决 voxel"这个理由本身成立，但"voxel 在这个数据上到底差多少"确实没验过，是实验空缺。

**L2-2 PillarAttention 的战绩：VoD 上 +3.6% 参数换 +3.3 mAP，但 MSR 上加宽到 64 通道成了"免费抽奖"、双头变体负向，板端它还留在两核 CPU 上跑直接吃 67.8ms 的预算——在 MSR 数据上究竟有什么受控证据支撑"部署模型必须保留 PAtt"，而不是换成 SE 或干脆删掉？**

答：先报核实结果：部署模型 `experiments/MSRV2_DATASET/YAML/msr_pp64_ch2d_trunk.yaml` 确实有 PillarAttention（32 通道 1 head），板端 front 段 CPU 跑的就是 `deployment/board/cpp/src/pillar_attention.cpp`，NEON 优化后仍占 20.3ms。MSR 数据上的受控证据：没有。MC 矩阵 46 个 YAML 里 44 个带 PAtt，仅有的 2 个不带的是 radarNeXt 复刻——backbone 整个不同，混杂变量，不构成对照；矩阵里唯一的 PAtt 实验是 head2（1→2 头）负向。PAtt 的支撑证据只有 VoD stage1 的 a0-a4 对照（+3.3 胜 SE），而那是公开集。保留它的真实理由是贪心链胜者固化加 `deployment/BOARD_DEPLOY.md` L49 的 L1 同构换权族定义（A 族 = PillarVFE 单层 + PAtt 单头 + ch2d_trunk），换 SE 或删掉需要重开部署链适配，当时没人提出也没有数据推动。这是缺口，我认：部署模型里最贵的 CPU 部件缺少自有数据上的存在性证明。

**L2-3 Ped +23.8~26.3 的换头红利，到底是"center heatmap 优于 anchor"这个范式差异，还是单纯"头容量翻了 2.3 倍"？anchor 基线的 anchor 尺寸先验认真调过吗，还是沿用默认？另外为什么从不考虑 Transformer 系检测头（DETR 类），是算力判死还是有别的原因？**

答：归因没拆干净，我认。anchor 先验只做过一次设定：实验 yaml 头注写明"anchor=GT 中位尺寸"，每类单 anchor 档、双朝向，从 GT 统计中位数一次性填入，没扫过 anchor 多档/尺寸容差——"认真调过"谈不上。容量解释有一个反向旁证：trunk 瘦身把 head 参数再砍 36% 精度反升（`mynote/实验结论整理1.0.md` 表 1-2），说明增益不是靠 head 容量堆的；但"anchor 头加容量到同参数量"这个正面对照没做，范式 vs 容量的严格归因确实没完成。DETR/Transformer 系检测头：全仓零记录，没考虑过——现在复盘的理由（边缘部署迭代慢、热图系已在 MSR 验证有效）是事后合理化，当时就是没进候选空间，认。

**L2-4 head 瘦身族砍了 hidden 卷积、砍了 3×3、砍了通道，唯独没碰 heatmap 本身——降 head 工作分辨率、heatmap 通道减半这类动结构的砍法为什么不在候选空间里？热图+五回归卷积是计算量 ×3.2~3.6 的主因，它还恰好是 INT8 最敏感的部件——最肥的一块肉为什么一刀没切？**

答：先纠正一半：通道减半其实切过——narrow 变体把共享 trunk 64→32（heatmap 头也走这个 trunk），-40% 参数近无损（表 1-2）。真正没进候选空间的是 heatmap 工作分辨率（stride 2→4）和按头砍热图类通道，这个指控成立。真实原因是时序错位，不是故意护着：head 瘦身在 8 月做，判据只有 AP/参数/FLOPs；INT8 敏感性是 9 月梯子终判才暴露的（hm 量化误差 mean_abs 0.502，出处 `salon.md`:277；`deployment/DEPLOY_GUIDE.md` L118 仅定性表述根因）——出事之后没有回头重开一轮"量化感知的 head 消融"。这直接暴露 L4-1 要认的更深问题：量化鲁棒性从来不在选型判据里，选型表 8 款没有 INT8 掉点列。

**L2-5 选型表白纸黑字把 radarNeXt 标为"上限参照/teacher"，ch3d_trunk 差它 0.77（保 99.2%），teacher 单帧 4.5G 推理成本完全可承受——把 radarNeXt 的 3D 能力蒸馏给轻量头这件事，为什么从头到尾没做？是试过失败没落盘，还是纯粹的优先级疏漏？**

答：优先级疏漏，不是失败未落盘。grep 全仓"蒸馏/distill/teacher"只有两处，都是 `mynote/实验结论整理1.0.md` 选型表里那句"作后续蒸馏 teacher（单帧 4.5G 成本可控）"——写下了意图，之后没有开过任何一个蒸馏 run，也没有失败记录。8 月矩阵收官后主线立刻切 MSRv2 套件和部署链，蒸馏被无限期后置。特别是 INT8 出事之后，"用 radarNeXt 蒸馏轻量头提升量化前精度余量"本该被重新提起，也没有。这是实打实的缺口。

**L2-6 tracker 状态恒 4 维 [x,y,vx,vy]：CTRV 需要的航向/角速度不在状态里，ch3d head 辛苦输出的 z 也不进滤波——这个极简状态是刻意设计还是将就？关联用 KM/GNN 硬指派，对 Ped 这种 4.6 点/gt、高频漏检的目标，为什么不用 JPDA/MHT 软关联？漏检一多，GNN + 存在概率记分不就是碎轨制造机？**

答：查了代码和 spec，4 维状态是刻意设计不是将就：`tracker/doc/design/spec/TrackFrame.md` L197 明写"BEV 4 维 [x,y,vx,vy] 常速度 KF；z/heading/lwh 不在此"，L213 说明 z/l/w/heading 走隐藏历史直通维护——设计意图是把运动学滤波和属性维护解耦；CTRV 的 yaw_rate 是外部量测输入（`tracker/filter.py` ctrv_predict 接收 yaw_rate_rad），不是状态估计量，车载 ego 转速表可得，所以 4 维状态对 CTRV 是自洽的。但 JPDA/MHT 软关联：grep 零结果，没试过，认。Ped 4.6 点/gt 高频漏检场景下硬指派的碎轨代价没有量化对照，"GNN+存在概率=碎轨制造机"这个方向性批评无法反驳——泛化 Ped MOTA 9.8% 我们定性为检测漏检主导（回波 40→5 点 + 模型内 0.2 阈值先裁），但 tracker 侧放大效应没拆过，这是未验证区。

---

## L3 实验与方法论

**L3-1 你们自己实测 run 级 σ 高达 4.6~9.4，比 <1 mAP 的"相当带"高一个数量级，却让整个 39 模型矩阵连同最终部署选型全部建立在单 seed 上——41 次训练一周跑完说明算力根本不稀缺，关键对的 3-seed 复验凭什么一直往后拖？单 seed 抽到 +1.25 的 ppmixrev_pn64 还被写进了部署选型表，这算什么纪律？**

答：没得辩，全认。σ 4.6~9.4 vs <1 判相当带差一个量级、41 训练一周跑完、单 seed 的 ppmixrev_pn64（+1.25，PN64 七对里唯一可判正）进 B1600 选型档——三件事都是仓库事实（`mynote/LESSONS_LEARNED.md` 2.5、`mynote/实验结论整理1.0.md` 第三节与表 2-1）。更难堪的是我自己在 LESSONS 2.5 写了"跨 run 报数需 ≥3 seed 配对"，写完没执行。真实原因是矩阵时代以"扫满宽度"为目标，种子复验被排进"实验空缺"清单垫底，随后 MSRv2/部署两条主线把算力和注意力全抽走。41 次训练的算力余量摆在那里，这个解释不构成理由，就是纪律债。

**L3-2 VoD stage0-4 逐段贪心，MSR 矩阵又把 backbone 实验全部固定在 ch2d_trunk 上——你们自己的文档承认 MDFEN 对比捆绑了 bs 与分辨率两个混杂变量、ch3d_trunk × 宽 backbone "3D 榜大概率未封顶"却从未验证。贪心链胜者凭什么让人相信是联合最优？这个空缺直接封顶 tracker 选型，为什么不优先补它？**

答：不凭什么信——贪心只能证伪不能证联合最优，且混杂我自己落盘了：MDFEN 对比捆绑 bs 8→4 加出口分辨率两个变量（表 1-4 混杂警告）、backbone 全程固定 ch2d_trunk、ch3d_trunk 只跑过 PP32 窄版、3D 榜大概率未封顶——这些全写在 `mynote/实验结论整理1.0.md` 第三节空缺清单第 1、5 条，我知道且写明了，但没补。没补的真实原因：矩阵收官时选型结论"两步法"（先定 2D/3D 再按档取最优）已经够当时部署决策用，补实验的边际收益让位于 MSRv2 套件。现在复盘这是短视：ch3d×pp64 一对实验就能定 tracker 档上限，成本一晚，一直欠着。

**L3-3 z/h 联合监督是全工程最漂亮的机制结论，但置信度自评 ~75%、n=1 seed，文档里设计好了三个一锤定音的判决实验（stop-grad 隔离、场景归一化 z、3-seed 配对）——为什么一个都没执行，就放心让它驱动 ch3d 家族选型和对外叙事？**

答：认。`mynote/LESSONS_LEARNED.md` 2.6 白纸黑字列了 stop-grad、场景归一化 z、≥3 seed 三个判决实验，grep 代码零实现——一个都没跑。z/h 结论却进了选型表（ch3d 家族"tracker 刚需"定位）、进了 salon 叙事。当时的真实状态：五臂受控加机制分析让我自己信了 ~75%，觉得判决实验是"加固"不是"判决"，分析收敛后精力直接转部署。但 75% 置信度的事驱动选型还对外讲，就该先把 stop-grad 跑了——一锤定音的实验最便宜，没做没有借口。

**L3-4 eval 内核已经错了三回（rotate_iou 致 Car_3d=0、has_label 硬编码致 AP 全空、5 类聚合致 Cyclist 假崩），每次错误数字都被下游决策消费过，现在 MSRv3 又出现 3D>BEV 违反合法性判据且未收口——除了"下次小心"，工程上落地了什么结构性防线？3D≤BEV 这类不变式为什么不能做成 eval 的自动断言？**

答：已落地的结构性防线有三件：test.py/test_cpu.py 双口径互证（GPU eval 存疑时纯 CPU 全链仲裁）、pickbest max+median 双落防单 epoch 尖峰、口径修正记录落盘成文档。但 3D≤BEV 合法性判据——grep `pcdet/datasets/kitti/kitti_object_eval_python/eval.py` 确认——只写进了文档纪律（salon 4.3"自此写进纪律"），没做成 eval 自动断言，MSRv3 又冒出 3D>BEV 未收口（工作区修复进行中），证明文档纪律挡不住下一次。这一问的批评成立：判据已经是明文知识，加十几行 assert 的工作量，三次事故之后还没做，是"纪律制度化"上的双重标准——CenterHead 碎调用我制度化进了 CLAUDE.md 强制规范，eval 断言没有同等对待。

**L3-5 横穿行人泛化缺口被定性为"数据问题"（训练 94% 行人在 ±12° 内），但你们的增广配置全在 DISABLE_AUG_LIST 里、模型内 0.2 阈值先裁也明知是帮凶——在下"必须补数据"的结论之前，旋转增广、阈值暴露这类零成本域内手段到底试过没有？试过的话数字在哪？**

答：先报核实结果：dataset yaml 基座（msrv2/msrv3）的 DISABLE_AUG_LIST 确实全禁 rotation/flip/scaling/gt_sampling，但实验 YAML 覆写后实际训练用 flip+scaling（flip 修复后还带 NEG_POINT_COLS 速度列同步取反）；random_world_rotation 在 MC/MSRV2/MSRV3 所有实验里从未启用过，gt_sampling 同样从未启用。模型内阈值：代码默认 0.1（`pcdet/models/dense_heads/radarnext_center_head.py`:819），部署 yaml 显式设 0.2；阈值敏感性只在 z/h 分析时实测过一次（0.2→0.1 使 gap +2.22→+0.90，LESSONS 2.5），没有作为横穿缺口的处置实验落盘。所以答案：没试，数字不存在。当时的定性依据是分布证据（94% 行人 ±12°、回波 40→5 点），直接跳到"扩训练分布/补数据"，旋转增广这个正对方位角分布的零成本手段被跳过了。这是排查顺序上的真实疏漏，认。

**L3-6 39 个模型共用一套训练配方，但你们自己实录过 bs 8→16 单项就能撬 ~9% AP——anchor 头与 center 头对配方的敏感度天然不同，凭什么相信"换头 +6.83"里没有一部分是"center 头恰好更喜欢这套配方"的伪影？做过任何 HP 敏感性抽查吗？**

答：bs 8→16 约 9% 的实录在 VoD 复现（LESSONS Ch1 1.5），39 模型统一配方（MC 46 个 yaml 逐值一致：bs=4/80ep/LR 0.003 同参；MSRV2 则是 SH 命令行实跑 bs=8/100ep），anchor 头与 center 头的配方敏感度差异没测过，HP 敏感性抽查没做过——都认。可辩护的边界：矩阵内比较是同配方单变量配对差，配方偏好若与头类型无交互就不影响相对结论；z/h 五臂也是同配方受控。但"头类型×bs/lr 交互"这个威胁模型确实没排除，+6.83 里含配方伪影的可能性无法证伪。正确做法是换头关键对补 2~3 配方点——和 3-seed 一样便宜，一样欠着。

---

## L4 工程部署与反思

**L4-1 INT8 worst -8.43pt 根因判定为 per-tensor 粒度 × heatmap 敏感，梯子六档全试，而 per-channel 本是卷积权重量化的行业默认——为什么在 vai_q 容器探测不到 API 入口时就止步，而不是更早换工具链或落地 16bit 混合？更根本的：手头有 anchor 系候选，量化鲁棒性为什么从来不是模型选型的判据之一？**

答：逐项对账：16bit 混合试了——d4 档就是它，mixbit 无效，d2/d3/d4/d5 四档 ap.json 完全相同；QAT +0.34 无效。per-channel 本在梯子 d1 档，卡在 vitis-ai-pytorch-cpu 3.5.0.306 容器的 nndct 无 per-channel API 入口，探测笔记本身还是个悬空引用（PROJECT_SUMMARY 附录 A：三处文档引用、文件本体不存在）。换工具链 = vai_q_onnx，获取卡点三条路（gh-proxy 拉 whl / Xilinx conda channel / 1ms.run 镜像）都卡在网络获取，被列为路线债后置——回看确实是"止步"早了，网络获取问题不该排在一个 -8.43pt 的硬阻塞前面。量化鲁棒性不是选型判据：属实，认——矩阵在 8 月、第一次量化终判在 9 月中，选型表从没加过 INT8 掉点列，`mynote/部署先验.md` P1 步骤 3 写了"INT8 掉点门禁"却从未在选型阶段执行。anchor 系候选手头有（heatmap 敏感 ⇒ anchor cls 结构或许更稳），没测。

**L4-2 全部署阶段的 headline 数字 67.8ms/帧，附录 A 自己承认一手对拍产物缺失、只剩文档转述——为什么这个数字没有在验收当场落袋归档？且余量只有 3%，这 67.8ms 是纯检测还是含 tracker？跟踪链路在板上的延迟预算到底排在哪？**

答：先报核实：`deployment/board_results_20260928/` 目录不存在，BOARD_DEPLOY.md 的引用悬空——67.8ms 目前只有文档转述加 optimize 阶梯数字，PROJECT_SUMMARY 附录 A 我自己挂了这笔账。为什么没落袋：板上结果按 `deployment/deploy_checklist.md` ⑥ step6 走 scp 取回，取回到了 WSL home 而非仓库，后续工作区清理时没有归档进 store——验收流程有"取回"步骤但没有"取回物入仓"的强制关卡，落袋纪律在最后一米断了，认。67.8ms 是纯检测全链（front CPU + DPU + back：radar_pipeline_board full），板上没有 tracker——cpp 源里无任何 tracker 文件。所以 tracker 板上延迟预算：不存在，没排过。SIL 回灌全在 PC 上做，跟踪上板是未规划项。

**L4-3 书面约定 DPU B1600，实际全部 xmodel 实编 B512，文档与产物不一致——哪个是有效事实？B512 真相对"B1600 档"选型表和 67.8ms 实测意味着什么？为什么没有人回头修正选型口径或补测？**

答：有效事实是 B512——板上实际跑的就是它（store 包 meta.json 实测口径）。书面 B1600 是 deploy_checklist ①给 FPGA 工程师的初始约定，而 `mynote/ZU3EG_部署评估与方案.md` §4 早就判定 B1600 需 ~60 BRAM 超 32 不可行、可行档是 B512/B800——两份文档自相矛盾，收 XSA 后按实际 arch.json 编出 B512，没人回头把 checklist 的书面约定改掉。salon 7.1 里挂了这笔债（"DPU 实交付 B512（书面约定 B1600）"），知道但没修，属于文档口径不同步的卫生债。补测 B1600 无意义（资源上不去），该做的是改书面口径。

**L4-4 选型叙事说"tracker 刚需 z/h 所以选 ch3d 家族"，ZU3EG 方案文档也写待部署模型是 centerhead3d_trunk，但实际部署包是 msrv2_pp64_ch2d_trunk——一个没有 z/h 输出的 2D 头。板上 tracker 到底吃哪个模型？"z/h 刚需"是怎么被悄悄放弃的？**

答：如实交代：板上没有 tracker，板上只有检测全链；tracker 在 PC 侧 SIL 回灌，跑的是五模型套件（含 ch2d_trunk 和 vel 头）。时间线是：8 月 `mynote/部署先验.md`/ZU3EG 方案以待部署模型 centerhead3d_trunk（"tracker 刚需 z/h 只有 ch3d 家族"是 v1 选型表口径）立项，9 月 12 日部署链启动时 `deployment/dep_common.py` 锚定的已是 msrv2_pp64_ch2d_trunk——切换发生在 MSRv2 套件出分之后（ch2d_trunk 是 v2 五类 BEV 均值最高 52.21），理由是部署链首要目标定为"B 段 2D 检测替换先跑通"。但这次切换没有显式决策记录，"z/h 刚需"的旧口径也没被正式撤销，两套叙事并存至今；vel head 后来长在 ch2d_trunk_vel 上，更是把"3D 头供 z/h"的必要性事实上架空了。质询说的"悄悄放弃"成立：是选型口径演进无记录，不是深思熟虑的推翻。这也是 tracker 侧欠的账——z 量测消费决策（部署先验 P1 步骤 1）从未正式闭环。

**L4-5 MSRv3 每帧约 30 个点，还沿用为密点云设计的 pillar 1.0×0.5m / 200×80 BEV——绝大多数 pillar 是空的，等效在给空图跑卷积。为什么明知极稀疏还整包继承 v2 架构？粗网格 pillar 甚至点级直接回归（30 个点上直接 PointNet+头）不是更贴合吗？**

答：核实无误：MSRv3 两个实验 YAML 的 VOXEL_SIZE [1.0,0.5,20]、range x[0,200]y[±20]、MAX_VOXELS 2048/3096 全部照抄 v2，pillar/范围/密度参数零调整（MAX_POINTS_PER_VOXEL 16 与 MAX_VOXELS 亦逐值照抄 v2，零稀疏适配）。为什么：入库走的是"CLASS_LABEL_MAP 覆写即可换代"的标准流程，当时明确把"极稀疏场景的 anchor/增强调参"列为 v3 定版的下一步（salon §10）——即先标准基线定锚、架构适配后置，这有记录。但"粗网格 pillar 或点级直接回归是否更贴合极稀疏数据"这个问题，连讨论记录都没有：30 点/帧下绝大多数 pillar 为空，pillar 伪图范式的适配性确实该被质疑而非默认继承。加上 v3 目前卡在 eval 口径修复（run status=blocked），架构适配一直没启动。后置是计划，没评估范式是缺口，两者都如实。

**L4-6 如果整个工程推倒重来，你会明确翻哪些决策的案——pillar-first 路线、单 seed 协议、per-tensor 量化路线、蒸馏的缺席，还是 VoD 复现优先的顺序？最后直言：这个工程当下最大的技术债是什么，下一步最值得投入的一件事是什么？**

答：翻案清单：单 seed 协议必翻——最便宜、最该做、所有选型数字的地基，3-seed 应该是矩阵设计的一部分而非事后补丁；per-tensor 量化教训要制度化成"量化掉点进选型表"的硬列；蒸馏缺席是真缺口，teacher 都标好了没接。不翻的案：pillar-first 不翻——部署约束（2D CNN 可上 DPU、算力包络）真实存在且 pillar 在三代数据上没输过；VoD 复现优先也不翻——移植保真是后面一切自研 claim 的锚点，4.4 个月里它回报最高。当下最大技术债两笔并列：INT8 -8.43pt（部署精度门禁未过，全部 store 包带红旗流通）和单 seed 协议（选型可信度地基）。下一步最值得投入的一件事：把 vai_q_onnx 拉下来跑通 per-channel 加敏感层 FP16——它是部署交付唯一硬阻塞，网络获取卡点再烦也是一次性成本；成了以后量化鲁棒性数据顺手回填选型表，一鱼两吃。

---

## 回答 agent 自查（如实性分类）

**(a) 有仓库证据支撑的回答**：L1-1（salon §1 链路定位；grep camera/fusion 零结果）、L1-2（ZU3EG 文档 §3"功能 A 已占用后剩"）、L1-3（`note/radarpillar复现结论.md`、stage0 对照）、L1-4（McRadarPillar.md 公式与"无论文依据"自注、代码 grep 零实现）、L2-2（部署 YAML 含 PAtt、板端 pillar_attention.cpp、MC 矩阵 44/46 带 PAtt 无受控对照）、L2-3（yaml 头注"anchor=GT 中位尺寸"、表 1-2 trunk 旁证）、L2-4（narrow 变体实测、DEPLOY_GUIDE L118 INT8 根因、时序=8月选型/9月量化）、L2-5（grep 仅选型表两处）、L2-6（TrackFrame.md L197/L213 spec、filter.py yaw_rate 输入、grep JPDA/MHT 零）、L3-1（LESSONS 2.5、实验结论整理三.4 与表 2-1）、L3-2（空缺清单第 1/5 条自载）、L3-3（LESSONS 2.6 三实验列明未执行）、L3-4（eval.py grep 无断言、已有防线 test_cpu/pickbest 有据）、L3-5（DISABLE_AUG_LIST 逐 yaml 核实、阈值默认 0.1 vs 部署 0.2、LESSONS 2.5 阈值实录）、L3-6（LESSONS Ch1 bs 实录）、L4-1（d2-d5 ap.json 相同、vai_q_onnx 三路卡点、悬空探测笔记）、L4-2（board_results 目录不存在、cpp 无 tracker）、L4-3（checklist B1600 vs meta.json B512 vs ZU3EG §4 判定）、L4-4（ZU3EG 文档待部署 ch3d → dep_common 20260912 锚 ch2d、7 包中 6 个模型包全 ch2d_trunk（另 1 个 Aries 系统卡不含模型）、切换无 commit/记录）、L4-5（MSRV3 yaml 参数逐一核实照抄 v2、salon §10 后置计划）。

**(b) 无记录、标注为"现在复盘/推测"的回答**：L1-1 radar+camera 排除原因（标定不可用/工程边界）与"对量产不可接受"的判断；L1-2 FPGA 收益复盘与 Jetson 未评估的坦白；L2-1 voxel 未做的归因（部署先验主导）是复盘；L2-3 DETR 排除理由明确标注为事后合理化；L4-4 切换动机（"v2 出分后首要目标定为 B 段 2D 先跑通"）基于时间线推断，无决策记录；L4-6 全部为复盘判断。

**(c) 直接承认的缺口**：L1-4（ispassable 无决策记录地放弃）、L2-1（voxel 对照未做）、L2-2（PAtt 在 MSR 无受控证据却占板端最大 CPU 预算）、L2-3（anchor 未认真调、范式 vs 容量未拆、DETR 没想过）、L2-4（heatmap 分辨率未进候选、量化后未重开 head 消融）、L2-5（蒸馏纯疏漏）、L2-6（JPDA/MHT 没试、碎轨放大效应未拆）、L3-1（3-seed 拖延、ppmixrev_pn64 入表）、L3-2（联合最优未验证且未补）、L3-3（判决实验零执行）、L3-4（3D≤BEV 自动断言没做）、L3-5（旋转增广/阈值处置实验没跑）、L3-6（HP 抽查没做）、L4-1（vai_q_onnx 后置过早、量化鲁棒性非选型判据、anchor 候选未测）、L4-2（一手产物未入仓、tracker 板上预算不存在）、L4-3（书面口径未修正）、L4-4（"悄悄放弃"成立）、L4-5（范式适配连讨论都没有）。

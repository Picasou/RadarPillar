# BUG.md — tracker 链路范式审查问题清单

- 审查日期：2026-08-25
- 复核日期：2026-09-03（9 域 fan-out agent 对全部 50 条论断逐条实证：file:line 取证 + pytest 18 项 + ctypes 回绕实测；勾选/描述/行号按当日工作区代码同步，含 6da5f28 与未提交改动）
- 范围：`tracker/` 全链路（tracker/loader/detector/matcher/filter/manager/evaluator/visualizer/schemas/utils + 全部 cfg）
- 方法：6 域并行深读 + 关键论断亲读实证（evaluator 空壳、trks 不重置、history 缩进、heading 字段类型）
- 背景：当前 cfg 均为 `mode=1`（纯检测），多数问题未触发；P0 与带 ★ 的 P1 是切 `mode=2`（全跟踪）前的硬门槛

## P0 —— 功能性错误，结果不可信

- [x] **P0-1** `self.trks` 跨序列不重置（2026-08-25 已修：`run()` 序列首行清 trks + `updater.reset()` 清 type_states/IMM bank；顺带修 P1-5 每帧剪枝 type_states；2026-09-03 复核属实，pytest 18 passed）
- [x] **P0-2** 评估链路三件套全断（2026-08-25 已修，2026-09-03 复核属实且为超集——另扩展 IDF1/DetA/AssA/HOTA/LocA/速度指标）：
  - [x] `evaluator.py` 实装：贪心最近邻 + 2m 门限 + 类别约束，TP/FP/FN/MOTA/MOTP/IDSW/Frag，per-class 分账 + 报告落盘 `eval.default/metrics_<template>.txt`（DATA.paths 公共父目录，需 `EVALUATE.report=1`）+ 每序列曲线 PNG（路径已非原文的 `output/tracker_eval/report.txt`）
  - [x] `loader._load_GTs` 实装：读 `gt.default/gt_radar_1200/1201.bin`（AUTOSIL 稠密标注，27B v2.0 紧凑为主 + 28B 旧格式兼容），无标注序列 warning——**但实际未跳过，见 P1-16**
  - [x] `history` 改帧循环内逐帧 `(gts, out_trks)` 配对
  - [x] eval_mode 语义理顺：1=online（逐帧记账+序列末打印）/ 2=offline（完整报告），mode!=2 不再误调 evaluate
  - 类别自适应对齐：数字类名（MSR 枚举即类名）按数值映射，名字类名按 Car/Pedestrian/Cyclist 映射；无对应类排除评估

## P1 —— 范式缺陷 / 边界隐患

### 编排 / 落盘
- [x] ★ **P1-1** `heading` 溢出（2026-09-03 已修：落盘两入口统一 `wrap180` 归一到 (-180,180] 再 ×100 量化，`utils/common.py:wrap180` + `tracker.py` 写入口；测试 `test_wrap180_quantize_safe`）
- [x] ★ **P1-2** `write()` 跳过逻辑无 mode 戳（2026-09-03 已修：overlap=0 时另存名按 mode 分档——mode=1→`0201.00001.bin`、mode=2→`0201.00002.bin`，检测/跟踪输出互不覆盖互不误跳）
- [x] **P1-3** 空序列 `frame` 未绑定 → NameError（2026-09-03 核实已修：`tracker.py:64-65` 空序列 `continue` 守卫 + frame 引用全在循环体内，run() 重构顺带消除）
- [x] **P1-16** 无 GT 序列未真正跳过评估（2026-09-03 已修：`loader.has_gt` 序列粒度判存 + tracker 评估三处入口按其门控，无 GT 序列不再产生 FP 污染；warning 文案改"该序列不参与评估"）

### 滤波
- [x] **P1-4** EKF 名存实亡（2026-09-03 核实已修，6da5f28：`updater.py:185-193` α-β β 通道由航向残差在线估计 yaw_rate，钳位 ±90 deg/s，EKF 消费点 `filter.py:243-247`；注意当前 cfg 均 `FILTER.type=2`（KF）不消费 ω，需切 type=3 生效；smooth=0 时退化清零，现 cfg 均 smooth=1）
- [x] **P1-5** type_states 无剪枝 + ID 复用污染（2026-08-25 已修：`predict` 每帧按存活剪枝 + 序列边界 reset；2026-09-03 复核属实）
- [x] ★ **P1-6** 配置陷阱：`cfg_gen_msr_*.yaml` 配 `dim=4` 与 MSR 检测 `vx/vy` 恒 0 冲突（2026-09-03 核实已消除，6da5f28：速度量测由 `trk.history` 滑窗差分自产并覆写 `obj.vx/vy`，`VELOCITY.enable` 默认 1；cfg 仍 dim=4 但速度分量不再来自检测的 0；仅出生首帧一次性零速瞬态；显式 enable=0 才回旧行为）
- [x] **P1-7** Q 无物理结构（2026-09-03 已修：Q 改由 `q_acc` (σa, m/s²) 按 CV 离散白噪声加速度模型 `cv_q(q_acc,dt)=σa²·GGᵀ` 逐帧展开，三处预测统一；cfg 全量 q 矩阵 → `q_acc: 5.0`；出生 P0 前期已修；测试 `test_cv_q_physical_structure`）

### 关联
- [ ] **P1-8** doppler 项死代码：`Obj.doppler` 全链无人填（detector/loader 均不填，`schemas.py:96` 注释"(loader/detector 填)"与事实不符），`gap_dim=3` 分支恒 (0−0)²（17 份 cfg 均 gap_dim=2，分支未激活）
- [ ] ~~**P1-9** 类别零成本混联~~（2026-09-03 标记：有意设计，不做类别门限；连带效应知悉——`_udt_type` 后验会缓慢改写吃错类轨的类型）
- [ ] **P1-10** 固定标量门：KF 协方差（coast 每帧 +Q 膨胀）不参与门控（matcher 不读 `trk.cov`），长漏检航迹难再捕获；gap_type=2 的"马氏"是目标外接椭圆非 KF 协方差，且 cfg 均 gap_type=1

### 管理
- [ ] ~~**P1-11** `obstacle_prob` 单向锁存~~（2026-09-03 标记：暂缓，暂不判断 obstacle_prob；另 `rw_struct.py:97` 该字段注释 "[0-100]" 与 0/1 锁存语义矛盾）
- [x] **P1-12** `death_heat` 可配 0（2026-09-03 已修：isvalid 下限 0→1，0 会连当帧刚量测航迹一并删除；测试 `test_isvalid_death_heat_zero_rejected`）
- [x] **P1-13** 落盘 `classification` 用 pcdet 1-based label（2026-09-03 已修：tracker 由数字类名建 `label2src` 映射，落盘前 pcdet label → 源 0201 枚举值；名字类名无映射原样落）
- [x] **P1-14** detector 丢弃 `box[2]/box[5]`（2026-09-03 已修：Obj 补 z/height 字段，detector 直传 box[2]/box[5]，出生继承 trk.z_m/height_m，两 mode 落盘均写真值；loader 回灌同步填充）
- [ ] **P1-15** 2031 存在时 `cycle_s` 硬编码 0.1（`loader.py:253`，行号自原文 191 漂移），cfg 值被忽略

## P2 —— 健壮性 / 死代码 / 风格

- [x] **P2-1** `RUN.delay` 定义+校验但零消费（2026-09-03 已修：schemas 与 19 份 cfg 全量移除）
- [ ] **P2-2** `MANAGER.dt/history_horizon` 零消费（部分过时：dt 已被 evaluator VDE 换算消费 `evaluator.py:416`；history_horizon 仍零消费，miss/death 仍按帧数）
- [x] **P2-3** TrkHistory 5 个数组无写入点（2026-09-03 核实已修，6da5f28：出生/matched/coast 三处入史 + `VelEstimator` 差分消费，已成速度量测链核心依赖）
- [ ] **P2-4** `v_r_comp` 列 detector 不读，死数据（每帧计算随 (N,7) 传递，全链无读取点）
- [x] **P2-5** METRICS 整组开关零消费（2026-09-03 已修：amota/amotp 随 AMOTA 实装入 METRIC_KEYS 生效，samota 死键自 cfg 移除；分数载体 = trk.det_score 最近检测置信度）
- [x] **P2-6** visualizer `range_xy` 跨序列残留（2026-09-03 已修：begin_seq 逐序列重置为 cfg range/None 再按 data_extent 回退）
- [x] **P2-7** mode=2 Pred 标题计数翻倍（2026-09-03 已修：标题分列 `Pred (det N + trk M)`，mode=1 保持 `Pred (N)`）
- [x] **P2-8** ID 池满 100 后新观测静默丢轨（2026-09-03 已修：池满逐帧告警打印丢弃计数）
- [x] **P2-9** `frame_cnt` uint16 回绕（2026-09-03 防护：Raw_TrkHead 契约字段不可升位，写入口超 65535 钳位 + 一次性告警，不再静默回绕）
- [x] **P2-10** `rw_struct.decode` 用 `__sizeof__()` 越界读（2026-09-03 已修：改 `len(data)` 校验，短 buffer 抛 ValueError；测试 `test_decode_short_buffer_raises`）
- [ ] **P2-11** 点 z 加 z_pos 与训练侧公式不一致，分布漂移（zpos≠0 时生效；当前 cfg 均 z_pos_m=0 不触发，但 2031 bin 内 zpos≠0 会静默引入）
- [x] **P2-12** 四个 min/max 生成器各全量扫一遍序列点云（2026-09-03 已修：tracker.run 单遍扫描四值同收）

## 遗漏 —— 相对标准 MOT 范式缺的环节（2026-09-03 修复后）

- **评估**：~~逐帧配对~~、~~per-class~~、~~GT id~~、~~MT/ML~~、~~AMOTA/AMOTP~~（`_calc_amota` 阈值扫描，score 载体 = trk.det_score）均已实装
- **管理**：tentative/confirmed 显式状态机（无降级路径）、max_age 时间制（上桌判定已改 `lifetime_s>=birth_heat*cycle_s` 标称时间制，删除仍帧数）仍缺；~~merge~~（中心距 < merge_dist 优者留）、~~FOV/ROI 边界删除~~（= point_cloud_range 出界即删）、~~birth 近距抑制~~（birth_min_range）、~~ID 池满告警~~ 已实装（2026-09-03）
- **关联**：χ² Mahalanobis 统计门（cov 已有未用）、doppler 一致性约束、age 级联/二次关联、置信度加权（Obj.score 已有数据，matcher 不读）
- **滤波**：~~yaw_rate 在线估计~~（6da5f28）、~~Joseph form~~（2026-09-03 kf_update 改 Joseph 保数值对称半正定）已实装；自适应 R 仍缺；heading 180° 翻转已有 sin/cos 向量融合结构性防护（瞬时翻转卡死已解），基于速度方向的显式消歧仍缺
- **时序**：`time_100us` 未用、无逐帧时间戳、dt 恒标称周期（`(gap+1)*cycle_s` 仅覆盖漏量测间隔，非实测 dt）

## 修复优先级建议（2026-09-03 修复后收敛）

1. ~~切 mode=2 前必改：P1-1~~ 已清零（P0 全部 + P1-1/2/3/6/7/12/13/14/16 已修）
2. 剩余跟踪质量项：P1-10（门控接入协方差）、P1-8（doppler 回填）、置信度建轨门限（关联域，用户暂不做类别门限）
3. 数据口径潜伏项：P1-15（cycle_s 硬编码）、P2-11（z_pos 公式）、P2-2 残余（history_horizon）、P2-4（v_r_comp 死列）
4. 管理范式残余：tentative/confirmed 状态机、max_age 时间制、obstacle_prob 下桌（暂缓项）

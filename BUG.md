# BUG.md — tracker 链路范式审查问题清单

- 审查日期：2026-08-25
- 范围：`tracker/` 全链路（tracker/loader/detector/matcher/updater/manager/evaluator/visualizer/schemas/utils + 全部 cfg）
- 方法：6 域并行深读 + 关键论断亲读实证（evaluator 空壳、trks 不重置、history 缩进、heading 字段类型）
- 背景：当前 cfg 均为 `mode=1`（纯检测），多数问题未触发；P0 与带 ★ 的 P1 是切 `mode=2`（全跟踪）前的硬门槛

## P0 —— 功能性错误，结果不可信

- [x] **P0-1** `self.trks` 跨序列不重置（2026-08-25 已修：`run()` 序列首行清 trks + `updater.reset()` 清 type_states/IMM bank；顺带修 P1-5 每帧剪枝 type_states；探针实测 5 序列 id 均从 1 重新分配）
- [x] **P0-2** 评估链路三件套全断（2026-08-25 已修）：
  - [x] `evaluator.py` 实装：贪心最近邻 + 2m 门限 + 类别约束，TP/FP/FN/MOTA/MOTP/IDSW/Frag，per-class 分账 + 报告落盘 `output/tracker_eval/report.txt`
  - [x] `loader._load_GTs` 实装：读 `gt.default/gt_radar_1200/1201.bin`（AUTOSIL 稠密标注，28B 记录），无标注序列 warning + 跳过
  - [x] `history` 改帧循环内逐帧 `(gts, out_trks)` 配对
  - [x] eval_mode 语义理顺：1=online（逐帧记账+序列末打印）/ 2=offline（完整报告），mode!=2 不再误调 evaluate
  - 类别自适应对齐：数字类名（MSR 枚举即类名）按数值映射，名字类名按 Car/Pedestrian/Cyclist 映射；无对应类排除评估

## P1 —— 范式缺陷 / 边界隐患

### 编排 / 落盘
- [ ] ★ **P1-1** `heading` 溢出：c_int16（`rw_struct.py:72`）× `heading_deg∈[0,360)` → (327.67°,360°) 静默回绕（实测 `c_int16(35999)→-29537`）；落盘前归一到 (-180,180]
- [ ] ★ **P1-2** `write()` 跳过逻辑无 mode 戳（`tracker.py:121`）：overlap=0 时 mode=2 先落 `0201.00001.bin`，再跑 mode=1 见文件存在即静默 return，检测永不出盘
- [ ] ★ **P1-3** 空序列 `frame` 未绑定 → NameError（`tracker.py:80`）

### 滤波
- [ ] **P1-4** EKF 名存实亡：`trk.yaw_rate_degs` 出生置 0 后全库无更新点，恒退化纯 CV，转弯场景过程模型失配
- [x] **P1-5** type_states 无剪枝 + ID 复用污染（2026-08-25 已修：`predict` 每帧按存活剪枝 + 序列边界 reset；与 P0-1 一并落地）
- [ ] ★ **P1-6** 配置陷阱：4 份 `cfg_gen_msr_*.yaml` 配 `dim=4`，与 MSR 检测 `vx/vy` 恒 0 冲突，mode=2 即速度量测被拉向零
- [ ] **P1-7** Q 无物理结构（diag(1,1,1,1)，等效 σa≈100 m/s²）平滑极弱、不随 dt 缩放；出生 P0=0 无物理依据

### 关联
- [ ] **P1-8** doppler 项死代码：`Obj.doppler` 全链无人填（detector/loader 均不填），`gap_dim=3` 分支恒 (0−0)²
- [ ] **P1-9** 类别零成本混联：cost 不含 type，Car 轨可直接吃 Pedestrian 检测
- [ ] **P1-10** 固定标量门：KF 协方差（coast 每帧 +Q 膨胀）不参与门控，长漏检航迹难再捕获

### 管理
- [ ] **P1-11** `obstacle_prob` 单向锁存：上桌后无下桌路径，`existence_prob` 对输出门控失效（仅作落盘 confidence）
- [ ] **P1-12** `death_heat` 可配 0（isvalid 放行）→ 当帧刚量测的航迹也被全删
- [ ] **P1-13** 落盘 `classification` 用 pcdet 1-based label，与源 0201 车载枚举域不一致，下游按源枚举解析会错类

### 数据
- [ ] **P1-14** detector 丢弃 `box[2]/box[5]` → 落盘 z/height 恒 0
- [ ] **P1-15** 2031 存在时 `cycle_s` 硬编码 0.1（`loader.py:191`），cfg 值被忽略

## P2 —— 健壮性 / 死代码 / 风格

- [ ] **P2-1** `RUN.delay` 定义+校验但零消费
- [ ] **P2-2** `MANAGER.dt/history_horizon` 零消费（miss/death 全按帧数）
- [ ] **P2-3** TrkHistory 5 个数组无写入点，补偿循环空转
- [ ] **P2-4** `v_r_comp` 列 detector 不读，死数据
- [ ] **P2-5** METRICS 整组开关（tp/fp/fn/ids/mota/motp/idf1/amota/amotp/samota）零消费
- [ ] **P2-6** visualizer `range_xy` 跨序列残留：第 2 序列起沿用第 1 序列坐标范围
- [ ] **P2-7** mode=2 Pred 标题计数翻倍（检测框+航迹框同画同一物理目标）
- [ ] **P2-8** ID 池满 100 后新观测静默丢轨，无告警
- [ ] **P2-9** `frame_cnt` uint16，帧号超 65535 回绕
- [ ] **P2-10** `rw_struct.decode` 用 `__sizeof__()` 而非 `len()`，截断文件越界读
- [ ] **P2-11** 点 z 加 z_pos 与训练侧公式不一致，分布漂移（zpos≠0 时生效）
- [ ] **P2-12** `aix_lim` 四个生成器各全量扫一遍序列点云（O(4N) 浪费）

## 遗漏 —— 相对标准 MOT 范式缺的环节

- [ ] **评估**：逐帧 GT-tracks 配对结构、MOTA/MOTP/IDSW/IDF1/MT/ML、AMOTA 阈值扫描、per-class 分层、GT `id` 字段（现缺，IDSW/IDF1 无从算起）
- [ ] **管理**：tentative/confirmed 显式状态机（无降级路径）、max_age 时间制、merge（`_man_merge_trks` 空壳）、FOV/ROI 边界删除、birth 近距抑制
- [ ] **关联**：χ² Mahalanobis 统计门（cov 已有未用）、doppler 一致性约束、age 级联/二次关联、置信度加权
- [ ] **滤波**：yaw_rate 在线估计、自适应 R、Joseph form、heading 180° 翻转歧义防护
- [ ] **时序**：`time_100us` 未用、无逐帧时间戳、dt 恒标称周期

## 修复优先级建议

1. 切 mode=2 前必改：P0-1（序列重置）、P0-2（评估拦截或实装）、P1-1/P1-2/P1-3/P1-6
2. 落盘正确性：P1-1、P1-2、P1-13
3. 跟踪质量：P1-9（类别 gating）、P1-10（门控接入协方差）、P1-8（doppler 回填）
4. 评估实装顺序：GT 数据源 → 逐帧配对结构 → CLEAR-MOT 指标 → 报告落盘

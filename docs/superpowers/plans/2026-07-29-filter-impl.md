# Filter 模块实装计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实装 α-β / KF / EKF(CTRV) / IMM 四种滤波器的 predict + update 两阶段，达成 pytest 闭环。

**Architecture:** 三层结构——
1. **kernel 层**：纯数学函数（functional，入参/出参均为 ndarray，无 Trk 副作用），各算法的 predict/update 公式核心；
2. **adapter 层**：`AlphaBetaFilter` / `KalmanFilter` / `EkfFilter` 做 Trk↔(x, P) 转换、原地写回、coast 标记；
3. **IMM 层**：`ImmFilter` 内部按 `trk.id` 维护模型组银行（模型概率 + 各模型 x/P），直接组合 kernel 运算，混合结果写回 Trk 标准字段。

状态与协方差全部复用 `Trk` 现有字段（`x_m/y_m/vx_mps/vy_mps` + `cov` 4×4 + 各 `*_std`），零 schema 改动。

**Tech Stack:** Python, NumPy, pytest

## Global Constraints

- 未经用户明确要求，**不得 git commit**（各 Task 无 commit 步骤）
- 注释简洁只说功能，不写编号/阶段映射等元说明
- 量测维度跟随 dim 配置：`dim=2 → z=[x,y]`，`dim=4 → z=[x,y,vx,vy]`；H 为单位阵（dim=2 取左上 2×2）
- Q/R 来自 cfg（dim×dim 矩阵，dim=2 时 R 取左上 2×2），`Cfg.isvalid` 已保证形状
- 新增函数接口以本计划为准，执行中不得擅自扩展对外接口
- 测试放 `tracker/tests/test_filter.py`，沿用 `test_tracker.py` 的 helper 风格（`sys.path` insert + 自建 `make_trk` / `make_cfg`）
- 中间产物（图、CSV、调试输出）一律落 `tracker/tests/.tmp/`，随时可删，任何模块不得引用；目录不存在时脚本自建
- `matches.matched` 契约由本计划钉死：`list[tuple[Trk, Obj]]`；`unmatched_trks: list[Trk]`；matcher 侧 `_post_result` 实装属 matcher 任务，filter 测试直接手工构造 `Matches`

## File Structure（本计划改动的全部文件）

| 路径 | 动作 | Task | 说明 |
|---|---|---|---|
| `tracker/utils/common.py` | 修改 | 0 | `compensate_trks` 四处状态写回去 `int(round())`，改 float 直写 |
| `tracker/tests/test_tracker.py` | 修改 | 0 | 既有断言改 `pytest.approx`；新增非整数量化用例 |
| `tracker/filter.py` | 修改 | 1-5 | 共享 I/O 工具函数 + α-β/KF/EKF-CTRV kernel + 三个 adapter 实装 + IMM 组合层 |
| `tracker/schemas.py` | 修改 | 1 | 仅 `Matches` docstring 钉契约，**零字段改动** |
| `tracker/tests/test_filter.py` | 新建 | 1-6 | 全部单元测试 + 端到端回归 |
| `tracker/cfg/cfg.yaml` | 修改 | 5 | `para_imm` 补示例配置（models 子项 + α-β 似然用 r） |
| `tools/scripts/debug/filter_mc.py` | 新建 | 7 | MC 仿真验证脚本（场景/量测/MC 引擎/统计绘图四块） |
| `tracker/tests/.tmp/filter_mc/` | 新建目录 | 7 | RMSE 图 + 统计 CSV 输出；**中间产物，随时可删** |

除上表外不动任何文件。

---

## Task 0: compensate_trks 量化修复（前置）

**Files:**
- Modify: `tracker/utils/common.py`（`compensate_trks` 内状态写回）
- Modify: `tracker/tests/test_tracker.py`（`test_straight_compensation` 等断言）

**Interfaces:**
- Produces: `compensate_trks` 写回 float 全精度状态（签名不变）

- [ ] **Step 1: 新增失败测试** —— `test_tracker.py` 增加非整数量化用例：trk x=100.4, 车速 3 m/s, dt=0.1 → 期望 `x_m ≈ 100.1`（`pytest.approx`）。当前 `int(round())` 下得 100，必 FAIL
- [ ] **Step 2: 修复写回** —— `compensate_trks` 中 `x_m/y_m/vx_mps/vy_mps` 四处写回去掉 `int(round())`，直接赋 float
- [ ] **Step 3: 同步既有断言** —— `test_straight_compensation` 等用 `pytest.approx` 比较（值不变，仅比较方式）
- [ ] **Step 4: 验收** —— `pytest tracker/tests/test_tracker.py -v` 全绿

---

## Task 1: 共享状态 I/O + matches 契约标注

**Files:**
- Modify: `tracker/filter.py`（模块级私有工具函数）
- Modify: `tracker/schemas.py`（`Matches` docstring 钉契约，不改字段）
- Test: `tracker/tests/test_filter.py`（新建，含 `make_trk` / `make_cfg` / `make_obj` helpers）

**Interfaces:**
- Produces（供 Task 2-5 使用）:
  - `_read_state(trk: Trk, dim: int) -> tuple[np.ndarray, np.ndarray]` — dim=2 取 x/y + `cov[:2,:2]`；dim=4 取 x/y/vx/vy + 全 `cov`
  - `_write_state(trk: Trk, x: np.ndarray, P: np.ndarray, dim: int) -> None` — 写回运动学字段 + `cov` + std 字段（`x_std_m=√P[0,0]`，dim=4 时含 `vx_std_mps=√P[2,2]` 等）
  - `_read_z(obj: Obj, dim: int) -> np.ndarray` — dim=2 → `[obj.x, obj.y]`；dim=4 → `[obj.x, obj.y, obj.vx, obj.vy]`
  - `_mark_coast(trks: list[Trk]) -> None` — `measurement_status = 0`

- [ ] **Step 1: 写测试** —— ① I/O roundtrip：write 后 read 回读一致；② std 一致性：`x_std_m == √cov[0,0]`；③ `_read_z` 两种 dim 的组装；④ `_mark_coast` 标记
- [ ] **Step 2: 运行确认 FAIL**（函数未定义）
- [ ] **Step 3: 实装四个工具函数 + `Matches` docstring 契约标注**
- [ ] **Step 4: 验收** —— `pytest tracker/tests/test_filter.py -v` 全绿

---

## Task 2: α-β 滤波（kernel + adapter）

**Files:**
- Modify: `tracker/filter.py`（`abf_predict` / `abf_update` kernel；`AlphaBetaFilter._predict/_update` 实装）
- Test: `tracker/tests/test_filter.py`

**Interfaces:**
- Consumes: Task 1 的 `_read_state/_write_state/_read_z`
- Produces:
  - `abf_predict(x: np.ndarray, dt: float) -> np.ndarray` — 常速度外推，仅位置 `x[:2] += x[2:]*dt`
  - `abf_update(x: np.ndarray, z: np.ndarray, alpha: float, beta: float, dt: float) -> tuple[np.ndarray, float]` — 残差 `r = z − x[:2]`；`x[:2] += α·r`；`x[2:] += β·r/dt`；返回 (x', likelihood)（likelihood 用残差对固定量测方差的高斯近似，供 IMM 用）
- 约束：α-β 不维护 P，`cov` 保持不动；`dt ≤ 0` 时速度项不更新（防除零）

- [ ] **Step 1: 写测试** —— ① 静止收敛：z 恒定 (10,5)，20 帧后 |x−z| < 0.1；② 匀速跟踪：z(t)=(t,0) 序列，vx → 1±0.1；③ dt=0 无异常且速度不变；④ matched 航迹 `measurement_status=1`
- [ ] **Step 2: 运行确认 FAIL**
- [ ] **Step 3: 实装 kernel + adapter**（adapter 仅遍历 `matches.matched` 修正；`unmatched_trks` 调 `_mark_coast`）
- [ ] **Step 4: 验收** —— 全绿

---

## Task 3: KF（kernel + adapter）

**Files:**
- Modify: `tracker/filter.py`（`kf_predict` / `kf_update` kernel；`KalmanFilter` 实装）
- Test: `tracker/tests/test_filter.py`

**Interfaces:**
- Consumes: Task 1 I/O
- Produces:
  - `kf_predict(x, P, dim, dt, Q) -> (x', P')` — F 构造：dim=4 为 CV 模型 `[[I, dt·I],[0, I]]`；dim=2 为 `I₂`（随机游走，不外推）；`P' = FPF' + Q`
  - `kf_update(x, P, z, H, R) -> (x', P', likelihood)` — `K = PH'(HPH'+R)⁻¹`；`x' = x + K(z−Hx)`；`P' = (I−KH)P`；likelihood = `N(z; Hx, HPH'+R)`
- `KalmanFilter.__init__` 持有 `dim, Q(asarray), R(dim=2 取左上 2×2), H(单位阵截取)`

- [ ] **Step 1: 写测试** —— ① 预测外推：x'=x+v·dt，trace(P) 增长；② 更新收缩：trace(P_post) < trace(P_pre)；③ 收敛：恒定 z 20 次更新后状态 → z；④ likelihood：近量测 > 远量测；⑤ P 保持对称
- [ ] **Step 2: 运行确认 FAIL**
- [ ] **Step 3: 实装 kernel + adapter**
- [ ] **Step 4: 验收** —— 全绿

---

## Task 4: EKF-CTRV（kernel + adapter）

**Files:**
- Modify: `tracker/filter.py`（`ctrv_predict` kernel；`EkfFilter` 实装，update 复用 `kf_update`）
- Test: `tracker/tests/test_filter.py`

**Interfaces:**
- Consumes: Task 3 的 `kf_update`
- Produces:
  - `ctrv_predict(x, P, dt, yaw_rate_rad, Q) -> (x', P')` — 内部 (vx,vy) → (v, ψ)；`|ω| < ε` 退化 CV（与 KF 数值一致）；CTRV 闭式：`x += v/ω·(sin(ψ+ωdt) − sinψ)`，`y += v/ω·(−cos(ψ+ωdt) + cosψ)`，`ψ += ωdt`，再转回 (vx,vy)；F 雅可比为 4×4 对 (x,y,vx,vy) 的解析偏导；`P' = FPF' + Q`
- 约束：EKF 语义只在 dim=4 成立；`dim=2` 时 `EkfFilter` 内部委托 `KalmanFilter` 行为（组合复用，不复制代码）；`yaw_rate` 取自 `trk.yaw_rate_degs`（deg/s → rad/s）

- [ ] **Step 1: 写测试** —— ① 直行退化：yaw_rate=0 与 `kf_predict` 输出逐元素一致；② 恒转弯解析解：v=10, ω=0.1, dt=1 后位置与 CTRV 闭式解吻合（容差 1e-6）；③ 雅可比校验：与有限差分（步长 1e-5）逐元素比对，容差 1e-4；④ P 对称
- [ ] **Step 2: 运行确认 FAIL**
- [ ] **Step 3: 实装 kernel + adapter**
- [ ] **Step 4: 验收** —— 全绿

---

## Task 5: IMM（组合层）

**Files:**
- Modify: `tracker/filter.py`（`ImmFilter` 实装 + 内部 `_ImmBank` 状态容器）
- Modify: `tracker/cfg/cfg.yaml`（`para_imm` 补示例配置：models 子项 `{type, ...参数, r}`，α-β 子模型的 `r` 为固定量测方差供似然计算）
- Test: `tracker/tests/test_filter.py`

**Interfaces:**
- Consumes: Task 2/3 kernel（`abf_predict/abf_update/kf_predict/kf_update`，update 均返回 likelihood）
- Produces:
  - `ImmFilter._states: dict[int, _ImmBank]`，`_ImmBank = {probs: (M,), banks: list[(x, P)]}`，键为 `trk.id`
  - predict 流程：新 id 初始化（probs 均匀，各模型 x/P 同取 trk 当前状态）→ 输入混合（`π_j = Σ_i markov[i,j]·probs_i`；`x̄_j = Σ_i (probs_i/π_j)·x_i`；`P̄_j = Σ_i (probs_i/π_j)·(P_i + ΔxΔx')`）→ 各模型 kernel 预测 → 惰性剪枝（dict 中不在当前 trks 的 id 删除）
  - update 流程：各模型 kernel 更新取 likelihood `Λ_j` → `probs_j ∝ Λ_j·π_j` 归一化 → 输出混合（x = Σ probs_j·x_j；P = Σ probs_j·(P_j + ΔxΔx')）→ `_write_state` 写回 Trk
- 约束：子模型↔kernel 映射按 cfg 子项 type（1=α-β，2=KF）；α-β 分支的 P 不进 kernel（kernel 只动 x），仅在输入/输出混合中流转；模型数 M = len(models)，markov 形状不符时 `_build_filter` 报错

- [ ] **Step 1: 写测试** —— ① 概率更新：匀速目标多帧后 KF 模型概率 > α-β；② 归一化：Σ probs = 1，P 正定对称；③ 剪枝：trk 移除后 `_states` 对应 id 消失；④ 新 trk 初始化 probs 均匀
- [ ] **Step 2: 运行确认 FAIL**
- [ ] **Step 3: 实装 `_ImmBank` + `ImmFilter._predict/_update` + cfg 示例**
- [ ] **Step 4: 验收** —— 全绿

---

## Task 6: facade 路由 + 端到端闭环

**Files:**
- Test: `tracker/tests/test_filter.py`（仅新增测试，不改实现）

**Interfaces:**
- Consumes: `Filter` facade（已就绪）+ 全部实装

- [ ] **Step 1: 写路由测试** —— `FILTER.type=1..4` 分别构建出 `AlphaBetaFilter / KalmanFilter / EkfFilter / ImmFilter`；非法 type 抛 `ValueError`
- [ ] **Step 2: 写端到端测试** —— 合成 30 帧匀速目标（cycle_s=0.1，z 含轻微噪声用固定 seed），经 `Filter.predict/update` 循环：末态位置误差 < 0.5 m，速度误差 < 0.2 m/s（KF 与 EKF 两条参数化）
- [ ] **Step 3: 写 coast 测试** —— 中间 5 帧该 trk 无匹配：期间 `measurement_status=0` 且 trace(cov) 增长，恢复匹配后 `=1` 且 trace 回落
- [ ] **Step 4: 全量回归** —— `pytest tracker/tests/ -v` 全绿

---

## Task 7: 蒙特卡洛仿真验证（离线脚本）

**Files:**
- Create: `tools/scripts/debug/filter_mc.py`
- Output: `tracker/tests/.tmp/filter_mc/`（PNG 图 + RMSE CSV，中间产物随时可删）

**Interfaces:**
- Consumes: `Filter` facade（type=1..4）、`Cfg` / `Trk` / `Obj` / `Matches`

脚本四块，块间低耦合：

1. **场景库 `scenarios()`** —— CV/CTRV 真值轨迹（输出 `(T, N, 4)` 即帧×目标×[x,y,vx,vy]，统一 300 帧 @10 Hz）：
   - S1 直线穿越：5 目标匀速直线、轨迹交叉、速度方向各异（KF 主场）
   - S2 匀速转弯：不同 ω 的圆弧运动（EKF 主场）
   - S3 直行+转弯拼接：单目标分段运动（IMM 主场）
2. **量测生成 `gen_measurements()`** —— `z = H·x_true + v, v ~ N(0, R_sim)`；噪声档 σ_pos ∈ {0.1, 0.3, 0.5, 1.0} m（σ_v 独立可配）；**oracle 关联**（目标 i 量测直接配航迹 i，matcher KM 未实装，且滤波验证须隔离关联误差）；每 MC run 独立 seed
3. **MC 引擎 `run_mc()`** —— 100 runs × 4 滤波器 × 3 场景 × 4 噪声档；航迹初始状态 = 首帧量测，P0 取大对角阵；逐帧逐目标收集估计误差
4. **统计绘图 `report()`** —— matplotlib：① 位置/速度 RMSE 时序曲线（每滤波器一条，MC 平均）② 稳态 RMSE–噪声 σ 曲线（burn-in 后窗口）③ 场景×滤波器平均 RMSE 柱状图；附 RMSE CSV

**验收判据（正确性锚点）:**
- S1 场景 KF 稳态位置 RMSE 与 Riccati 迭代理论解（脚本内数值迭代 `P = FPF' + Q − FPH'(HPH'+R)⁻¹HPF'` 至收敛，取对角开方）偏差 < 10%
- S1 场景 EKF 与 KF 的 RMSE 时序曲线全程差 < 5%（直行退化一致性）
- 所有滤波器稳态 RMSE 随 σ 单调递增
- S3 场景 IMM 平均 RMSE ≤ 1.1 × 最优单模型
- α-β 在 S1 的 RMSE ≤ 1.5 × KF（cfg 默认 α/β 参数有效性）

- [ ] **Step 1: 场景库 + 量测生成** —— 三个场景函数 + 噪声注入，真值/量测可单独可视化自检
- [ ] **Step 2: MC 引擎** —— 跑通单场景单噪声档，落地误差数组
- [ ] **Step 3: Riccati 理论基准** —— 独立函数算 KF 理论下界并打印对比
- [ ] **Step 4: 全量扫描 + 绘图** —— 100 runs 全组合，出图出 CSV，逐条核对验收判据

---

## 备注：已知外围依赖

- matcher `_post_result` 尚未实装，须按本计划 `matched: list[tuple[Trk, Obj]]` 契约执行（matcher 自有任务）
- dim=4 量测用到 `Obj.vx/vy`，需 detector 填；未填场景（默认 0）由 R 吸收，测试以合成数据为准
- α-β 不更新 cov，与 `gap_type=2`（马氏）组合时门限用的是历史 cov，配置层面知悉即可

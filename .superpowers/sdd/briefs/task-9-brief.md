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

# RadarPillar + Tracker

## 项目概述

- **检测模型**：`RadarPillar` 深度学习目标检测
- **跟踪链路**：`tracker/` 下的跟踪仿真项目，基于检测结果实现完整跟踪流程

## 项目目标

以 `RadarPillar` 为检测 backbone，后续搭建 tracker 全链路，仿真验证将深度学习雷达检测引入跟踪系统的性能与可行性。

## 架构原则

- **低耦合、高内聚**：模块间通过清晰接口交互，独立演进
- **模块化编程**：每个模块（loader、preprocessor、detector、matcher、filter、manager、evaluator）职责单一，可独立测试
- **分层设计**：数据层 → 处理层 → 评估层，层层解耦

## 沟通规范

- 专业名词保持英文，其余使用中文
- 计划与设计文档使用顶层抽象描述，避免暴露实现细节
- 言简意赅，直接回应核心问题
- 回答问题

## 计划规范
 
+ **计划完备性**：计划需详细到可一次执行到底，避免中途因信息缺失而中断
+ **上下文管理**：上下文达 80% 时，将任务进度与关键信息写入临时文件后执行 `/compact` 删减上下文
+ **临时文件规范**：统一存放于 `.tmp/`，单任务单文件，命名含任务标识；保持简洁但不丢失任务关键信息，禁止堆叠多任务进度
+ **收尾清理**：任务完成后删除所有临时文件与临时测试文件，保持工作区干净

## 训练 & 测试脚本

+ 训练还是测试任务都需要仿照/tools/scripts 中，新建对应的训练/测试脚本进行任务的完成

## 可视化规范

BEV 检测可视化（`tools/utils/visual_utils/`，MSR 参照 `visualize_msr.py`）统一遵守：

- **布局**：三面板 `[Camera | BEV+GT | BEV+pred]`，GT 与 pred 分面板展示；两 BEV 面板同 range + 共享 colorbar，便于逐框对比
- **BEV 朝向**：x 前（屏幕上）/ y 左（屏幕左），车规惯例
- **点着色**：doppler_gnd 对地多普勒，蓝↔灰↔红 diverging 对称色标
- **类色与点云分离**：类色避开点云蓝红色域——Car=黄、Pedestrian=品红、Cyclist=草绿、Truck=青绿
- **GT 框**：细实线（linewidth 1.2）+ 无填充
- **pred 框**：粗虚线（linewidth 2.8）+ 半透明类色填充（α≈0.25），边线保持实色
- **标注**：框上不写类名/分数，类别统一看 legend（全类固定槽位色，两面板各一份）；面板 title 标各自目标数，如 `GT (11)` / `Pred (16)`；title 简洁
- **绘制入口**：GT/pred 统一走 `viz_common.draw_box_bev`（Polygon 角点法 + 朝向短线，`swap_xy` 支持车规朝向），勿散写 Rectangle
- **选帧**：`pick_frames` 分段覆盖全程 + 类多样性优先 + GT 签名去重；非 testing split 强制排除测试集帧（防信息泄露）

## 开发规范

- 用户未明确要求 commit，则不提交
- 新增函数需与用户沟通确认

## 性能规范

- **循环内禁止碎 GPU 调用**：for 循环体内不做 `torch.tensor(..., device='cuda')`、`.cpu()`/`.item()`、单元素小张量操作；攒成 batch 一次搬运，KB 级小数据（如 BEV 小图）直接留 CPU 用 numpy
- **性能问题先测量再动手**：GPU util 低 + 主进程单核 100% + worker 0% 为碎调用指纹，用 py-spy record 定位热点；禁止凭猜测改参数验证（先例：`radarnext_center_head.py` get_targets_single 逐目标循环占 CPU 70%，CenterHead 系训练慢 2.4×）
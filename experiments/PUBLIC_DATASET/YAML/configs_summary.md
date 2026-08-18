# experiments/YAML 配置统计表

mAP = VoD val R40 moderate 三类（Car/Ped/Cyclist）均值；来源为落盘 eval 结果，未重跑。

| 名字 | VFE | 3D backbone | 2D backbone | neck | head | NMS | 2D map (BEV) | 3D map (3D) |
|---|---|---|---|---|---|---|---|---|
| a0 | PillarVFE | 无 | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 52.99 | 45.43 |
| a1 | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 57.78 | 48.71 |
| a2 | PillarVFE | SEBlock | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 51.73 | 44.55 |
| a3 | PillarVFE | SEDWConv | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 53.02 | 45.16 |
| a4 | PillarVFE | PillarAttentionRes | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无¹ | 无¹ |
| a4_lnpost | PillarVFE | PillarAttentionResA | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无² | 无² |
| a4_rezero | PillarVFE | PillarAttentionResB | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无² | 无² |
| b1 | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无³ | 44.85 |
| b2 | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无³ | 40.92 |
| b3 | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无³ | 43.14 |
| b4 | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无³ | 44.60 |
| b5 | PillarVFE | PillarAttention | RepDWCNoneBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无³ | 43.46⁴ |
| b6 | PillarVFE | PillarAttention | RepDWCNoneBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无³ | 43.24⁴ |
| b7 | PillarVFE | PillarAttention | RepDWCNoneBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无³ | 43.62⁴ |
| b8 | PillarVFE | PillarAttention | RepDWCNoneBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无³ | 46.47 |
| b9 | PillarVFE | PillarAttention | RepDWCNoneBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无³ | 38.10⁴ |
| e1 | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无² | 无² |
| e2 | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无² | 无² |
| e3 | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无² | 无² |
| f1 | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无² | 无² |
| f3 | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无² | 无² |
| head_2d | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | RadarNeXtCenterHead2D | nms_gpu(0.1) | 无² | 无² |
| head_anchor | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无² | 无² |
| head_b1_2d | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | RadarNeXtCenterHead2DNoZ | nms_gpu(0.1) | 48.14⁵ | 0⁶ |
| head_b1_2d_lossup | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | RadarNeXtCenterHead2DLossUp | nms_gpu(0.1) | 51.84 | 0⁶ |
| head_b1_abl_h | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | RadarNeXtCenterHeadAblH | nms_gpu(0.1) | 50.52 | 3.89⁷ |
| head_b1_abl_z | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | RadarNeXtCenterHeadAblZ | nms_gpu(0.1) | 49.71 | 0⁶ |
| head_b1_center | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | RadarNeXtCenterHead | nms_gpu(0.1) | 51.86⁵ | 44.18⁵ |
| head_b8_2d | PillarVFE | PillarAttention | RepDWCNoneBackbone | 无 | RadarNeXtCenterHead2DNoZ | nms_gpu(0.1) | 50.23 | 0⁶ |
| head_b8_center | PillarVFE | PillarAttention | RepDWCNoneBackbone | 无 | RadarNeXtCenterHead | nms_gpu(0.1) | 51.41 | 44.26 |
| head_center | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | RadarNeXtCenterHead | nms_gpu(0.1) | 无² | 无² |
| head_n3_2d | PillarVFE | PillarAttention | PPMDFENBackbone | 无 | RadarNeXtCenterHead2DNoZ | nms_gpu(0.1) | 48.81 | 0⁶ |
| head_n3_center | PillarVFE | PillarAttention | PPMDFENBackbone | 无 | RadarNeXtCenterHead | nms_gpu(0.1) | 50.87 | 42.56 |
| head_n6_2d | PillarVFE | PillarAttention | RadarNeXtMDFENBackbone | 无 | RadarNeXtCenterHead2DNoZ | nms_gpu(0.1) | 49.11 | 0⁶ |
| head_n6_center | PillarVFE | PillarAttention | RadarNeXtMDFENBackbone | 无 | RadarNeXtCenterHead | nms_gpu(0.1) | 46.79 | 39.95 |
| head_n7_2d | PillarVFE | PillarAttention | RadarNeXtMDFENBackbone | 无 | RadarNeXtCenterHead2DNoZ | nms_gpu(0.1) | 无⁸ | 无⁸ |
| n1 | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无³ | 45.11 |
| n2 | PillarVFE | PillarAttention | PPFPNBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无³ | 41.99 |
| n3 | PillarVFE | PillarAttention | PPMDFENBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无³ | 41.35 |
| n4 | PillarVFE | PillarAttention | RepDWCNoneBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无³ | 45.66 |
| n5 | PillarVFE | PillarAttention | RadarNeXtFPNBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无³ | 43.08 |
| n6 | PillarVFE | PillarAttention | RadarNeXtMDFENBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无³ | 42.79 |
| n7 | PillarVFE | PillarAttention | RadarNeXtMDFENBackbone | 无 | AnchorHeadSingle | nms_gpu(0.1) | 无³ | 42.68 |
| n8 | PillarVFE | PillarAttention | RadarNeXtMDFENBackbone | 无 | RadarNeXtCenterHead | nms_gpu(0.2) | 无² | 无² |
| rp_mask_a0 | PillarVFE | 无 | BaseBEVBackbone | 无 | RadarPillarAnchorHeadSingle | nms_gpu(0.1) | 34.34 | 31.02 |
| rp_mask_center | PillarVFE | PillarAttention | BaseBEVBackbone | 无 | RadarPillarCenterHead | nms_gpu(0.1) | 40.12 | 35.85 |

¹ a4 训练失败（残差 + RepDWC reparam 路径致 Car 退化，见 stage2 报告）。
² 未独立训练：e/f 系列为特征组合 cfg 探索（结论吸收进 stage2 固定项 E2/F3）；head_2d/head_anchor/head_center 为 cfg 修复验证件；a4_lnpost/a4_rezero 仅产出对症资产；n8 仅建脚本。
³ anchor-head 系列仅记录 3D R40，BEV 口径未落盘。
⁴ stage2 csv 旧值受 eval 3D-IoU bug 污染（低估 ~8×），此处为 2026-07-31 修复后重评值（repdwc_concat_fair_comparison.md）。
⁵ 0814 复跑（z/h 消融同批，pickbest=多 ckpt）：head_b1_center 42.94/52.59，head_b1_2d BEV 51.87。
⁶ 2D head 无 z 通道，3D 恒 0 为设计，BEV 为其有效指标。
⁷ abl_h 有 h 无 z（z 按类查表），3D 近似失效。
⁸ ckpt 就绪但 eval 未跑（results.json 空）。

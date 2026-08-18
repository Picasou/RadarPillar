# 基于 MC-A1 DATA 的 RadarPillar 自适应

## 数据集
### 数据集结构
    MC_Single_Radar简称**MSR**：  
        |-IMAGES - 存图   （xxxxxxxx.png/img ）  
        |-LABELS - 存真值 （xxxxxxxx.bin + 对应bin的结构体.json）  
        |-POINTS - 存点云 （xxxxxxxx.bin + 对应bin的结构体.json）  
        |-PARAMS - 存参数 （1. 雷达静态参数 + 对应bin的结构体.json; 2.摄像头静态、标定参数; 3.每一帧的动态参数 xxxxxxxx.bin + 对应bin的结构体.json）  
        |-IMAGESETS 存划分结果（training/val/testing .txt）  
### 数据结构
#### 点云
- [range, doppler, azi, elv, rcs, snr, doppler_anti_amb_confi, exist_confi, frame, beam, extra_cnt]  

#### GT
- [x,y,z,length,width,height,heading,vx,vy,type,ispassble]  

**P.S.**  
- 2D目标 ： [x,y,length,width,heading,vx,vy,type,ispassble]  
- 3D目标 ： [x,y,z,length,width,height,heading,vx,vy,type,ispassble]  

## Pillar 默认范围
 - x [0:1:100]  
 - y [-20:0.5:20]  
 - z [-10:20:10]  

## Loss
  
L = w1×L(location) + w2×L(size) + w3×L(heading) + w4×L(type) + w5×L(vel) + w6×L(ispassable)  

其中：  
- **L(location)**：中心位置回归（x/y 亚像素偏移 + z 高度），L1 回归
  $$L_{loc} = \tfrac{1}{N}\sum\big|\widehat{(dx,dy)} - (g_x,g_y)\big| + \big|\hat z - g_z\big|$$

- **L(size)**：尺寸回归（length/width/height），L1 回归 + IoU/dIoU 辅助监督框重叠
  $$L_{size} = \tfrac{1}{N}\sum\big|\hat d - \log(g_{lwh})\big| + w_{IoU}\sum\big|IoU_p - IoU_r\big| + w_{dIoU}\sum(1-dIoU)$$
  其中  
  $IoU_r$ 为预测框与真值框的 aligned 3D IoU；  
  $IoU_p$ 为网络预测的 IoU 分数； 
  $dIoU = IoU_{3D} - \dfrac{\rho^{2}(b,b^{gt})}{d^{2}}$（$\rho$=中心欧氏距，$d$=外接对角线）

- **L(heading)**：朝向回归（sin/cos 双通道消周期性），L1 回归
  $$L_{head} = \tfrac{1}{N}\sum\big|\widehat{(s,c)} - (\sin\theta,\cos\theta)\big|$$


- **L(type)**：类别分类，BEV heatmap 中心点 Focal Loss
  $$L_{type} = -\tfrac{1}{N}\sum_{ijc}\Big[(1-\hat y_{ijc})^{\alpha}\log\hat y_{ijc}\cdot\mathbb{1}_{y=c} + (1-y_{ijc})^{\beta}\hat y_{ijc}^{\alpha}\log(1-\hat y_{ijc})\cdot\mathbb{1}_{y\neq c}\Big]$$
  Focal 默认 $\alpha{=}2,\beta{=}4$（CornerNet 口径）

- **L(vel)**：速度回归（vx/vy，仅 head 含 vel 通道时启用），L1 回归

- **L(ispassable)**：可通行性二分类，MC-A1 独有字段，无论文依据，建议 BCE 独立头

## 输出
- 2D目标 ： [x,y,length,width,heading,vx,vy,type,ispassble]  
- 3D目标 ： [x,y,z,length,width,height,heading,vx,vy,type,ispassble]  

## 网络内部结构

### **DataPre**
**input  :** [N_p ,N_f] + Dym_Para + pillar_cfg  
**output :** [N_f ,H ,W]  
**keyWord :** 特征工程提取特征后pillar化  
#### 特征工程
- feature_store : [range, azi, elv, rcs, dop_x, dop_y, x, y, z, dop_x_gnd, dop_y_gnd]  
- 原始特征：range, azi, elv, rcs， doppler
- 特征工程：x, y, z, dop_x, dop_y, dop_x_gnd, dop_y_gnd
- 输出 ：PointList [N_p, N] -- N_p：点云个数，N：特征个数（其中N_p候选）
#### pillar化
- 通过pillar_cfg + PointList
- pillar最多点云数 + 统计
- 输出：[N_f ,H ,W] 

### **VFE**
**input  :** [N_v, M, N_f]（N_v: pillar，M:单 pillar 最大点数）  
**output :** [N_v, N]  
**keyWord :** pointnNet风格抽象block[N,N,N], 逐个pillar提取freature map    
**candidate:** N候选32,64,128  
**pipeline :**   

### **Attention**
**input  :** [N_f ,H ,W]  
**output :** [N_Fatt ,H ,W]  
**keyWord :** gatter非空pillar, 使用pillarattion(self-attention) ,scatter成bevmap  [N_Fatt ,H ,W]  
**candidate:** Head数候选1,2  
**pipeline :**     


### **Backbone**
**input  :** [N_Fatt ,H ,W]   
**output :** 多尺度 [N_Fbev, {H, H/2, H/4}]（下采样金字塔）  
**keyWord :** 重复 block 提取多尺度 BEV 特征金字塔  
**candidate:** 
- block : [pointpiller 风格块 ； repDwc 风格块]
- N_Fbev(f_dim) : [[32,32,32],[64,64,64],[32,64,128]]
- pyramid : [stride[1,2,2] → {H,H/2,H/4}（论文口径，voxel_x=1 自洽）；stride[2,2,2] → {H/2,H/4,H/8}（OpenPCDet 默认，需 voxel 可整除 8）]

**pipeline :**  

### **Neck**
**input  :** 多尺度 [N_Fbev, {H, H/2, H/4}]（Backbone 金字塔）  
**output :** [N_Fneck ,H ,W]（融合后单尺度 → Head）  
**keyWord :** 多尺度金字塔融合成单张 BEV 图  
**candidate:**  
- block : [直接concat, FPN, MDFEN]
- N_Fneck : concat→压缩定值 ; FPN/MDFEN→各级加总

**pipeline :**

### **Head**
**input  :** [N_Fneck ,H ,W]   
**output :** [N_cand × (location, size, vel, type, ispassable)]  
**keyWord :** 从 BEV feature map 解码目标（location/size/vel/type/ispassable）  
**candidate:**   
- AnchorHeadSingle (anchor 回归) ；
- RadarNeXtCenterHead (center heatmap)

**pipeline :**   

### **NMS**
**input  :** Head 候选 [N_cand × (score, location, size, vel, type, ispassable)]  
**output :** 最终目标 [N_det × (location, size, vel, type, ispassable)]  
**keyWord :** 后处理去重--推理期 pillar mask 只放行非空 pillar 位置候选，省 NMS  
**candidate:** 
- mask 触发点 : [anchor head→非空位置 anchor cls 置 -1e9 ； center head→非空位置 heatmap logit 压 -1e9]  
- NMS_PRE_MAXSIZE : [4096(dense) → 1024(mask)]   

**pipeline :** 

# MSR Dataset 设计 spec

## 目标

为新数据集 MSR(MC_Single_Radar)实现:数据结构 load、dataset 注册、info pkl 生成脚本,并适配 RadarPillar 既有工程约定(仿 VoD)。

## 数据集现状(已实测)

- 路径:`/mnt/d/DataSet/11111111111`(WSL),目录结构:
  - `IMAGES/` - 图 `.png`
  - `LABELS/` - GT `.bin`(28B/框)+ `struct.json`(schema)
  - `POINTS/` - 点云 `.bin`(24B/点)+ `struct.json`
  - `PARAMS/` - 动态参数 `.bin`(84B/帧)+ `radar_dynamic.struct.json`;静态参数 `radar_static.struct.json`
  - `IMAGESETS/` - `training.txt` / `val.txt` / `testing.txt`(每行 8 位 index `00000000`)
- 597 帧,4285 框,type 取值仅 {1,4,5}(轿车/二轮车/卡车),ghost 全 0
- 数据为**自车静止**场景:hostVelocity=0 是真实值(非占位);yawrate 在 offset[32-35] 有值
- GT 坐标 BEV 平面:z 全 0,长宽高有真实值

## 架构分层(对齐 VoD 既有约定)

```
读 bin → [reader 加工] → 选列 → 归一化(可选) → encoder(纯选列) → processor(voxelize)
         get_radar 内
```

- 加工(极坐标→xyz、速度分解、gnd 补偿)落在 **reader `get_radar` 内**(与 VoD 的 get_lidar 同层)
- PointFeatureEncoder 不改,只按 used_feature_list 选列
- 基类 dataset.py、processor、augmentor 不动

## 动态 schema 加载(核心机制,不写死 dtype)

1. 读 `<目录>/struct.json` 的 `fields`,按 `name + type + type_mapping` 动态构造 `np.dtype`
2. 拿 `structures[0].total_size` 与 fields 累加 size 做差,差值 > 0 → 末尾补 `('_pad', u1×差值)` 对齐磁盘(当前 LABELS 27→28 补 1B;POINTS 24=24 不补)
3. 按 `fields[].output_name` 作为特征列名契约取值;json 缺某 output_name → `warning + 该列补 0`(数据降级不报错)
4. schema 变更(增删字段)时代码自动适配

### scale 字段可信度(实测发现 json 多处不准)

| 字段 | json scale | 实测判定 | 处理 |
|---|---|---|---|
| POINTS.range | 0.01 | 正确 | 用 json |
| POINTS.azi/elv | 0.0001745 | 正确 | 用 json |
| POINTS.doppler | 0.01 | 正确 | 用 json |
| LABELS.x/y/z/w/l/h | 1.0(注释说×0.01) | **实际 ×0.01**(raw 几千→米) | 用 json 的 scale=1.0 会错 |
| LABELS.heading | 1.0(注释说×0.01) | **实际 ×0.01**(raw 2435 当度>360 不合法,×0.01=24° 合法) | 用 json 的 scale=1.0 会错 |

规则:LABELS 的物理量字段(cm/deg)一律强制 `×0.01`(不信任 json 的 `scale:1.0`);POINTS 的 scale 字段可信,用 json。转换时统一:LABELS 几何量乘 0.01,heading 再 ×π/180 转弧度。此规则在代码注释里标注「json scale 不准,实测强制 0.01」。

## 文件清单

| 新增 | 职责 |
|---|---|
| `pcdet/datasets/msr/msr_dataset.py` | MsrDataset 类 + create_msr_infos + CLI 入口 |
| `pcdet/datasets/msr/__init__.py` | 导出 MsrDataset |
| `tools/cfgs/dataset/msr_dataset.yaml` | 数据集配置 |
| `tools/scripts/data/create_msr_data.py` | pkl 生成 wrapper |
| `tools/scripts/data/check_msr.py` | 验证脚本 |

注册:`pcdet/datasets/__init__.py` 加 `from .msr.msr_dataset import MsrDataset` + `__all__['MsrDataset']`。

## reader 函数

- `get_radar(idx)`:读 `POINTS/{idx}.bin` + `struct.json` 动态解包 → 产出 18 列 src_feature_list
- `get_label(idx)`:读 `LABELS/{idx}.bin` + `struct.json` 动态解包 → gt_boxes + gt_names
- `get_dynamic_param(idx)`:读 `PARAMS/{idx}.bin` + `radar_dynamic.struct.json` → ego_speed/yawrate

## 特征工程(get_radar 内,产出 18 列 src_feature_list)

原始直传列名**以 json 的 output_name 为准**(range_m/doppler_mps/ang_rad/elv_rad/exist_confidence);加工列(x/y/z/dop_*)自定名。

| 步骤 | 特征 |
|---|---|
| 原始直传(11) | range_m, doppler_mps, ang_rad, elv_rad, rcs, snr, doppler_anti_amb_confi, exist_confidence, frame, beam, extra_cnt |
| 极坐标→笛卡尔(3) | x, y, z |
| 速度分解-雷达系(2) | dop_x, dop_y |
| 速度分解-地面系(2) | dop_x_gnd, dop_y_gnd |

dtype 字段名也用 output_name(契约名),保证 build/parse 的 col() 按 output_name 取数与 dtype 字段一致。

- x/y/z:`x=range·cos(azi)cos(elv)` 等(azi/elv 已是弧度,scale 在 json 里)
- dop_x/dop_y:径向 doppler 投影到 x/y(`doppler·cos(azi)`, `doppler·sin(azi)`)
- dop_x_gnd/dop_y_gnd:用 get_dynamic_param 的 ego_speed/yawrate 做自车运动补偿;ego_speed=0 时等于 dop_x/dop_y
- `id`/`flags` 是元数据,不进特征列

## GT 处理

- `get_label`:解 28B → `gt_boxes[x,y,z,l,w,h,heading]`(单位 cm→m,heading deg→rad,heading 取值见 schema scale)+ `gt_names`(type 直接 `str()`)
- CLASS_NAMES = `['1','4','5']`(底层数字传递);mapping 表 `{1:轿车,2:行人,4:二轮车,5:卡车,7:静态障碍物}` 写代码注释备查
- ghost 不过滤;无 calib/camera 依赖(纯雷达 BEV),不调用 KITTI calib 链

## pkl 生成

`create_msr_infos` 照 VoD 流程:
- `set_split` → `get_infos`(多线程装 point/image/annos/param 四块)→ dump `msr_infos_{train,val,test}.pkl` + `msr_infos_trainval.pkl`
- `create_groundtruth_database` 产 `msr_dbinfos_train.pkl` + `gt_database/`
- 简化点(相对 VoD):无 calib_info(填 None/占位)、无 FOV 裁剪、annos 字段精简(无 KITTI 相机系字段)

## yaml 关键配置

- `DATASET: 'MsrDataset'`
- `DATA_PATH`
- `POINT_CLOUD_RANGE`:z 默认 `[-10, 10]`(可配)
- `DATA_SPLIT`、`INFO_PATH`(指向 msr_infos_*.pkl)
- `POINT_FEATURE_ENCODING`:`src_feature_list`(18 列)/ `used_feature_list`(配置选)
- `POINT_FEATURE_NORMALIZATION`:默认关(USE_NORM: False)
- `USE_GND_VELOCITY`:是否读取动态参数算 gnd 特征(默认 True)

## 验证脚本

`tools/scripts/data/check_msr.py`:读 1 帧,打印 points/label 解析 shape + 字段 + 动态 dtype 对齐情况 + BEV 散点图(matplotlib),确认结构体对齐无误。

## 范围外(YAGNI)

- 不做 eval(评测接口,后续按需)
- 不做相机/标定链路
- 不做 VDC 多帧补偿(MSR 单帧,无 time 多帧语义)
- 不改 encoder/processor/基类

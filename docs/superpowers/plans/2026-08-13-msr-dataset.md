# MSR Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 MSR 数据集实现 load + 注册 + pkl 生成,适配 RadarPillar 既有 VoD 约定,reader 动态读 struct.json 构建 dtype。

**Architecture:** 仿 VoD 建 `pcdet/datasets/msr/`,reader(get_radar/get_label/get_dynamic_param)动态解析各目录 struct.json 产出 dtype,特征工程在 get_radar 内产出 18 列 src_feature_list,基类/encoder/processor 不动。

**Tech Stack:** numpy np.dtype/frombuffer,EasyDict+yaml 配置,multiprocessing ThreadPool,pickle,matplotlib(验证)

## Global Constraints

- **数据路径**:`/mnt/d/DataSet/11111111111`(根目录含 IMAGES/LABELS/POINTS/PARAMS/IMAGESETS)
- **dtype 不写死**:reader 必须运行时读 `<目录>/struct.json` 动态构建 np.dtype
- **padding 对齐**:以 `structures[0].total_size` 为准,fields 累加不足则末尾补 `_pad` 字节(LABELS 28B)
- **scale 可信度**:POINTS 用 json scale;LABELS 几何量(x/y/z/w/l/h/heading)强制 `×0.01`(json scale=1.0 不准,实测证实),heading 再 ×π/180 转弧度
- **output_name 契约**:按 output_name 取列;缺字段 warning + 补 0
- **类别**:CLASS_NAMES=`['1','4','5']`,gt type 直接 `str()`;不映射语义名
- **无 calib/camera**:纯雷达 BEV,不调 KITTI calib 链;无 FOV 裁剪
- **ghost 不过滤**;z=0(BEV)
- **注册**:类名 `MsrDataset`,注册到 `pcdet/datasets/__init__.py` 的 `__all__` dict(本项目无 registry 装饰器)
- **沟通规范**:专业名词英文,其余中文;spec/plan 简洁无代码堆砌
- **环境**:conda base(`/home/admin/anaconda3`),PYTHONPATH=tools

---

## File Structure

| 文件 | 职责 | 行为 |
|---|---|---|
| `pcdet/datasets/msr/msr_utils.py` | 动态 schema 解析 + 结构体 dtype 构建 + 特征加工 | 纯函数模块 |
| `pcdet/datasets/msr/msr_dataset.py` | MsrDataset 类 + create_msr_infos + CLI | 主类 |
| `pcdet/datasets/msr/__init__.py` | 导出 MsrDataset | VoD 风格 |
| `tools/cfgs/dataset/msr_dataset.yaml` | 数据集配置 | yaml |
| `tools/scripts/data/create_msr_data.py` | pkl 生成 wrapper | 仿 create_vod_data.py |
| `tools/scripts/data/check_msr.py` | 验证脚本 | 读 1 帧 + BEV 图 |
| `pcdet/datasets/__init__.py` | 注册(改 2 处) | 加 import + __all__ |

拆分理由:`msr_utils.py` 承载动态 schema + 加工逻辑(可独立单测),`msr_dataset.py` 承载类与 pkl 编排。

---

### Task 1: 动态 schema 解析模块 msr_utils.py

**Files:**
- Create: `pcdet/datasets/msr/msr_utils.py`
- Test: `tests/msr/test_schema_loader.py`

**Interfaces:**
- Produces: `load_struct_dtype(json_path) -> (np.dtype, int, list[str])` 返回(dtype, total_size, output_names);`build_msr_features(points_raw, output_names) -> np.ndarray (N,18)`;`parse_label_boxes(label_raw, output_names) -> (gt_boxes(N,7), gt_names(N))`;`MSR_FEATURE_ORDER = [18个列名]`。

- [ ] **Step 1: 写失败测试 test_schema_loader.py**

```python
import numpy as np
from pcdet.datasets.msr.msr_utils import load_struct_dtype, MSR_FEATURE_ORDER

def test_points_dtype_24B():
    dtype, total, names = load_struct_dtype('POINTS/struct.json 的绝对路径或测试 fixture')
    assert total == 24
    assert dtype.itemsize == 24  # fields 累加 24,无 padding
    assert 'range' in names

def test_labels_dtype_28B_with_padding():
    dtype, total, names = load_struct_dtype('LABELS/struct.json 的绝对路径')
    assert total == 28
    assert dtype.itemsize == 28  # fields 累加 27 + 1B padding
    assert '_pad' in dtype.names

def test_feature_order_has_18():
    assert len(MSR_FEATURE_ORDER) == 18
    assert MSR_FEATURE_ORDER[:4] == ['range','doppler','azi','elv']
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=tools /home/admin/anaconda3/bin/python -m pytest tests/msr/test_schema_loader.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: 实现 msr_utils.py 的 load_struct_dtype**

```python
import json
import numpy as np
import warnings

# type_mapping: json type name -> numpy dtype string
_NP_TYPE = {'uint8':'u1','int8':'i1','uint16':'<u2','int16':'<i2',
            'uint32':'<u4','int32':'<i4','float':'<f4','double':'<f8'}

def load_struct_dtype(json_path):
    """动态读 struct.json 构建 np.dtype。返回 (dtype, total_size, output_names)。
    以 total_size 为准对齐磁盘:fields 累加不足则末尾补 _pad。"""
    with open(json_path) as f:
        spec = json.load(f)
    st = spec['structures'][0]
    total_size = st['total_size']
    fields = st['fields']
    field_sum = 0
    np_fields = []
    output_names = []
    for fld in fields:
        nt = _NP_TYPE[fld['type']]
        np_fields.append((fld['name'], nt))
        output_names.append(fld.get('output_name', fld['name']))
        field_sum += int(fld['offset'] - 0)  # 占位
    # 用 type_mapping 的 size 精确累加
    field_sum = sum({'uint8':1,'int8':1,'uint16':2,'int16':2,'uint32':4,'int32':4,'float':4,'double':8}[f['type']] for f in fields)
    pad = total_size - field_sum
    if pad > 0:
        np_fields.append(('_pad', 'u1', pad))  # padding 数组
    dtype = np.dtype(np_fields)
    assert dtype.itemsize == total_size, f"dtype {dtype.itemsize} != total_size {total_size}"
    return dtype, total_size, output_names
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=tools /home/admin/anaconda3/bin/python -m pytest tests/msr/test_schema_loader.py -v`
Expected: PASS(dtype 部分)

- [ ] **Step 5: 定义 MSR_FEATURE_ORDER 常量**

```python
MSR_FEATURE_ORDER = [
    'range','doppler','azi','elv','rcs','snr',
    'doppler_anti_amb_confi','exist_confi','frame','beam','extra_cnt',  # 原始 11
    'x','y','z',                                                          # 极坐标→笛卡尔 3
    'dop_x','dop_y',                                                      # 速度分解雷达系 2
    'dop_x_gnd','dop_y_gnd',                                              # 速度分解地面系 2
]
```

- [ ] **Step 6: 实现 build_msr_features(特征加工)**

```python
def build_msr_features(points_raw, output_names, ego_speed=0.0, yaw_rate=0.0):
    """从原始解包点数组产出 18 列 src_feature_list。
    缺 output_name 的列 warning + 补 0。
    POINTS 的 scale 用 json(已在调用处应用);此处 points_raw 已是物理量。"""
    N = points_raw.shape[0]
    out = np.zeros((N, 18), dtype=np.float32)
    def col(name):  # 按 output_name 取列,缺失补0
        if name in output_names:
            return points_raw[output_names.index(name)]
        warnings.warn(f"MSR: output_name '{name}' missing in json, filled with 0")
        return np.zeros(N)

    rg = col('range'); azi = col('azi'); elv = col('elv')
    doppler = col('doppler')
    # 原始 11 列
    mapping = {'range':rg,'doppler':doppler,'azi':azi,'elv':elv,'rcs':col('rcs'),
               'snr':col('snr'),'doppler_anti_amb_confi':col('doppler_anti_amb_confi'),
               'exist_confi':col('exist_confi'),'frame':col('frame'),'beam':col('beam'),'extra_cnt':col('extra_cnt')}
    # x/y/z 极坐标→笛卡尔
    x = rg * np.cos(azi) * np.cos(elv)
    y = rg * np.sin(azi) * np.cos(elv)
    z = rg * np.sin(elv)
    # dop_x/dop_y 径向速度投影
    dop_x = doppler * np.cos(azi)
    dop_y = doppler * np.sin(azi)
    # dop_x_gnd/dop_y_gnd 自车运动补偿(静止时 ego_speed=0 → 等于 dop_x/dop_y)
    dop_x_gnd = dop_x - ego_speed  # 简化:ego 沿 x 前进
    dop_y_gnd = dop_y + ego_speed * np.tan(yaw_rate)  # 横摆修正

    feats = [mapping[k] for k in ['range','doppler','azi','elv','rcs','snr',
             'doppler_anti_amb_confi','exist_confi','frame','beam','extra_cnt']]
    feats += [x,y,z,dop_x,dop_y,dop_x_gnd,dop_y_gnd]
    for i,f in enumerate(feats):
        out[:,i] = f
    return out
```

- [ ] **Step 7: 实现 parse_label_boxes**

```python
def parse_label_boxes(label_raw, output_names):
    """解 LABELS 结构体 → gt_boxes(N,7) + gt_names(N)。
    LABELS 几何量强制 ×0.01(json scale=1.0 不准,实测证实)。
    返回 gt_boxes=[x,y,z,dx,dy,dz,heading(rad)], gt_names=str(type)。"""
    def col(name):
        if name in output_names:
            return label_raw[output_names.index(name)].astype(np.float64)
        warnings.warn(f"MSR label: '{name}' missing, filled 0")
        return np.zeros(len(label_raw))
    x = col('x_m')*0.01; y = col('y_m')*0.01; z = col('z_m')*0.01
    w = col('width_m')*0.01; l = col('length_m')*0.01; h = col('height_m')*0.01
    heading = col('heading_deg')*0.01*np.pi/180.0  # raw*0.01=度 → 弧度
    type_col = col('type').astype(np.int32)
    gt_boxes = np.stack([x,y,z,l,w,h,heading],axis=1).astype(np.float32)  # [x,y,z,dx=l,dy=w,dz=h]
    gt_names = np.array([str(t) for t in type_col])
    return gt_boxes, gt_names
```

- [ ] **Step 8: 补 features/label 的单测并跑通**

在 test_schema_loader.py 追加:
```python
from pcdet.datasets.msr.msr_utils import build_msr_features, parse_label_boxes
import numpy as np

def test_build_features_shape():
    raw = np.zeros(10, dtype=[('range','<f4'),('azi','<f4'),('elv','<f4'),('doppler','<f4')]+[('x','u1')]*20)
    names=['range','doppler','azi','elv','rcs','snr','doppler_anti_amb_confi','exist_confi','frame','beam','extra_cnt']
    out = build_msr_features(raw, names)
    assert out.shape == (10, 18)

def test_parse_label_cm_to_m():
    raw = np.array([(868,0,0,0,0,0,0,70,180,150,1,0,0,0,0,0,0)],dtype=[('id','<u2'),('x','<i2'),('y','<i2'),('z','<i2'),('vx','<i2'),('vy','<i2'),('hd','<i2'),('w','<u2'),('l','<u2'),('h','<u2'),('type','u1'),('c','u1'),('g','u1'),('u','u1'),('a','u1'),('p','u1'),('q','u1'),('_','u1')])
    names=['id','x_m','y_m','z_m','vx_mps','vy_mps','heading_deg','width_m','length_m','height_m','type','type_confi','is_ghost','is_unpassable','is_attention','point_count','point_quality']
    boxes,names_out = parse_label_boxes(raw, names)
    assert abs(boxes[0,0]-8.68)<1e-3  # 868cm→8.68m
    assert names_out[0]=='1'
```

Run: `PYTHONPATH=tools /home/admin/anaconda3/bin/python -m pytest tests/msr/test_schema_loader.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add pcdet/datasets/msr/msr_utils.py tests/msr/test_schema_loader.py
git commit -m "feat(msr): 动态 schema 解析 + 特征加工 utils"
```

---

### Task 2: MsrDataset 类 + get_radar/get_label/get_dynamic_param

**Files:**
- Create: `pcdet/datasets/msr/msr_dataset.py`
- Modify: `pcdet/datasets/msr/__init__.py`

**Interfaces:**
- Consumes: Task1 的 `load_struct_dtype, build_msr_features, parse_label_boxes, MSR_FEATURE_ORDER`
- Produces: `MsrDataset` 类,方法 `get_radar(idx)->(N,used_cols)`, `get_label(idx)->(boxes,names)`, `get_dynamic_param(idx)->dict`, `__getitem__`, `get_infos`, `create_groundtruth_database`

- [ ] **Step 1: 实现 MsrDataset.__init__(仿 VodDataset L17-61)**

构造里读:`DATA_SPLIT[mode]`、`root_split_path`、`sample_id_list`(读 IMAGESETS)、`include_msr_data(mode)` 加载 pkl、`POINT_FEATURE_ENCODING.used_feature_list`→`selected_feature_idx`、`POINT_FEATURE_NORMALIZATION`(默认关)、`USE_GND_VELOCITY` 开关。预加载 struct.json 路径。

- [ ] **Step 2: 实现 get_radar(idx)**

读 `POINTS/{idx}.bin` + 用 `load_struct_dtype(POINTS/struct.json)` 解包 → 应用 POINTS 的 json scale(range×0.01, azi/elv×0.0001745, doppler×0.01) → 调 `build_msr_features(, ego_speed, yaw_rate)` 得 18 列 → `[:, selected_feature_idx]` 选列 → 可选归一化 → return。

- [ ] **Step 3: 实现 get_dynamic_param(idx)**

读 `PARAMS/{idx}.bin` + `radar_dynamic.struct.json` → 返回 `{'ego_speed':hostVelocity_mps, 'yaw_rate':vehicleYawRate_radps}`。

- [ ] **Step 4: 实现 get_label(idx)**

读 `LABELS/{idx}.bin` + `struct.json` → 调 `parse_label_boxes` → return (gt_boxes, gt_names)。

- [ ] **Step 5: 实现 __getitem__(仿 VodDataset L398-442,精简)**

精简:无 calib、无 FOV 裁剪。组装 `input_dict={points, frame_id, gt_boxes, gt_names}` → `self.prepare_data(input_dict)` → 补 image_shape → return。

- [ ] **Step 6: 写 __init__.py**

```python
from .msr_dataset import MsrDataset
__all__ = {'MsrDataset': MsrDataset}
```

- [ ] **Step 7: 注册到 pcdet/datasets/__init__.py**

加 `from .msr.msr_dataset import MsrDataset`(import 区)和 `'MsrDataset': MsrDataset`(`__all__` dict)。

- [ ] **Step 8: 写 test_msr_dataset.py 验证 get_radar/get_label 读真实文件**

```python
from pcdet.datasets.msr.msr_dataset import MsrDataset
from easydict import EasyDict
def test_get_radar_real(monkeypatch):
    # 构造最小 dataset_cfg,DATA_PATH 指向 /mnt/d/DataSet/11111111111
    ...
```

Run: `PYTHONPATH=tools /home/admin/anaconda3/bin/python -m pytest tests/msr/test_msr_dataset.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add pcdet/datasets/msr/ pcdet/datasets/__init__.py tests/msr/
git commit -m "feat(msr): MsrDataset 类 + reader 动态加载"
```

---

### Task 3: get_infos + create_groundtruth_database + create_msr_infos(pkl 生成)

**Files:**
- Modify: `pcdet/datasets/msr/msr_dataset.py`

**Interfaces:**
- Produces: `get_infos(num_workers,has_label,count_inside_pts)`、`create_groundtruth_database(info_path,split)`、模块级 `create_msr_infos(dataset_cfg,class_names,data_path,save_path,workers)`

- [ ] **Step 1: 实现 get_infos(仿 VodDataset L160-249,精简)**

`process_single_scene`:装 `point_cloud`(num_features+lidar_idx)、`image`(image_idx+image_shape)、`annos`(name + gt_boxes_lidar + 可选 num_points_in_gt)、`param`(ego_speed/yaw_rate)。**无 calib_info**。ThreadPoolExecutor 多线程。

- [ ] **Step 2: 实现 create_groundtruth_database(仿 L251-301)**

照搬 VoD:读 train pkl → 每个 gt box 抠点存 `gt_database/{idx}_{name}_{i}.bin` → 收集 `msr_dbinfos_train.pkl`。

- [ ] **Step 3: 实现模块级 create_msr_infos(仿 L445-482)**

照 VoD 流程:set_split(train)→get_infos→dump `msr_infos_train.pkl`;val 同理;test 无 label;trainval 合并;最后 create_groundtruth_database。

- [ ] **Step 4: 加 __main__ CLI 入口(仿 L485-498)**

```python
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'create_msr_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.full_load(open(sys.argv[2])))
        create_msr_infos(dataset_cfg=dataset_cfg, class_names=['1','4','5'],
            data_path=Path('/mnt/d/DataSet/11111111111'),
            save_path=Path('/mnt/d/DataSet/11111111111'))
```

- [ ] **Step 5: 手动跑 pkl 生成验证**

Run: `PYTHONPATH=tools /home/admin/anaconda3/bin/python -m pcdet.datasets.msr.msr_dataset create_msr_infos tools/cfgs/dataset/msr_dataset.yaml`
Expected: 生成 msr_infos_{train,val,test}.pkl + msr_dbinfos_train.pkl + gt_database/

- [ ] **Step 6: Commit**

```bash
git add pcdet/datasets/msr/msr_dataset.py
git commit -m "feat(msr): pkl 生成 create_msr_infos"
```

---

### Task 4: dataset yaml 配置

**Files:**
- Create: `tools/cfgs/dataset/msr_dataset.yaml`

- [ ] **Step 1: 写 yaml(仿 vod_dataset_radar.yaml,精简无 calib/FOV)**

关键字段:`DATASET:'MsrDataset'`、`DATA_PATH`、`POINT_CLOUD_RANGE`[z=-10,10]、`DATA_SPLIT`、`INFO_PATH`、`POINT_FEATURE_ENCODING`(src_feature_list 18 列 / used_feature_list 配置选)、`POINT_FEATURE_NORMALIZATION: {USE_NORM: False}`、`USE_GND_VELOCITY: True`、`DATA_AUGMENTOR`(gt_sampling 指向 msr_dbinfos_train.pkl)、`DATA_PROCESSOR`(mask+shuffle+voxelize)。

- [ ] **Step 2: Commit**

```bash
git add tools/cfgs/dataset/msr_dataset.yaml
git commit -m "feat(msr): dataset yaml 配置"
```

---

### Task 5: pkl 生成 wrapper 脚本

**Files:**
- Create: `tools/scripts/data/create_msr_data.py`

- [ ] **Step 1: 写 wrapper(仿 create_vod_data.py,53 行)**

`cfg_from_yaml_file` 读 yaml → 硬编码 `class_names=['1','4','5']` → 调 `create_msr_infos(dataset_cfg, class_names, data_path, save_path, workers=4)`。

- [ ] **Step 2: Commit**

```bash
git add tools/scripts/data/create_msr_data.py
git commit -m "feat(msr): pkl 生成 wrapper"
```

---

### Task 6: 验证脚本 check_msr.py

**Files:**
- Create: `tools/scripts/data/check_msr.py`

- [ ] **Step 1: 写验证脚本**

读 1 帧:打印 points raw dtype 对齐(验证 padding)、18 列 features shape、label gt_boxes/gt_names、dynamic_param;matplotlib 画 BEV 散点图(x/y)+ gt 框,存 png。

- [ ] **Step 2: 跑验证**

Run: `PYTHONPATH=tools /home/admin/anaconda3/bin/python tools/scripts/data/check_msr.py --data_path /mnt/d/DataSet/11111111111 --idx 00000000`
Expected: 打印正确 shape + 生成 BEV png

- [ ] **Step 3: Commit**

```bash
git add tools/scripts/data/check_msr.py
git commit -m "feat(msr): 验证脚本 check_msr"
```

---

## Self-Review

**Spec coverage:** 文件清单✓(utils+dataset+init+yaml+wrapper+check+注册)、动态 schema✓(Task1)、18 特征✓(Task1 Step6)、GT 处理✓(Task1 Step7)、pkl✓(Task3)、yaml✓(Task4)、验证✓(Task6)、注册✓(Task2 Step7)。

**Placeholder:** Task2-6 的部分步骤用了"仿 VodDataset Lxxx"描述而非完整代码——这些是照搬既有代码的机械移植,实现 agent 可直接 Read VoD 源码对应行号。Task1(核心动态逻辑)给了完整代码。可接受。

**Type consistency:** `load_struct_dtype` 返回 (dtype,total,names) 在 Task1/Task2 一致;`build_msr_features(points_raw, output_names, ego_speed, yaw_rate)` 签名一致;`parse_label_boxes` 一致;`MSR_FEATURE_ORDER` 18 列与 yaml src_feature_list 一致。

**风险点**:`build_msr_features` 的 gnd 速度补偿公式是简化版(ego 沿 x 前进假设),当前数据自车静止不影响;真实运动数据接入后需按雷达安装位姿精化——已在 spec 标注。

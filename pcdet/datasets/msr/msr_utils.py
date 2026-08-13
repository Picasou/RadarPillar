"""MSR dataset utils: 动态 schema 解析 + 结构体 dtype 构建 + 特征加工。

核心机制:
- `load_struct_dtype` 运行时读 `<目录>/struct.json` 动态构建 np.dtype,以 total_size
  为准对齐磁盘(fields 累加不足则末尾补 `_pad` 数组字段,如 LABELS 27→28 补 1B)。
- scale 规则:
    POINTS  的 scale 字段可信(range×0.01, azi/elv×0.0001745, doppler×0.01),
           由调用处(msr_dataset.get_radar)应用,本模块的 `build_msr_features`
           接收的 points_raw 已是物理量。
    LABELS  几何量(x/y/z/w/l/h/heading)json scale=1.0 不准(实测),
           一律强制 ×0.01;heading 再 ×π/180 转弧度。
"""

import json
import warnings

import numpy as np

# json type name -> (numpy dtype string, size_in_bytes)
_TYPE_INFO = {
    'uint8': ('u1', 1),
    'int8': ('i1', 1),
    'uint16': ('<u2', 2),
    'int16': ('<i2', 2),
    'uint32': ('<u4', 4),
    'int32': ('<i4', 4),
    'float': ('<f4', 4),
    'double': ('<f8', 8),
}

# 18 列 src_feature_list 顺序(get_radar 产出, PointFeatureEncoder 选列)
MSR_FEATURE_ORDER = [
    'range', 'doppler', 'azi', 'elv', 'rcs', 'snr',
    'doppler_anti_amb_confi', 'exist_confi', 'frame', 'beam', 'extra_cnt',  # 原始直传 11
    'x', 'y', 'z',                                                          # 极坐标→笛卡尔 3
    'dop_x', 'dop_y',                                                       # 速度分解-雷达系 2
    'dop_x_gnd', 'dop_y_gnd',                                               # 速度分解-地面系 2
]


def load_struct_dtype(json_path):
    """动态读 struct.json 构建 np.dtype。

    Returns:
        (dtype, total_size, output_names):
            dtype         : np.dtype,itemsize == total_size(以 total_size 为准补 padding)
            total_size    : structures[0].total_size
            output_names  : list[str],按 fields 顺序的 output_name(缺则 fallback 到 name)
    """
    with open(json_path) as f:
        spec = json.load(f)
    st = spec['structures'][0]
    total_size = st['total_size']
    fields = st['fields']

    np_fields = []
    output_names = []
    # 用 type_mapping 的 size 精确累加(不依赖 json 的 offset)
    field_sum = 0
    for fld in fields:
        np_type, size = _TYPE_INFO[fld['type']]
        np_fields.append((fld['name'], np_type))
        output_names.append(fld.get('output_name', fld['name']))
        field_sum += size

    pad = total_size - field_sum
    if pad > 0:
        # 数组字段语法:可变长度 padding 对齐磁盘
        np_fields.append(('_pad', 'u1', pad))
    elif pad < 0:
        raise ValueError(
            f"fields size {field_sum} > total_size {total_size} in {json_path}")

    dtype = np.dtype(np_fields)
    assert dtype.itemsize == total_size, \
        f"dtype {dtype.itemsize} != total_size {total_size}"
    return dtype, total_size, output_names


def build_msr_features(points_raw, output_names, ego_speed=0.0, yaw_rate=0.0):
    """从原始解包点数组产出 18 列 src_feature_list(按 MSR_FEATURE_ORDER 顺序)。

    Args:
        points_raw   : np.ndarray(结构化 dtype),已应用 POINTS scale(物理量)
        output_names : list[str],与 points_raw.dtype.names 顺序对齐(契约存在性判定)
        ego_speed    : 自车纵向速度(m/s),静止时 0
        yaw_rate     : 自车横摆角速度(rad/s)

    Returns:
        np.ndarray shape=(N,18) dtype=float32

    缺 output_name 的列 warning + 补 0。
    """
    N = points_raw.shape[0]
    out = np.zeros((N, 18), dtype=np.float32)

    def col(name):
        # 按 output_name 契约判定存在性;取数直接用字段名(output_names 仅作 in 判定)
        if name in output_names:
            return points_raw[name].astype(np.float32)
        warnings.warn(
            f"MSR: output_name '{name}' missing in json, filled with 0")
        return np.zeros(N, dtype=np.float32)

    rg = col('range'); azi = col('azi'); elv = col('elv')
    doppler = col('doppler')

    # 原始直传 11 列
    feats = [
        rg, doppler, azi, elv,
        col('rcs'), col('snr'),
        col('doppler_anti_amb_confi'), col('exist_confi'),
        col('frame'), col('beam'), col('extra_cnt'),
    ]

    # 极坐标 → 笛卡尔(azi/elv 已是弧度)
    x = rg * np.cos(azi) * np.cos(elv)
    y = rg * np.sin(azi) * np.cos(elv)
    z = rg * np.sin(elv)
    feats += [x, y, z]

    # 径向速度投影到雷达系 x/y
    dop_x = doppler * np.cos(azi)
    dop_y = doppler * np.sin(azi)
    feats += [dop_x, dop_y]

    # 地面系补偿(简化:ego 沿 x 前进;ego_speed=0 时等于 dop_x/dop_y)
    dop_x_gnd = dop_x - ego_speed
    dop_y_gnd = dop_y + ego_speed * np.tan(yaw_rate)
    feats += [dop_x_gnd, dop_y_gnd]

    assert len(feats) == 18
    for i, f in enumerate(feats):
        out[:, i] = f
    return out


def parse_label_boxes(label_raw, output_names):
    """解 LABELS 结构体 → (gt_boxes, gt_names)。

    LABELS 几何量(json scale=1.0 不准,实测)一律强制 ×0.01:cm→m。
    heading 再 ×π/180 转弧度(raw×0.01 = 度)。

    Args:
        label_raw    : np.ndarray(结构化 dtype)
        output_names : list[str],契约存在性判定

    Returns:
        gt_boxes : np.ndarray shape=(N,7) [x,y,z,dx=l,dy=w,dz=h,heading(rad)] float32
        gt_names : np.ndarray shape=(N,) str(type)
    """
    def col(name):
        if name in output_names:
            return label_raw[name].astype(np.float64)
        warnings.warn(f"MSR label: '{name}' missing, filled 0")
        return np.zeros(len(label_raw), dtype=np.float64)

    # 几何量 ×0.01(cm → m);heading ×0.01 → 度,再 ×π/180 → 弧度
    x = col('x_m') * 0.01
    y = col('y_m') * 0.01
    z = col('z_m') * 0.01
    w = col('width_m') * 0.01
    l = col('length_m') * 0.01
    h = col('height_m') * 0.01
    heading = col('heading_deg') * 0.01 * np.pi / 180.0

    type_col = col('type').astype(np.int32)
    # gt_boxes 顺序: [x,y,z, dx=l, dy=w, dz=h, heading]
    gt_boxes = np.stack([x, y, z, l, w, h, heading], axis=1).astype(np.float32)
    gt_names = np.array([str(t) for t in type_col])
    return gt_boxes, gt_names

import numpy as np
from pcdet.datasets.msr.msr_utils import (
    load_struct_dtype,
    MSR_FEATURE_ORDER,
    build_msr_features,
    parse_label_boxes,
)

POINTS_STRUCT = '/mnt/d/DataSet/11111111111/POINTS/struct.json'
LABELS_STRUCT = '/mnt/d/DataSet/11111111111/LABELS/struct.json'


# ---------- Step 1: load_struct_dtype ----------

def test_points_dtype_24B():
    dtype, total, names = load_struct_dtype(POINTS_STRUCT)
    assert total == 24
    assert dtype.itemsize == 24  # fields 累加 24,无 padding
    assert 'range_m' in names  # 真实 struct.json 的 output_name


def test_labels_dtype_28B_with_padding():
    dtype, total, names = load_struct_dtype(LABELS_STRUCT)
    assert total == 28
    assert dtype.itemsize == 28  # fields 累加 27 + 1B padding
    assert '_pad' in dtype.names
    assert dtype['_pad'].shape == () or dtype['_pad'].itemsize == 1


def test_feature_order_has_18():
    assert len(MSR_FEATURE_ORDER) == 18
    # 前 3 列强制 xyz(满足基类 PointFeatureEncoder 断言)
    assert MSR_FEATURE_ORDER[:3] == ['x', 'y', 'z']


# ---------- Step 8: build_msr_features / parse_label_boxes ----------

def test_build_features_shape():
    # 10 个点,11 个真实 output_name(以 json 为准)+ padding(不重名字段)
    np_fields = [('range_m', '<f4'), ('doppler_mps', '<f4'),
                 ('ang_rad', '<f4'), ('elv_rad', '<f4'),
                 ('rcs', '<i2'), ('snr', '<i2'),
                 ('doppler_anti_amb_confi', 'i1'), ('exist_confidence', 'u1'),
                 ('frame', '<i2'), ('beam', 'u1'), ('extra_cnt', 'u1')]
    # 11 真实字段=4+2+2+2+2+2+1+1+2+1+1=20B,补 4B padding 凑 24B
    pad_fields = [('p%d' % i, 'u1') for i in range(4)]
    raw = np.zeros(10, dtype=np_fields + pad_fields)
    raw['range_m'] = np.arange(10) * 1.0
    raw['ang_rad'] = np.arange(10) * 0.1
    raw['elv_rad'] = np.arange(10) * 0.05
    raw['doppler_mps'] = np.arange(10) * 0.2
    names = ['range_m', 'doppler_mps', 'ang_rad', 'elv_rad', 'rcs', 'snr',
             'doppler_anti_amb_confi', 'exist_confidence', 'frame', 'beam', 'extra_cnt']
    out = build_msr_features(raw, names)
    assert out.shape == (10, 18)


def test_build_features_xyz_correct():
    """极坐标 → 笛卡尔公式正确性(单点构造)。"""
    np_fields = [('range_m', '<f4'), ('doppler_mps', '<f4'),
                 ('ang_rad', '<f4'), ('elv_rad', '<f4')]
    raw = np.zeros(1, dtype=np_fields + [('p%d' % i, 'u1') for i in range(20)])
    raw['range_m'] = [10.0]
    raw['ang_rad'] = [0.0]    # x 方向
    raw['elv_rad'] = [0.0]
    raw['doppler_mps'] = [1.0]
    names = ['range_m', 'doppler_mps', 'ang_rad', 'elv_rad']
    out = build_msr_features(raw, names)
    # MSR_FEATURE_ORDER(新序): idx 0=x, 1=y, 2=z, 14=dop_x, 15=dop_y
    assert abs(out[0, 0] - 10.0) < 1e-4   # x=10*cos0*cos0=10
    assert abs(out[0, 1] - 0.0) < 1e-4    # y=10*sin0*cos0=0
    assert abs(out[0, 2] - 0.0) < 1e-4    # z=10*sin0=0
    assert abs(out[0, 14] - 1.0) < 1e-4   # dop_x=1*cos0=1
    assert abs(out[0, 15] - 0.0) < 1e-4   # dop_y=1*sin0=0


def test_parse_label_cm_to_m():
    # 构造一个 label 原始记录(结构对齐真实 LABELS/struct.json 的字段顺序)
    # tuple 按字段顺序:id=0, x_m=868(cm), 其余几何清0, type=1
    raw = np.array(
        [(0, 868, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)],
        dtype=[
            ('id', '<u2'), ('x_m', '<i2'), ('y_m', '<i2'), ('z_m', '<i2'),
            ('vx_mps', '<i2'), ('vy_mps', '<i2'), ('heading_deg', '<i2'),
            ('width_m', '<u2'), ('length_m', '<u2'), ('height_m', '<u2'),
            ('type', 'u1'), ('type_confi', 'u1'), ('is_ghost', 'u1'),
            ('is_unpassable', 'u1'), ('is_attention', 'u1'),
            ('point_count', 'u1'), ('point_quality', 'u1'), ('_pad', 'u1'),
        ],
    )
    names = ['id', 'x_m', 'y_m', 'z_m', 'vx_mps', 'vy_mps', 'heading_deg',
             'width_m', 'length_m', 'height_m', 'type', 'type_confi',
             'is_ghost', 'is_unpassable', 'is_attention', 'point_count', 'point_quality']
    boxes, names_out = parse_label_boxes(raw, names)
    assert abs(boxes[0, 0] - 8.68) < 1e-3  # 868cm → 8.68m
    assert names_out[0] == '1'

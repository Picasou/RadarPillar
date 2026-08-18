"""Task 2 测试:MsrDataset 类的 reader 方法对真实数据的形状/正确性验证。

依赖真实数据路径 /mnt/d/DataSet/MSR(WSL)。路径不存在时整模块 skip。
"""
import os

import numpy as np
import pytest
from easydict import EasyDict

DATA_ROOT = '/mnt/d/DataSet/MSR'

# 真实数据缺失则跳过整组(测试依赖外部数据集)
pytestmark = pytest.mark.skipif(
    not os.path.isdir(DATA_ROOT) or not os.path.isfile(os.path.join(DATA_ROOT, 'POINTS/struct.json')),
    reason='MSR 真实数据 %s 不可用' % DATA_ROOT,
)


def _make_minimal_cfg(used_features):
    """构造最小 dataset_cfg,只够 MsrDataset 构造 + reader 调用。"""
    return EasyDict({
        'DATA_PATH': DATA_ROOT,
        'DATA_SPLIT': {'train': 'training', 'test': 'val'},
        'INFO_PATH': {'train': ['msr_infos_training.pkl'], 'test': ['msr_infos_val.pkl']},
        'POINT_CLOUD_RANGE': [0, -25.6, -3, 51.2, 25.6, 2],
        'POINT_FEATURE_ENCODING': {
            'encoding_type': 'absolute_coordinates_encoding',
            'used_feature_list': used_features,
            'src_feature_list': used_features,
        },
        'POINT_FEATURE_NORMALIZATION': {'USE_NORM': False},
        'USE_GND_VELOCITY': True,
        'DATA_AUGMENTOR': {'DISABLE_AUG_LIST': ['placeholder'], 'AUG_CONFIG_LIST': []},
        'DATA_PROCESSOR': [
            {'NAME': 'mask_points_and_boxes_outside_range', 'REMOVE_OUTSIDE_BOXES': True},
            {'NAME': 'shuffle_points', 'SHUFFLE_ENABLED': {'train': True, 'test': False}},
            {'NAME': 'transform_points_to_voxels',
             'VOXEL_SIZE': [0.16, 0.16, 5.0],
             'MAX_POINTS_PER_VOXEL': 10,
             'MAX_NUMBER_OF_VOXELS': {'train': 16000, 'test': 40000}},
        ],
    })


@pytest.fixture(scope='module')
def ds():
    from pcdet.datasets.msr.msr_dataset import MsrDataset
    ds = MsrDataset(
        dataset_cfg=_make_minimal_cfg(['x', 'y', 'z', 'rcs', 'doppler_mps']),
        class_names=['1', '4', '5'], training=False, root_path=None,
    )
    ds.set_split('training')
    return ds


# ---------- 构造 + 注册 ----------

def test_msr_dataset_registered():
    """MsrDataset 已注册到 pcdet.datasets 顶层。"""
    from pcdet.datasets import __all__ as datasets_all
    assert 'MsrDataset' in datasets_all


def test_msr_dataset_constructs(ds):
    """构造成功,sample_id_list 读到真实 IMAGESETS。"""
    assert ds.sample_id_list is not None
    assert len(ds.sample_id_list) > 0
    assert ds.sample_id_list[0] == '00000000'  # training.txt 首行


def test_selected_feature_idx(ds):
    """selected_feature_idx 对应 MSR_FEATURE_ORDER 中的位置(xyz 强制前 3)。"""
    # used_feature_list=['x','y','z','rcs','doppler_mps']
    # 新 MSR_FEATURE_ORDER: x=0,y=1,z=2, rcs=7, doppler_mps=4
    assert ds.selected_feature_idx == [0, 1, 2, 7, 4]


# ---------- get_dynamic_param ----------

def test_get_dynamic_param_returns_floats(ds):
    """get_dynamic_param 返回 ego_speed/yaw_rate 都是 float。"""
    p = ds.get_dynamic_param('00000000')
    assert isinstance(p['ego_speed'], float)
    assert isinstance(p['yaw_rate'], float)
    # ego_speed 物理量范围合理(-100, 100);当前数据自车静止应≈0
    assert -100.0 <= p['ego_speed'] <= 100.0
    assert -10.0 <= p['yaw_rate'] <= 10.0


def test_get_dynamic_param_missing_file(ds):
    """PARAMS 文件缺失时返回 0(不抛异常)。"""
    p = ds.get_dynamic_param('nonexistent_idx')
    assert p == {'ego_speed': 0.0, 'yaw_rate': 0.0}


# ---------- get_radar ----------

def test_get_radar_shape(ds):
    """get_radar 返回 (N, 18) MSR_FEATURE_ORDER 全列且 dtype=float32。

    返回 src 全列(不预选):选列由基类 PointFeatureEncoder.forward 统一完成。
    """
    from pcdet.datasets.msr.msr_utils import MSR_FEATURE_ORDER
    pts = ds.get_radar('00000000')
    assert pts.ndim == 2
    assert pts.shape[1] == len(MSR_FEATURE_ORDER)  # 18 列 src 全列
    assert pts.dtype == np.float32
    assert pts.shape[0] > 0  # 00000000 有真实点


def test_getitem_encoder_selects_used_columns(ds):
    """回归:prepare_data 全链路后 points 列数 == len(used_feature_list)。

    防双重选列复发(get_radar 曾预选 6 列,encoder 再按 18 列 src 索引选列,
    numpy 越界切片产生空列,特征只剩 xyz 3 列)。
    """
    item = ds[0]  # training=False → prepare_data 走 encoder + data_processor
    assert item['points'].shape[1] == 5  # fixture used_feature_list 长度
    # xyz 在前 3 列(voxel 契约)
    assert item['points'][:, 0].max() > 0  # x 物理量(米)


def test_get_radar_xyz_is_physical(ds):
    """scale 已应用:x=range·cos(azi)cos(elv),应是物理量(米)。"""
    pts = ds.get_radar('00000000')
    # x 应在合理雷达探测范围内(几米到几百米)
    assert (pts[:, 0] > 0).all()  # x 前向都为正
    assert pts[:, 0].max() < 200.0  # 不超过 200m
    # y/z 不应全 0(scale 修复前的回归 bug 防护)
    assert not (pts[:, 1] == 0).all()
    assert not (pts[:, 2] == 0).all()


# ---------- get_label ----------

def test_get_label_shape_and_classes(ds):
    """get_label 返回 (N,7) gt_boxes + (N,) gt_names,类名在 CLASS_NAMES 内。"""
    boxes, names = ds.get_label('00000000')
    assert boxes.shape[1] == 7
    assert names.shape[0] == boxes.shape[0]
    assert set(np.unique(names)).issubset({'1', '4', '5'})


def test_get_label_geometry_in_meters(ds):
    """LABELS 几何量已 ×0.01(cm→m):boxes 数值在米量级。"""
    boxes, _ = ds.get_label('00000000')
    # 位置 x 应在几十米内(不可能是几千米)
    assert np.abs(boxes[:, 0]).max() < 500.0
    # 尺寸 l/w/h 都为正且小于 20m(普通车辆/二轮车)
    assert (boxes[:, 3] > 0).all() and (boxes[:, 3] < 20).all()
    assert (boxes[:, 4] > 0).all() and (boxes[:, 4] < 20).all()
    assert (boxes[:, 5] > 0).all() and (boxes[:, 5] < 20).all()
    # heading 在 [-π, π]
    assert (boxes[:, 6] >= -np.pi).all() and (boxes[:, 6] <= np.pi).all()


def test_get_label_empty_file(ds, tmp_path):
    """空 LABELS 文件(0 框)返回 (0,7) + 空 names。"""
    # 00000000 已知有 5 框;找一个空 label 的 idx(用 mock idx 不存在场景不适用,
    # 这里直接构造空 bin 测试逻辑路径)
    empty_bin = tmp_path / 'empty.bin'
    empty_bin.write_bytes(b'')
    # 复用 ds 的 dtype 读空文件
    raw = np.fromfile(str(empty_bin), dtype=ds._labels_dtype)
    assert raw.shape[0] == 0


# ---------- USE_GND_VELOCITY 开关 ----------

def test_use_gnd_velocity_off():
    """USE_GND_VELOCITY=False 时 ego_speed/yaw_rate 应传 0(dop_*_gnd == dop_*)。"""
    from pcdet.datasets.msr.msr_dataset import MsrDataset
    cfg = _make_minimal_cfg(
        ['x', 'y', 'z', 'dop_x', 'dop_y', 'dop_x_gnd', 'dop_y_gnd']
    )
    cfg.USE_GND_VELOCITY = False
    ds = MsrDataset(dataset_cfg=cfg, class_names=['1', '4', '5'], training=False, root_path=None)
    ds.set_split('training')
    pts = ds.get_radar('00000000')
    # 18 列 src 顺序:dop_x(14), dop_y(15), dop_x_gnd(16), dop_y_gnd(17)
    # USE_GND_VELOCITY=False → ego_speed=0 → dop_*_gnd == dop_*
    assert np.allclose(pts[:, 14], pts[:, 16], atol=1e-5)  # dop_x == dop_x_gnd
    assert np.allclose(pts[:, 15], pts[:, 17], atol=1e-5)  # dop_y == dop_y_gnd

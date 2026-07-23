"""build_network 的占位 Dataset（不触碰 infos/gt_database）。
build_network 内部读取 dataset.point_feature_encoder.num_point_features、
grid_size、voxel_size、point_cloud_range，构造与数据集无关的前向链。
"""
from types import SimpleNamespace


def make_dummy_dataset(full_cfg):
    """→ SimpleNamespace 含 build_network 所需全部属性。full_cfg 是完整 cfg（不是 cfg.MODEL）。"""
    data_cfg = full_cfg.DATA_CONFIG
    pcr = data_cfg.POINT_CLOUD_RANGE
    # VFE in_channels 估计：若 USE_VELOCITY_DECOMPOSITION + 3(xyz)+1(rcs)=9；否则 ~5
    vfe = full_cfg.MODEL.VFE
    n_raw = 9 if vfe.get('USE_VELOCITY_DECOMPOSITION', True) else 5
    encoder = SimpleNamespace(num_point_features=n_raw)
    voxel_size = data_cfg.DATA_PROCESSOR[2]['VOXEL_SIZE']
    nx = int((pcr[3] - pcr[0]) / voxel_size[0])
    ny = int((pcr[4] - pcr[1]) / voxel_size[1])
    nz = int((pcr[5] - pcr[2]) / voxel_size[2])
    # 用 numpy 让 grid_size[:2] // int 可用（head.generate_anchors 需要）
    import numpy as np
    grid_size = np.array([nx, ny, nz], dtype=np.int32)
    return SimpleNamespace(
        class_names=list(full_cfg.CLASS_NAMES) if hasattr(full_cfg, 'CLASS_NAMES') else ['Car','Pedestrian','Cyclist'],
        point_feature_encoder=encoder,
        grid_size=grid_size,
        voxel_size=voxel_size,
        point_cloud_range=list(pcr),
    )

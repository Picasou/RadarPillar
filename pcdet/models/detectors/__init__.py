from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .pointnet_seg import PointNetSeg
from .centerpoint import CenterPoint

# 原版 OpenPCDet 移植（第一批：无编译 op 依赖的 detector）
# 注：voxel_rcnn 缺 voxelrcnn_head(voxel_pool_modules)，mppnet/mppnet_e2e 缺 mppnet_head(mppnet_utils)，
# 这三者的 head 依赖暂不搬，故 detector 不注册，文件保留待后续批次。
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .pillarnet import PillarNet
from .voxelnext import VoxelNeXt
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .transfusion import TransFusion

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'PointNetSeg': PointNetSeg,
    'CenterPoint': CenterPoint,
    # 移植（第一批）
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'PillarNet': PillarNet,
    'VoxelNeXt': VoxelNeXt,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'TransFusion': TransFusion,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model

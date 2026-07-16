from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .roi_head_template import RoIHeadTemplate

# 原版 OpenPCDet 移植（第一批）
# 注：voxelrcnn_head 缺 voxel_pool_modules，mppnet_head/mppnet_memory_bank_e2e 缺 mppnet_utils，
# 这三者依赖暂不搬，故不注册，文件保留待后续批次。
from .second_head import SECONDHead

__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PVRCNNHead': PVRCNNHead,
    'PointRCNNHead': PointRCNNHead,
    # 移植（第一批）
    'SECONDHead': SECONDHead,
}

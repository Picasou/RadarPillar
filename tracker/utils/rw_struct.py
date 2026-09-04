# -*-coding:utf-8-*-
import os
from ctypes import Structure, string_at, addressof, sizeof, memmove
from ctypes import (c_uint8, c_uint16, c_uint32, c_int8, c_int16, c_int32, c_float)


class c_struct(Structure):
    def __init__(self):
        super().__init__()

    def encode(self):
        return string_at(addressof(self), sizeof(self))

    def decode(self, data):
        if len(data) < sizeof(self):
            raise ValueError(f'decode buffer {len(data)}B < struct {sizeof(self)}B (截断输入)')
        memmove(addressof(self), data, sizeof(self))
        return sizeof(self)


# ==================================================
# -------------------- Bin 文件格式 ----------------
# 仅保留 4 类数据的二进制内存布局: 点云 / 目标 / 动态参数 / 静态参数
# ==================================================

class Raw_DetHead(c_struct):
    """点云帧头"""
    _fields_ = [
        ('version', c_uint16),
        ('frame_cnt', c_uint16),
        ('det_num', c_uint16),
        ('reserved', c_uint16)]


class Raw_Det(c_struct):
    """点云单点"""
    _fields_ = [
        ('id', c_uint16),
        ('flags', c_uint16),
        ('range', c_int32),
        ('doppler', c_int16),
        ('azimuth', c_int16),
        ('elevation', c_int16),
        ('rcs', c_int16),
        ('snr', c_int16),
        ('doppler_anti_amb_confi', c_int8),
        ('exist_confi', c_uint8),
        ('frame', c_int16),
        ('beam', c_uint8),
        ('extra_cnt', c_uint8)]


class Raw_TrkHead(c_struct):
    """目标帧头"""
    _fields_ = [
        ('version', c_uint16),
        ('frame_cnt', c_uint16),
        ('trk_num', c_uint16),
        ('reserved', c_uint16)]


class Raw_Trk(c_struct):
    """单个跟踪目标"""
    _fields_ = [
        ('id', c_uint16),
        
        # 位置
        ('x_m', c_int16),
        ('y_m', c_int16),
        ('z_m', c_int16),
        
        # 速度
        ('vx_mps', c_int16),
        ('vy_mps', c_int16),
        
        # 加速度
        ('ax_mps2', c_int16),
        ('ay_mps2', c_int16),
        
        # 朝向
        ('heading_deg', c_int16),
        
        # 尺寸
        ('width_m', c_uint16),
        ('length_m', c_uint16),
        ('height_m', c_uint16),
        
        # 语义信息
        ('type', c_uint8),              # classification
        ('type_confi', c_uint8),        # confidence [0-100]
        
        ('lifetime_s', c_uint16),
        
        # 状态信息
        ('motion_status', c_uint8),     # 0:静止 | 1:运动 | 2:慢速
        ('measurement_status', c_uint8),# 0:coasting | 1:normal
        ('existence_prob', c_uint8),    # [0-100]
        ('obstacle_prob', c_uint8),     # [0-100]
        ('passable_status', c_uint8),   # 0:不可通行 | 1:可通行    
        ('rel_vel', c_uint8),            # 0:absolute | 1:relative
        ('rel_acc', c_uint8),            # 0:absolute | 1:relative
        
        # 速度不确定性
        ('vx_std_mps', c_uint16),
        ('vy_std_mps', c_uint16),
        ('xy_vel_cov', c_uint16),
        
        # 加速度不确定性
        ('ax_std_mps2', c_uint16),
        ('ay_std_mps2', c_uint16),
        ('xy_acc_cov', c_uint16),
        
        # 位置不确定性
        ('x_std_m', c_uint16),
        ('y_std_m', c_uint16),
        ('z_std_m', c_uint16),
        ('xy_pos_cov', c_uint16),
        
        # 朝向不确定性
        ('heading_std', c_uint8),
        
        # 横摆角速度 (注意C结构体中yaw_rate在heading_std之后)
        ('yaw_rate_degs', c_int16),
        ('yaw_rate_std', c_uint8),
        
        # 尺寸不确定性
        ('length_std', c_uint8),
        ('width_std', c_uint8),
        ('height_std', c_uint8),]


class Raw_Vdd(c_struct):
    """动态参数"""
    _fields_ = [
        ('stLen', c_uint32),
        ('stType', c_uint16),
        ('stVer', c_uint16),
        ('hostVelocity_mps', c_float),
        ('hostRawVelocity_mps', c_float),
        ('hostAccelX_mpss', c_float),
        ('hostAccelY_mpss', c_float),
        ('laneRadius_m', c_float),
        ('laneRadiusDot_mps', c_float),
        ('vehicleYawRate_radps', c_float),
        ('steeringWheelAngle_rad', c_float),
        ('vehicleRotPtX_m', c_float),
        ('vehicleRotPtY_m', c_float),
        ('wfModelId', c_uint8),
        ('sceneID', c_uint8),
        ('sceneProbability', c_uint8),
        ('reserved3', c_uint8),
        ('maxDetectionRange_m', c_float),
        ('maxDetectionDoppler_mps', c_float),
        ('driving_direction', c_uint8),
        ('driveGearEngaged', c_uint8),
        ('turningLeft', c_uint8),
        ('turningRight', c_uint8),
        ('frontLeftDoorClosed', c_uint8),
        ('frontRightDoorClosed', c_uint8),
        ('rearLeftDoorClosed', c_uint8),
        ('rearRightDoorClosed', c_uint8),
        ('frontLeftLocked', c_uint8),
        ('frontRightLocked', c_uint8),
        ('rearLeftLocked', c_uint8),
        ('rearRightLocked', c_uint8),
        ('leftLampFault', c_uint8),
        ('rightLampFault', c_uint8),
        ('ignitionOn', c_uint8),
        ('bReady', c_uint8),
        ('keySt', c_uint8 * 2),
        ('time_100us', c_uint16),
        ('spare', c_uint16),
        ('crc16', c_uint16)]

    def __init__(self):
        super().__init__()
        self.hostVelocity_mps = 0.0
        self.vehicleYawRate_radps = 0.0
        self.driving_direction = 1
        self.driveGearEngaged = 1
        self.bReady = 1


class Raw_Vds(c_struct):
    """静态参数"""
    _fields_ = [
        ('stLen', c_uint32),
        ('stType', c_uint16),
        ('stVer', c_uint16),
        ('vehLength_m', c_float),
        ('vehWidth_m', c_float),
        ('wheelbase_m', c_float),
        ('wheelcircumference_m', c_float),
        ('overhang_m', c_float),
        ('sg_stLen', c_uint32),
        ('sg_stType', c_uint16),
        ('sg_stVer', c_uint16),
        ('xpos', c_float),
        ('ypos', c_float),
        ('zpos', c_float),
        ('rotation', c_float),
        ('location', c_uint32),
        ('ft_stLen', c_uint32),
        ('ft_stType', c_uint16),
        ('ft_stVer', c_uint16),
        ('azim_align_fitting_type', c_uint32),
        ('delta_channel_offset', c_int16),
        ('elev_channel_offset', c_int16),
        ('sum_channel_offset', c_int8),
        ('ft_pad', c_uint8),
        ('ft_crc16', c_uint16),
        ('orientation', c_uint8),
        ('sensorAddr', c_uint8),
        ('sg_crc16', c_uint16),
        ('bReady', c_uint8),
        ('pad', c_uint8),
        ('crc16', c_uint16)]


# ==================================================
# -------------------- 读写函数 --------------------
# ==================================================

def struct_read(filepath, c_item, limit=None):
    """从 bin 文件读取结构体列表"""
    c_list = []
    if os.path.exists(filepath):
        item_size = sizeof(c_item)
        item_num = os.path.getsize(filepath) // item_size
        if limit is not None and 0 < limit <= item_num:
            item_num = limit
        with open(filepath, 'rb') as f:
            for i in range(item_num):
                payload = f.read(item_size)
                item = c_item()
                item.decode(payload)
                c_list.append(item)
    else:
        print(filepath + ' not exist !')
    return c_list


def struct_write(filepath, c_list, heads=None, head_filepath=None):
    """
    结构体写出: c_list → bin; heads 非空时写 head_filepath 帧头文件(仿数据源 0200/0201 布局)
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    if heads is not None and head_filepath is not None:
        os.makedirs(os.path.dirname(head_filepath) or '.', exist_ok=True)
        with open(head_filepath, 'wb') as f:
            for h in heads:
                f.write(h.encode())
    with open(filepath, 'wb') as f:
        for item in c_list:
            f.write(item.encode())

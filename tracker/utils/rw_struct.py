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
        size = min(data.__sizeof__(), sizeof(self))
        memmove(addressof(self), data, size)
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
        ('x', c_int16),
        ('y', c_int16),
        ('z', c_int16),
        ('vx', c_int16),
        ('vy', c_int16),
        ('ax', c_int16),
        ('ay', c_int16),
        ('heading', c_int16),
        ('width', c_uint16),
        ('length', c_uint16),
        ('height', c_uint16),
        ('classification', c_uint8),
        ('confidence', c_uint8),
        ('pad1', c_uint16),
        ('pad2', c_uint16),
        ('pad3', c_uint16)]


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


def struct_write(filepath, c_list):
    """将结构体列表写入 bin 文件"""
    with open(filepath, 'wb') as f:
        for item in c_list:
            payload = item.encode()
            f.write(payload)

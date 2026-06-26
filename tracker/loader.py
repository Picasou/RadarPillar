import os
import numpy as np
from schemas import (Cfg, VDS, VDD, PT, PTs, GT, GTs, Trk, Trks, FRAME, FRAMEs)
from utils.rw_struct import (struct_read, Raw_DetHead, Raw_Det, Raw_TrkHead, Raw_Trk, Raw_Vdd, Raw_Vds)


class Loader:
    """加载与切片 — 解析数据路径，产出 FRAME 序列与 VDS。"""

    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        self.relpath = 'radar.default'

    # ---- 公共接口 ----

    def getframes(self, path: str) -> FRAMEs:
        data_path = os.path.join(path, self.relpath)
        pts_list = self._load_PTs(data_path)
        gts_list = self._load_GTs(data_path)
        vdd_list = self._load_VDD(data_path)

        # 取最小长度确保对齐
        n = min(len(pts_list), len(vdd_list))
        frames = []
        for i in range(n):
            frame = FRAME(
                gts=gts_list[i] if i < len(gts_list) else GTs(num=0, Lst=[]),
                pts=pts_list[i],
                vdd=vdd_list[i]
            )
            frames.append(frame)
        return FRAMEs(num=len(frames), Lst=frames)

    def getvds(self, path: str) -> VDS:
        data_path = os.path.join(path, self.relpath)
        vds_file = self._find_vds_file(data_path)
        if vds_file is not None:
            return self._load_vds(vds_file)
        return self._vds_from_cfg()

    # ---- 内部方法 ----

    def _load_PTs(self, path: str) -> list[PTs]:
        """加载点云: 0100(帧头) + 0101(单点) → schemas.PTs 列表"""
        file_0100 = os.path.join(path, '0100.00000.bin')
        file_0101 = os.path.join(path, '0101.00000.bin')
        if not os.path.exists(file_0100) or not os.path.exists(file_0101):
            return []

        det_head_list = struct_read(file_0100, Raw_DetHead)
        det_list = struct_read(file_0101, Raw_Det)

        # 按帧头 det_num 切片，定点转浮点 (/100)
        pts_list = []
        offset, limit = 0, len(det_list)
        for head in det_head_list:
            num = head.det_num
            if offset + num > limit:
                num = limit - offset
            if num < 0:
                break
            pts = []
            for j in range(num):
                d = det_list[offset + j]
                pt = PT(
                    beam=d.beam,
                    extra_cnt=d.extra_cnt,
                    exist_confidence=d.exist_confi,
                    doppler_anti_amb_confi=d.doppler_anti_amb_confi,
                    id=d.id,
                    flags=d.flags,
                    rcs=d.rcs,
                    snr=d.snr,
                    frame=d.frame,
                    range_m=d.range / 100,
                    ang_rad=np.radians(d.azimuth / 100),
                    elv_rad=np.radians(d.elevation / 100),
                    doppler_mps=d.doppler / 100
                )
                pts.append(pt)
            pts_list.append(PTs(num=len(pts), Lst=pts))
            offset += head.det_num
        return pts_list

    def _load_TKs(self, path: str) -> list[Trks]:
        """加载目标: 0200(帧头) + 0201(单目标) → schemas.Trks 列表"""
        file_0200 = os.path.join(path, '0200.00000.bin')
        file_0201 = os.path.join(path, '0201.00000.bin')
        if not os.path.exists(file_0200) or not os.path.exists(file_0201):
            return []

        trk_head_list = struct_read(file_0200, Raw_TrkHead)
        trk_list = struct_read(file_0201, Raw_Trk)

        # 按帧头 trk_num 切片，定点转浮点 (/100)
        trks_list = []
        offset, limit = 0, len(trk_list)
        for head in trk_head_list:
            num = head.trk_num
            if offset + num > limit:
                num = limit - offset
            if num < 0:
                break
            trks = []
            for j in range(num):
                t = trk_list[offset + j]
                trk = Trk(
                    id=t.id,
                    x_m=t.x / 100.0,
                    y_m=t.y / 100.0,
                    z_m=t.z / 100.0,
                    vx_mps=t.vx / 100.0,
                    vy_mps=t.vy / 100.0,
                    ax_mps2=t.ax / 100.0,
                    ay_mps2=t.ay / 100.0,
                    heading_deg=t.heading / 100.0,
                    yaw_rate_degs=0,
                    width_m=t.width / 100.0,
                    height_m=t.height / 100.0,
                    length_m=t.length / 100.0,
                    lifetime_s=0,
                    x_std_m=0, y_std_m=0, z_std_m=0,
                    vx_std_mps=0, vy_std_mps=0, ax_std_mps2=0, ay_std_mps2=0,
                    xy_pos_cov=0, xy_vel_cov=0, xy_acc_cov=0,
                    width_std_m=0, height_std_m=0, length_std_m=0,
                    heading_std_deg=0, yaw_rate_std_degs=0,
                    type=t.classification,
                    type_confi=t.confidence,
                    obstacle_prob=0, existence_prob=0,
                    motion_status=0, measurement_status=0, passable_status=0,
                    rel_vel=0, rel_acc=0,
                    cov=None, history=None
                )
                trks.append(trk)
            trks_list.append(Trks(num=len(trks), Lst=trks))
            offset += head.trk_num
        return trks_list

    def _load_GTs(self, path: str) -> list[GTs]:
        """加载真值（暂留空，待补 GT bin 文件名）"""
        return []

    def _load_VDD(self, path: str) -> list[VDD]:
        """加载动态参数: 2021 → schemas.VDD 列表"""
        file_2021 = os.path.join(path, '2021.00000.bin')
        if not os.path.exists(file_2021):
            return []

        vdd_raw_list = struct_read(file_2021, Raw_Vdd)
        vdd_list = []
        for b in vdd_raw_list:
            vdd = VDD(
                speed_ms=b.hostVelocity_mps,
                yaw_rate=b.vehicleYawRate_radps,
                gear=b.driveGearEngaged
            )
            vdd_list.append(vdd)
        return vdd_list

    def _find_vds_file(self, path: str):
        """查找静态参数文件"""
        vds_file = os.path.join(path, '2031.00000.bin')
        if os.path.exists(vds_file):
            return vds_file
        return None

    def _load_vds(self, vds_file) -> VDS:
        """加载静态参数: 2031 → schemas.VDS"""
        vds_raw_list = struct_read(vds_file, Raw_Vds, 1)
        if not vds_raw_list:
            return self._vds_from_cfg()
        b = vds_raw_list[0]
        return VDS(
            wheelbase_m=b.wheelbase_m,
            x_pos_m=b.xpos,
            y_pos_m=b.ypos,
            z_pos_m=b.zpos,
            rotation_rad=b.rotation,
            cycle_s=0.1,
            oritation=b.orientation
        )

    def _vds_from_cfg(self) -> VDS:
        """从 cfg 降级构造 VDS"""
        v = self.cfg.RUN.vds
        return VDS(
            wheelbase_m=v.wheelbase_m,
            x_pos_m=v.x_pos_m,
            y_pos_m=v.y_pos_m,
            z_pos_m=v.z_pos_m,
            rotation_rad=0.0,
            cycle_s=v.cycle_s,
            oritation=0,
        )

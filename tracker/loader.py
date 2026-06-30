from __future__ import annotations
import os
import warnings
import numpy as np

from schemas import (Cfg, VDS, VDD, PT, PTs, GT, GTs,
                     Obj, Objs, FRAME, FRAMEs)
from utils.rw_struct import (struct_read,
                             Raw_DetHead, Raw_Det,
                             Raw_TrkHead, Raw_Trk,
                             Raw_Vdd, Raw_Vds)

IS_TURNING_THRESHOLD = 1e-4


class Loader:
    """加载与切片 — 解析数据路径,产出 FRAME 序列与 VDS。"""

    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        self.relpath = 'radar.default'

    # ---- 公共接口 ----

    def getframes(self, path: str) -> FRAMEs:
        data_path = os.path.join(path, self.relpath)
        pts_list = self._load_PTs(data_path)
        gts_list = self._load_GTs(data_path)
        vdd_list = self._load_VDD(data_path)
        objs_list = self._load_objs(data_path)

        n = min(len(pts_list), len(vdd_list), len(objs_list))
        if n != len(pts_list) or n != len(vdd_list) or n != len(objs_list):
            warnings.warn(
                f"[loader] 帧级长度不一致, 已截断到 {n} "
                f"(pts={len(pts_list)}, vdd={len(vdd_list)}, objs={len(objs_list)})",
                RuntimeWarning, stacklevel=2
            )

        frames = []
        for i in range(n):
            frame = FRAME(
                gts=gts_list[i] if i < len(gts_list) else GTs(num=0, Lst=[]),
                pts=pts_list[i],
                vdd=vdd_list[i],
                objs=objs_list[i]
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
        file_0100 = os.path.join(path, '0100.00000.bin')
        file_0101 = os.path.join(path, '0101.00000.bin')
        if not os.path.exists(file_0100) or not os.path.exists(file_0101):
            return []

        det_head_list = struct_read(file_0100, Raw_DetHead)
        det_list = struct_read(file_0101, Raw_Det)

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

    def _load_objs(self, path: str) -> list[Objs]:
        file_0200 = os.path.join(path, '0200.00000.bin')
        file_0201 = os.path.join(path, '0201.00000.bin')
        if not os.path.exists(file_0200) or not os.path.exists(file_0201):
            return []

        trk_head_list = struct_read(file_0200, Raw_TrkHead)
        trk_list = struct_read(file_0201, Raw_Trk)

        # ego 补偿需要每帧 VDD
        vdd_path = os.path.join(path, '2021.00000.bin')
        vdd_raw_list = struct_read(vdd_path, Raw_Vdd) if os.path.exists(vdd_path) else []
        cycle_s_list = self._compute_cycle_s(vdd_raw_list)

        objs_list = []
        offset, limit = 0, len(trk_list)
        for frame_i, head in enumerate(trk_head_list):
            num = head.trk_num
            if offset + num > limit:
                num = limit - offset
            if num < 0:
                break

            cycle_s = cycle_s_list[frame_i] if frame_i < len(cycle_s_list) else 0.05
            v = vdd_raw_list[frame_i] if frame_i < len(vdd_raw_list) else None

            objs = []
            for j in range(num):
                t = trk_list[offset + j]
                x, y, vx, vy = self._compensate_state(
                    t.x / 100.0, t.y / 100.0,
                    t.vx / 100.0, t.vy / 100.0,
                    v, cycle_s
                )
                objs.append(Obj(
                    id=t.id,
                    x=x, y=y, vx=vx, vy=vy,
                    length=t.length / 100.0,
                    width=t.width / 100.0,
                    heading=t.heading / 100.0,
                    type=t.classification,
                    isghost=0,
                    ispassable=0,
                ))
            objs_list.append(Objs(num=len(objs), Lst=objs))
            offset += head.trk_num
        return objs_list

    def _load_GTs(self, path: str) -> list[GTs]:
        return []

    def _load_VDD(self, path: str) -> list[VDD]:
        file_2021 = os.path.join(path, '2021.00000.bin')
        if not os.path.exists(file_2021):
            return []
        vdd_raw_list = struct_read(file_2021, Raw_Vdd)
        return [VDD(speed_ms=b.hostVelocity_mps,
                    yaw_rate=b.vehicleYawRate_radps,
                    gear=b.driveGearEngaged)
                for b in vdd_raw_list]

    def _find_vds_file(self, path: str):
        vds_file = os.path.join(path, '2031.00000.bin')
        return vds_file if os.path.exists(vds_file) else None

    def _load_vds(self, vds_file) -> VDS:
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

    # ---- ego 补偿 ----

    def _compute_cycle_s(self, vdd_raw_list: list) -> list[float]:
        """从 B_2021.time_100us 帧差计算 cycle_s (秒)。"""
        cycle_s_list = []
        prev_ts = None
        for b in vdd_raw_list:
            ts = b.time_100us
            if prev_ts is None:
                cycle_s_list.append(0.05)
            else:
                dt_ticks = (ts - prev_ts) & 0xFFFF
                cycle_s_list.append(dt_ticks * 1e-4 if dt_ticks != 0 else 0.05)
            prev_ts = ts
        return cycle_s_list

    def _compensate_state(self, x, y, vx, vy, vdd_raw, cycle_s):
        """ego 补偿: 平移 + 旋转到当前自车坐标系。"""
        if vdd_raw is None:
            return x, y, vx, vy

        hostv = vdd_raw.hostVelocity_mps
        yaw_rate = vdd_raw.vehicleYawRate_radps
        dx = hostv * cycle_s
        wt = yaw_rate * cycle_s

        if abs(yaw_rate) > IS_TURNING_THRESHOLD:
            cos_wt = np.cos(wt)
            sin_wt = np.sin(wt)
            dx_pos = x - dx
            dy_pos = y
            x_new = dx_pos * cos_wt + dy_pos * sin_wt
            y_new = dx_pos * (-sin_wt) + dy_pos * cos_wt
            vx_new = vx * cos_wt + vy * sin_wt
            vy_new = vx * (-sin_wt) + vy * cos_wt
        else:
            x_new = x - dx
            y_new = y
            vx_new = vx
            vy_new = vy

        return x_new, y_new, vx_new, vy_new
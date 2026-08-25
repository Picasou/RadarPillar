"""BEV 可视化: 复用 visual_utils/viz_common.draw_box_bev, 布局/风格照抄 visualize_msr.py."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw

# 复用 tools 侧可视化公共件(与 visualize_msr_seq.py 同款 sys.path 方式)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'utils' / 'visual_utils'))
from viz_common import draw_box_bev  # type: ignore[import-not-found]

# CJK 字体注册: 序列名含中文, DejaVu 无 CJK 字形; 依次尝试系统 Noto Sans CJK / WSL Windows SimHei
for _fp in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/mnt/c/Windows/Fonts/simhei.ttf'):
    try:
        matplotlib.font_manager.fontManager.addfont(_fp)  # type: ignore[attr-defined]
        plt.rcParams['font.sans-serif'] = [
            matplotlib.font_manager.FontProperties(fname=_fp).get_name()] + plt.rcParams['font.sans-serif']  # type: ignore[attr-defined]
        break
    except Exception:
        pass

from .schemas import FRAME, Trk

# ---- 风格常量: 全部照抄 visualize_msr.py ----
DEFAULT_CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']
CLASS_DISPLAY = {'1': 'Car', '2': 'Pedestrian', '4': 'Cyclist', '5': 'Truck'}   # MSR 类别号 → 显示名
CLASS_COLORS = ['#f1c40f', '#d63ee0', '#7cb342', '#00b3a4']   # 类色避开 doppler 蓝红色域
FALLBACK_COLOR = '#eda100'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
DIV_BLUE, DIV_GRAY, DIV_RED = '#2a78d6', '#444444', '#e34948'   # 中间深灰, 不隐入白画布
DOPPLER_CMAP = LinearSegmentedColormap.from_list('trk_div', [DIV_BLUE, DIV_GRAY, DIV_RED])
OUT_ROOT = Path('output/tracker_viz')     # 默认输出根; PNG 在 <OUT_ROOT>/<序列名>/, GIF/MP4 在 <OUT_ROOT>/<序列名>.gif/.mp4
FIG_DPI = 120                                            # 画布渲染 dpi(PNG/视频同源; 120 兼顾清晰与渲染耗时)
VFRAME_WIDTH = 1600                                      # 视频帧宽 GIF/MP4 共用(清晰度与内存折中)
TEST_BORDER_W = 14                                       # test/val 数据黑边框宽(px)


class Visualizer:
    """
    逐帧出图(三面板 [Camera | BEV+GT | BEV+pred]): step7 调 run, 序列结束调 on_seq_end
    """
    def __init__(self, cfg, class_names: list[str] | None = None) -> None:
        v = cfg.VISUAL
        self.enable = (v.enable == 1)
        self.save = v.save
        self.show = v.show
        self.cycle_s = cfg.RUN.vds.cycle_s
        # 落盘总门: mode=0/1 图恒落盘; mode=2 与航迹数据同受 RUN.save 门控
        self.save_enabled = (cfg.RUN.mode != 2) or (cfg.RUN.save == 1)
        self.class_names = list(class_names) if class_names else list(DEFAULT_CLASS_NAMES)
        self.cam_rotate = v.cam_rotate
        # 固定坐标范围(xlo,xhi,ylo,yhi): cfg.VISUAL.range 显式指定;
        self.range_xy = tuple(v.range) if v.range else None
        self._cur_seq = 'seq'
        self._is_test = False
        self._vid_frames: list[tuple] = []    # [(png_bytes, w, h)] 压缩缓存防 OOM(长序列整段 RGB 会爆内存)
        self._cam = None
        self._cam_fps = 30.0
        # figure 复用: 布局/legend/坐标轴只建一次, 每帧只更新动态元素(散点/框/标题)
        self._fig = None  # 惰性创建, begin_seq 后非 None; matplotlib 动态属性族见各行 ignore
        self._ax = {}
        self._dyn = {}                        # 每帧重建的动态 artist 容器

    # ---- 对外入口 ----
    def begin_seq(self, seq_name: str, seq_path: str | None = None,
                  data_extent: tuple[float, float, float, float] | None = None,
                  is_test: bool = False) -> None:
        """
        序列切换: 记录序列名/相机流; data_extent=(xmin,xmax,ymin,ymax) 全程点云外沿,
        照 visualize_msr 口径 外沿+3m 边距(含 0) 定死固定范围; is_test 命中 test/val 时图加黑边框
        """
        self._cur_seq = seq_name
        self._is_test = is_test
        self._vid_frames = []
        self._teardown_fig()                          # 跨序列布局可能变(相机有无), figure 重建
        self._close_cam()
        if data_extent is not None and self.range_xy is None:   # cfg 显式 range 优先
            xmin, xmax, ymin, ymax = data_extent
            m = 3.0
            self.range_xy = (min(xmin, 0) - m, xmax + m,
                             min(ymin, 0) - m, ymax + m)
        if seq_path:
            cam_dir = Path(seq_path) / 'camera.frontmiddle'
            mp4s = sorted(cam_dir.glob('*.mp4')) if cam_dir.exists() else []
            if mp4s:
                import cv2
                self._cam = cv2.VideoCapture(str(mp4s[0]))
                self._cam_fps = self._cam.get(cv2.CAP_PROP_FPS) or 30.0

    def run(self, frame: FRAME, objs: list, trks: list[Trk]) -> None:
        """
        单帧出图: Camera + BEV+GT + BEV+pred(检测+航迹), 固定坐标范围;
        figure 复用(布局/legend 只建一次), 单次渲染字节 PNG/视频共用
        """
        if not self.enable:
            return
        if not self.save or not self.save_enabled:
            return                                      # 无落盘格式: 不渲染不出图
        pts = frame.proc.points if frame.proc.points is not None else np.zeros((0, 7))
        cam_img = self._read_cam(frame)
        if self._fig is None or bool(self._ax.get('img')) != (cam_img is not None):
            self._build_fig(has_cam=cam_img is not None)    # 首帧/相机流起止 → (重)建布局

        fig = self._fig
        ax_gt, ax_pred = self._ax['gt'], self._ax['pred']
        panels = [ax_gt, ax_pred]

        # ---- 动态元素: 清上一帧残留, 重画本帧散点/框/相机 ----
        for a in self._dyn.pop('artists', []):
            try:
                a.remove()
            except Exception:
                pass
        artists = []
        if cam_img is not None:
            artists.append(self._ax['img'].imshow(cam_img))
        if self.show.get('points', 1) and pts.shape[0]:
            c = pts[:, 5]
            vmax = np.percentile(np.abs(c), 99) or 1.0
            norm = Normalize(vmin=-vmax, vmax=vmax)  # type: ignore[arg-type]
            for ax in panels:
                artists.append(ax.scatter(pts[:, 1], pts[:, 0], c=c, cmap=DOPPLER_CMAP,
                                          norm=norm, s=2, linewidths=0, alpha=0.9, zorder=2))
            cb = self._dyn.get('cb')
            if cb is None:
                sc = artists[-1]
                cb = fig.colorbar(sc, ax=panels, fraction=0.046, pad=0.02)  # type: ignore[union-attr]
                cb.set_label('doppler_gnd', fontsize=9, color=INK2)
                cb.ax.tick_params(colors=MUTED, labelsize=8)
                cb.outline.set_edgecolor(GRID)  # type: ignore[union-attr]
                self._dyn['cb'] = cb
            else:                                       # 复用 colorbar, 只更新归一化域
                cb.norm.vmin, cb.norm.vmax = -vmax, vmax
                cb.update_normal(artists[-1])

        # 框 artist 收集: 画框前记基线, 画完把新增 patches/lines 全纳入动态清理(防跨帧残留成"轨迹")
        bases = [(ax, len(ax.patches), len(ax.lines)) for ax in panels]
        n_gt = self._draw_gts(ax_gt, frame)
        n_obj = self._draw_objs(ax_pred, objs)
        n_trk = self._draw_tracks(ax_pred, trks)
        for ax, np0, nl0 in bases:
            artists.extend(ax.patches[np0:])
            artists.extend(ax.lines[nl0:])
        for ax in panels:
            artists.append(ax.scatter([0], [0], marker='o', s=5, color=INK, zorder=5))
        artists.append(ax_gt.annotate('ego', (0, 0), textcoords='offset points',
                                      xytext=(6, 6), fontsize=8, color=INK2))
        self._dyn['artists'] = artists

        # ---- 面板计数 + 两行总标题(只改文字) ----
        ax_gt.set_title('GT (%d)' % n_gt, fontsize=12, color=INK2, pad=8)
        ax_pred.set_title('Pred (%d)' % (n_obj + n_trk), fontsize=12, color=INK2, pad=8)
        vdd = frame.vdd
        fig.suptitle('%s\nframe %s  |  pts - %d, gts - %d, objs - %d, trks - %d  |  '  # type: ignore[union-attr]
                     'ego - %.1f m/s, yaw - %.3f'
                     % (self._cur_seq, frame.frame_id, pts.shape[0], n_gt, n_obj, n_trk,
                        vdd.speed_ms if vdd else 0.0, vdd.yaw_rate if vdd else 0.0),
                     fontsize=13, color=INK, y=0.97, va='top')

        # ---- 单次渲染: PNG 字节一份, 写盘 + 视频缓存共用; test/val 加黑边框 ----
        if self._is_test:
            from matplotlib.patches import Rectangle
            fig.patches.append(Rectangle((0, 0), 1, 1, transform=fig.transFigure,  # type: ignore[union-attr]
                                         fill=False, edgecolor='black',
                                         linewidth=TEST_BORDER_W, zorder=100))
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=FIG_DPI, facecolor='white')  # type: ignore[union-attr]
        data = buf.getvalue()
        buf.close()
        if self._is_test:
            fig.patches.pop()  # type: ignore[union-attr]
        out_dir = OUT_ROOT / self._cur_seq
        if 2 in self.save:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / ('%s.png' % frame.frame_id)).write_bytes(data)
        if 1 in self.save or 3 in self.save:
            with Image.open(io.BytesIO(data)) as im:    # 仅取宽高, 合成期才解码缩放
                self._vid_frames.append((data, im.width, im.height))

    def _build_fig(self, has_cam: bool) -> None:
        """
        布局构建(每序列一次): [Camera | BEV+GT | BEV+pred], 固定边距/坐标轴/图例/ego 点
        """
        self._teardown_fig()
        if has_cam:
            fig, (ax_img, ax_gt, ax_pred) = plt.subplots(
                1, 3, figsize=(28, 10), dpi=FIG_DPI, width_ratios=[0.8, 1, 1],
                gridspec_kw={'wspace': 0.22})
            ax_img.set_title('Camera', fontsize=12, color=INK2, pad=8)
            ax_img.axis('off')
            self._ax['img'] = ax_img
        else:
            fig, (ax_gt, ax_pred) = plt.subplots(
                1, 2, figsize=(20, 10), dpi=FIG_DPI, gridspec_kw={'wspace': 0.18})
        # 固定边距(禁 tight 裁剪): 每帧画布像素级同尺寸, 防视频抖动
        fig.subplots_adjust(left=0.05, right=0.97, top=0.86, bottom=0.10)
        self._fig, self._ax['gt'], self._ax['pred'] = fig, ax_gt, ax_pred

        if self.range_xy is not None:
            xlo, xhi, ylo, yhi = self.range_xy
        else:
            xlo, xhi, ylo, yhi = -10, 210, -25, 25
        for ax in (ax_gt, ax_pred):
            ax.set_xlim(ylo, yhi)
            ax.set_ylim(xlo, xhi)
            ax.invert_xaxis()                              # +y 朝左
            ax.set_aspect('equal')
            ax.set_xlabel('y (m)', fontsize=9, color=INK2)
            ax.set_ylabel('x (m)', fontsize=9, color=INK2)
            ax.tick_params(colors=MUTED, labelsize=8)
            for s in ax.spines.values():
                s.set_color(GRID)
            ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
            ax.set_axisbelow(True)

        class_handles = [Line2D([0], [0], color=CLASS_COLORS[i], linewidth=2,
                                label=self._class_name(i))
                         for i in range(min(len(self.class_names), len(CLASS_COLORS)))]
        line_handles = [Line2D([0], [0], color=INK2, linewidth=1.2, ls='-', label='Det'),
                        Line2D([0], [0], color=INK2, linewidth=2.8, ls='--', label='Trk')]
        fig.legend(handles=class_handles + line_handles, loc='lower center',
                   ncol=len(class_handles) + 2, frameon=False, fontsize=11,
                   labelcolor=INK2, bbox_to_anchor=(0.5, 0.012))

    def _teardown_fig(self) -> None:
        """
        figure 释放: 关闭复用画布与动态容器(序列切换/析构时)
        """
        if self._fig is not None:
            plt.close(self._fig)
        self._fig, self._ax, self._dyn = None, {}, {}

    def on_seq_end(self) -> None:
        """
        序列收尾: 攒的帧按 save 格式落盘(GIF / MP4, 恒覆盖)
        """
        want = [f for f in (1, 3) if f in self.save]   # 1=GIF 3=MP4
        self._teardown_fig()                            # 序列结束释放复用 figure
        if not (self.enable and self.save_enabled and want and self._vid_frames):
            self._vid_frames = []
            self._close_cam()
            return
        frames = self._vid_frames
        seq_dir = OUT_ROOT / self._cur_seq              # 序列目录: PNG/GIF/MP4 统一入数据名文件夹
        seq_dir.mkdir(parents=True, exist_ok=True)
        if 1 in want:
            out = seq_dir / ('%s.gif' % self._cur_seq)
            gen = self._iter_frames(frames, self._is_test)
            first = next(gen)
            first.save(out, save_all=True, append_images=gen,
                       duration=int(self.cycle_s * 1000), loop=0)
            print('  [visualizer] %d frames -> %s' % (len(frames), out))
        if 3 in want:
            out = seq_dir / ('%s.mp4' % self._cur_seq)
            self._write_mp4(out, self._iter_frames(frames, self._is_test))
            print('  [visualizer] %d frames -> %s' % (len(frames), out))
        self._vid_frames = []
        self._close_cam()

    @staticmethod
    def _iter_frames(frames: list, test_border: bool = False):
        """
        帧流式生成: 逐帧解码→宽归一→白底补齐到统一高(布局切换帧高不一), 偶数化兼容 yuv420p;
        test_border 时画等宽黑边框(防泄露警示)
        """
        W = VFRAME_WIDTH
        H = max(h for _, _, h in frames)
        H += H % 2
        bw = max(TEST_BORDER_W, W * TEST_BORDER_W // VFRAME_WIDTH)
        for b, w, h in frames:
            im = Image.open(io.BytesIO(b)).convert('RGB').resize((W, h))
            canvas = Image.new('RGB', (W, H), 'white')
            canvas.paste(im, (0, (H - h) // 2))
            if test_border:
                ImageDraw.Draw(canvas).rectangle([0, 0, W - 1, H - 1],
                                                 outline='black', width=bw)
            yield canvas

    def _write_mp4(self, out: Path, frames) -> None:
        """
        MP4 合成: 流式写帧 → h264 (fps=1/cycle_s, yuv420p, CRF18 高画质)
        """
        import imageio.v2 as imageio
        fps = max(1, round(1.0 / self.cycle_s))
        with imageio.get_writer(out, fps=fps, codec='libx264', macro_block_size=1,  # type: ignore[union-attr]
                                ffmpeg_params=['-pix_fmt', 'yuv420p', '-crf', '18']) as w:
            for im in frames:
                w.append_data(np.asarray(im))  # type: ignore[attr-defined]

    # ---- 相机 ----
    def _read_cam(self, frame: FRAME):
        """
        读当前帧对应相机图: 雷达时间 i*cycle_s 映射视频帧号
        """
        if self._cam is None:
            return None
        import cv2
        t = int(frame.frame_id) * self.cycle_s if frame.frame_id.isdigit() else 0.0
        idx = int(t * self._cam_fps)
        self._cam.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = self._cam.read()
        if not ok:
            return None
        if self.cam_rotate:
            bgr = np.ascontiguousarray(np.rot90(bgr, self.cam_rotate))
        return bgr[:, :, ::-1]

    def _close_cam(self) -> None:
        if self._cam is not None:
            self._cam.release()
            self._cam = None

    # ---- 绘制件 ----
    def _draw_gts(self, ax, frame: FRAME) -> int:
        """
        GT 框绘制: 含 ghost 计数入标题, ghost 框不画(幽灵目标非真实物体)
        """
        if not self.show.get('gts', 1):
            return 0
        for g in frame.gts.Lst:
            if getattr(g, 'isghost', 0):
                continue
            draw_box_bev(ax, [g.x, g.y, g.z, g.length, g.width, g.height, g.heading],
                         self._color(g.type), linestyle='-', linewidth=1.2,
                         zorder=4, swap_xy=True)
        return len(frame.gts.Lst)          # 计数含 ghost(与 GT 面板 title 口径一致)

    def _draw_objs(self, ax, objs: list) -> int:
        if not self.show.get('objs', 1):
            return 0
        for o in objs:
            # Obj.heading 契约为度, draw_box_bev 吃弧度(与 _draw_tracks 的 np.radians 同口径)
            draw_box_bev(ax, [o.x, o.y, 0, o.length, o.width, 1.5, np.radians(o.heading)],
                         self._color(o.type), linestyle='-', linewidth=1.2,
                         zorder=4, swap_xy=True)
        return len(objs)

    def _draw_tracks(self, ax, trks: list[Trk]) -> int:
        if not self.show.get('tracks', 1):
            return 0
        n = 0
        for t in trks:
            if not t.obstacle_prob:                       # 只画上桌航迹, 与输出层口径一致
                continue
            color = self._color(t.type)
            h = np.radians(t.heading_deg)                 # Trk 契约: heading_deg 为度
            draw_box_bev(ax, [t.x_m, t.y_m, t.z_m, t.length_m, t.width_m, t.height_m, h],
                         color, linestyle='--', linewidth=2.8, zorder=5,
                         facecolor=color, alpha=0.25, swap_xy=True)
            n += 1
        return n

    # ---- 小工具 ----
    def _color(self, type_idx: int) -> str:
        if 0 <= type_idx - 1 < len(CLASS_COLORS):
            return CLASS_COLORS[type_idx - 1]             # detector label 为 1-based
        return FALLBACK_COLOR

    def _class_name(self, i: int) -> str:
        raw = self.class_names[i] if i < len(self.class_names) else 'cls%d' % (i + 1)
        return CLASS_DISPLAY.get(raw, raw)                     # 类别号 → 文字类名

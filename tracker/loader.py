"""数据加载: 读入连续数据, 切割成单帧."""
from pathlib import Path


class FrameLoader:
    """读取连续数据并按自定义逻辑切割为逐帧."""

    def __init__(self, in_dir, frame_step=1):
        self.in_dir = Path(in_dir)        # 连续数据根目录
        self.frame_step = frame_step      # 单帧切片步长/大小
        self._seq = None                  # 连续数据缓存
        self._idx = 0                     # 当前帧索引

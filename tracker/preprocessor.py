"""预处理: 点云裁剪、去噪、特征规整."""


class Preprocessor:
    """单帧点云净化."""

    def __init__(self, x_range=(-5, 50), y_range=(-25, 25), z_range=(-3, 2)):
        self.x_range = x_range            # 感知范围 (m)
        self.y_range = y_range
        self.z_range = z_range

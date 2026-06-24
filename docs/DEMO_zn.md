# 快速 Demo

这里我们提供一个快速 demo，用于在自定义点云数据上测试预训练模型并可视化预测结果。

我们假定你已经按照 [INSTALL.md](INSTALL.md) 成功安装了 `OpenPCDet` 仓库。

1. 按照 [README.md](../README.md) 中说明下载提供的预训练模型。

2. 确认你已经安装了 `mayavi` 可视化工具。如果没有，可按以下方式安装：

   ```
   pip install mayavi
   ```

3. 准备你自己的点云数据（若直接使用原始 KITTI 数据，可跳过该步骤）。
   * 你需要将自定义点云的坐标变换到 `OpenPCDet` 的统一规范坐标系：x 轴指向前方，y 轴指向左方，z 轴指向上方。
   * （可选）点云坐标系的 z 轴原点最好在地面上方约 1.6m 处，因为目前提供的模型都是在 KITTI 数据集上训练的。
   * 设置好强度信息，并将变换后的自定义数据保存为 `numpy file`：

   ```python
   # 变换你的点云数据
   ...

   # 保存到文件
   # points 的形状应为 (num_points, 4)，即 [x, y, z, intensity]
   # 如果没有强度信息，可全部置零
   # 如果有强度信息，应归一化到 [0, 1]
   points[:, 3] = 0
   np.save(`my_data.npy`, points)
   ```

4. 使用预训练模型（例如 PV-RCNN）和你的自定义点云数据运行 demo：

```shell
python demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt pv_rcnn_8369.pth \
    --data_path ${POINT_CLOUD_DATA}
```

其中 `${POINT_CLOUD_DATA}` 可以是以下任一形式：

* 形如 `my_data.npy` 的单个 numpy 文件（已变换的自定义数据）。
* 已变换的自定义数据目录，可一次测试多个点云。
* `data/kitti` 下的原始 KITTI `.bin` 数据，例如 `data/kitti/training/velodyne/000008.bin`。

随后你将看到附带点云可视化的预测结果：

<p align="center">
  <img src="demo.png" width="99%">
</p>

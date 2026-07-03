# 快速上手

数据集相关的配置位于 [tools/cfgs/dataset](../tools/cfgs/dataset)，
模型相关的配置位于 [tools/cfgs](../tools/cfgs)，按不同数据集分别组织。

## 数据集准备

目前我们提供 KITTI 数据集和 NuScenes 数据集的数据加载器，更多数据集的支持正在开发中。

### KITTI 数据集

* 请下载官方的 [KITTI 3D 目标检测](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) 数据集，并将下载的文件按以下结构组织（地面平面 road planes 可从 [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing) 下载，该项可选，用于数据增强）：
* 注意：如果你已有 `pcdet v0.1` 生成的数据 infos，可以选择复用旧 infos 并在 `tools/cfgs/dataset/kitti_dataset.yaml` 中将 `DATABASE_WITH_FAKELIDAR` 选项设为 `True`。另一种选择是重新生成 infos 和 gt database，并保持配置不变。

```
OpenPCDet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

* 通过运行以下命令生成数据 infos：

```python
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset/kitti_dataset.yaml
```

### NuScenes 数据集

* 请下载官方的 [NuScenes 3D 目标检测数据集](https://www.nuscenes.org/download)，并按以下结构组织下载的文件：

```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval（或使用 mini 时选 v1.0-mini）
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval
├── pcdet
├── tools
```

* 安装 `nuscenes-devkit` 1.0.5 版本：

```shell script
pip install nuscenes-devkit==1.0.5
```

* 通过运行以下命令生成数据 infos（可能需要数小时）：

```python
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset/nuscenes_dataset.yaml \
    --version v1.0-trainval
```

## 训练与测试

### 使用预训练模型进行测试与评估

* 使用预训练模型进行测试：

```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* 若要测试某一训练配置下保存的所有权重，并在 Tensorboard 上绘制性能曲线，可添加 `--eval_all` 参数：

```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```

* 使用多卡进行测试：

```shell script
sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}

# 或

sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```

### 训练模型

你可以通过附加命令行参数 `--batch_size ${BATCH_SIZE}` 和 `--epochs ${EPOCHS}` 来指定自定义参数。

* 使用多卡或多机训练：

```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# 或

sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

* 使用单卡训练：

```shell script
python train.py --cfg_file ${CONFIG_FILE}
```

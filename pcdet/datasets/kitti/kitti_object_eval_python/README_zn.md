# kitti-object-eval-python

**说明**：本评估脚本借鉴自 [traveller59/kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python)

在 Python 中快速完成 KITTI 目标检测评估（10 秒内完成），支持 2D / BEV / 3D / AOS，支持 COCO 风格 AP。如果使用命令行接口，numba 需要一些时间来 JIT 编译函数。

## 依赖

仅支持 Python 3.6+，需要 `numpy`、`skimage`、`numba`、`fire`。如果你使用 Anaconda，只需在 Anaconda 中安装 `cudatoolkit`。否则，请参考 [该页面](https://github.com/numba/numba#custom-python-environments) 为 numba 配置 LLVM 与 CUDA。

* 通过 conda 安装：

```
conda install -c numba cudatoolkit=x.x  (8.0、9.0、9.1，取决于你的环境)
```

## 用法

* 命令行接口：

```
python evaluate.py evaluate --label_path=/path/to/your_gt_label_folder --result_path=/path/to/your_result_folder --label_split_file=/path/to/val.txt --current_class=0 --coco=False
```

* Python 接口：

```python
import kitti_common as kitti
from eval import get_official_eval_result, get_coco_eval_result

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

det_path = "/path/to/your_result_folder"
dt_annos = kitti.get_label_annos(det_path)
gt_path = "/path/to/your_gt_label_folder"
gt_split_file = "/path/to/val.txt"  # 来自 https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
val_image_ids = _read_imageset_file(gt_split_file)
gt_annos = kitti.get_label_annos(gt_path, val_image_ids)

print(get_official_eval_result(gt_annos, dt_annos, 0))  # 在我的机器上耗时 6 秒
print(get_coco_eval_result(gt_annos, dt_annos, 0))      # 在我的机器上耗时 18 秒
```

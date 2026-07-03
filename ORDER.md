# RadarPillar 常用命令

本文档记录在 `/home/dministrator1/RadarPillar` 中训练、验证、测试和可视化 nuScenes RadarPillar 的常用命令。

## 0. 进入项目和环境

每次训练前先执行：

```bash
cd /home/dministrator1/RadarPillar
. /home/dministrator1/miniconda3/etc/profile.d/conda.sh
conda activate angle
```

确认环境：

```bash
python - <<'PY'
import torch, numpy, scipy
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('numpy:', numpy.__version__)
print('scipy:', scipy.__version__)
PY
```

## 1. 数据构建

nuScenes 数据目录：

```text
/home/dministrator1/RadarPillar/data/nuscenes
```

如果需要重新生成 nuScenes radar info 文件，优先查看项目中 dataset 脚本支持的参数：

```bash
python pcdet/datasets/nuscenes/nuscenes_radar_dataset.py --help
```

常用检查：

```bash
ls data/nuscenes
ls data/nuscenes/*.pkl
```

重点确认这些文件是否存在：

```text
nuscenes_infos_radar_1sweeps_train.pkl
nuscenes_infos_radar_1sweeps_val.pkl
```

## 2. 普通训练

使用 mini 数据集或当前配置训练：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml \
  --batch_size 1 \
  --workers 0 \
  --epochs 20 \
  --extra_tag train_debug
```

正式训练时可以按显存调整：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml \
  --batch_size 2 \
  --workers 4 \
  --epochs 80 \
  --extra_tag full_train
```

训练输出一般位于：

```text
output/cfgs/nuscenes_models/radarpillar_nuscenes/<extra_tag>/
```

## 3. 1-Batch Overfit 训练

1-batch overfit 用于验证工程链路是否闭环，不用于评价泛化性能。

推荐使用专用脚本：

```bash
CUDA_VISIBLE_DEVICES=0 python /mnt/c/Users/Administrator/Documents/openDet/train_and_plot_overfit_one_batch.py \
  --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes_overfit1.yaml \
  --iters 1200 \
  --lr 0.003 \
  --score_thresh 0.1 \
  --nms_post_maxsize 80 \
  --out_dir output/overfit1_plot
```

更严格的可视化版本：

```bash
CUDA_VISIBLE_DEVICES=0 python /mnt/c/Users/Administrator/Documents/openDet/train_and_plot_overfit_one_batch.py \
  --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes_overfit1.yaml \
  --iters 3000 \
  --lr 0.002 \
  --score_thresh 0.3 \
  --nms_post_maxsize 40 \
  --out_dir output/overfit1_plot_tighter
```

注意：

```text
batch_size=1 不等于 1-batch overfit。
真正的 1-batch overfit 要保证 dataset_len=1。
```

如果使用标准训练脚本做 overfit，需要更多 epoch：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes_overfit1.yaml \
  --batch_size 1 \
  --workers 0 \
  --epochs 1200 \
  --extra_tag overfit1_1200
```

## 4. 验证 Val

对某个 checkpoint 做验证：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml \
  --batch_size 1 \
  --workers 0 \
  --ckpt output/cfgs/nuscenes_models/radarpillar_nuscenes/full_train/ckpt/checkpoint_epoch_80.pth
```

验证输出一般位于：

```text
output/cfgs/nuscenes_models/radarpillar_nuscenes/<extra_tag>/eval/
```

如果是 overfit1 配置，官方 nuScenes eval 可能提示 split 数量不匹配。这通常不是训练失败，而是 debug 数据集只有 1 个样本。

## 5. 测试 Test

测试命令与验证类似，重点是更换配置、checkpoint 和 split：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml \
  --batch_size 1 \
  --workers 0 \
  --ckpt output/cfgs/nuscenes_models/radarpillar_nuscenes/full_train/ckpt/checkpoint_epoch_80.pth
```

如果需要保存预测结果，检查输出目录中的：

```text
result.pkl
```

## 6. 查看训练 Loss

查看训练日志：

```bash
ls output/cfgs/nuscenes_models/radarpillar_nuscenes/*/log_train_*.txt
```

快速搜索 loss、recall 和预测数量：

```bash
grep -E "loss|recall|Average predicted" \
  output/cfgs/nuscenes_models/radarpillar_nuscenes/*/log_train_*.txt
```

overfit 专用脚本会直接打印：

```text
iter
loss
cls loss
loc loss
recall
num_pred_after_nms
top_scores
```

## 7. 检查 result.pkl

查看预测数量、类别和分数：

```bash
python - <<'PY'
import pickle

p = 'output/cfgs/nuscenes_models/radarpillar_nuscenes_overfit1/0701/eval/eval_with_train/epoch_100/val/result.pkl'
with open(p, 'rb') as f:
    data = pickle.load(f)

print('num_samples:', len(data))
if data:
    a = data[0]
    print('frame_id:', a.get('frame_id'))
    print('num_pred:', len(a.get('name', [])))
    print('names_top10:', a.get('name', [])[:10])
    print('scores_top10:', a.get('score', [])[:10])
    print('boxes_shape:', a.get('boxes_lidar').shape)
PY
```

注意：

```text
如果 OUTPUT_RAW_SCORE=True，score 可能是 raw logit，出现负数是可能的。
如果要看概率分数，建议设置 OUTPUT_RAW_SCORE=False。
```

## 8. 画图 Plot

推荐直接使用 overfit 专用脚本，它会训练并画图：

```bash
CUDA_VISIBLE_DEVICES=0 python /mnt/c/Users/Administrator/Documents/openDet/train_and_plot_overfit_one_batch.py \
  --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes_overfit1.yaml \
  --iters 1200 \
  --lr 0.003 \
  --score_thresh 0.1 \
  --nms_post_maxsize 80 \
  --out_dir output/overfit1_plot
```

输出图片：

```text
output/overfit1_plot/overfit1_radar_gt_pred.png
```

复制到 Windows 工作区：

```bash
cp output/overfit1_plot/overfit1_radar_gt_pred.png \
  /mnt/c/Users/Administrator/Documents/openDet/overfit1_radar_gt_pred.png
```

## 9. 常见问题排查

没有预测框时按顺序查：

```text
1. conda angle 是否正确激活
2. dataset_len 是否符合预期
3. valid_gt 是否大于 0
4. 点云和 GT 是否在同一坐标系
5. loss 是否下降
6. raw cls score 是否升高
7. SCORE_THRESH 是否过高
8. NMS_POST_MAXSIZE 是否过小或过大
```

预测框过多时重点查：

```text
1. SCORE_THRESH 是否太低
2. NMS_POST_MAXSIZE 是否太大
3. OUTPUT_RAW_SCORE 是否导致分数理解错误
4. 是否画了 NMS 前预测
```

GT 和点云不重合时重点查：

```text
1. global -> ego 变换
2. sensor -> ego 变换方向
3. sweep transform 是否应用
4. velocity 是否旋转到 ego 坐标系
5. 可视化坐标轴是否符合约定
```

## 10. 推荐判断标准

1-batch overfit 合格标准：

```text
loss 明显下降
valid_gt 大于 0
recall 接近训练 GT 数量
NMS 后有预测框
top score 明显升高
预测框主体接近 GT
```

普通训练合格标准：

```text
训练 loss 持续下降
验证集 Average predicted number 不为 0
result.pkl 中有合理预测框
可视化中点云、GT、预测大致对齐
官方 mAP/NDS 不再全 0
```

## 11. Post-Train Visualization

`tools/train.py` 在训练结束（或 `--skip_eval` 提前返回）后会自动调用
`run_post_train_artifacts`，由 `cfg.TRAIN_POSTPROCESS` 控制行为。默认全部禁用，不开就不会产生额外文件。

配置块示例（写入 YAML，或通过 `--set KEY VAL` 临时启用）：

```yaml
TRAIN_POSTPROCESS:
    ENABLE: True
    PLOT_LOSS_CURVE: True          # 渲染 loss_curve.png
    PLOT_SAMPLES:
        ENABLE: True               # 渲染 sample 可视化（首版为 stub）
        NUM_SAMPLES: 8             # 计划中预留，本版未消费
        REQUIRE_CAMERAS: True      # VoD 等无相机数据集必须跳过相机面板
        SKIP_IF_UNSUPPORTED_DATASET: True
        SAVE_DIR: auto             # 本版未消费，输出固定在 artifacts_dir 下
```

字段语义：

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `ENABLE` | `False` | 整个 post-train 流程的总开关；关闭时直接返回 `{"status": "skipped", "reason": "disabled"}` |
| `PLOT_LOSS_CURVE` | `False` | 复用 `tools/scripts/plot_loss.py`，从最新 train log 渲染 `loss_curve.png` |
| `PLOT_SAMPLES.ENABLE` | `False` | 渲染 sample 面板（当前实现为 stub，会返回 `count=1` 占位） |
| `PLOT_SAMPLES.REQUIRE_CAMERAS` | `True` | 相机面板不支持时跳过 |
| `PLOT_SAMPLES.SKIP_IF_UNSUPPORTED_DATASET` | `True` | 不支持的数据集（如 VoD）默认安全跳过，不会抛异常 |

artifact 输出位置：

```text
output/cfgs/<group>/<tag>/<extra_tag>/post_train_artifacts/
├── loss_curve.png         # 启用 PLOT_LOSS_CURVE 时生成
└── samples/               # 启用 PLOT_SAMPLES 时创建（首版可能为空目录）
```

训练日志末尾会出现一行：

```text
Post-train artifacts: {'status': 'ok', 'artifacts_dir': '.../post_train_artifacts', 'steps': {'loss_curve': {...}, 'samples': {...}}}
```

常见场景：

1. nuScenes overfit1 默认已开启 `TRAIN_POSTPROCESS`（在 `radarpillar_nuscenes_overfit1.yaml` 中），正常训练就能看到 loss_curve.png。
2. VoD 默认未开启。临时启用并验证"安全跳过 sample"：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python tools/train.py \
     --cfg_file tools/cfgs/vod_models/vod_radarpillar_rot.yaml \
     --batch_size 1 --workers 0 --epochs 1 \
     --extra_tag post_train_skip_smoke --skip_eval \
     --set TRAIN_POSTPROCESS.ENABLE True \
           TRAIN_POSTPROCESS.PLOT_LOSS_CURVE True \
           TRAIN_POSTPROCESS.PLOT_SAMPLES.ENABLE True \
           TRAIN_POSTPROCESS.PLOT_SAMPLES.REQUIRE_CAMERAS True \
           TRAIN_POSTPROCESS.PLOT_SAMPLES.SKIP_IF_UNSUPPORTED_DATASET True
   ```
   预期：`loss_curve` 为 `created`，`samples` 为 `{"status": "skipped", "reason": "camera_panel_unsupported"}`。

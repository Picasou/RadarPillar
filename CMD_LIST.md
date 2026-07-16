# RadarPillar 常用命令

## 0. 环境

```bash
cd /home/dministrator1/RadarPillar
. /home/dministrator1/miniconda3/etc/profile.d/conda.sh
conda activate angle
```

## 1. 数据

确认 pkl 存在：

```bash
ls data/nuscenes/*radar*.pkl
```

需要 `nuscenes_infos_radar_1sweeps_train.pkl` 和 `_val.pkl`。

重新生成：

```bash
python pcdet/datasets/nuscenes/nuscenes_radar_dataset.py --cfg_file <yaml> --version v1.0-mini
```

---

## 2. train.py 完整参数

```bash
python tools/train.py [OPTIONS]
```

### 基础参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--cfg_file` | **必填** | 配置文件路径，如 `tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml` |
| `--batch_size` | 配置文件中值 | 每 GPU batch size |
| `--epochs` | 配置文件中值 | 训练轮数 |
| `--workers` | 8 | DataLoader 并行数 |
| `--extra_tag` | `'default'` | 实验标记，用于区分输出目录 |

### 断点续训

| 参数 | 说明 |
|---|---|
| `--ckpt` | 从指定 checkpoint 继续训练，填 pth 文件路径 |
| `--pretrained_model` | 加载预训练权重（只加载参数，不�� optimizer/scheduler） |

### 分布式训练

| 参数 | 说明 |
|---|---|
| `--launcher` | `none` / `pytorch` / `slurm`，单机多卡用 `pytorch`，集群用 `slurm` |
| `--tcp_port` | 分布式通信端口，默认 18888 |
| `--local_rank` | 分布式 GPU 编号，自动设置 |

### 训练控制

| 参数 | 说明 |
|---|---|
| `--sync_bn` | 开启 SyncBatchNorm（多卡训练用） |
| `--fix_random_seed` | 固定随机种子为 666（可复现） |
| `--ckpt_save_interval` | 每 N 个 epoch 保存一次 checkpoint，默认 1 |
| `--max_ckpt_save_num` | 最多保留多少个 checkpoint，默认 30 |
| `--merge_all_iters_to_one_epoch` | 把所有 iters 合并为一个 epoch（用于 debug） |

### 评估与日志

| 参数 | 说明 |
|---|---|
| `--use_wandb` | 启用 Weights & Biases 记录实验 |
| `--skip_eval` | 训练结束后跳过评估（适合 overfit 等小数据集） |
| `--save_to_file` | 评估结果保存到文件 |
| `--max_waiting_mins` | 评估时最长等待分钟数 |
| `--start_epoch` | 评估起始 epoch |

### 动态配置

| 参数 | 说明 |
|---|---|
| `--set KEY VAL` | 覆盖配置文件中的参数，可多次使用 |

### 示例

```bash
# 单卡训练 80 epoch
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml \
  --batch_size 2 --workers 4 --epochs 80 --extra_tag full_train

# 从断点继续
python tools/train.py --cfg_file <yaml> --ckpt output/.../ckpt/checkpoint_epoch_50.pth

# 临时覆盖配置
python tools/train.py --cfg_file <yaml> --set DATA_CONFIG.MAX_SWEEPS 3

# 多卡训练
python tools/train.py --cfg_file <yaml> --launcher pytorch --batch_size 4
```

---

## 3. test.py 完整参数

```bash
python tools/test.py [OPTIONS]
```

### 基础参数

| 参数 | 说明 |
|---|---|
| `--cfg_file` | **必填** | 配置文件路径 |
| `--batch_size` | 配置文件中值 | 测试 batch size |
| `--workers` | 4 | DataLoader 并行数 |
| `--ckpt` | **必填** | 要评测的 checkpoint 路径 |

### 分布式与评估

| 参数 | 说明 |
|---|---|
| `--launcher` | `none` / `pytorch` / `slurm` |
| `--tcp_port` | 分布式通信端口 |
| `--local_rank` | GPU 编号 |
| `--eval_all` | 评测 ckpt_dir 下所有 checkpoint |
| `--ckpt_dir` | 指定 checkpoint 目录，自动评测目录下所有 epoch |
| `--eval_tag` | 评测结果子目录标记 |
| `--save_to_file` | 保存预测结果 |

### 评估控制

| 参数 | 说明 |
|---|---|
| `--max_waiting_mins` | 评测轮询等待分钟数 |
| `--start_epoch` | 起始 epoch |
| `--set KEY VAL` | 覆盖配置 |

### 示例

```bash
# 单卡测试
python tools/test.py \
  --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml \
  --batch_size 1 --workers 0 \
  --ckpt output/.../checkpoint_epoch_80.pth

# 评测目录下所有 checkpoint
python tools/test.py --cfg_file <yaml> --ckpt_dir output/.../ckpt --eval_all

# 分布式多卡测试
python tools/test.py --cfg_file <yaml> --launcher slurm --ckpt <path> --batch_size 4
```

---

## 4. Slurm 集群脚本

### slurm_train.sh

```bash
bash tools/scripts/slurm_train.sh <PARTITION> <JOB_NAME> <GPUS> [PY_ARGS...]
```

| 参数 | 说明 |
|---|---|
| `PARTITION` | 集群分区名 |
| `JOB_NAME` | 作业名称 |
| `GPUS` | 使用 GPU 数量 |
| `PY_ARGS` | 传给 train.py 的参数 |

环境变量（可预设）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GPUS_PER_NODE` | 8 | 每节点 GPU 数 |
| `CPUS_PER_TASK` | 5 | 每任务 CPU 数 |
| `SRUN_ARGS` | `""` | 其他 srun 参数 |

```bash
# 示例：4 卡训练
bash tools/scripts/slurm_train.sh partition_name exp1 4 \
  --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml \
  --batch_size 8 --epochs 80 --extra_tag slurm_train
```

### slurm_test_single.sh

```bash
bash tools/scripts/slurm_test_single.sh <PARTITION> [PY_ARGS...]
```

单卡测试，自动设置 `GPUS=1`。

```bash
bash tools/scripts/slurm_test_single.sh partition_name \
  --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml \
  --ckpt output/.../checkpoint_epoch_80.pth
```

### slurm_test_mgpu.sh

```bash
bash tools/scripts/slurm_test_mgpu.sh <PARTITION> <GPUS> [PY_ARGS...]
```

| 参数 | 说明 |
|---|---|
| `PARTITION` | 分区名 |
| `GPUS` | GPU 数量，自动设为每节点 GPU 数 |

```bash
bash tools/scripts/slurm_test_mgpu.sh partition_name 4 \
  --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml \
  --ckpt output/.../checkpoint_epoch_80.pth --batch_size 4
```

---

## 5. TensorBoard 查看

训练时自动记录，输出目录：

```text
output/<exp_group>/<tag>/<extra_tag>/tensorboard/
```

启动服务：

```bash
tensorboard --logdir output/ --port 6006
```

浏览器打开：`http://localhost:6006`

查看特定实验：

```bash
tensorboard --logdir output/cfgs/nuscenes_models/radarpillar_nuscenes/full_train/tensorboard/
```

---

## 6. 1-Batch Overfit

使用专用脚本：

```bash
python /mnt/c/Users/Administrator/Documents/openDet/train_and_plot_overfit_one_batch.py \
  --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes_overfit1.yaml \
  --iters 1200 --lr 0.003 --score_thresh 0.1 --nms_post_maxsize 80 \
  --out_dir output/overfit1_plot
```

输出：`output/overfit1_plot/overfit1_radar_gt_pred.png`

> batch_size=1 ≠ 1-batch overfit，必须 dataset_len=1。

---

## 7. 查看训练状态

### Loss 日志

```bash
grep -E "loss|recall|Average predicted" output/cfgs/nuscenes_models/radarpillar_nuscenes/*/log_train_*.txt
```

### result.pkl

```bash
python - <<'PY'
import pickle
with open('result.pkl路径', 'rb') as f: data = pickle.load(f)
print('samples:', len(data))
a = data[0]
print('pred:', len(a.get('name',[])), 'scores:', a.get('score',[])[:5])
PY
```

> `OUTPUT_RAW_SCORE=True` 时 score 是 raw logit，可能为负。

---

## 8. 排查

| 现象 | 重点查 |
|---|---|
| 无预测框 | valid_gt>0、loss 下降、cls score 升高、SCORE_THRESH/NMS_POST_MAXSIZE |
| 预测框过多 | SCORE_THRESH 太低、NMS_POST_MAXSIZE 太大、OUTPUT_RAW_SCORE 误读 |
| GT 与点云不重合 | global→ego 方向、sensor→ego 方向、sweep transform、velocity 旋转 |

---

## 9. 合格标准

**Overfit**：loss 下降、recall≈GT 数、NMS 后有框、top score 升高、预测框≈GT

**正式训练**：loss 下降、val 有预测、可视化对齐、mAP/NDS>0

---

## 10. Post-Train Visualization

`TRAIN_POSTPROCESS` 控制训练后自动画图，默认关闭。overfit1 配置已默认开启。

```yaml
TRAIN_POSTPROCESS:
    ENABLE: True
    PLOT_LOSS_CURVE: True
    PLOT_SAMPLES:
        ENABLE: True
        REQUIRE_CAMERAS: True
        SKIP_IF_UNSUPPORTED_DATASET: True
```

输出：`output/.../post_train_artifacts/loss_curve.png`
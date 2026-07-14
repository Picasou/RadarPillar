#!/bin/bash

#  两模型对比可视化: 同时跑 VAL + TRAIN 两个 split.
#
#  用法:
#      bash tools/scripts/compare_ckpt.sh
#
#  必改: CKPT1 / CKPT2 / EXTRA1 / EXTRA2  (见下方对照区).
#
#  依赖:
#      - 两份 result.pkl 已存在 (用 tools/scripts/eval_vod.sh 各跑一遍).
#      - tools/utils/visual_utils/compare_two_models.py 可调用.

set -e

# ════════════════════════════════════════════════════════════════
#  必改 (两份模型各自的 ckpt + 对应 result.pkl)
# ════════════════════════════════════════════════════════════════
# val 与 train 各自的 extra_tag (train 用 *_train 后缀避免与 val 的输出混在同目录)
EXTRA1="base"                          # val extra_tag (model1)
EXTRA1_TRAIN="base_train"              # train extra_tag (model1)
EXTRA2="radarpiller_0709"              # val extra_tag (model2)
EXTRA2_TRAIN="radarpiller_0709_train"  # train extra_tag (model2)

OUTPUT_ROOT1="output/cfgs/model/vod_models/vod_radarpillar/${EXTRA1}"
OUTPUT_ROOT2="output/cfgs/model/vod_models/vod_radarpillar/${EXTRA2}"
OUTPUT_ROOT1_TRAIN="output/cfgs/model/vod_models/vod_radarpillar/${EXTRA1_TRAIN}"
OUTPUT_ROOT2_TRAIN="output/cfgs/model/vod_models/vod_radarpillar/${EXTRA2_TRAIN}"

# val 上的 result.pkl
VAL_PKL1="${OUTPUT_ROOT1}/eval/epoch_56/val/default/result.pkl"
VAL_PKL2="${OUTPUT_ROOT2}/eval/epoch_100/val/default/result.pkl"

# train 上的 result.pkl (需要先用 tools/test.py 在 train split 上跑一次)
TRAIN_PKL1="${OUTPUT_ROOT1_TRAIN}/eval/epoch_56/train/default/result.pkl"
TRAIN_PKL2="${OUTPUT_ROOT2_TRAIN}/eval/epoch_100/train/default/result.pkl"

NAME1="best_map52.56"                  # 图例里的 model1 名字
NAME2="ckpt_epoch_100"                 # 图例里的 model2 名字

DATAROOT="./data/VoD/view_of_delft_PUBLIC/radar_5frames"

OUTPUT_DIR="output/radarpiller_compare"

# ════════════════════════════════════════════════════════════════
#  可选参数 (一般不用改)
# ════════════════════════════════════════════════════════════════
N_SAMPLES=20                  # 每个 split 采样帧数
SCORE_THRESHOLDS="0.1 0.3"    # score 阈值列表 (空格分隔)

# 点云颜色 (uniform | rcs | doppler)
POINT_COLOR_MODE="uniform"
DOPPLER_FIELD="v_r"            # 仅 doppler 模式生效

# ════════════════════════════════════════════════════════════════
#  执行
# ════════════════════════════════════════════════════════════════
cd "$(dirname "$0")/../.."
source /home/dministrator1/miniconda3/etc/profile.d/conda.sh
conda activate angle

mkdir -p "${OUTPUT_DIR}"
rm -f "${OUTPUT_DIR}"/*.png

# ---- val ----
python -u tools/utils/visual_utils/compare_two_models.py \
    --ckpt1_result "${VAL_PKL1}" \
    --ckpt2_result "${VAL_PKL2}" \
    --name1 "${NAME1}" --name2 "${NAME2}" \
    --dataroot "${DATAROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_samples "${N_SAMPLES}" \
    --score_thresholds ${SCORE_THRESHOLDS} \
    --split val \
    --point_color_mode "${POINT_COLOR_MODE}" \
    $( [ "${POINT_COLOR_MODE}" = "doppler" ] && echo "--doppler_field ${DOPPLER_FIELD}" ) \
    --seed 42

# ---- train ----
if [ -f "${TRAIN_PKL1}" ] && [ -f "${TRAIN_PKL2}" ]; then
    python -u tools/utils/visual_utils/compare_two_models.py \
        --ckpt1_result "${TRAIN_PKL1}" \
        --ckpt2_result "${TRAIN_PKL2}" \
        --name1 "${NAME1}" --name2 "${NAME2}" \
        --dataroot "${DATAROOT}" \
        --output_dir "${OUTPUT_DIR}" \
        --n_samples "${N_SAMPLES}" \
        --score_thresholds ${SCORE_THRESHOLDS} \
        --split train \
        --point_color_mode "${POINT_COLOR_MODE}" \
        $( [ "${POINT_COLOR_MODE}" = "doppler" ] && echo "--doppler_field ${DOPPLER_FIELD}" ) \
        --seed 42
else
    echo "[warn] train result.pkl missing, skip train split:"
    echo "       ${TRAIN_PKL1}"
    echo "       ${TRAIN_PKL2}"
    echo "       (先用 tools/test.py 在 train split 上各跑一次两个模型)"
fi

echo ""
echo "DONE -> ${OUTPUT_DIR}/"
ls "${OUTPUT_DIR}" | head

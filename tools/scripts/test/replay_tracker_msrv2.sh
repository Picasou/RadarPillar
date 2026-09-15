#!/usr/bin/env bash
# MSRv2 三模型 检测(mode1)/跟踪(mode2) 回灌 runner (自含: 可被 Claude 会话或 OS cron 兜底触发)
# 源(只读): /mnt/d/DataSet/{.trained,.generalization}/v2/.base → 每模型×每集×每mode 一份拷贝
# 依赖: 各模型训练整链完成(output/train_log/msr/<ts>_rpillar_<tag>_<tag>/<tag>.done + resbag/)
# 幂等: 拷贝目录 .replay_done 标记跳过; 全局锁 /tmp/msrv2_stage3.running(mkdir 原子); 完成 /tmp/msrv2_stage3.done
# 用法: bash tools/scripts/test/replay_tracker_msrv2.sh [model_tag ...]   (缺省全部 3 个)
set -uo pipefail
cd "$(dirname "$0")/../../.."   # 工程根

# conda 自探测 (env=base 本机唯一环境)
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _c in "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        [ -f "$_c/etc/profile.d/conda.sh" ] && { source "$_c/etc/profile.d/conda.sh"; break; }
    done
fi
conda activate base

LOCK=/tmp/msrv2_stage3.running
DONE_ALL=/tmp/msrv2_stage3.done
LOG_DIR=.tmp/replay_msrv2
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/replay_$(date +%Y%m%d-%H%M%S).log"

mkdir "$LOCK" 2>/dev/null || { echo "[stage3] 已有实例在跑 ($LOCK), 退出"; exit 0; }
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(msr_pp64_ch2d_trunk msr_ppmix_ch2d_trunk msr_radarpillar)

echo "[stage3] start $(date)  models=${MODELS[*]}" | tee -a "$LOG"
FAIL=()

for model in "${MODELS[@]}"; do
    # 训练整链完成核验: root(marker 所在) + resbag 产物
    root="$(ls -1d output/train_log/msr/*_rpillar_${model}_${model} 2>/dev/null | tail -n1)"
    if [ -z "$root" ] || [ ! -f "${root}/${model}.done" ] || [ ! -f "${root}/resbag/best.pth" ]; then
        echo "[stage3] WARN: ${model} 训练未完成(root=${root:-无}), 跳过该模型" | tee -a "$LOG"
        FAIL+=("${model}:训练未完成")
        continue
    fi
    for src in .trained .generalization; do
        base="/mnt/d/DataSet/${src}/v2/.base"
        for mode in 1 2; do
            target="/mnt/d/DataSet/${src}/v2/${model}/mode${mode}"
            short="$([ "$src" = ".trained" ] && echo trained || echo gen)"
            cfg="tracker/cfg/cfg_v2_${short}_${model}_mode${mode}.yaml"
            echo "=== [stage3] ${model} ${short} mode${mode} → ${target}" | tee -a "$LOG"
            if [ -f "${target}/.replay_done" ]; then
                echo "    已完成, 跳过" | tee -a "$LOG"; continue
            fi
            # 重入清理: 只清本任务自建的拷贝目录, .base 永不触碰
            rm -rf "$target"
            mkdir -p "$target"
            cp -a "${base}/." "${target}/" || { echo "    cp 失败" | tee -a "$LOG"; FAIL+=("${model}/${short}/mode${mode}:cp"); continue; }
            nseq=$(find "$target" -mindepth 1 -maxdepth 1 -type d | wc -l)
            echo "    拷贝 ${nseq} 序列" | tee -a "$LOG"
            python tools/scripts/test/gen_replay_cfg_msrv2.py \
                --base "tracker/cfg/cfg_trained_${model}.yaml" \
                --mode "$mode" --out "$cfg" \
                --model-cfg "$(pwd)/${root}/resbag/cfg.yaml" \
                --model-ckpt "$(pwd)/${root}/resbag/best.pth" \
                --seq-root "$target" 2>&1 | tee -a "$LOG" || { echo "    gen cfg 失败" | tee -a "$LOG"; FAIL+=("${model}/${short}/mode${mode}:gencfg"); continue; }
            python -u -m tracker.tracker --cfg "$cfg" 2>&1 | tee -a "$LOG" || { echo "    tracker 失败" | tee -a "$LOG"; FAIL+=("${model}/${short}/mode${mode}:tracker"); continue; }
            touch "${target}/.replay_done"
            echo "    ✓ done" | tee -a "$LOG"
        done
    done
done

if [ ${#FAIL[@]} -eq 0 ]; then
    touch "$DONE_ALL"
    echo "[stage3] ALL DONE $(date)" | tee -a "$LOG"
else
    echo "[stage3] 部分失败: ${FAIL[*]} (重跑本脚本可续)" | tee -a "$LOG"
fi

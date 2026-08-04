#!/bin/bash
# run_all_repdwc.sh — RepDWC concat 公平对照: 5 模型串行 driver (long-term-task-plan 编排层)
#
# 职责 (修复对抗审查致命缺陷):
#   1. 确定性串行 for-loop (不依赖会话存活逐个发 nohup)
#   2. 每模型传正确三元组 MODEL/CFG_FILE/EXTRA_TAG (避免全跑成 a4)
#   3. 失败 || 兜住 continue (OOM/NaN/crash 不卡死链路)
#   4. 每模型收尾 checkpoint.py save (会话崩溃也能恢复)
#   5. 启动 oracle: best.pth + model_store.yaml 都在 = 已完成, 跳过
#
# 用法 (前台):
#   bash .claude/skills/model-train/scripts/run_all_repdwc.sh
# 后台 (脱离终端, 会话死也继续):
#   setsid nohup bash .claude/skills/model-train/scripts/run_all_repdwc.sh \
#     > /tmp/run_all_repdwc.log 2>&1 < /dev/null &

set -uo pipefail   # 不用 -e, 让单模型失败被兜住后 continue

cd "$(dirname "$0")/../../../.."   # -> 仓库根 (.claude/skills/model-train/scripts/ 上 4 级)

TASK="train-repdwc-concat-b5-b9"

# 顺序: 小容量优先, 大容量 (b7 concat=448) 最后
# b9 训练已完成, eval/pickbest/resbag 单独补全后用 oracle 跳过; 此队列从 b5 起跑
TAGS=(b5 b6 b8 b7)
N=${#TAGS[@]}

# 每 tag 的 CFG_FILE — 决不可漏, 否则全跑成 a4 (MODEL 仅影响路径/resbag tag)
declare -A CFG=( [b9]=experiments/YAML/b9.yaml [b5]=experiments/YAML/b5.yaml \
                 [b6]=experiments/YAML/b6.yaml [b8]=experiments/YAML/b8.yaml \
                 [b7]=experiments/YAML/b7.yaml )

echo "[driver] start $(date)  tags=${TAGS[*]}"

for i in $(seq 0 $((N-1))); do
    tag=${TAGS[$i]}
    if [ $((i+1)) -lt $N ]; then nxt=${TAGS[$((i+1))]}; else nxt="done"; fi
    echo "========================================================"
    echo "[driver] === 模型 $tag (next=$nxt)  $(date)"

    # ---- oracle: 已完成则跳过 (幂等恢复) ----
    OR=$(ls -td output/train_log/vod/*_"${tag}"_"${tag}" output/train_log/vod/*_"${tag}" 2>/dev/null | head -1)
    if [ -n "$OR" ] && [ -f "$OR/best.pth" ] && [ -f "$OR/model_store.yaml" ]; then
        echo "[driver] $tag 已完成 (oracle 命中 $OR), 跳过"
        python .claude/skills/long-term-task-plan/checkpoint.py save \
            --task "$TASK" --stage "训练${tag}" \
            --artifact "output=$OR" \
            --result "status=skipped_already_done" \
            --next_start "训练${nxt}" \
            --note "oracle 命中, 跳过" >/dev/null 2>&1
        continue
    fi

    # ---- 跑 unified pipeline (前台; set +e 兜住失败) ----
    # MODEL=rpillar_<tag> 匹配 experiments/SH/train_rpillar_<tag>.sh (resbag train.sh 源)
    set +e
    MODEL=rpillar_${tag} \
    CFG_FILE="${CFG[$tag]}" \
    EXTRA_TAG=${tag} \
    EPOCHS=80 BATCH_SIZE=16 GPU=0 \
    LAST_N_EVAL=10 RUN_VIZ=false \
    RUN_PICKBEST=true RUN_RESBAG=true \
        bash .claude/skills/model-train/scripts/pipeline.sh
    RC=$?
    set -e 2>/dev/null || true   # 恢复 (本脚本顶层未设 -e, 此行 no-op)

    # ---- 收尾: 无论成败都 checkpoint save (不依赖会话存活) ----
    NEWOR=$(ls -td output/train_log/vod/*_"${tag}"_"${tag}" 2>/dev/null | head -1)
    if [ $RC -eq 0 ] && [ -n "$NEWOR" ] && [ -f "$NEWOR/model_store.yaml" ]; then
        STATUS="ok"
        echo "[driver] $tag 成功  OUTPUT_ROOT=$NEWOR"
    else
        STATUS="fail_rc${RC}"
        echo "[driver] $tag 失败 (rc=$RC), 记档后继续下一个"
    fi
    python .claude/skills/long-term-task-plan/checkpoint.py save \
        --task "$TASK" --stage "训练${tag}" \
        --artifact "output=${NEWOR:-none}" \
        --result "status=${STATUS}" \
        --next_start "训练${nxt}" \
        --note "rc=$RC" >/dev/null 2>&1
done

echo "========================================================"
echo "[driver] ALL MODELS DONE  $(date)"
echo "[driver] 跨实验总览:"
python .claude/skills/resbag/resbag.py list --dataset vod 2>/dev/null || true

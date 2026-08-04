#!/bin/bash
# scripts/verify_hardening.sh — 自检: 5 类硬化是否都已在代码中
#
# 用法: bash scripts/verify_hardening.sh
# 输出: 每项 ✓ / ✗ + 总分

set -uo pipefail
SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

PASS=0
FAIL=0
check() {
    local name="$1" file="$2" pattern="$3"
    if grep -qE "$pattern" "$SKILL_DIR/$file" 2>/dev/null; then
        echo "  ✓ $name"
        PASS=$((PASS+1))
    else
        echo "  ✗ $name  ($file 缺: $pattern)"
        FAIL=$((FAIL+1))
    fi
}

echo "=== H1: pipeline.sh 内置 retry ==="
check "RETRY_MAX env"        scripts/pipeline.sh 'RETRY_MAX'
check "retry for loop"       scripts/pipeline.sh 'for try in'
check "NaN 守卫 regex"       scripts/pipeline.sh 'NAN_PATTERN'
check "OOM 守卫 regex"       scripts/pipeline.sh 'OOM_PATTERN'
check "DONE_MARKER 检查"     scripts/pipeline.sh 'DONE_MARKER'

echo ""
echo "=== H2: watchdog cron 自我守护 ==="
check "pgrep cron"           helpers/watchdog.sh 'pgrep -x cron'
check "systemctl 重启"       helpers/watchdog.sh 'systemctl restart cron'
check "日志记录"             helpers/watchdog.sh 'cron 死了'

echo ""
echo "=== H3: brief GPU 显存告警 ==="
check "nvidia-smi 调用"      scripts/brief.sh 'nvidia-smi'
check "显存阈值 7000"        scripts/brief.sh '7000'

echo ""
echo "=== H4: done_notifier marker 校验 ==="
check "MARKERS_FILE 参数"    helpers/done_notifier.sh 'MARKERS_FILE'
check "marker 缺失检测"      helpers/done_notifier.sh 'MISSING'
check "complete 状态"        helpers/done_notifier.sh 'complete'
check "partial 状态"         helpers/done_notifier.sh 'partial'
check "POST_HOOK 触发"       helpers/done_notifier.sh 'POST_HOOK'

echo ""
echo "=== H5: generate_workflow.sh 语法检查 ==="
check "bash -n 检查"         scripts/generate_workflow.sh 'bash -n'

echo ""
echo "=== 通用: 4 类机制文件存在 ==="
check "tmux_spawn"           helpers/tmux_spawn.sh '^# helpers/tmux_spawn.sh'
check "watchdog"             helpers/watchdog.sh '^# helpers/watchdog.sh'
check "brief"                scripts/brief.sh '^# scripts/brief.sh'
check "done_notifier"        helpers/done_notifier.sh '^# helpers/done_notifier.sh'
check "pipeline"             scripts/pipeline.sh '^# scripts/pipeline.sh'
check "generate_workflow"    scripts/generate_workflow.sh '^# scripts/generate_workflow.sh'

echo ""
echo "=== 总分: $PASS ✓ / $FAIL ✗ ==="
[ "$FAIL" -eq 0 ] && echo "✓ 全部就绪" || echo "✗ 有缺失, 查上方 ✗ 行"
exit $FAIL
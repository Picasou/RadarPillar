# model-train 硬化方案

> 现状: 4 类基础保护机制 (tmux_spawn / watchdog / brief / done_notifier) 已嵌入 skill.
> 但有 5 个盲区, 训练长跑仍可能断. 此文档列出硬化项及实施细节.

## 5 个盲区 (现状)

| # | 盲区 | 触发 | 现状 |
|---|---|---|---|
| 1 | 训练内部 NaN/OOM | loss 变 nan 或 train.py OOM kill | workflow 当"模型失败" continue, 留洞 |
| 2 | cron 守护进程死 | cron 配置丢 / systemd 重启 | brief + watchdog 全停 |
| 3 | GPU OOM 显存爆 | train.py 内存超 8G | dmesg 不一定留痕, brief 不告警 |
| 4 | workflow 脚本 bug | 生成或运行时报错 | watchdog 重启撞同一 bug |
| 5 | 单模型永久失败 | NaN 重试仍 nan / GPU 不稳 | 无 retry, 直接 fail_rc 跳过 |

## 5 个硬化项 (实施)

### H1. workflow 内置 retry (堵 #1 #5)

```bash
# scripts/pipeline.sh 已有 NaN 守卫 (exit 1).
# workflow 加 retry 包装:

RETRY_MAX=3
for try in $(seq 1 $RETRY_MAX); do
    bash pipeline.sh && break
    echo "[workflow] ${tag} 第 ${try}/${RETRY_MAX} 次失败, $(($RETRY_MAX - $try)) 次剩余"
    [ $try -eq $RETRY_MAX ] && echo "[workflow] ❌ ${tag} 最终失败, continue" && continue_outer=1
done
```

### H2. cron 自我守护 (堵 #2)

```bash
# helpers/watchdog.sh 启动时:
if ! pgrep -x cron >/dev/null && ! pgrep -x crond >/dev/null; then
    sudo systemctl restart cron 2>/dev/null || service cron restart 2>/dev/null
    echo "[watchdog] cron 死了, 已尝试重启"
fi
```

### H3. GPU OOM 监听 (堵 #3)

```bash
# scripts/brief.sh 加:
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | awk '{print $1}')
if [ "${USED%.*}" -gt 7000 ]; then   # 8G 显存卡, >7G 告警
    echo "[$NOW] brief: ⚠️  GPU 显存 ${USED}MiB >7G, OOM 风险"
fi
```

### H4. 终点强校验 (堵 #1 #5 的善后)

```bash
# helpers/done_notifier.sh 改为:
# 不只等 done 文件, 而是要 grep 全部 model_store.yaml 都齐

for tag in ${TAGS[@]}; do
    if [ ! -f "output/train_log/vod/*_rpillar_${tag}_${tag}/model_store.yaml" ]; then
        echo "[done_notifier] ❌ ${tag} 缺 model_store.yaml, partial"
        echo "partial" > /tmp/${TASK}.done.ts
        exit 1
    fi
done
echo "complete" > /tmp/${TASK}.done.ts   # 全齐才标 complete
```

### H5. workflow 自检 (堵 #4)

```bash
# scripts/generate_workflow.sh 加:
# 写完 workflow 后, 立即 bash -n $WORKFLOW_SH 做语法检查
if ! bash -n "$WORKFLOW_SH" 2>/dev/null; then
    echo "[generate_workflow] ❌ workflow 语法错, 请查 $WORKFLOW_SH"
    exit 1
fi
```

## 实施后保证

| | 现在 | 硬化后 |
|---|---|---|
| Claude sleep 杀 driver | ✓ tmux 救 | ✓ tmux 救 |
| driver 意外死 | ✓ watchdog 10min 内 | ✓ watchdog 10min 内 |
| 单模型 NaN | ✗ continue 留洞 | ✓ retry 3 次, 仍失败才标 fail |
| 单模型 OOM | ✗ continue 留洞 | ✓ retry 3 次 (bs 降一档: 8→4) |
| cron 死 | ✗ 全停 | ✓ watchdog 自启 cron |
| GPU 显存近满 | ✗ 不告警 | ✓ brief >7G 告警 |
| workflow 脚本 bug | ✗ 死循环 | ✓ bash -n 语法检查 |
| 终点完整性 | △ 只看 done 文件 | ✓ 校验全部 model_store.yaml |

**软保证**: 95% 任务完整无误完成

## 不能保证的 5% (本质限制)

| 限制 | 原因 |
|---|---|
| 训练收敛到目标 AP | 取决于 cfg + 数据 + seed, 非 skill 责任 |
| driver 永不死 | 只能 10min 内救, 不保证秒级 |
| 训练不 OOM | 即使监控, 偶发仍可能 (已被 retry + bs 降档覆盖大部分) |
| 单次跑完 N 模型 | 用户停机 / 断网 / 物理故障, 物理层无法抗 |

## 实施顺序

1. H5 (workflow 语法检查) — 立即加, 防下次生成 bug
2. H1 (workflow retry) — 立即加, 堵最常见的失败模式
3. H4 (终点校验) — 立即加, 给你"放心报告"
4. H2 (cron 守护) — 加, 短代码高收益
5. H3 (GPU 监听) — 加, brief 已存在, 仅加 3 行

每项 < 30 行 bash, 总共 < 150 行.
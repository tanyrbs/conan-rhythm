# Conan Rhythm Branch

当前分支聚焦两条线：

1. **maintained teacher-first 主线**
   - `teacher_offline`
   - `student_kd`
   - `student_retimed`
2. **更高上限的 experimental extension**
   - `student_ref_bootstrap`（stage-2.5 external-reference bootstrap）

## 当前 AutoDL 真实状态（2026-04-06 UTC）

- `train100` formal binary 已完成：`data/binary/libritts_train100_formal_rhythm_v5`
- Stage-1 warm up 已完成到 **20k**：
  - `checkpoints/teacher_offline_train100_warmup/model_ckpt_steps_20000.ckpt`
- 最新可信 validation（step `19000`）显示：
  - `exec_total_corr = 0.8907`
  - `pause_event_f1 = 0.5957`
  - `prefix_drift_l1 = 25.7781`
  - `budget_projection_repair_ratio_mean = 0.0`
- `export smoke` 已通过
- 当前正在恢复 `train360` metadata shards，接下来会切到：
  - `train360 train-only binary`
  - `train100|train360` mixed formal Stage-1

## 当前最重要的结论

- 现在**不是**继续折腾 `train100 warm up`
- 现在的正确动作是：
  1. 完成 `train360` metadata 恢复
  2. 接管为 `train360 train-only binary`
  3. 用 `warmup 20k ckpt` 启动新的 `100+360` formal teacher
- `student_kd` 只是**后移**，不是取消
- `student_ref_bootstrap` 是 teacher 之后、用于冲上限的 stage-2.5，不是现在直接替代 teacher-first 主链

## 推荐阅读顺序

1. `docs/autodl_train100_formal_progress.md`
2. `docs/autodl_train100_formal_quickstart.md`

## 与最新主线相比，本地额外强化了什么

在吸收最新 `origin/main` external-reference bootstrap 改动后，本地额外补了：

- `student_ref_bootstrap` 的显式 stage 识别与 contract fail-fast
- `sample_ref/external + cached_only/prefer_cache` 的监督冲突硬错误
- runtime external-reference fail-fast 与 reference self/external observability
- Stage-3 acoustic scalar ramp，避免 early `student_retimed` 被 `L_base` 完全淹没
- streaming audit 中更可靠的 phase non-retro 统计

## 当前最短操作命令

```bash
cd /root/autodl-tmp/project-1/conan-rhythm
tail -f logs/live_recovery_monitor.log
# 等 train360 metadata 完成

bash logs/stage1_takeover_from_existing_train360_metadata_trainonly.sh
# 接管 metadata -> train-only binary -> mixed preflight/smoke -> formal Stage-1
```

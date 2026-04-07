# Conan Rhythm Branch

## 当前唯一维护中的训练路线

虽然仓库里仍保留多种 stage 配置，但**当前项目计划已经切成 teacher-first 路线**：

1. `teacher_offline` Stage 1（当前正在跑）
2. teacher Stage 2（同合同 teacher polish / upper-bound continuation）
3. teacher v2.5（强节奏变化 / external-reference teacher 增强）
4. teacher v3（鲁棒性 / cache 刷新 / 必要时 soft-boundary 重建）
5. 固化并归档各 teacher 阶段最佳 checkpoint / audit 结果 / export 资产
6. **最后**再进入学生蒸馏，优先考虑 `student_kd`
7. `student_ref_bootstrap`、`student_retimed` 现在都不是眼前主线

> 也就是说：**先把 teacher 提到上限，再做学生。**

---

## 当前状态（2026-04-07 UTC）

### 已有资产

- `train100` processed：`data/processed/libritts_train100_formal`
- `train100` binary：`data/binary/libritts_train100_formal_rhythm_v5`
- `train360` processed：`data/processed/libritts_train360_formal_trainset`
- `train360` binary：`data/binary/libritts_train360_formal_trainset_rhythm_v5`
- `train100` warmup ckpt：`checkpoints/teacher_offline_train100_warmup/model_ckpt_steps_20000.ckpt`

### 当前正在跑什么

- mixed `train100 + train360` 的 `teacher_offline` formal Stage 1
- 启动脚本：`scripts/autodl_recovery_mixed100360_v3_trainonly.sh`
- 状态文件目前只会停在：
  - `logs/stage1_recovery_mixed100360_v3_trainonly.status`
  - `FORMAL_STAGE1_RUN_START`
- **真实进度要看训练日志 / ckpt**，不要只看 status 文件

### 当前最新已观察到的有效验证窗口

来自 `logs/stage1_recovery_mixed100360_v3_trainonly.log`：

- step `65000`
- `total_loss = 0.2219`
- `L_exec_pause = 0.1138`
- `L_prefix_state = 0.0337`
- `rhythm_metric_pause_event_precision = 0.6692`
- `rhythm_metric_pause_event_recall = 0.5904`
- `rhythm_metric_pause_event_f1 = 0.5861`
- `rhythm_metric_prefix_drift_l1 = 23.5898`
- `rhythm_metric_exec_total_corr = 0.8873`
- `rhythm_metric_budget_projection_repair_ratio_mean = 0.0`
- `L_base = 0.0`
- `L_pitch = 0.0`

结论：

- Stage 1 还在继续
- 阶段契约目前是对的
- 但还没有到“teacher 已经定稿，可以开始学生蒸馏”的时点

---

## 为什么现在不先做 student

因为当前最重要的不是“尽快蒸一个 student”，而是：

- 先把 teacher 的 pause / prefix / 强节奏变化学稳
- 先把 teacher 的听感和控制上限做出来
- 先把 teacher 的 target 语义做干净、做稳定

否则后面学生蒸馏只会把一个还不够成熟的 teacher 更快复制一遍。

---


## cache / binary 注意事项

Repository head 现在把 soft-boundary `ref_rhythm_trace` 语义记为 **cache version 6**。

这意味着：

- 当前**已经在跑**的 Stage 1 进程不受影响
- 但未来任何基于当前代码头的新 `cached_only` 启动，都不能再假装沿用旧 `v5` binary
- 如果要真正声明 soft-boundary 生效，必须先恢复 raw，再重建 binary/cache

## 当前文档入口

1. `README.md`
2. `docs/autodl_training_handoff.md`
3. `docs/training_plan.md`

这 3 份是当前唯一维护的训练文档。

---

## 当前建议命令

启动/续跑当前 teacher Stage 1：

```bash
cd /root/autodl-tmp/project-1/conan-rhythm
bash scripts/autodl_recovery_mixed100360_v3_trainonly.sh
```

监控当前 teacher Stage 1：

```bash
cat logs/stage1_recovery_mixed100360_v3_trainonly.status
tail -f logs/stage1_recovery_mixed100360_v3_trainonly.log
python scripts/monitor_stage1_metrics.py \
  --log logs/stage1_recovery_mixed100360_v3_trainonly.log \
  --tail 5
```

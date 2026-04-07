# AutoDL formal 训练进度记录（2026-04-07 UTC）

> 这份文档保留 train100 warmup 证据，但已经更新为当前真实总状态；不要再把它理解成“train360 还在做 metadata shards”。

## 1. 当前真实进度

### 已完成

- `train100` formal processed / binary：完成
- `teacher_offline_train100_warmup`：完成到 `20k`
- `train360` formal processed / binary：完成
- export smoke：已通过

### 尚未完成

- mixed `train100 + train360` formal stage-1：**未正式启动成功**
- `student_ref_bootstrap`：未开始
- `student_retimed`：未开始
- `student_kd`：未开始（当前不作为默认下一步）

## 2. warmup 关键结论

最新可信 validation 点仍是：

- `step=19000`
- `exec_total_corr = 0.8907`
- `pause_event_f1 = 0.5957`
- `prefix_drift_l1 = 25.7781`
- `budget_projection_repair_ratio_mean = 0.0`

结论：

- warmup 可以作为 mixed stage-1 的 warm-start
- 当前不该继续围绕 train100 warmup 本身反复折腾

## 3. train360 这轮到底发生了什么

已经不是“还在跑 train360 metadata shards”。

真实情况是：

- `data/processed/libritts_train360_formal_trainset` 已生成
- `data/binary/libritts_train360_formal_trainset_rhythm_v5` 已生成
- v2 恢复脚本失败在 mixed preflight dry-run，而不是 train360 binarize

失败证据：

- `logs/stage1_recovery_mixed100360_v2_20260406_091257.status`
- `logs/stage1_recovery_mixed100360_v2_20260406_091257.log`

关键错误：

- `not enough values to unpack (expected 3, got 2)`

## 4. 当前真正 blocker

当前 blocker 是：

- mixed preflight / runtime bug

不是：

- train360 数据没准备好
- train100 数据没准备好
- 磁盘不够继续 stage-1（当前 binary 已保住，raw 已清掉）

## 5. 当前磁盘策略

已经删掉：

- `train-clean-100` raw
- `train-clean-360` raw
- train360 shards 中间产物

当前保留：

- `train100` binary
- `train360` binary
- `dev-clean`
- `test-clean`

因此现在的策略是：

- **不要重做 train100 / train360 binarize**
- **直接复用现有 binary 开 mixed stage-1**

## 6. 下一步唯一正确动作

```bash
cd /root/autodl-tmp/project-1/conan-rhythm
bash scripts/autodl_recovery_mixed100360_v3_trainonly.sh
```

然后监控：

```bash
tail -f logs/stage1_recovery_mixed100360_v3_trainonly.status
```

## 7. 当前项目优先训练顺序

1. mixed `teacher_offline` formal stage-1
2. `student_ref_bootstrap`
3. `student_retimed`
4. `student_kd`（可选 baseline / ablation / 稳定性支线）

## 8. 为什么不是先 `student_kd`

批判性判断：

- `student_kd` 更偏 cached/self-conditioned teacher surface 的稳定蒸馏
- 当前最核心的问题是 external-ref 是否真的被使用
- 所以 teacher 之后，当前更该优先做的是 `student_ref_bootstrap`

但也要说严谨：

- 当前 v2.5 仍然使用 `algorithmic` target source
- 所以它更准确地说是 teacher warm-start + external-ref bootstrap
- 不是最终摆脱 algorithmic ceiling 的终局版本

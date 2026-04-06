# AutoDL 当前可执行 quickstart

更新日期：2026-04-06 UTC

## 1. 当前项目状态

### 已完成

- `train100` formal processed/binary 已完成
- stage-1 warmup 已完成到 **20000**
- warmup export smoke 已完成

关键路径：

- warmup ckpt：
  - `checkpoints/teacher_offline_train100_warmup/model_ckpt_steps_20000.ckpt`
- train100 processed：
  - `data/processed/libritts_train100_formal`
- train100 binary：
  - `data/binary/libritts_train100_formal_rhythm_v5`

### 进行中

- `train360` metadata 分片恢复中
- 当前后台脚本：
  - `logs/stage1_recovery_mixed100360_v2_20260406_091257.sh`

### 接下来

- 分片完成后切到：
  - `logs/stage1_takeover_from_existing_train360_metadata_trainonly.sh`
- 目标是开启 formal：
  - `teacher_offline_train100_360_stage1`

## 2. 现在最重要的命令

### 2.1 看 train360 恢复状态

```bash
cd /root/autodl-tmp/project-1/conan-rhythm
cat logs/stage1_recovery_mixed100360_v2_20260406_091257.status
tail -n 40 logs/live_recovery_monitor.log
```

### 2.2 分片完成后接管

```bash
cd /root/autodl-tmp/project-1/conan-rhythm
nohup bash logs/stage1_takeover_from_existing_train360_metadata_trainonly.sh \
  > logs/stage1_takeover_from_existing_train360_metadata_trainonly.log 2>&1 < /dev/null &
```

这个 takeover 会自动做：

1. merge train360 metadata shards
2. train360 `train-only` binarize
3. train360 preflight
4. mixed `train100|train360` preflight
5. mixed real smoke
6. formal stage-1 启动

## 3. 正式 Stage-1 训练语义

当前不是生成一个统一超大 binary，而是：

- `binary_data_dir = train100`
- `train_sets = train100|train360`

即：

- valid/test 继续锚定在 `train100`
- train360 只作为额外训练集暴露量

这条路在当前磁盘条件下是工程上更合理的折中。

## 4. 为什么 train360 改成 train-only

原因：

- mixed validation/test 仍然走 `train100` 的 `valid/test`
- train360 在 formal stage-1 里只通过 `train_sets` 提供训练样本

所以：

- **不需要**再给 train360 单独构建 `valid/test`
- 这样能省磁盘、省时间，也不影响当前 formal Stage-1 训练

## 5. warmup 是否正常

以 `19000` valid 为准，当前判断是 **正常**：

- 三个 stage flag 都是 `1.0`
- `exec_total_corr = 0.8907`
- `pause_event_f1 = 0.5957`
- `prefix_drift_l1 = 25.7781`
- `budget_projection_repair_ratio_mean = 0.0`

注意：

- `phase_nonretro_rate` 目前不适合作为标准 val 的硬 gate
- 真要看它，应看 chunkwise / streaming 审计

## 6. 升级后的训练路线

### 默认正式路线

先做强 teacher：

1. `teacher_offline`
2. `student_kd`
3. `student_retimed`

### 更高上限实验路线

在 `student_kd` 和 `student_retimed` 之间，可插：

4. `student_pairwise_ref_runtime_teacher`
   - alias：`student_ref_bootstrap`

它和旧主线的关键区别是：

- 不再只用 self-conditioned cached rhythm target
- 改成 external same-speaker ref 驱动的 runtime teacher supervision

## 7. 当前已经吸收的主线改进

- external-reference bootstrap 配置
- external-reference fail-fast
- `reference_self_rate / reference_external_rate` 观测
- stage-3 acoustic scalar ramp
- chunkwise phase_nonretro 流式审计增强

## 8. git / 分支建议

当前建议在新分支上整理本地 AutoDL 工作：

```bash
cd /root/autodl-tmp/project-1/conan-rhythm
git checkout -b autodl
```

然后只针对明确文件做定向提交，不要 `git add -A`。

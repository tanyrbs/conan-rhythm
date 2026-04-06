# AutoDL stage-1 / formal 100+360 进度

更新日期：2026-04-06 UTC

## 1. 当前结论

- `train100` warmup 已完成到 **20000 step**
- 当前最可信 validation 点是 **19000**
- `train360` 仍在做 **metadata 分片恢复**
- 当前正式路线仍是：**先把 teacher 做强，再进 student**

## 2. 已完成内容

### 2.1 warmup 20k 已完成

关键产物：

- ckpt：
  - `checkpoints/teacher_offline_train100_warmup/model_ckpt_steps_20000.ckpt`
- warmup 复核报告：
  - `logs/stage1_warmup20k_formal100360_pipeline_20260406_075555.report.json`
- export smoke：
  - `artifacts/warmup20k_export_smoke/manifest.json`

### 2.2 warmup 关键指标（step=19000 valid）

- `rhythm_metric_module_only_objective = 1.0`
- `rhythm_metric_skip_acoustic_objective = 1.0`
- `rhythm_metric_disable_acoustic_train_path = 1.0`
- `total_loss = 0.2166`
- `L_rhythm_exec = 0.1753`
- `L_exec_speech = 0.0680`
- `L_exec_pause = 0.1073`
- `L_budget = 0.0066`
- `L_prefix_state = 0.0347`
- `L_stream_state = 0.0413`
- `rhythm_metric_exec_total_corr = 0.8907`
- `rhythm_metric_pause_event_f1 = 0.5957`
- `rhythm_metric_prefix_drift_l1 = 25.7781`
- `rhythm_metric_budget_projection_repair_ratio_mean = 0.0`
- `rhythm_metric_phase_nonretro_rate = null`

判定：

- **阶段语义正常**
- **control 面收敛方向正常**
- `phase_nonretro_rate` 仍不能作为当前标准 val 的硬 gate，应看 chunkwise / streaming 审计指标

## 3. train360 当前状态

当前后台恢复脚本：

- `logs/stage1_recovery_mixed100360_v2_20260406_091257.sh`

当前状态文件：

- `logs/stage1_recovery_mixed100360_v2_20260406_091257.status`
- 当前仍为：`TRAIN360_METADATA_SHARDED_START`

截至最近监控：

- `live_recovery_monitor.log` 最近均值进度约 **63.8%**
- 线性外推 metadata 完成时间约 **2026-04-06 18:02 ~ 18:12 UTC**

当前磁盘：

- `/root/autodl-tmp` 约 **51G used / 37G avail**
- `/` 约 **19G used / 12G avail**

## 4. 正在采用的恢复/接管方案

不再给 `train360` 单独构建 `valid/test`，只构建 `train`：

- 目标脚本：
  - `logs/stage1_takeover_from_existing_train360_metadata_trainonly.sh`

接管后的顺序：

1. merge 现有 train360 metadata shards
2. `train360` 做 **train-only binarize**
3. 对 `train360 train` 做 preflight
4. 对 `train100|train360` mixed config 做 preflight
5. 跑 `mixed real smoke`
6. 正式启动 `teacher_offline_train100_360_stage1`

## 5. 下一步要做什么

### 5.1 眼前动作

- 继续盯 `train360` metadata 分片，不中断当前进程
- 分片完成后，切到 `train-only takeover`

### 5.2 formal Stage-1

formal stage-1 语义：

- `binary_data_dir = train100`
- `train_sets = train100|train360`
- `load_ckpt = warmup 20k ckpt`
- `load_ckpt_strict = True`
- 新 exp：
  - `teacher_offline_train100_360_stage1`

### 5.3 teacher 之后的路线

当前不是永久跳过 `student_kd`，而是：

- **先完成强 teacher**
- 然后默认仍可走：
  - `teacher_offline -> student_kd -> student_retimed`

如果要冲更高上限，可在中间插入实验性 stage-2.5：

- `student_pairwise_ref_runtime_teacher`
- alias：`student_ref_bootstrap`

## 6. 和 origin/main 的差异与已吸收改进

当前本地已吸收的主线优点包括：

1. **external-reference bootstrap**
   - 增加 `student_ref_bootstrap / student_pairwise_ref_runtime_teacher`
   - 防止 `sample_ref + cached self targets` 的监督冲突

2. **external reference fail-fast + observability**
   - `rhythm_require_external_reference: true`
   - `rhythm_reference_is_self`
   - `rhythm_metric_reference_self_rate`
   - `rhythm_metric_reference_external_rate`

3. **stage-3 acoustic stabilization**
   - retimed acoustic loss scalar warmup
   - 观测项：
     - `rhythm_metric_stage3_acoustic_loss_scale`
     - `rhythm_metric_retimed_acoustic_loss_scale`

4. **streaming phase 审计增强**
   - 补充 chunkwise `phase_nonretro` 相关流式指标

## 7. 目前最准确的一句话

当前项目进度是：

- `train100` warmup 20k 已通过
- `train360` 正在恢复 metadata
- formal `100+360` teacher 还没正式开跑
- 代码上已经补齐了 stage-2.5 external-reference bootstrap 与 stage-3 acoustic stabilization 的主干能力

# AutoDL formal 训练进度记录

更新时间：2026-04-06 UTC

## 1. 当前真实状态

### 1.1 warmup

- `teacher_offline_train100_warmup` 已到 **20000 steps**
- 当前准备用于 formal Stage-1 warm-start 的 ckpt：
  - `checkpoints/teacher_offline_train100_warmup/model_ckpt_steps_20000.ckpt`
- 最新可信 validation 点：
  - `step=19000`
  - 证据文件：`logs/stage1_warmup20k_formal100360_pipeline_20260406_075555.report.json`

### 1.2 warmup 结论

19k validation 关键指标：

- `total_loss = 0.2166`
- `L_rhythm_exec = 0.1753`
- `L_exec_speech = 0.0680`
- `L_exec_pause = 0.1073`
- `L_budget = 0.0066`
- `L_prefix_state = 0.0347`
- `L_stream_state = 0.0413`
- `rhythm_metric_module_only_objective = 1.0`
- `rhythm_metric_skip_acoustic_objective = 1.0`
- `rhythm_metric_disable_acoustic_train_path = 1.0`
- `rhythm_metric_exec_total_corr = 0.8907`
- `rhythm_metric_pause_event_f1 = 0.5957`
- `rhythm_metric_prefix_drift_l1 = 25.7781`
- `rhythm_metric_budget_projection_repair_ratio_mean = 0.0`
- `rhythm_metric_phase_nonretro_rate = null`

结论：

- warmup **通过**
- 阶段语义没有跑偏
- “该关的 loss” 已关住
- 结构指标方向正常
- `phase_nonretro_rate` 仍然**不是当前标准 val 的可靠 gate**
  - 要看 streaming/chunkwise 指标，不要误用标准 validation 的空值

### 1.3 smoke 状态

- export smoke：**通过**
  - `artifacts/warmup20k_export_smoke/manifest.json`
- integration smoke：**未通过**
  - 原因是 `train-clean-100` raw wav 已删，导致流程资产断裂
  - **不是** warmup loss/metric 异常

证据目录：

- `artifacts/warmup_20k_evidence/`

## 2. train360 当前进度

当前在跑的是：

- `logs/stage1_recovery_mixed100360_v2_20260406_091257.sh`

当前阶段：

- `TRAIN360_METADATA_SHARDED_START`

按 `logs/prepare_train360_shards/shard_*.log` 最新进度估算：

- shard 平均进度约 **64.5%**
- 估计 metadata 分片结束时间约：
  - **2026-04-06 18:03 UTC**

说明：

- 现在还**没进入** train360 binarize
- 现在主要在做 `train-clean-360` metadata 分片构建

## 3. 当前磁盘状态

当前磁盘：

- `/root/autodl-tmp`：约 **51G used / 37G avail**
- 系统 overlay（`/root` / `/mnt`）：约 **12G avail**

当前重要占用：

- `data/binary/libritts_train100_formal_rhythm_v5`：约 **13G**
- `/root/autodl-tmp/data/LibriTTS/train-clean-360`：约 **33G**
- `/root/autodl-tmp/data/LibriTTS/dev-clean`：约 **1.6G**
- `/root/autodl-tmp/data/LibriTTS/test-clean`：约 **1.5G**

### 3.1 批判性判断

formal takeover 脚本的“train360 train-only binarize”本身是对的；
但**磁盘仍然偏紧**。

原因不是 valid/test，而是：

- raw `train-clean-360` 还在
- train100 binary 还在
- train360 binary 很可能也会很大

所以在 metadata 完成后、正式开始 train360 binarize 前，要再次检查 `/root/autodl-tmp` 可用空间。

保守要求：

- 如果 `/root/autodl-tmp` 可用空间还低于 **46~50G**，不要盲目直接开 binarize
- 先继续清空间，或把一部分 `train100` binary 临时挪到 overlay `/root`/`/mnt`

## 4. 和 upstream main 的差异

当前远端主线比本地 `main` 新两次提交，核心新增点是：

1. **stage-3 acoustic scalar ramp**
   - 目的：减轻 `student_retimed` 早期被 acoustic loss 压制
2. **stage-2.5 external-reference bootstrap**
   - 目的：让 rhythm 路径真正响应 external same-speaker ref，而不是继续吃 self-conditioned cached surface

本地 AutoDL 侧已经吸收并保留的关键改进：

- `student_retimed` acoustic warmup / observability
- `student_ref_bootstrap` / pairwise runtime-teacher 配置
- external-ref 与 cached self target 的**硬合同防呆**
- runtime external-ref fail-fast
  - `rhythm_require_external_reference: true`
- reference self/external 监控指标
  - `rhythm_metric_reference_self_rate`
  - `rhythm_metric_reference_external_rate`
- 本地额外补充：
  - streaming 侧 phase non-retro 指标
  - 显式 `student_ref_bootstrap` stage 名称，便于本地 contract fail-fast

## 5. 当前最重要的判断

### 5.1 teacher 计划

当前仍然是：

- **teacher first**

也就是：

1. 先把 `teacher_offline` formal Stage-1（100+360）跑起来
2. 再做 teacher export / student rebuild
3. 再决定 student 链路

### 5.2 会不会永久跳过 `student_kd`

不会。

更准确地说：

- **现在先不跑 `student_kd`**
- 不是“彻底删除 `student_kd`”

当前默认正式链仍然是：

1. `teacher_offline`
2. export teacher targets / rebuild student cache
3. `student_kd`
4. 可选 upper-bound：`student_ref_bootstrap`
5. `student_retimed`

## 6. 下一步要做什么

按顺序：

1. 等 train360 metadata 分片完成
2. takeover：
   - merge shards
   - train360 **train-only** binarize
   - train360 preflight
   - mixed preflight
   - mixed real smoke
3. 开 formal Stage-1：
   - `train_sets='train100|train360'`
   - 新 exp
   - `--reset`
   - `load_ckpt='...model_ckpt_steps_20000.ckpt'`
   - `load_ckpt_strict=True`
4. formal teacher 跑到：
   - 最低：`100k~120k`
   - 更稳：`150k~180k`

## 7. 当前命令层判断

### 7.1 takeover / formal Stage-1 脚本要不要因为 upstream 重写

**现在不用重写。**

原因：

- upstream 这次核心变化主要在 `student_ref_bootstrap` 和 `student_retimed`
- 当前正在做的是 `teacher_offline` formal Stage-1
- 所以现有：
  - `logs/stage1_takeover_from_existing_train360_metadata_trainonly.sh`
  - `scripts/autodl_train_stage1.sh`

对 teacher formal 启动仍然是可用的

### 7.2 需要调整的不是 teacher 命令，而是后续 student 命令

后面如果进入 upper-bound 路线，应优先使用：

- `egs/conan_emformer_rhythm_v2_student_ref_bootstrap.yaml`
- `egs/conan_emformer_rhythm_v2_student_pairwise_ref_runtime_teacher.yaml`

而不是只把 `rhythm_cached_reference_policy` 从 `self` 改成 `sample_ref`

因为那样会把 external ref 条件和 self cache target 混在一起，监督冲突。

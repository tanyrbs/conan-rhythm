# AutoDL Training Handoff（2026-04-07 UTC）

## 1. 当前项目处于什么状态

### 1.1 已完成的数据资产

- `data/processed/libritts_train100_formal`
- `data/binary/libritts_train100_formal_rhythm_v5`
- `data/processed/libritts_train360_formal_trainset`
- `data/binary/libritts_train360_formal_trainset_rhythm_v5`
- `checkpoints/teacher_offline_train100_warmup/model_ckpt_steps_20000.ckpt`

### 1.2 raw 音频状态

为了控制磁盘，以下 raw 已删除：

- `/root/autodl-tmp/data/LibriTTS/train-clean-100`
- `/root/autodl-tmp/data/LibriTTS/train-clean-360`

这意味着：

- **继续训练当前 Stage 1 没问题**，因为直接复用现成 binary
- **如果要重做 binarize / cache**，必须先重新挂载 raw 音频

### 1.3 当前正在跑的训练

当前主线是：

- mixed `train100 + train360`
- `teacher_offline` formal Stage 1
- 启动脚本：`scripts/autodl_recovery_mixed100360_v3_trainonly.sh`

注意：

- status 文件只会显示 `FORMAL_STAGE1_RUN_START`
- 真正的训练进度请看：
  - `logs/stage1_recovery_mixed100360_v3_trainonly.log`
  - `checkpoints/teacher_offline_train100_360_stage1/`

### 1.4 当前最新可引用的验证窗口

截至 **2026-04-07 UTC**，日志里最新完整 validation 为：

- step `75000`
- `total_loss = 0.2155`
- `L_exec_pause = 0.1133`
- `L_prefix_state = 0.0314`
- `rhythm_metric_pause_event_precision = 0.7111`
- `rhythm_metric_pause_event_recall = 0.5666`
- `rhythm_metric_pause_event_f1 = 0.5778`
- `rhythm_metric_prefix_drift_l1 = 22.8119`
- `rhythm_metric_exec_total_corr = 0.8943`
- `rhythm_metric_budget_projection_repair_ratio_mean = 0.0`
- `L_base = 0.0`
- `L_pitch = 0.0`

解释：

- teacher Stage 1 仍在健康推进
- 契约没有跑歪
- 当前最关键短板已不再是 budget，而是 pause placement 的 recall/precision 平衡
- prefix consistency 仍未完全收住，所以现在还不是 teacher 定稿点

---


## 1.5 cache lineage 说明

- 当前仓库头已经把 soft-boundary `ref_rhythm_trace` 语义提升到 **`rhythm_cache_version: 6`**
- 当前正在跑的 Stage 1 是旧 cache lineage 上启动的进程，**会继续跑完**
- 但未来任何新开的 `cached_only` 实验，如果想声明 soft-boundary 生效，必须：
  1. 恢复 raw
  2. 重建 binary/cache
  3. 在 version 6 下重新 preflight

### 2026-04-07 运行补充

- 当前 mixed lineage 的实际 binary 仍是：
  - `data/binary/libritts_train100_formal_rhythm_v5`
  - `data/binary/libritts_train360_formal_trainset_rhythm_v5`
- 当前机器上 `/root/autodl-tmp/data/LibriTTS/...` raw wav 已不存在，因此**现在不能直接重建 v6 binary**
- 为了让 `teacher_offline_train100_360_stage1` 从 `80000 -> 82500` 精确续训，同时不“默默假装 soft-boundary 已生效”，恢复脚本现在显式加入：
  - `rhythm_cache_version=5`
  - `rhythm_allow_legacy_cache_resume=True`
- 这条恢复链的含义是：
  - **允许旧 v5 cache 明确续训**
  - **pause-recall loss 改动生效**
  - **soft-boundary cache 语义暂不生效**
  - 后续如果要正式宣称 soft-boundary，有且只有一条路：恢复 raw 后重建 train100 + train360 两套 binary/cache 到 v6

## 2. 当前唯一正确的总计划

不是：

- teacher -> 立刻 `student_ref_bootstrap`
- teacher -> 立刻 `student_kd`

而是：

1. **先把 teacher Stage 1 跑稳**
2. **继续 teacher Stage 2 / v2.5 / v3，把 teacher 上限做高**
3. **保留每个 teacher 阶段的最佳 checkpoint**
4. **固定 teacher audit 集，持续做人耳 + descriptor 审计**
5. **等 teacher 家族成熟后，再做学生蒸馏**

---

## 3. 当前不该做什么

- 不要因为 student 配置已经存在，就提前切到学生
- 不要现在把 `student_ref_bootstrap` 当默认下一步
- 不要为了 soft boundary 立刻打断当前 Stage 1 去重做 binary
- 不要只看 `val_loss` 就宣布 teacher 已经够好

---

## 4. 当前最该怎么盯 Stage 1

### S 级指标

- `L_base`
- `L_pitch`
- `rhythm_metric_module_only_objective`
- `rhythm_metric_skip_acoustic_objective`
- `rhythm_metric_disable_acoustic_train_path`
- `L_exec_pause`
- `L_exec_pause_value`
- `L_pause_event`
- `L_prefix_state`
- `rhythm_metric_prefix_drift_l1`
- `rhythm_metric_pause_event_f1`
- `rhythm_metric_pause_event_precision`
- `rhythm_metric_pause_event_recall`
- `rhythm_metric_budget_projection_repair_ratio_mean`

### A 级指标

- `L_exec_speech`
- `L_budget`
- `L_stream_state`
- `rhythm_metric_exec_total_corr`
- `rhythm_metric_exec_pause_l1`
- `rhythm_metric_exec_speech_l1`

### 当前判断

- 60k 还属于 teacher Stage 1 主收敛期
- 第一批认真挑 teacher ckpt 的窗口仍建议放在：
  - **80k ~ 120k**

---

## 5. 关于 5k 验证节奏与断点续训

### 5.1 `val_check_interval = 5000` 会不会影响 pause 收敛

不会直接影响 pause objective 本身。

当前 trainer 里，`5000` 主要决定：

- 什么时候跑 validation
- 什么时候刷新 best checkpoint
- 我们每 5k 复盘一次指标的节奏

它**不会直接改 pause loss / projector / optimizer 路径**。  
所以我们现在保留 `5000`，是为了：

- 让恢复训练后的每一档调参都正好对应一个完整评估窗口
- 便于人工比较 `75k -> 80k -> 85k -> 90k`

### 5.2 当前支持“同 exp + 新配置 + 精确续训”

当前代码支持下面这种恢复方式：

- 同一个 `exp_name`
- `RESET=1`
- 不删除旧 checkpoint

效果：

1. 新 YAML 会覆盖 `checkpoints/<exp_name>/config.yaml`
2. trainer 会自动恢复该 `exp_name` 下最新 `model_ckpt_steps_*.ckpt`
3. `global_step` 和 optimizer state 都会继续

也就是说：

- 这是**真正的断点续训**
- 不是重新从 0 开始 warm-start

### 5.3 当前 pause recall 续训入口

新增：

- 配置：`egs/conan_emformer_rhythm_v2_teacher_offline_train100_360_pause_recall.yaml`
- 脚本：`scripts/autodl_resume_stage1_pause_recall.sh`

默认逻辑：

- 自动找到当前 `teacher_offline_train100_360_stage1` 的最新 checkpoint
- 自动把 `max_updates` 设成 `latest + 5000`
- 用新的 pause-recall 配置继续下一档

如果当前老训练还在跑，脚本默认会拒绝启动第二个 writer。  
要切换时，可显式：

```bash
STOP_EXISTING=1 bash scripts/autodl_resume_stage1_pause_recall.sh
```

---

## 6. 交接时最少要记住的命令

```bash
cd /root/autodl-tmp/project-1/conan-rhythm
bash scripts/autodl_recovery_mixed100360_v3_trainonly.sh
```

```bash
tail -f logs/stage1_recovery_mixed100360_v3_trainonly.log
```

```bash
python scripts/monitor_stage1_metrics.py \
  --log logs/stage1_recovery_mixed100360_v3_trainonly.log \
  --tail 5
```

```bash
STOP_EXISTING=1 bash scripts/autodl_resume_stage1_pause_recall.sh
```

# Training Plan（Teacher-First，2026-04-07 UTC）

## 1. 当前训练主线

### T1-current：train360-only / v5 / split-head / teacher_offline

当前主线不是 old consistent line，也不是 mixed v6 line，而是：

- `teacher_offline_train360only_pause_split_heads_stage1`
- `teacher_offline + teacher_as_main`
- `train360-only`
- `legacy v5 binary`
- `split-head pause support/allocation`
- 从 `teacher_offline_train360only_pause_recall_consistent_stage1@115000` 做 **weight-only warm-start**

配置：

- `egs/conan_emformer_rhythm_v2_teacher_offline_train100_360_pause_split_heads_resume.yaml`

数据：

- `binary_data_dir='data/binary/libritts_train360_formal_trainset_rhythm_v5'`
- `processed_data_dir='data/processed/libritts_train360_formal_trainset'`
- `train_sets='data/binary/libritts_train360_formal_trainset_rhythm_v5'`
- `rhythm_cache_version=5`
- `rhythm_allow_legacy_cache_resume=True`

这条线的语义要始终写清：

- **不是 exact resume**
- **不是 mixed 100+360**
- **不是 v6 cache line**
- 是一条 **legacy v5 结构修复验证线**

---

## 2. 为什么当前主线是 split-head

当前 pause 训不好的主因，不是 budget 本身，而是：

- guidance target 稠密且偏 boundary
- planner 的 pause distribution 以前由单一 softmax 同时承担 support + allocation
- sparse projector + boundary prior 形成后置瓶颈
- event / support / value 的目标面曾经不完全同向

split-head 的目标就是把：

- **哪里该停**
- **停顿量怎么分**

拆开处理。

当前已经落地的关键修复：

1. `pause_support_head` + `pause_amount_weight_unit`
2. `pause_support_event` / `pause_support_count` 监督
3. `pause_event_boundary_weight=0.0`
4. projector boundary prior 改为 **gain** 语义
5. teacher 阶段恢复可训练的 soft pause selection 路径

---

## 3. 当前主线的最新观察

### 已有 validation

#### `@2500`

- pause P/R/F1 = **0.8009 / 0.6426 / 0.6679**
- `prefix_drift_l1 = 26.5305`
- `exec_total_corr = 0.9122`
- `repair_ratio = 0.0`
- `L_budget = 0.0022`

#### `@5000`

- pause P/R/F1 = **0.8745 / 0.6198 / 0.6813**
- `prefix_drift_l1 = 31.8658`
- `exec_total_corr = 0.8794`
- `repair_ratio = 0.0`
- `L_budget = 0.0035`

#### `@7500`

- pause P/R/F1 = **0.8504 / 0.6952 / 0.6928**
- `prefix_drift_l1 = 31.5229`
- `exec_total_corr = 0.8820`
- `repair_ratio = 0.0`

#### `@10000`

- pause P/R/F1 = **0.8846 / 0.6176 / 0.6789**
- `prefix_drift_l1 = 38.0078`
- `exec_total_corr = 0.8833`
- `repair_ratio = 0.0`

#### `@12500`

- pause P/R/F1 = **0.8453 / 0.6051 / 0.6570**
- `prefix_drift_l1 = 30.2024`
- `exec_total_corr = 0.8967`
- `repair_ratio = 0.0`

#### `@15000`

- pause P/R/F1 = **0.7769 / 0.6412 / 0.6543**
- `prefix_drift_l1 = 31.9455`
- `exec_total_corr = 0.8831`
- `repair_ratio = 0.0`

### 当前判断

- 相对旧 consistent `115000` 起点，split-head **方向有效**
- 但在 split-head 线内部：
  - `5000` 比 `2500` **更高 precision / 略高 f1**
  - 但 **recall / prefix / corr 回退**

所以当前最准确的结论是：

- **按综合稳定性，最好 checkpoint 暂时仍记 `2500`**
- **按 pause recall / f1 峰值，`7500` 是目前这条线最强的一次**
- `10000`、`12500`、`15000` 之间仍然来回波动，说明当前线还在波动，不能过早下最终结论

---

## 4. 当前该盯哪些指标

### S 级

- `rhythm_metric_pause_event_recall`
- `rhythm_metric_pause_event_f1`
- `rhythm_metric_pause_event_precision`
- `rhythm_metric_prefix_drift_l1`
- `rhythm_metric_exec_total_corr`
- `rhythm_metric_pause_support_cover_at_topk`
- `rhythm_metric_pause_recall_drop_post_from_planner`
- `rhythm_metric_pause_target_over_topk_rate`
- `rhythm_metric_pause_event_recall_boundary`
- `rhythm_metric_pause_event_recall_nonboundary`
- `rhythm_metric_budget_projection_repair_ratio_mean`

### 阶段契约

- `L_base = 0`
- `L_pitch = 0`
- `rhythm_metric_module_only_objective = 1`
- `rhythm_metric_skip_acoustic_objective = 1`
- `rhythm_metric_disable_acoustic_train_path = 1`

### A 级辅助指标

- `L_pause_event`
- `L_pause_support`
- `L_pause_support_event`
- `L_pause_support_count`
- `L_budget`
- `L_prefix_state`
- threshold sweep：`t02 / t03 / t05`
- `best_f1 / best_threshold`

---

## 5. 新 binary 与当前实现的关系

### 5.1 结论

**当前实现兼容完整 v6 binary，但目前数据侧还没完整 ready。**

原因：

- split-head 是模型侧 / loss 侧改动
- 它不要求重定义新的 binary schema
- 因此 **完整 v6 binary 可以直接喂给当前实现**

### 5.2 当前磁盘状态

当前已确认：

- `data/binary/libritts_train100_formal_rhythm_v6`：仅见 `valid/test`
- `data/binary/libritts_train360_formal_trainset_rhythm_v6`：**不存在**
- `data/binary/libritts_train360_formal_trainset_rhythm_v5`：当前主线正在使用

所以现在不能说：

- “已经可以切 mixed v6 训练”

最多只能说：

- **实现兼容，等完整 v6 binary 落盘后即可切新实验**

### 5.3 当前不允许的混搭

不要把下面这种混搭当正式主线：

- `train100 v6` + `train360 v5`

原因：

- mixed lineage 不干净
- cache contract 不一致
- 当前 v6 语义下不应继续沿用 v5 legacy 数据

---

## 6. 切到新 binary 的计划

### 6.1 硬条件

必须同时满足：

1. `data/binary/libritts_train100_formal_rhythm_v6` 是完整 train/valid/test 目录
2. `data/binary/libritts_train360_formal_trainset_rhythm_v6` 已落盘
3. preflight 确认两侧 cache version 都是 v6

### 6.2 启动语义

切到新 binary 后，推荐：

```yaml
binary_data_dir='data/binary/libritts_train100_formal_rhythm_v6'
processed_data_dir='data/processed/libritts_train100_formal'
train_sets='data/binary/libritts_train100_formal_rhythm_v6|data/binary/libritts_train360_formal_trainset_rhythm_v6'
rhythm_cache_version=6
# no rhythm_allow_legacy_cache_resume
```

并且：

- `binary_data_dir` 继续选 **train100 base**
- **新开 exp**
- **weight-only warm-start**
- **不要 exact resume 当前 v5 主线**

### 6.3 推荐顺序

1. 继续把当前 v5 split-head 线盯到下一批 validation
2. 同步等待完整 v6 binary 落盘
3. v6 ready 后，新开一个 **mixed 100+360 split-head** 实验
4. 先跑 2.5k~5k 短 block 验证方向

---

## 7. 当前的最短结论

当前项目应被描述为：

> **legacy v5 / train360-only / split-head pause structural-fix run**

而不是：

> mixed v6 exact resume line

新 binary 线的最短结论是：

> **实现兼容，但数据尚未完整 ready；ready 后应新开 mixed v6 实验，而不是直接把当前 v5 run 改名续训。**

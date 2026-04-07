# AutoDL Training Handoff（2026-04-07 UTC）

## 1. 当前数据与代码状态

### 1.1 当前磁盘上已确认的数据资产

- `data/processed/libritts_train100_formal`
- `data/processed/libritts_train360_formal_trainset`
- `data/binary/libritts_train360_formal_trainset_rhythm_v5`：**完整可训练**（train/valid/test）
- `data/binary/libritts_train100_formal_rhythm_v6`：**当前仅看到 valid/test**，还不能当作完整 mixed train100 训练根目录使用
- `data/binary/libritts_train360_formal_trainset_rhythm_v6`：**当前未落盘**

补充：

- 当前仓库维护缓存版本是 **v6**
- 当前 active run 之所以还能跑 v5，是因为显式使用：
  - `rhythm_cache_version=5`
  - `rhythm_allow_legacy_cache_resume=True`

### 1.2 当前代码实现与新 binary 的兼容性

结论：**兼容，但前提是 v6 binary 真的完整落盘。**

原因：

- split-head 改动是**模型侧 / loss 侧**改动，不要求重新定义新的 binary schema
- 新增的字段主要是运行时 planner 输出：
  - `pause_support_logit_unit`
  - `pause_support_prob_unit`
  - `pause_amount_weight_unit`
  - `pause_candidate_score_unit`
- 当前 loss 仍主要使用已有 cached target，例如 `pause_exec_tgt`

所以：

- **完整的 v6 binary 与当前实现兼容**
- **split-head 不是“必须重做 binary schema 才能用”的实现**
- 但**不能混用** `train100 v6 + train360 v5` 作为一条正式主线

---

## 2. 当前最新训练主线

### 2.1 当前 active run 的精确定义

当前真正的主训练线是：

- **teacher_offline**
- **teacher_as_main**
- **train360-only**
- **legacy v5 binary**
- **pause split-head support/allocation**
- **projector boundary gain**
- 从旧 consistent 线 `115000` 做 **weight-only warm-start**

具体信息：

- 配置：`egs/conan_emformer_rhythm_v2_teacher_offline_train100_360_pause_split_heads_resume.yaml`
- 实验：`teacher_offline_train360only_pause_split_heads_stage1`
- warm-start ckpt：`checkpoints/teacher_offline_train360only_pause_recall_consistent_stage1/model_ckpt_steps_115000.ckpt`
- 当前 work dir：`checkpoints/teacher_offline_train360only_pause_split_heads_stage1`
- 日志：`logs/stage1_train360only_pause_split_heads_from115000.log`
- 训练数据：`data/binary/libritts_train360_formal_trainset_rhythm_v5`
- `val_check_interval=2500`
- `max_updates=135000`

这条线的语义必须写清楚：

- **不是 exact resume**
- **不是 mixed 100+360**
- **不是 v6 cache 线**
- 是一条 **legacy v5 / train360-only / split-head 结构验证线**

### 2.2 当前启动语义

```bash
CUDA_VISIBLE_DEVICES=0 \
RESET=1 \
MAX_UPDATES=135000 \
VAL_CHECK_INTERVAL=2500 \
HP_EXTRA="binary_data_dir='data/binary/libritts_train360_formal_trainset_rhythm_v5',\
processed_data_dir='data/processed/libritts_train360_formal_trainset',\
train_sets='data/binary/libritts_train360_formal_trainset_rhythm_v5',\
load_ckpt='checkpoints/teacher_offline_train360only_pause_recall_consistent_stage1/model_ckpt_steps_115000.ckpt',\
load_ckpt_strict=False,\
rhythm_cache_version=5,\
rhythm_allow_legacy_cache_resume=True" \
bash scripts/autodl_train_stage1.sh \
  egs/conan_emformer_rhythm_v2_teacher_offline_train100_360_pause_split_heads_resume.yaml \
  teacher_offline_train360only_pause_split_heads_stage1
```

### 2.3 当前主线配置里最关键的参数

```yaml
rhythm_offline_teacher_split_pause_heads: true

rhythm_pause_support_weight: 0.05
rhythm_pause_support_event_weight: 0.12
rhythm_pause_support_count_weight: 0.03
rhythm_pause_support_threshold: 0.20
rhythm_pause_support_pos_weight: 2.0
rhythm_pause_support_loss_type: focal
rhythm_pause_support_focal_gamma: 2.0
rhythm_pause_support_focal_alpha: 0.75

rhythm_pause_event_weight: 0.20
rhythm_pause_event_threshold: 0.30
rhythm_pause_event_temperature: 0.22
rhythm_pause_event_pos_weight: 2.0
rhythm_pause_event_boundary_weight: 0.0

rhythm_pause_source_boundary_weight: 0.06
rhythm_boundary_feature_scale: 0.22
rhythm_boundary_source_cue_weight: 0.55
rhythm_pause_boundary_weight: 0.22

rhythm_projector_pause_selection_mode: sparse
rhythm_projector_pause_topk_ratio: 0.55
rhythm_projector_pause_topk_ratio_train_start: 0.50
rhythm_projector_pause_topk_ratio_train_end: 0.55
rhythm_projector_pause_topk_ratio_anneal_steps: 15000
rhythm_projector_pause_soft_temperature: 0.20
rhythm_projector_pause_boundary_mode: gain
rhythm_projector_pause_boundary_bias_weight: 0.08
rhythm_projector_pause_min_boundary_weight: 0.0
```

---

## 3. 当前 split-head 线的最新训练观察

### 3.1 当前运行状态

截至当前检查：

- 训练**仍在运行**
- 日志最新 step 已超过 **15000**
- 已保存：
  - `model_ckpt_steps_2500.ckpt`
  - `model_ckpt_steps_5000.ckpt`
  - `model_ckpt_steps_7500.ckpt`
  - `model_ckpt_steps_10000.ckpt`
  - `model_ckpt_steps_12500.ckpt`
  - `model_ckpt_steps_15000.ckpt`
  - `model_ckpt_best.pt`（当前仍是早期 best）

### 3.2 已完成的 validation

#### `@2500`

- pause precision / recall / f1 = **0.8009 / 0.6426 / 0.6679**
- `prefix_drift_l1 = 26.5305`
- `exec_total_corr = 0.9122`
- `budget_projection_repair_ratio_mean = 0.0`
- `L_budget = 0.0022`
- `L_pause_support_event = 0.0067`
- `L_pause_support_count = 0.0013`

#### `@5000`

- pause precision / recall / f1 = **0.8745 / 0.6198 / 0.6813**
- `prefix_drift_l1 = 31.8658`
- `exec_total_corr = 0.8794`
- `budget_projection_repair_ratio_mean = 0.0`
- `L_budget = 0.0035`
- `L_pause_support_event = 0.0079`
- `L_pause_support_count = 0.0004`

#### `@7500`

- pause precision / recall / f1 = **0.8504 / 0.6952 / 0.6928**
- `prefix_drift_l1 = 31.5229`
- `exec_total_corr = 0.8820`
- `budget_projection_repair_ratio_mean = 0.0`

#### `@10000`

- pause precision / recall / f1 = **0.8846 / 0.6176 / 0.6789**
- `prefix_drift_l1 = 38.0078`
- `exec_total_corr = 0.8833`
- `budget_projection_repair_ratio_mean = 0.0`

#### `@12500`

- pause precision / recall / f1 = **0.8453 / 0.6051 / 0.6570**
- `prefix_drift_l1 = 30.2024`
- `exec_total_corr = 0.8967`
- `budget_projection_repair_ratio_mean = 0.0`

#### `@15000`

- pause precision / recall / f1 = **0.7769 / 0.6412 / 0.6543**
- `prefix_drift_l1 = 31.9455`
- `exec_total_corr = 0.8831`
- `budget_projection_repair_ratio_mean = 0.0`

### 3.3 当前结论

对这条 split-head 新线，当前最保守、最准确的判断是：

- **相对旧 consistent@115000 起点，split-head 方向明显有效**
  - recall 提高
  - f1 提高
  - prefix drift 改善
- 但在 split-head 线内部：
  - `5000` 相比 `2500` 只是 **f1 小升**
  - `recall / prefix / exec corr` 反而回退

所以当前应记为：

- **按综合稳定性 / best checkpoint，当前仍是 `2500` 最稳**
- **按 pause recall / f1 峰值，当前 `7500` 是这条线到目前为止最强的一次**
- `10000`、`12500`、`15000` 之间仍在来回波动，说明这条线还没有真正稳定收敛

---

## 4. 已落地的关键代码修复

### 4.1 split-head support / allocation

当前已实现：

- `pause_support_head` 负责 support
- 原 `pause_head` 负责 amount / allocation
- `pause_candidate_score = support_prob * amount_weight`
- 再归一化得到 `pause_weight_unit`

对应主要文件：

- `modules/Conan/rhythm/offline_teacher.py`
- `modules/Conan/rhythm/projector.py`
- `modules/Conan/rhythm/contracts.py`
- `tasks/Conan/rhythm/losses.py`
- `tasks/Conan/rhythm/metrics.py`

### 4.2 pause_event 默认不再 boundary-weighted

当前默认语义：

- `l_exec_pause_value` 可继续偏重 boundary
- `l_pause_event` 默认走 `unit_mask`
- `pause_event_boundary_weight = 0.0`

这一步是为了让 event loss 真正承担 recall 补漏，而不是继续奖励少数强边界点。

### 4.3 projector boundary prior 已改为 gain 语义

目标不是取消 boundary，而是避免 post-softmax absolute bias 直接盖过 planner 的细粒度 pause surface。

### 4.4 单测状态

本轮核心节奏相关测试已通过：

```bash
OMP_NUM_THREADS=1 python -m unittest \
  tests.rhythm.test_loss_components \
  tests.rhythm.test_projector_invariants \
  tests.rhythm.test_metrics_masking \
  tests.rhythm.test_factorization_contract
```

结果：`40 passed`

---

## 5. 新 binary 什么时候能切

### 5.1 现在为什么还不能直接切到 mixed v6

当前原因有两个：

1. `data/binary/libritts_train360_formal_trainset_rhythm_v6` **还不存在**
2. `data/binary/libritts_train100_formal_rhythm_v6` **当前仅看到 valid/test**，还不能确认它已是完整 train/valid/test 训练目录

因此当前不能把项目表述成：

- “已经可以切新 binary mixed 训练”

最多只能表述成：

- **当前实现兼容 v6，新 binary 线接近可切，但数据侧尚未完整 ready**

### 5.2 切到新 binary 的硬条件

必须同时满足：

1. `data/binary/libritts_train100_formal_rhythm_v6` 为**完整训练目录**
2. `data/binary/libritts_train360_formal_trainset_rhythm_v6` 真正落盘
3. preflight 确认两侧 `rhythm_cache_version=6`
4. mixed lineage 显式指定，不能被 YAML 默认值偷换

推荐写法：

```yaml
binary_data_dir='data/binary/libritts_train100_formal_rhythm_v6'
processed_data_dir='data/processed/libritts_train100_formal'
train_sets='data/binary/libritts_train100_formal_rhythm_v6|data/binary/libritts_train360_formal_trainset_rhythm_v6'
rhythm_cache_version=6
# 不再开启 rhythm_allow_legacy_cache_resume
```

### 5.3 为什么 `binary_data_dir` 仍应选 train100 base

因为在这个项目里：

- `train_sets` 决定训练集
- `binary_data_dir` 同时承担：
  - valid/test 默认根目录
  - shared artifact / json / condition map 基准目录
  - train_sets contract check 的基准目录

而 train360 这边本来是 **train-only** 语义，所以 mixed 线的 base 仍应放在 **train100**。

### 5.4 切 v6 时的训练语义

切到新 binary / 新 cache 线时，推荐：

- **新开 exp**
- **weight-only warm-start**
- **不要 exact resume 当前 v5 split-head 线**

原因：

- 数据 lineage 已变化
- cache 语义已变化
- 继续复用旧 optimizer / scheduler / global_step 会让实验语义变脏

### 5.5 明确禁止的混搭

不建议也不应把下面这种当正式主线：

- `train100 v6` + `train360 v5`

原因：

- 语义不干净
- cache contract 不一致
- 当前代码的 v6 run 不会把 v5 当作同版本兼容线收进来

---

## 6. 当前建议的执行顺序

### 6.1 现在先做什么

- 继续盯当前 `train360-only / v5 / split-head` 主线
- 重点看后续 validation 是否能超过 `@2500`
- 当前如果要选“最好 checkpoint”，优先记 `2500`

### 6.2 新 binary ready 后再做什么

一旦 `train100 v6` 与 `train360 v6` 都 ready：

- 新开一个 **mixed 100+360 v6 split-head** 实验
- 从当前较好的 split-head checkpoint 做 **weight-only warm-start**
- 先跑一个短 block（2.5k~5k）
- 重点盯：
  - `pause_event_recall / f1`
  - `pause_support_cover_at_topk`
  - `pause_recall_drop_post_from_planner`
  - `prefix_drift_l1`
  - `cache_version_mean`

---

## 7. 现在最重要的一句话

当前项目的最新主线不是 “mixed v6 exact resume”，而是：

> **legacy v5 / train360-only / split-head pause structural-fix run**

而新 binary 线目前的最准确认知是：

> **实现已兼容，但数据侧还没完整 ready；等完整 v6 binary 落盘后，再新开 mixed v6 实验。**

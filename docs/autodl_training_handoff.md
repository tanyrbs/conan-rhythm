# AutoDL Training Handoff（2026-04-08 UTC）

## 1. 当前唯一维护主线

当前唯一维护中的训练主线已经收敛为：

- `teacher_offline`
- `mixed train100 + train360`
- `v6 binary / cache lineage`
- `split-head pause support + allocation`
- `weight-only warm-start`
- 统一入口：`scripts/autodl_train_stage1_mixed_v6_split_heads.sh`

### 明确排除
以下都**不是当前主线**：
- `train360-only / legacy v5`
- `pause_recall exact-resume`
- `recovery_v3_trainonly` 旧恢复脚本
- `mixed v5` 或 `train100 v6 + train360 v5` 混搭线

---

## 2. 当前数据与 checkpoint 语义

### 2.1 train100 v6
- processed: `data/processed/libritts_train100_formal`
- binary: `data/binary/libritts_train100_formal_rhythm_v6`
- 角色：
  - mixed 的 `binary_data_dir`
  - valid/test contract 基座
  - base dataset

### 2.2 train360 v6
- processed: `data/processed/libritts_train360_formal_trainset`
- binary: `data/binary/libritts_train360_formal_trainset_rhythm_v6`
- 角色：
  - train-only supplemental dataset
  - 通过 `train_sets='train100|train360'` 拼进 mixed stage-1

### 2.3 warm-start checkpoint
- 默认 bootstrap 位置：`checkpoints/bootstrap/model_ckpt_steps_17500.ckpt`
- 语义：`weight-only warm-start`
- 不是：`exact resume`

---

## 3. 当前标准训练入口

### 3.1 标准配置
- `egs/conan_emformer_rhythm_v2_teacher_offline_train100_360_pause_split_heads_resume.yaml`

### 3.2 标准脚本
- `scripts/autodl_train_stage1_mixed_v6_split_heads.sh`

它统一封装：
- mixed v6 data override
- split-head config
- preflight
- `MAX_UPDATES=80000`
- `VAL_CHECK_INTERVAL=5000`
- checkpoint 保留策略：
  - 最近 `3` 个 step ckpt
  - `model_ckpt_best.pt`（整体最优）
  - `model_ckpt_pause_best.pt`（pause 最优，按 `rhythm_metric_pause_event_f1`）

### 3.3 等价关键 override
```text
binary_data_dir='data/binary/libritts_train100_formal_rhythm_v6'
processed_data_dir='data/processed/libritts_train100_formal'
train_sets='data/binary/libritts_train100_formal_rhythm_v6|data/binary/libritts_train360_formal_trainset_rhythm_v6'
load_ckpt='checkpoints/bootstrap/model_ckpt_steps_17500.ckpt'
load_ckpt_strict=False
rhythm_cache_version=6
num_ckpt_keep=3
save_best=True
extra_valid_monitor_key='rhythm_metric_pause_event_f1'
extra_valid_monitor_mode='max'
extra_valid_monitor_filename='model_ckpt_pause_best.pt'
```

---

## 4. pause 路径当前已收敛的控制点

### 4.1 smoke / helper / CI
- `scripts/smoke_test_rhythm_v2.py` 已与当前 projector helper 签名对齐
- strict/simple smoke 场景改为**显式传 projector knobs**，不再依赖已经漂移的隐式默认值
- cache compatibility smoke 已与当前 `RHYTHM_CACHE_VERSION=6` 契约对齐

### 4.2 pause top-k anchor
- `scripts/autodl_resume_stage1_pause_recall.sh` 已补入 `PAUSE_TOPK_ANCHOR_STEP`
- 与 `runtime_adapter.py` 的 `rhythm_projector_pause_topk_anchor_step` 调度逻辑对齐

### 4.3 monitor
- `scripts/monitor_stage1_metrics.py` 已补充 pause 结构性诊断指标：
  - `rhythm_metric_pause_support_cover_at_topk`
  - `rhythm_metric_pause_recall_drop_post_from_planner`
  - `rhythm_metric_pause_f1_drop_post_from_planner`
  - `rhythm_metric_pause_target_over_topk_rate`
  - `rhythm_metric_pause_event_recall_boundary`
  - `rhythm_metric_pause_event_recall_nonboundary`

---

## 5. 当前推荐命令

### 5.1 启动训练
```bash
cd /root/autodl-tmp/project/conan-rhythm
bash scripts/autodl_train_stage1_mixed_v6_split_heads.sh
```

### 5.2 只做 preflight
```bash
cd /root/autodl-tmp/project/conan-rhythm
PREFLIGHT_ONLY=1 \
BOOTSTRAP_CKPT=checkpoints/bootstrap/model_ckpt_steps_17500.ckpt \
bash scripts/autodl_train_stage1_mixed_v6_split_heads.sh
```

### 5.3 监控指标
```bash
python scripts/monitor_stage1_metrics.py \
  --log logs/teacher_offline_train100_360_v6_split_heads_warm17500.log \
  --tail 5
```

---

## 6. 旧线如何看待

以下内容保留，仅作历史记录或迁移参考：
- `train360-only / legacy v5 / split-head` 历史观察
- `scripts/autodl_recovery_mixed100360_v3_trainonly.sh`
- `scripts/autodl_resume_stage1_pause_recall.sh` 的 legacy exact-resume 用法

如果未来需要回看旧线，请在实验记录中明确标注：
- cache lineage
- train set scope
- whether exact resume or weight-only warm-start
- whether split-head or old pause_recall path

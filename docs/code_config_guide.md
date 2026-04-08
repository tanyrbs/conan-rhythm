# Code / Config Guide

更新日期：2026-04-08 UTC

## 1. 当前唯一维护主线

当前唯一维护中的训练主线：

- stage: `teacher_offline`
- data: `mixed train100 + train360`
- lineage: `v6`
- pause path: `split-head support + allocation`
- bootstrap semantics: `weight-only warm-start`

标准入口：
- launcher: `scripts/autodl_train_stage1_mixed_v6_split_heads.sh`
- config: `egs/conan_emformer_rhythm_v2_teacher_offline_train100_360_pause_split_heads_resume.yaml`

---

## 2. 当前标准 launcher

### 2.1 主脚本
`scripts/autodl_train_stage1_mixed_v6_split_heads.sh`

这个脚本统一封装了：
- mixed v6 data override
- split-head config 选择
- preflight dry-run
- `MAX_UPDATES=80000`
- `VAL_CHECK_INTERVAL=5000`
- checkpoint 保留策略

### 2.2 底层调用
主脚本最终调用：
- `scripts/autodl_train_stage1.sh`
- 再落到底层 `python -m tasks.run --config ... --exp_name ... --reset -hp ...`

---

## 3. 当前标准 config

### 3.1 主配置
`egs/conan_emformer_rhythm_v2_teacher_offline_train100_360_pause_split_heads_resume.yaml`

这条 config 对当前主线负责：
- `teacher_offline` stage1
- split-head pause support / allocation
- mixed v6 teacher-first 训练语义
- pause boundary gain / sparse top-k 等主线路径

### 3.2 当前主线关键 hparams
当前主线的关键 override 为：

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

## 4. 当前数据与 checkpoint 约定

### 4.1 train100 v6
- processed: `data/processed/libritts_train100_formal`
- binary: `data/binary/libritts_train100_formal_rhythm_v6`
- 角色：mixed 的 base / valid / test contract

### 4.2 train360 v6
- processed: `data/processed/libritts_train360_formal_trainset`
- binary: `data/binary/libritts_train360_formal_trainset_rhythm_v6`
- 角色：mixed 的 train-only supplemental dataset

### 4.3 bootstrap ckpt
- 默认：`checkpoints/bootstrap/model_ckpt_steps_17500.ckpt`
- 语义：**weight-only warm-start**
- 不是：**exact resume**

---

## 5. 当前 checkpoint 策略

主线 stage1 统一保留：
- 最近 `3` 个 step ckpt
- `model_ckpt_best.pt`：整体最优（`val_loss`）
- `model_ckpt_pause_best.pt`：pause 最优（`rhythm_metric_pause_event_f1`）

当前 launcher 默认：
- `NUM_CKPT_KEEP=3`
- `EXTRA_VALID_MONITOR_KEY=rhythm_metric_pause_event_f1`
- `EXTRA_VALID_MONITOR_MODE=max`
- `EXTRA_VALID_MONITOR_FILENAME=model_ckpt_pause_best.pt`

---

## 6. 当前推荐命令

### 6.1 preflight
```bash
cd /root/autodl-tmp/project/conan-rhythm
PREFLIGHT_ONLY=1 \
BOOTSTRAP_CKPT=checkpoints/bootstrap/model_ckpt_steps_17500.ckpt \
bash scripts/autodl_train_stage1_mixed_v6_split_heads.sh
```

### 6.2 正式训练
```bash
cd /root/autodl-tmp/project/conan-rhythm
BOOTSTRAP_CKPT=checkpoints/bootstrap/model_ckpt_steps_17500.ckpt \
MAX_UPDATES=80000 \
VAL_CHECK_INTERVAL=5000 \
bash scripts/autodl_train_stage1_mixed_v6_split_heads.sh
```

### 6.3 监控日志
```bash
python scripts/monitor_stage1_metrics.py \
  --log logs/teacher_offline_train100_360_v6_split_heads_restart17500_fix1.log \
  --tail 5
```

---

## 7. 当前应该重点关注的指标

### 核心健康指标
- `rhythm_metric_pause_event_f1`
- `rhythm_metric_exec_total_corr`
- `rhythm_metric_prefix_drift_l1`
- `rhythm_metric_budget_projection_repair_ratio_mean`

### pause 结构指标
- `rhythm_metric_pause_support_cover_at_topk`
- `rhythm_metric_pause_recall_drop_post_from_planner`
- `rhythm_metric_pause_target_over_topk_rate`
- `rhythm_metric_pause_event_recall_boundary`
- `rhythm_metric_pause_event_recall_nonboundary`

### 阶段契约
- `L_base = 0`
- `L_pitch = 0`
- `rhythm_metric_module_only_objective = 1`
- `rhythm_metric_skip_acoustic_objective = 1`
- `rhythm_metric_disable_acoustic_train_path = 1`

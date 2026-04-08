# Conan Rhythm Branch

## 当前唯一维护主线（2026-04-08 UTC）

当前仓库唯一维护中的训练主线已经收敛为：

- `teacher_offline`
- `train100 + train360` mixed stage-1
- `v6` binary / cache lineage
- `split-head pause support + allocation`
- `weight-only warm-start`
- 统一启动入口：`scripts/autodl_train_stage1_mixed_v6_split_heads.sh`

> 结论：现在不要再把 `train360-only / legacy v5`、`pause_recall exact-resume`、`recovery_v3_trainonly` 视为当前主线。

---

## 当前数据资产

### train100 v6
- processed: `data/processed/libritts_train100_formal`
- binary: `data/binary/libritts_train100_formal_rhythm_v6`
- 角色：`base dataset`
  - 提供 train / valid / test
  - mixed 训练时承担验证与测试语义基座

### train360 v6
- processed: `data/processed/libritts_train360_formal_trainset`
- binary: `data/binary/libritts_train360_formal_trainset_rhythm_v6`
- 角色：`train-only supplemental dataset`
  - mixed 训练时通过 `train_sets='train100|train360'` 拼接进入训练
  - 不承担 valid / test 基座语义

---

## 当前训练语义

### warm-start
- 语义：`weight-only warm-start`
- 不是：`exact resume`
- 默认 bootstrap 位置：`checkpoints/bootstrap/model_ckpt_steps_17500.ckpt`

### 主配置
- `egs/conan_emformer_rhythm_v2_teacher_offline_train100_360_pause_split_heads_resume.yaml`

### 主脚本
- `scripts/autodl_train_stage1_mixed_v6_split_heads.sh`

这个入口会统一做：
- mixed v6 data override
- split-head config 选择
- preflight dry-run
- `val_check_interval=5000`
- `max_updates=80000`
- checkpoint 保留最近 `3` 个 step ckpt
- 同时保留：
  - `model_ckpt_best.pt`（整体最优）
  - `model_ckpt_pause_best.pt`（pause 最优，按 `rhythm_metric_pause_event_f1`）

---

## 当前建议启动命令

```bash
cd /root/autodl-tmp/project/conan-rhythm
bash scripts/autodl_train_stage1_mixed_v6_split_heads.sh
```

如需显式指定 17500 warm-start：

```bash
cd /root/autodl-tmp/project/conan-rhythm
BOOTSTRAP_CKPT=checkpoints/bootstrap/model_ckpt_steps_17500.ckpt \
MAX_UPDATES=80000 \
VAL_CHECK_INTERVAL=5000 \
bash scripts/autodl_train_stage1_mixed_v6_split_heads.sh
```

---

## 当前重点监控指标

基础指标：
- `total_loss`
- `L_exec_pause`
- `L_prefix_state`
- `rhythm_metric_pause_event_precision`
- `rhythm_metric_pause_event_recall`
- `rhythm_metric_pause_event_f1`
- `rhythm_metric_prefix_drift_l1`
- `rhythm_metric_exec_total_corr`
- `rhythm_metric_budget_projection_repair_ratio_mean`

pause 结构性诊断指标：
- `rhythm_metric_pause_support_cover_at_topk`
- `rhythm_metric_pause_recall_drop_post_from_planner`
- `rhythm_metric_pause_f1_drop_post_from_planner`
- `rhythm_metric_pause_target_over_topk_rate`
- `rhythm_metric_pause_event_recall_boundary`
- `rhythm_metric_pause_event_recall_nonboundary`

监控命令：

```bash
python scripts/monitor_stage1_metrics.py \
  --log logs/teacher_offline_train100_360_v6_split_heads_warm17500.log \
  --tail 5
```

---

## 历史/旧线说明

以下内容仍保留在仓库里，但不再作为默认主线入口：

- `scripts/autodl_recovery_mixed100360_v3_trainonly.sh`
- `scripts/autodl_resume_stage1_pause_recall.sh`
- `train360-only / legacy v5` 相关 handoff 记录

它们只应被视为：
- 历史实验记录
- legacy/debug 辅助脚本
- 对照/迁移参考

---

## 当前文档入口

1. `README.md`
2. `docs/autodl_training_handoff.md`
3. `docs/training_plan.md`

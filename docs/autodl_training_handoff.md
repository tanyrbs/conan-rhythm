# AutoDL Training Handoff（2026-04-07 UTC）

## 1. 当前项目状态

### 1.1 已完成、可直接复用的数据资产

- `data/processed/libritts_train100_formal`
- `data/binary/libritts_train100_formal_rhythm_v5`
- `data/processed/libritts_train360_formal_trainset`
- `data/binary/libritts_train360_formal_trainset_rhythm_v5`
- teacher warmup ckpt：`checkpoints/teacher_offline_train100_warmup/model_ckpt_steps_20000.ckpt`

这些资产足够继续做 **legacy v5 lineage** 的 teacher Stage-1 / pause-recall 实验。

### 1.2 raw 音频状态

为了控制磁盘，以下 raw 已删除：

- `/root/autodl-tmp/data/LibriTTS/train-clean-100`
- `/root/autodl-tmp/data/LibriTTS/train-clean-360`

因此：

- **继续训练当前 v5 binary 没问题**
- **任何需要重建 cache/binary 的实验都暂时不能直接做**
- 包括：`soft-boundary v6`、修改 guidance target 构造、修改 teacher cached target 的导出语义

---

## 2. 当前主训练线

### 2.1 当前 active run

当前主线已进一步切换为：

- **train360-only**
- **teacher_offline + teacher_as_main**
- **pause split-head support/allocation**
- 从旧 `consistent` 线的 **115000 ckpt** 做 **weight-only warm-start**

具体信息：

- 配置：`egs/conan_emformer_rhythm_v2_teacher_offline_train100_360_pause_split_heads_resume.yaml`
- 目标实验：`teacher_offline_train360only_pause_split_heads_stage1`
- warm-start ckpt：`checkpoints/teacher_offline_train360only_pause_recall_consistent_stage1/model_ckpt_steps_115000.ckpt`
- 新 work dir：`checkpoints/teacher_offline_train360only_pause_split_heads_stage1`
- 日志：`logs/stage1_train360only_pause_split_heads_from115000.log`
- 训练数据：`data/binary/libritts_train360_formal_trainset_rhythm_v5`
- `val_check_interval = 2500`
- `max_updates = 135000`

说明：

- 这条新线**不是** exact resume，因为新增了 `pause_support_head`
- 因此保留旧 `consistent` 线作为对照，新 split-head 线从 step `0` 重新记步
- warm-start 时允许新 head 缺失权重，其他主干权重从 `115000.ckpt` 载入

### 2.2 为什么停掉 boundary-relaxed 线

旧的 `teacher_offline_train360only_pause_recall_boundary_relaxed_stage1` 已停止。

原因不是 budget，也不是单一 bug，而是当前 stage 的 pause 训练存在**结构性错配**：

1. **target 面**：主监督是 `guidance`，而 guidance pause surface 本身偏 boundary-heavy
2. **planner 面**：teacher_offline 的 `pause_weight_unit` 是 softmax simplex，天然更偏“少数赢家”
3. **projector 面**：pause 在 `sparse top-k + boundary bias + 低温 soft gate` 下形成稀疏瓶颈
4. **loss 面**：event/support/value 目标之前不完全同向，容易把系统推成“边界点少而准”

所以会稳定出现：

- `pause_event_precision` 高
- `pause_event_recall` 低
- `L_budget` 与 repair 看起来正常
- 但 `prefix_drift_l1` 仍难降

---

## 3. 当前已经落地的代码修复

### 3.0 新增：pause support / allocation 分头

本轮已新增一条真正对准“support / allocation 解耦”的结构修复：

- planner 新增 `pause_support_head`
- `pause_support_logit_unit -> sigmoid -> pause_support_prob_unit`
- 原 `pause_head` 继续承担 amount / allocation
- `pause_amount_weight_unit -> softmax`
- `pause_candidate_score_unit = support_prob * amount_weight`
- 兼容输出 `pause_weight_unit = normalized(candidate_score)`

同时新增两类 support 监督：

- `rhythm_pause_support_event_weight`
- `rhythm_pause_support_count_weight`

其中：

- `pause_support_weight` 继续作为 planner allocation / shape KL
- `pause_support_event` 用稀疏正例 supervision 修 support
- `pause_support_count` 约束 support 激活率不要塌

对应新配置：

- `egs/conan_emformer_rhythm_v2_teacher_offline_train100_360_pause_split_heads_resume.yaml`

对应关键文件：

- `modules/Conan/rhythm/offline_teacher.py`
- `modules/Conan/rhythm/projector.py`
- `modules/Conan/rhythm/contracts.py`
- `tasks/Conan/rhythm/losses.py`
- `tasks/Conan/rhythm/targets.py`
- `tasks/Conan/rhythm/task_runtime_support.py`
- `tasks/Conan/rhythm/metrics.py`

### 3.1 已修：pause_event 默认不再 boundary-weighted

文件：

- `tasks/Conan/rhythm/losses.py`
- `tasks/Conan/rhythm/targets.py`
- `tasks/Conan/rhythm/task_runtime_support.py`

当前默认语义：

- `l_exec_pause_value` 仍可按 boundary-weighted mask 训练
- `l_pause_event` 默认走 `unit_mask`
- `pause_event_boundary_weight` 默认 `0.0`

这让 event loss 真正承担“全局补 recall”的职责，而不是继续偏向少数强边界点。

### 3.2 已修：teacher 阶段不再因 `force_full_commit=True` 被动退化成 hard top-k

文件：

- `modules/Conan/rhythm/projector.py`
- `modules/Conan/rhythm/module.py`

当前修复：

- projector 新增 `allow_soft_pause_selection_with_force_full_commit`
- `forward_teacher()` 显式传 `True`

效果：

- teacher 阶段仍可 full commit
- 但 pause support 训练时不再默认被 hard top-k 卡死
- 这是当前最重要的结构性修复之一

### 3.3 已补：pause 诊断指标

文件：

- `tasks/Conan/rhythm/metrics.py`

已新增：

- planner vs post-projector recall/f1
- `pause_support_cover_at_topk`
- `pause_target_over_topk_rate`
- threshold sweep (`t02/t03/t05`)
- `best_f1 / best_threshold`
- boundary / non-boundary recall split
- `pause_soft_selection_active`
- `pause_topk_ratio_used_mean`
- `pause_soft_temperature_mean`
- `force_full_commit_mean`

这些指标是当前判断主因最重要的证据链。

### 3.4 当前测试状态

已通过：

```bash
OMP_NUM_THREADS=1 /root/miniconda3/envs/conan/bin/python -m unittest \
  tests.rhythm.test_projector_invariants \
  tests.rhythm.test_loss_components \
  tests.rhythm.test_task_runtime_support \
  tests.rhythm.test_metrics_masking
```

结果：

- `50 passed`

补充：

- split-head 实现本轮新增后，已再次通过核心节奏单测回归
- 本地通过：

```bash
OMP_NUM_THREADS=1 python -m unittest \
  tests.rhythm.test_loss_components \
  tests.rhythm.test_projector_invariants \
  tests.rhythm.test_metrics_masking \
  tests.rhythm.test_factorization_contract
```

- 结果：`40 passed`

---

## 4. 当前 consistent resume 配置的真实意图

`conan_emformer_rhythm_v2_teacher_offline_train100_360_pause_recall_consistent_resume.yaml` 的目标不是“彻底解决 pause”，而是做一条**方向一致**的续训线：

- 降低 active path 的 boundary 过度强化
- 开大 support capacity
- 提高 soft gate 温度
- 把 event threshold 从 `0.50` 降到 `0.30`
- 保持 legacy v5 数据不变，先验证当前结构修复能否带来真正的 recall 收益

当前关键参数：

```yaml
rhythm_pause_event_weight: 0.20
rhythm_pause_support_weight: 0.06
rhythm_pause_event_threshold: 0.30
rhythm_pause_event_temperature: 0.22
rhythm_pause_event_boundary_weight: 0.0

rhythm_pause_source_boundary_weight: 0.06
rhythm_boundary_feature_scale: 0.22
rhythm_boundary_source_cue_weight: 0.55

rhythm_projector_pause_boundary_bias_weight: 0.10
rhythm_pause_boundary_weight: 0.22

rhythm_projector_pause_topk_ratio: 0.50
rhythm_projector_pause_topk_ratio_train_start: 0.44
rhythm_projector_pause_topk_ratio_train_end: 0.50
rhythm_projector_pause_topk_ratio_anneal_steps: 15000
rhythm_projector_pause_soft_temperature: 0.18
```

---

## 5. 现在还没改、且必须区分“在线旋钮”和“需要重建 cache 的旋钮”

### 5.1 现在可直接通过 resume 生效的

- `rhythm_pause_event_*`
- `rhythm_pause_support_weight`
- `rhythm_pause_source_boundary_weight`
- `rhythm_boundary_feature_scale`
- `rhythm_boundary_source_cue_weight`
- `rhythm_projector_pause_boundary_bias_weight`
- `rhythm_projector_pause_topk_ratio*`
- `rhythm_projector_pause_soft_temperature`

### 5.2 现在**不能**靠 resume 直接宣称已验证的

- `rhythm_guidance_pause_strength`
- `rhythm_guidance_boundary_strength`
- soft-boundary `ref_rhythm_trace` / `rhythm_cache_version: 6`
- 任何 teacher/export cache 语义变化

原因：

这些都属于 **cache/binary lineage** 的一部分；没有 raw 就无法重建，当前只能继续把这轮实验表述为：

- **legacy v5 train360-only pause-recall structural fix line**

而不是：

- **v6 / rebuilt guidance target line**

---

## 6. 当前最重要的监控指标

### S 级

- `rhythm_metric_pause_event_recall`
- `rhythm_metric_pause_event_f1`
- `rhythm_metric_pause_event_precision`
- `rhythm_metric_prefix_drift_l1`
- `rhythm_metric_planner_pause_event_recall`
- `rhythm_metric_pause_recall_drop_post_from_planner`
- `rhythm_metric_pause_support_cover_at_topk`
- `rhythm_metric_pause_target_over_topk_rate`
- `rhythm_metric_pause_event_recall_t02 / t03 / t05`
- `rhythm_metric_pause_event_best_f1`
- `rhythm_metric_pause_event_best_threshold`
- `rhythm_metric_pause_event_recall_boundary`
- `rhythm_metric_pause_event_recall_nonboundary`
- `rhythm_metric_pause_soft_selection_active`

### 阶段契约

- `L_base = 0`
- `L_pitch = 0`
- `rhythm_metric_module_only_objective = 1`
- `rhythm_metric_skip_acoustic_objective = 1`
- `rhythm_metric_disable_acoustic_train_path = 1`

### A 级

- `L_exec_pause`
- `L_pause_event`
- `L_pause_support`
- `L_prefix_state`
- `rhythm_metric_exec_total_corr`
- `rhythm_metric_budget_projection_repair_ratio_mean`

---

## 7. 当前最合理的后续 A/B 顺序

### A/B-1（当前已在跑）

- consistent sparse line
- 目标：先验证 teacher soft pause selection + 一致化 boundary 下降 是否足以修 recall

### A/B-2（若当前线仍卡住）

最优先短诊断：

- `rhythm_projector_pause_selection_mode: simple`
- 只跑 2k~5k

目的：

- 快速确认根因是不是 sparse projector bottleneck 本身

### A/B-3（需要 raw 后再做）

- guidance target 去 boundary 化
- 例如：`guidance_boundary_strength 1.25 -> 0.75~1.00`
- 并重建 cache/binary

---

## 8. 当前最少要记住的命令

查看日志：

```bash
cd /root/autodl-tmp/project-1/conan-rhythm

tail -f logs/stage1_train360only_pause_recall_consistent_from105000.log
```

查看最新 validation：

```bash
python scripts/monitor_stage1_metrics.py \
  --log logs/stage1_train360only_pause_recall_consistent_from105000.log \
  --tail 5
```

如果要确认当前 source ckpt：

```bash
ls checkpoints/teacher_offline_train360only_pause_recall_stage1/model_ckpt_steps_105000.ckpt
```

# Training Plan（Teacher-First，2026-04-07 UTC）

## 1. 总原则

当前项目继续执行 **teacher-first**：

- 先把 teacher 做强
- 先把 pause placement / prefix consistency / strong-rhythm 做稳
- 先把 teacher surface 语义理顺
- **最后**再考虑 student KD

当前 pause 训不好的主因不是 budget，也不是一个单独 bug，而是当前 stage 存在：

- **稠密 guidance target**
- **softmax planner pause surface**
- **sparse projector bottleneck**
- **boundary 多层注入**
- **event/support/value 监督口径不完全一致**

合起来会把系统推向：

- 边界点少而准
- 非强边界但应该停的位置放不出来
- `precision` 高、`recall` 低
- `prefix_drift_l1` 难降

---

## 2. 当前 active 训练路线

### T1-current：train360-only teacher_offline pause split-head line

当前主线已经切到 split-head：

- `teacher_offline_train360only_pause_split_heads_stage1`
- 从 `teacher_offline_train360only_pause_recall_consistent_stage1@115000` 做 weight-only warm-start
- 使用现成 `train360 v5 binary`
- `val_check_interval = 2500`

配置：

- `egs/conan_emformer_rhythm_v2_teacher_offline_train100_360_pause_split_heads_resume.yaml`

目标：

1. 直接验证 support / allocation 分头能否提升 recall
2. 验证 projector 的 `boundary gain` 是否比 additive floor 更稳
3. 在 legacy v5 lineage 下先把**在线可修**的问题修掉

### 为什么不再继续把 old consistent line 当主线

旧 `teacher_offline_train360only_pause_recall_consistent_stage1` 已保留为对照，但不再是主线。

已完成验证的关键点：

- `@107500`: recall `0.4231`, f1 `0.5221`, prefix drift `31.4913`, exec corr `0.8863`
- `@112500`: recall `0.5067`, f1 `0.5950`, prefix drift `30.3021`, exec corr `0.8866`
- `@115000`: recall `0.4447`, f1 `0.5264`, prefix drift `40.1634`, exec corr `0.8664`
- `@117500`: recall `0.3628`, f1 `0.4276`, prefix drift `35.2436`, exec corr `0.8785`

结论：

- 这条线在 `112500` 出现过短暂最好点
- 但 `115000` 相比 `105000` 起训阶段附近的最近可比验证点（`107500`）**没有稳定进步**
- 因此选 `115000` 作为 split-head 的 warm-start，只是出于“主干已收敛、可直接加新头试结构修复”，不是因为 `115000` 本身是 pause 最优点

### 为什么不是继续跑 boundary-relaxed

因为旧 boundary-relaxed 线虽然开始意识到 boundary 过强，但仍没有彻底解决：

- teacher 训练实际被 `force_full_commit=True` 顺手关掉了 soft pause selection
- guidance target 仍是 boundary-heavy
- sparse top-k + low temperature + additive boundary bias 仍是硬瓶颈
- event threshold 仍偏硬

所以 consistent line 的改动更加同向一致。

---

## 3. 当前已经落地的训练相关修复

### 3.0 新增：split-head support / allocation

当前最重要的新修复是：

- `support head` 负责“哪里该停”
- `allocation head` 负责“停顿质量怎么分”

并新增：

- `L_pause_support_event`
- `L_pause_support_count`

这样不再让单个 softmax 同时承担：

- support 激活
- pause budget 分配

### 3.1 teacher 阶段启用可训练的 soft pause selection

现已修复：

- `forward_teacher()` 不再因为 `force_full_commit=True` 自动退化成 hard top-k
- teacher 阶段现在可以 full commit + soft pause selection 同时成立

这一步优先级高于继续微调 top-k 0.40 -> 0.42 这类小改动。

### 3.2 event loss 默认不再继续偏向 boundary

现已修复：

- `pause_event_boundary_weight` 默认 `0.0`
- `pause_event` 默认承担全局 recall 补漏，而不是“边界点别漏太多”

### 3.3 已补 pause 结构诊断指标

当前实验不再只看：

- `pause_event_precision / recall / f1`

还要看：

- planner vs post-projector recall
- `pause_support_cover_at_topk`
- `pause_target_over_topk_rate`
- threshold sweep (`t02/t03/t05`)
- `best_f1 / best_threshold`
- boundary / non-boundary recall split

---

## 4. 当前 consistent line 的训练假设

### 假设 A：主因之一是 sparse projector bottleneck

因此当前线做了：

- `pause_topk_ratio: 0.50`
- `pause_soft_temperature: 0.18`
- teacher 阶段 soft pause selection 重新打开

### 假设 B：后置 boundary 强化过强

因此当前线做了：

- `pause_source_boundary_weight: 0.06`
- `boundary_feature_scale: 0.22`
- `boundary_source_cue_weight: 0.55`
- `projector_pause_boundary_bias_weight: 0.10`
- `pause_boundary_weight: 0.22`

### 假设 C：event 阈值过硬

因此当前线做了：

- `pause_event_threshold: 0.30`

### 假设 D：support loss 不是主刀，只是辅助手段

因此当前线：

- 保留 `pause_support_weight`
- 但不把主要希望押在它身上

---

## 5. 当前阶段该怎么判断“有效”

### 最值得盯的不是 budget，而是这几组

#### 支撑链路是否通了

- `rhythm_metric_planner_pause_event_recall`
- `rhythm_metric_pause_recall_drop_post_from_planner`
- `rhythm_metric_pause_support_cover_at_topk`
- `rhythm_metric_pause_target_over_topk_rate`

#### 真正 pause 是否起来了

- `rhythm_metric_pause_event_recall`
- `rhythm_metric_pause_event_f1`
- `rhythm_metric_pause_event_precision`
- `rhythm_metric_pause_event_recall_t02 / t03 / t05`
- `rhythm_metric_pause_event_best_f1`

#### 是否只会在边界点停

- `rhythm_metric_pause_event_recall_boundary`
- `rhythm_metric_pause_event_recall_nonboundary`
- `rhythm_metric_pause_fn_boundary_mean`

#### prefix 是否跟着改善

- `rhythm_metric_prefix_drift_l1`
- `L_prefix_state`

#### feasibility 是否仍干净

- `L_budget`
- `rhythm_metric_budget_projection_repair_ratio_mean`

---

## 6. 当前 run 后面的最优先 A/B

### A/B-1：继续看 consistent sparse line

这是当前主线，不中断。

### A/B-2：若 consistent 线仍旧 recall 不起

最优先短诊断不是继续微调 boundary，而是：

```yaml
rhythm_projector_pause_selection_mode: simple
```

只跑一个 2k~5k block。

它最能回答：

- planner 本身是否已经学到 pause support
- 还是 sparse projector 仍是主瓶颈

### A/B-3：有 raw 之后再做

当 raw 恢复后，再做：

- guidance target 去 boundary 化
- rebuild cache/binary
- v6/soft-boundary lineage

这一步不能和当前 legacy v5 run 混在一起叙事。

---

## 7. 当前不建议做什么

### 7.1 不建议继续“开一点 top-k，同时再加一点 boundary bias”

这是方向打架。

### 7.2 不建议只继续加 `pause_support_weight`

如果 projector 还是主瓶颈，这样收益有限。

### 7.3 不建议现在切到 algorithmic teacher 作为 primary target

因为 algorithmic teacher 本身仍更 boundary-heavy，当前 recall 已低时很可能更糟。

---

## 8. 当前训练的最低表达

当前这轮实验应被描述为：

- **legacy v5 train360-only teacher pause-recall structural-fix run**

而不是：

- soft-boundary v6 已验证
- rebuilt guidance target 已验证
- final teacher 已定稿

---

## 9. 当前最小执行命令

日志：

```bash
tail -f logs/stage1_train360only_pause_recall_consistent_from105000.log
```

最近验证：

```bash
python scripts/monitor_stage1_metrics.py \
  --log logs/stage1_train360only_pause_recall_consistent_from105000.log \
  --tail 5
```

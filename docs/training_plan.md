# Training Plan（Teacher-First，2026-04-07 UTC）

## 1. 总原则

当前项目改成 **teacher-first 上限路线**：

- 先把 teacher 做强
- 先把强节奏变化、pause placement、prefix consistency 做稳
- 先把 teacher target 语义做干净
- **最后**再做 student 蒸馏

核心原则：

1. **teacher 上限优先于 student 速度**
2. **所有 teacher 阶段都保留最佳 checkpoint**
3. **每个 teacher 阶段都做固定 audit set 审计**
4. **尽量不破坏后续蒸馏合同**

---

## 2. 当前训练总路线

### Phase T1：Teacher Stage 1（当前正在跑）

目标：

- 学稳 scheduler / controller / projector / prefix consistency
- 得到可执行、合同干净的 base teacher

当前入口：

- `egs/conan_emformer_rhythm_v2_teacher_offline_train100_360.yaml`
- `scripts/autodl_recovery_mixed100360_v3_trainonly.sh`

当前观察重点：

- `pause_event_recall` 继续涨，但不要让 `precision` 快速塌掉
- `prefix_drift_l1` / `deadline_final_abs_mean` 继续缓慢下降
- 说明当前最关键短板已经不是 `L_budget`，而是 pause placement recall 与 prefix 收口

### Phase T2：Teacher Stage 2（Stage 1 后的 teacher polish）

目标：

- 在 **不改 teacher 主合同** 的前提下继续提升上限
- 重点打磨：
  - `pause_event_f1`
  - `prefix_drift_l1`
  - 强节奏变化下的稳定性
- 优先做法：
  - 延续 teacher-only 训练
  - 降学习率
  - 小幅调权重，不重写结构

原则：

- 不把 teacher 搞成 acoustic-heavy
- 不大改 target source
- 不在这一阶段破坏后续可蒸馏性

### Phase T2.5：Teacher v2.5（强节奏变化 / external-ref teacher 增强）

目标：

- 先在 teacher 身上验证更强的节奏变化能力
- 先把 external-reference / strong-rhythm 相关能力做出来
- 先把 one-to-many 下的节奏分离能力做强

说明：

- 这是**teacher 侧增强阶段**，不是当前默认 student 阶段
- 现有 student 配置不能直接等价替代这一步的 teacher 目标

### Phase T3：Teacher v3（鲁棒性 / cache 刷新 / binary 重建）

目标：

- 在 teacher 侧做鲁棒性补强
- 例如：
  - soft boundary trace 生效
  - cache / binary 重建
  - exhaustion-aware 相关增强
  - short-ref / long-source robustness

注意：

- 这一步如果涉及 `ref_rhythm_trace` 分布变化，必须单独管理
- raw 音频恢复前，不能真的重做 train100/train360 binarize

---


## 2.1 cache version gate

soft-boundary `ref_rhythm_trace` 的定义变化已经被记为 **cache version 6**。

因此：

- v6 之后的任何新 `cached_only` 实验，都必须使用重建后的 binary/cache
- 旧 v5 binary 只能被视为旧 lineage，不应和 soft-boundary 结果混做同一 ablation 叙事
- 当前正在跑的 teacher Stage 1 可以继续，因为它是在旧 lineage 上已经启动的进程

### 当前操作约束（2026-04-07）

- mixed teacher Stage 1 真正可用的 binary 还是：
  - `data/binary/libritts_train100_formal_rhythm_v5`
  - `data/binary/libritts_train360_formal_trainset_rhythm_v5`
- 本机 raw LibriTTS 路径已经缺失，所以**不能立刻重建 v6**
- 因此当前 pause-recall 续训采用的是：
  - 相同 `exp_name/work_dir` 精确 resume
  - mixed lineage 不变
  - `val_check_interval=2500`
  - checkpoint 也随 validation 每 2500 step 落一次
  - 显式 `rhythm_cache_version=5 + rhythm_allow_legacy_cache_resume=True`

这意味着：

- 当前继续训练的是 **legacy v5 cache lineage**
- 当前比较 pause recall / prefix 的实验结论仍然有效
- 但**不能**把这轮结果表述成“soft-boundary v6 已验证”

## 3. 每个 teacher 阶段结束后必须做什么

### 3.1 固化 checkpoint

每个 teacher 阶段至少保留：

- best by validation
- best by pause/prefix 指标
- best by audit 听感

建议命名：

- `teacher_stage1_best_val`
- `teacher_stage1_best_pause_prefix`
- `teacher_stage1_best_audit`
- `teacher_stage2_best_*`
- `teacher_v25_best_*`
- `teacher_v3_best_*`

### 3.2 固定 audit set

固定一组 10 条音频，后续所有 teacher 阶段都重复用：

- 长短不同
- pause 风格不同
- 节奏变化强弱不同
- 至少覆盖若干强节奏变化样本

每次审计导出：

- 音频结果
- `global_rate`
- `pause_ratio`
- `local_rate_trace`
- `boundary_trace`

### 3.3 审计通过才允许进入下一阶段

teacher 不应该只凭 `val_loss` 进入下一阶段。

至少要同时满足：

- 阶段契约正确
- pause / prefix 指标达标
- 人耳听感通过
- projector repair 没有异常依赖

---

## 4. 当前 Stage 1 的退出标准

当前建议不是“跑满 200k 再说”，而是分窗口判断。

### 4.1 当前推荐观察窗口

- 第一认真挑 ckpt 窗口：**80k ~ 120k**
- 200k 是上限，不是必须跑满的目标

### 4.2 必看指标

#### 阶段契约

- `L_base = 0`
- `L_pitch = 0`
- `rhythm_metric_module_only_objective = 1`
- `rhythm_metric_skip_acoustic_objective = 1`
- `rhythm_metric_disable_acoustic_train_path = 1`

#### 核心 teacher 质量

- `L_exec_pause`
- `L_prefix_state`
- `rhythm_metric_pause_event_f1`
- `rhythm_metric_prefix_drift_l1`
- `rhythm_metric_exec_total_corr`
- `rhythm_metric_budget_projection_repair_ratio_mean`

### 4.4 当前 pause recall 专项修正策略

截至当前 mixed `train100 + train360` Stage 1，最关键短板不是 `L_budget`，而是：

- `pause_event_precision` 明显高于 `pause_event_recall`
- 模型更像是“保守地打 pause support”
- 漏 pause 会直接拖慢 `prefix_drift_l1` 的收敛

因此当前 Stage 1 后段不建议只继续硬跑原配置，而建议改成 **5k 一档** 的保守调参续训。

#### 为什么继续保持 `val_check_interval = 5000`

`5000` 这个验证频率**不会直接改变 pause objective 本身**。  
它在当前 trainer 里的作用主要是：

- 评估频率
- best checkpoint 刷新频率
- 我们每 5k 一次人工复盘的节奏锚点

也就是说：

- **影响 pause 的，是配置变化本身**
- **不是 5k 验证间隔本身**

#### 当前推荐的 5k staged plan

##### Block A：`75k -> 80k`

目标：

- 先补 recall
- 但不要一下子把 precision 打崩

配置：

- `egs/conan_emformer_rhythm_v2_teacher_offline_train100_360_pause_recall.yaml`

关键参数：

```yaml
rhythm_pause_event_weight: 0.15
rhythm_pause_event_threshold: 0.50
rhythm_pause_event_temperature: 0.20
rhythm_pause_event_pos_weight: 2.0
rhythm_projector_pause_topk_ratio: 0.40
rhythm_projector_pause_boundary_bias_weight: 0.18
rhythm_pause_boundary_weight: 0.45
```

##### Block B：`80k -> 85k`

如果满足：

- recall 仍然没有明显抬升
- precision 仍然高于约 `0.68`

则可以加一点 support 容量：

```yaml
rhythm_pause_event_weight: 0.20
rhythm_projector_pause_topk_ratio: 0.45
rhythm_projector_pause_boundary_bias_weight: 0.22
```

##### Block C：`85k -> 90k`

如果出现：

- precision 明显下滑
- F1 没有提升

则不要继续猛加 top-k，而是先稳住：

```yaml
rhythm_pause_event_weight: 0.15
rhythm_projector_pause_topk_ratio: 0.40
rhythm_pause_event_pos_weight: 2.0
```

#### 每个 5k block 结束后怎么判断是否继续

重点看：

- `pause_event_recall`
- `pause_event_f1`
- `prefix_drift_l1`
- `exec_total_corr`
- `L_exec_pause_value`
- `L_pause_event`

预期顺序通常是：

1. `L_pause_event` 先降
2. recall 先抬
3. precision 可能小幅回落
4. F1 上升
5. `prefix_drift_l1` 跟着更快下降

### 4.3 什么时候可以结束 Stage 1

满足下面条件后，可以从 Stage 1 退出并进入 teacher Stage 2：

- 指标连续 2~3 个 validation window 改善很小
- `pause_event_f1` 基本平台
- `prefix_drift_l1` 基本平台
- `repair_ratio_mean` 持续低位
- 固定 audit set 听感过关

### 4.5 断点续训语义

当前代码路径支持 **同 exp 精确续训 + 覆盖新配置**：

- 使用同一个 `exp_name`
- `RESET=1`
- 不删除旧 checkpoint

这样会发生两件事：

1. `config.yaml` 会被新的 YAML 覆盖
2. trainer 仍会从该 `exp_name` 下最新 `model_ckpt_steps_*.ckpt` 恢复

因此：

- **global_step 会接着走**
- **optimizer state 会接着走**
- **新的 pause 配置会立刻生效**

当前建议用：

- `scripts/autodl_resume_stage1_pause_recall.sh`

默认行为就是：

- 找到当前 exp 最新 ckpt
- 目标步数自动设成 `latest + 5000`
- 用新 pause-recall YAML 继续跑下一档 5k

---

## 5. 为什么现在不直接做学生蒸馏

因为当前我们更想要的是：

- 一个**上限更高**的 teacher
- 一个**强节奏变化更好**的 teacher
- 一个后续可以被多个 student 阶段复用的 teacher 家族

这样做的好处是：

- 后续 student_kd 会更快
- 蒸出来的 student 上限更高
- 不必在 teacher 还没成熟时频繁返工 student

---

## 6. 学生阶段什么时候开始

只有在下面都成立时，才进入学生：

1. teacher Stage 1 / 2 / 2.5 / 3 至少完成到计划中的停止点
2. 已保留各 teacher 阶段最佳 ckpt
3. 已完成固定 audit set 审计
4. 已确定“哪一个 teacher / 哪一阶段的 teacher target”要拿来蒸馏

当前默认顺序：

1. 先 teacher 家族
2. 再 `student_kd`
3. 之后再决定是否进入其他 student 分支

---

## 7. 当前不该做的事

- 不要把现有 student 配置当成当前默认主线
- 不要在 teacher 还没成熟时急着蒸 student
- 不要在当前 Stage 1 中途大改 cache contract
- 不要把 soft boundary 的 binary 重建插进当前正在跑的 teacher Stage 1

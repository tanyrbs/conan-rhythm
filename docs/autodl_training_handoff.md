# AutoDL Training Handoff（2026-04-07 UTC）

这份文档只回答 3 个问题：

1. **项目现在到底进行到哪里了？**
2. **每个训练阶段的目的是什么？**
3. **我们现在为什么先不走 `student_kd`，而是 teacher 后优先上 v2.5？**

---

## 1. 当前项目真实状态

### 1.0 本次相对 `origin/main` 的代码侧结论

这轮没有盲目追 main，而是做了“吸收 upstream 明显优点 + 保留本地更保守训练策略”的组合：

- **吸收 upstream**
  - stage-3 projector 显式默认切到 `sparse + boundary_commit_guard + render_plan`
- **本地补强**
  - `_sample_trace_pair()` 三返回值 contract 修复
  - stage-3 acoustic ramp 改成 runtime 锚点，而不是绝对 `global_step`
  - stage-3 pitch loss 默认跟随 acoustic curriculum
  - algorithmic teacher 的 pause top-k 改成 per-sample row-wise
  - online retimed target 权重接入 `trace_reliability.local_gate`
  - zero-speech descriptor robustness 修复
- **需要注意**
  - soft `boundary_trace` 已改到代码，但**只有重建 binary/cache 后才会生效**
  - 当前正在跑的 stage-1 teacher 不会自动吃到这项 cache 侧更新

### 1.1 已经完成的资产

#### train100

- processed：`data/processed/libritts_train100_formal`
- binary：`data/binary/libritts_train100_formal_rhythm_v5`
- warmup checkpoint：`checkpoints/teacher_offline_train100_warmup/model_ckpt_steps_20000.ckpt`

#### train360

- processed：`data/processed/libritts_train360_formal_trainset`
- binary：`data/binary/libritts_train360_formal_trainset_rhythm_v5`

#### 当前保留的关键大小

- `data/processed/libritts_train100_formal`：约 `140M`
- `data/binary/libritts_train100_formal_rhythm_v5`：约 `13G`
- `data/processed/libritts_train360_formal_trainset`：约 `375M`
- `data/binary/libritts_train360_formal_trainset_rhythm_v5`：约 `33G`

### 1.2 已经删掉的东西

为防止磁盘再次溢出，下面这些 raw 音频已经删除：

- `/root/autodl-tmp/data/LibriTTS/train-clean-100`
- `/root/autodl-tmp/data/LibriTTS/train-clean-360`

这意味着：

- **继续跑 mixed stage-1 训练没有问题**，因为它复用现成 binary
- **如果要重做 train100 / train360 metadata 或 binarize，就必须重新挂载 raw 音频**

### 1.3 当前没有在跑的东西

截至 **2026-04-07 UTC**，没有检测到正在运行的：

- mixed100+360 stage-1 正式训练
- preflight
- binarize

也就是说：

> **现在不是“训练已经结束”，而是“mixed formal stage-1 还没真正启动起来”。**

---

## 2. 最近一次失败到底是什么

最近一次恢复流水线：

- `logs/stage1_recovery_mixed100360_v2_20260406_091257.sh`

最终状态：

- `logs/stage1_recovery_mixed100360_v2_20260406_091257.status`
- 内容：`2026-04-06T18:19:23Z PIPELINE_FAILED rc=1`

### 2.1 它失败在什么地方

不是失败在：

- train360 metadata
- train360 binarize

而是失败在：

- **mixed preflight 的 model dry-run**

日志关键错误：

- `Model dry-run failed for split 'train': not enough values to unpack (expected 3, got 2)`

### 2.2 根因

根因已定位：

- 文件：`modules/Conan/rhythm/module.py`
- 函数：`_sample_trace_pair()`

问题是：

- 调用方期望返回 `3` 个值
- 实际函数只返回 `2` 个值

### 2.3 当前修复状态

本地已经补了修复：

- `_sample_trace_pair()` 现在返回：
  - `trace_context`
  - `planner_trace_context`
  - `reliability`

同时本地新增了更安全的恢复脚本：

- `scripts/autodl_recovery_mixed100360_v3_trainonly.sh`

它的关键作用是：

- **直接复用已经完成的 train360 processed / binary**
- 跳过已经做完的 metadata / binarize
- 继续执行：
  - `TRAIN360_PREFLIGHT`
  - `MIXED_PREFLIGHT`
  - `MIXED_REAL_SMOKE`
  - `FORMAL_STAGE1_RUN`

---

## 3. 现在训练的真正目的是什么

当前不是泛泛而谈“训练模型”，而是有明确阶段目标。

### 3.1 Stage 1：`teacher_offline`

配置主入口：

- `egs/conan_emformer_rhythm_v2_teacher_offline.yaml`
- 当前 mixed 入口：`egs/conan_emformer_rhythm_v2_teacher_offline_train100_360.yaml`

#### 这个阶段的目的

- 先把 **offline teacher** 学出来
- 先把：
  - scheduler
  - controller
  - projector
  - prefix consistency
  - budget / execution 结构
  学稳
- 保留 full/global teacher 的全局规划能力
- 为后续 external-ref student 阶段提供强初始化

#### 这个阶段不是干什么

- 不是最终部署模型
- 不是最终 acoustic closure
- 不是 external-ref 上限实验本身

#### 当前这一阶段的真实目标

> 用现有 `train100` + `train360` binary 跑通 mixed formal stage-1，拿到真正可用的 teacher checkpoint。

---

### 3.2 为什么 teacher 后先不走 `student_kd`

这是当前策略变化里最重要的一点。

#### 先说结论

对我们现在的核心目标，**teacher 之后优先上 `student_ref_bootstrap`，比先上 `student_kd` 更对焦。**

#### 原因

因为当前真正想回答的问题不是：

- student 能不能稳定复现 cached/self-conditioned teacher surface

而是：

- 模型是否真的使用了外部 `B`
- one-to-many 下会不会 collapse 成平均计划
- descriptor 语义是否真的跟着 `B` 变化

而 `student_kd` 当前更偏向：

- `cached_only`
- `teacher_target_source: learned_offline`
- `distill_surface: cache`

它更适合作为：

- baseline
- ablation
- 稳定性对照

但**不是当前最直接回答 external-reference 问题的第一 student 阶段。**

---

### 3.3 teacher 结束后，优先进入的阶段：`student_ref_bootstrap`

配置：

- `egs/conan_emformer_rhythm_v2_student_ref_bootstrap.yaml`
- `egs/conan_emformer_rhythm_v2_student_pairwise_ref_runtime_teacher.yaml`

#### 这个阶段的目的

- external reference / pairwise bootstrap
- 检查模型是否真的根据外部 `B` 改变 rhythm，而不是：
  - 输出平均计划
  - 轻度依赖 B 但不够强
  - 干脆忽略 B
- 把它作为 **teacher 之后的第一 student-facing distillation / bootstrap 阶段**

#### 为什么它现在更该优先做

因为它已经正面对准了：

- `runtime_only`
- `sample_ref`
- `rhythm_require_external_reference: true`
- `lambda_rhythm_descriptor_consistency`
- `lambda_rhythm_pairwise_contrastive`
- `lambda_rhythm_pairwise_diversity`

也就是说，它已经不只是“换个 ref”，而是在训练机制上开始逼模型：

- 输出的 rhythm semantics 要像 `B`
- same-A 面对不同 `B` 时必须分开
- 不能塌成平均答案

#### 但必须说清楚的限制

当前这一步要说严谨：

- 它仍然是 `runtime_only`
- `rhythm_teacher_target_source` 仍然是 `algorithmic`

所以更准确地说，它是：

> **teacher warm-start + external-ref bootstrap / first student-facing distillation stage**

而**还不是**完全摆脱 algorithmic ceiling 的最终 learned-teacher distill 终局。

这点文档里必须说清楚，不能自我催眠。

---

### 3.4 `student_kd` 现在在项目中的定位

配置：

- `egs/conan_emformer_rhythm_v2_student_kd.yaml`

#### 这个阶段现在还保不保留

保留。

#### 但它现在不再是什么

它**不再是 teacher 之后的默认下一步**。

#### 它现在更像什么

- baseline
- ablation
- 稳定性验证分支
- 对照“cached teacher surface 能蒸到什么程度”

所以当前更准确的说法是：

- `student_kd` 不是被删除
- 而是从“主链默认下一步”降级成“辅助支线”

---

### 3.5 Stage 3：`student_retimed`

配置：

- `egs/conan_emformer_rhythm_v2_student_retimed.yaml`

#### 这个阶段的目的

- 把 rhythm control 真正闭环到 retimed acoustic canvas 上
- 让输出声学结果也服从节奏控制，而不只是控制层指标好看

#### 为什么它不是现在要做的事

因为：

- 它依赖前面 teacher / v2.5 student 路径先成立
- 它还依赖 retimed mel / F0 sidecar 等更重的资产合同
- 当前真正卡住的，不是 stage-3，而是 mixed stage-1 还没正式跑起来

---

## 4. 当前最重要的批判性结论

### 4.1 我们现在不是“数据没准备好”

不是。

至少对 mixed stage-1 来说：

- train100 binary 已有
- train360 binary 已有

所以现在的 blocker 不是数据准备，而是：

- v2 脚本在 mixed preflight model dry-run 上挂掉
- 需要切到本地修过 bug 的 v3 流程

### 4.2 我们现在也不是“该切 student 了”

也不是。

因为当前最关键的 teacher formal mixed stage-1 还没跑起来。

所以现在如果跳 student，会把训练链条打乱。

### 4.3 当前正确叙述

最准确的说法应该是：

> `train100` warmup 已完成，`train360` processed/binary 已完成；当前下一步是启动并跑通 mixed `teacher_offline` formal stage-1。教师训练好后，优先进入 `student_ref_bootstrap`，而不是先跑 `student_kd`。

---

## 5. 现在就该怎么做

### 5.1 立即动作

```bash
cd /root/autodl-tmp/project-1/conan-rhythm
bash scripts/autodl_recovery_mixed100360_v3_trainonly.sh
```

### 5.2 监控

```bash
tail -f logs/stage1_recovery_mixed100360_v3_trainonly.status
```

如果你是用 shell 重定向方式启动，例如：

```bash
bash scripts/autodl_recovery_mixed100360_v3_trainonly.sh > logs/stage1_recovery_mixed100360_v3_trainonly.log 2>&1 &
```

那么再看：

```bash
tail -f logs/stage1_recovery_mixed100360_v3_trainonly.log
```

如果外层 wrapper 需要：

```bash
bash logs/stage1_recovery_mixed100360_v3_trainonly.sh
```

### 5.3 预期正确的状态推进

应依次看到：

1. `RECOVERY_V3_TRAINONLY_START`
2. `TRAIN360_REUSE_EXISTING_ARTIFACTS`
3. `TRAIN360_PREFLIGHT_DONE`
4. `MIXED_PREFLIGHT_DONE`
5. `MIXED_REAL_SMOKE_DONE`
6. `FORMAL_STAGE1_RUN_START`

如果最终正式训练在跑，说明当前主阻塞已经解除。

---

## 6. stage 之后到底要干什么

这是当前最容易被说乱的地方。

### 当前项目优先总链路

1. `teacher_offline`
2. 固化 teacher checkpoint / 准备 pair manifest / grouped batch
3. `student_ref_bootstrap`
4. `student_retimed`
5. `student_kd`（可选 baseline / ablation / 稳定性分支）

### 一句话理解

- `teacher_offline`：先把老师练强
- `student_ref_bootstrap`：先逼 student 真正学会用外部 ref
- `student_retimed`：把节奏控制闭环到最终声学输出
- `student_kd`：作为 cached/self-conditioned 对照支线保留

### teacher export 现在还要不要做

要，但要说清楚它的定位：

- **建议做**，因为它有利于审计、对照和后续 cached-teacher baseline
- 但它**不再是 v2.5 启动的硬前置条件**
- 它仍然是 `student_kd` 的硬前置

---

## 7. 当前最短结论

截至 **2026-04-07 UTC**：

- `train100` binary：**完成**
- `train360` binary：**完成**
- mixed `teacher_offline` formal stage-1：**尚未正式启动成功**
- 当前正确动作：**切到 v3，复用现有 train360 资产，启动 mixed stage-1**
- teacher 之后的当前项目优先顺序：**student_ref_bootstrap -> student_retimed**
- `student_kd`：**保留，但不作为默认下一步**

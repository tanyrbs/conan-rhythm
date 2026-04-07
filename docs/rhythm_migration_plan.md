# Rhythm Migration Plan（2026-04-07 UTC）

这份文档只保留当前仍然有效的迁移结论，不再复述已经失效的“train360 仍在分片中”之类状态。

## 1. 代码层有哪些 stage

当前仓库里明确支持的关键 stage 有：

- `teacher_offline`
- `student_kd`
- `student_ref_bootstrap`
- `student_retimed`

## 2. 当前项目优先训练链

批判性更新后，当前项目优先训练链不再写成：

- `teacher_offline -> student_kd -> student_retimed`

而是写成：

1. `teacher_offline`
2. `student_ref_bootstrap`
3. `student_retimed`
4. `student_kd`（可选 baseline / ablation / 稳定性支线）

## 3. 为什么当前先不走 `student_kd`

因为当前真正要回答的问题是：

- external ref 是否真的被使用
- one-to-many 下会不会平均化 / collapse
- descriptor semantics 是否真的跟着 `B` 变化

而 `student_kd` 当前更适合回答的是：

- cached/self-conditioned teacher surface 能不能稳定蒸给 student

所以：

- `student_kd` 不是被删掉
- 它只是从“主链默认下一步”降级成“辅助对照支线”

## 4. 对 v2.5 的严谨表述

当前 `student_ref_bootstrap` 应该被描述为：

- teacher warm-start
- runtime-only external-reference bootstrap
- first student-facing distillation / bootstrap stage

但不能夸张说成：

- 最终 fully learned-teacher distill 终局

因为当前它仍然：

- `rhythm_dataset_target_mode: runtime_only`
- `rhythm_teacher_target_source: algorithmic`

所以它还没有完全摆脱 algorithmic ceiling。

## 5. 当前项目处于哪一步

当前项目实际停在：

- **stage-1 mixed `teacher_offline` 启动前**

更准确地说：

- train100 processed / binary：完成
- train360 processed / binary：完成
- mixed stage-1：尚未正式跑起来

## 6. 当前训练目标

### 6.1 眼前目标

把 mixed `teacher_offline` formal stage-1 跑起来。

目的：

- 学强 offline teacher
- 得到后续 v2.5 warm-start 所需的 teacher checkpoint

### 6.2 紧接着的目标

做 `student_ref_bootstrap`。

目的：

- 先验证 student 是否真的学会 external-reference-driven rhythm transfer
- 先解决 same-A / multi-B collapse 风险

### 6.3 再下一步目标

做 `student_retimed`。

目的：

- 把 rhythm control 闭环到最终 acoustic canvas

### 6.4 `student_kd` 的定位

- baseline
- ablation
- 稳定性对照
- cached-teacher 支线

## 7. 当前真正的技术阻塞

当前不是数据阻塞，而是运行链阻塞。

最近一次失败发生在：

- mixed preflight model dry-run

根因：

- `modules/Conan/rhythm/module.py`
- `_sample_trace_pair()` 返回值个数不一致

这也是为什么当前优先级是：

- 修复 runtime contract
- 复用现成 binary
- 直接启动 v3 mixed stage-1

## 8. 训练阶段的职责边界

### `teacher_offline`

职责：

- 训练 learned-offline teacher
- 稳住 planner / controller / projector / prefix consistency
- 给后续 v2.5 提供强初始化

### `student_ref_bootstrap`

职责：

- external reference / pairwise bootstrap
- 验证模型是否真的使用外部 `B`
- 通过 descriptor-consistency / pairwise contrastive / diversity 约束降低 collapse 风险

### `student_retimed`

职责：

- 把 rhythm control 真正闭环到 acoustic canvas
- 解决最终声学输出是否服从节奏控制

### `student_kd`

职责：

- cached/self-conditioned teacher surface 的稳定蒸馏
- 作为 baseline / ablation / 稳定性支线保留

## 9. 当前建议

### 现在就做

```bash
bash scripts/autodl_recovery_mixed100360_v3_trainonly.sh
```

### 现在不要做

- 不要重做 train100 / train360 binarize
- 不要先跳到 `student_kd`
- 不要把 `student_retimed` 当成当前主阻塞
- 不要把“teacher 后优先 v2.5”错误说成“已经完成了最终 teacher distill”

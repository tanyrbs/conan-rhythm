# Conan Rhythm Branch

当前仓库里同时存在 4 个关键 stage 配置：

- `teacher_offline`
- `student_kd`
- `student_ref_bootstrap`
- `student_retimed`

但**当前 AutoDL 项目优先训练路径**，已经不是最早那版默认叙述，而是：

1. `teacher_offline`
2. `student_ref_bootstrap`（把 v2.5 当成 teacher 之后的第一 student 阶段）
3. `student_retimed`
4. `student_kd` 降级为 **baseline / ablation / 稳定性对照分支**，不是默认下一步

## 为什么现在先不走 `student_kd`

批判性判断：

- `student_kd` 更擅长把 **cached / self-conditioned teacher surface** 稳定蒸给 student
- 但它**不是**当前最直接回答“模型是否真的看外部 reference B”的阶段
- 你们当前最核心的问题是：
  - external ref 是否真的被使用
  - same-A / multi-B 下是否会 collapse 成平均计划
  - descriptor 语义是否真的跟着 B 变化

因此，在当前研究目标下，更合理的优先顺序是：

- 先把 full/global `teacher_offline` 练强
- 然后先进入 `student_ref_bootstrap`
- 再做 `student_retimed`

## 当前真实状态（2026-04-07 UTC）

### 已完成

- `train100` formal processed：`data/processed/libritts_train100_formal`
- `train100` formal binary：`data/binary/libritts_train100_formal_rhythm_v5`
- `teacher_offline_train100_warmup` 已到 **20k steps**
  - checkpoint：`checkpoints/teacher_offline_train100_warmup/model_ckpt_steps_20000.ckpt`
  - 最新可信 validation（step `19000`）：
    - `exec_total_corr = 0.8907`
    - `pause_event_f1 = 0.5957`
    - `prefix_drift_l1 = 25.7781`
    - `budget_projection_repair_ratio_mean = 0.0`
- `train360` formal processed：`data/processed/libritts_train360_formal_trainset`
- `train360` formal binary：`data/binary/libritts_train360_formal_trainset_rhythm_v5`
- 为节省磁盘，`train-clean-100` / `train-clean-360` raw audio 已删除；当前保留的是可复用的 processed / binary 资产

### 未完成

- **mixed `train100 + train360` formal stage-1 (`teacher_offline`) 还没有正式跑起来**
- `student_ref_bootstrap`、`student_retimed`、`student_kd` 都还没开始正式训练

### 最近一次真正的阻塞点

- v2 恢复流水线在 **2026-04-06 18:19:23 UTC** 失败：
  - 状态文件：`logs/stage1_recovery_mixed100360_v2_20260406_091257.status`
  - 原因：mixed preflight 的 model dry-run 报错
    - `not enough values to unpack (expected 3, got 2)`
- 根因已经定位到：
  - `modules/Conan/rhythm/module.py`
  - `_sample_trace_pair()` 调用方解包 3 个返回值，但函数只返回了 2 个
- 本地已经补了修复，并新增了可复用 train360 资产的 v3 启动脚本：
  - `scripts/autodl_recovery_mixed100360_v3_trainonly.sh`

## 当前唯一正确的主任务

不是继续纠结 train100 warmup，也不是现在就去跑 `student_kd`。

**当前唯一正确的主任务是：**

1. 用现有 `train100` / `train360` binary 直接启动 mixed formal stage-1
2. 跑通 `teacher_offline_train100_360_stage1`
3. 教师训练好后，优先进入 `student_ref_bootstrap`
4. 再做 `student_retimed`
5. `student_kd` 只作为后续 baseline / ablation / 稳定性分支

## 每个训练 stage 的目的

| Stage | 配置 | 目的 | 当前定位 |
|---|---|---|---|
| Stage 1 | `teacher_offline` | 先把 scheduler / controller / projector / prefix consistency 学稳，得到强的 full/global teacher | **当前正在准备正式 mixed 100+360 运行** |
| Stage 2（优先） | `student_ref_bootstrap` | external reference / pairwise bootstrap；强迫模型真的使用 `B`，而不是学平均 | **teacher 之后的第一 student 阶段** |
| Stage 3 | `student_retimed` | 在 retimed acoustic canvas 上闭环，把 rhythm control 真正压到声学结果里 | v2.5 之后 |
| Optional | `student_kd` | cached/self-conditioned baseline；用于对照、稳定性验证、ablation | **不是默认下一步** |

## 关键批判性提醒

当前 `student_ref_bootstrap` 虽然会放在 teacher 后优先做，但要说严谨：

- 它现在仍是 `runtime_only`
- 目标源仍是 `rhythm_teacher_target_source: algorithmic`
- 所以它更准确地说是：
  - **teacher warm-start + external-ref bootstrap / first student-facing distillation stage**
- 它还不是“完全摆脱 algorithmic ceiling 的最终 learned-teacher distill 终局”

这也是为什么：

- teacher 仍然必须先练强
- `student_kd` 仍然保留，只是不再作为当前默认下一步

## stage 之间应该做什么

- `teacher_offline` 结束后：
  1. 固化 teacher ckpt
  2. 准备 pair manifest / grouped batch / external-ref 训练入口
  3. 优先开 `student_ref_bootstrap`
  4. teacher export 仍然建议做，但它不再是 v2.5 的硬前置
- `student_ref_bootstrap` 结束后：
  1. 准备 retimed cache / F0 sidecars
  2. 再开 `student_retimed`
- `student_kd`：
  - 放到后面作为 baseline / ablation / 稳定性支线
  - 或者在需要 cached-teacher 对照时再单独跑

## 相比最新 `origin/main`，本地这次有意识吸收/保留的改动

不是盲目跟 main，而是只吸收当前确实对 AutoDL 主线有帮助的部分。

### 已吸收

- stage-3 projector 显式默认：
  - `rhythm_projector_pause_selection_mode: sparse`
  - `rhythm_projector_use_boundary_commit_guard: true`
  - `rhythm_projector_build_render_plan: true`

### 本地额外补强

- 修复 `_sample_trace_pair()` 返回值 contract：现在稳定返回
  - `trace_context`
  - `planner_trace_context`
  - `trace_reliability`
- stage-3 acoustic ramp 改为**锚定当前 runtime 的首个 stage-3 step**，避免 warm-start 直接跳满
- pitch loss 默认跟随 stage-3 acoustic curriculum，避免 early retimed 阶段出现“mel 缩了但 pitch 仍满强度”
- algorithmic teacher 的 pause top-k 改成 **per-sample row-wise**，不再被同 batch 极端样本污染
- online retimed acoustic target 权重额外乘上 `trace_reliability.local_gate`
- `boundary_trace` 改为 soft boundary strength；但这项**需要重建 binary/cache 后才会真正影响训练数据**
- zero-speech / pause-only 边界样本下：
  - `progress` 回到 uniform fallback
  - `global_rate=0`，不再错误映射成正常语速

### 有意不直接照搬

- 默认 `student_retimed` 先走更保守 baseline：
  - `rhythm_retimed_target_mode: cached`
  - `rhythm_online_retimed_target_start_steps: 40000`
- bounded EMA loss balancing 保留在
  - `egs/conan_emformer_rhythm_v2_student_retimed_balanced.yaml`
  作为偏实验型 stronger variant，而不是直接写死进首个 formal baseline

## 当前最短操作路径

```bash
cd /root/autodl-tmp/project-1/conan-rhythm
bash scripts/autodl_recovery_mixed100360_v3_trainonly.sh
```

建议同时监控：

```bash
tail -f logs/stage1_recovery_mixed100360_v3_trainonly.status
```

## 建议阅读顺序

1. `docs/autodl_training_handoff.md`
2. `docs/autodl_train100_formal_progress.md`
3. `docs/autodl_train100_formal_quickstart.md`
4. `docs/rhythm_migration_plan.md`

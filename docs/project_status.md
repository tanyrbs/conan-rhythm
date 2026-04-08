# Project Status

更新时间：2026-04-08 UTC

## 1. 当前项目状态

### 当前唯一维护主线
- stage: `teacher_offline`
- data: `mixed train100 + train360`
- lineage: `v6`
- pause path: `split-head support + allocation`
- semantic: `weight-only warm-start`

### 当前 active experiment
- exp: `teacher_offline_train100_360_v6_split_heads_restart17500_fix1`
- log: `logs/teacher_offline_train100_360_v6_split_heads_restart17500_fix1.log`
- ckpt dir: `checkpoints/teacher_offline_train100_360_v6_split_heads_restart17500_fix1`
- bootstrap: `/root/autodl-tmp/project/conan-rhythm/checkpoints/model_ckpt_steps_17500.ckpt`

### 当前运行状态
- 训练进程仍在运行
- latest observed train step token in log: **33236**
- 已完成 validation step：`5000 / 10000 / 15000 / 20000 / 25000 / 30000`

> 注意：当前正在运行的进程早于最近一次代码提交启动，因此它反映的是**旧的进程内代码**，不是当前仓库 HEAD 对应的新逻辑。

---

## 2. 当前训练进度记录

| val step | pause_event_f1 | exec_total_corr | prefix_drift_l1 | support_cover_at_topk | recall_drop_post_from_planner | boundary recall |
|---|---:|---:|---:|---:|---:|---:|
| 5000  | 0.8200 | 0.9192 | 21.2524 | 0.9357 | 0.0380 | 0.0019 |
| 10000 | 0.8114 | 0.9171 | 22.9551 | 0.9307 | 0.0356 | 0.0019 |
| 15000 | 0.8215 | 0.9172 | 21.4709 | 0.9344 | 0.0460 | 0.0018 |
| 20000 | 0.8268 | 0.9180 | 21.3356 | 0.9366 | 0.0491 | 0.0019 |
| 25000 | 0.8226 | 0.9203 | 21.4935 | 0.9367 | 0.0474 | 0.0019 |
| 30000 | 0.8271 | 0.9185 | 21.8889 | 0.9355 | 0.0531 | 0.0019 |

### 当前 best 记录
- overall best (`model_ckpt_best.pt`): **@15000**，按 `val_loss=0.22422`
- pause best (`model_ckpt_pause_best.pt`): **@30000**，按 `rhythm_metric_pause_event_f1=0.82708`
- 最近 step ckpt：`20000 / 25000 / 30000`

---

## 3. 当前判断

### 结论
**全局健康，但 boundary 仍明显滞后。**

### 支持这个判断的事实
- `pause_event_f1` 整体维持在 `0.81~0.83` 区间，没有主线崩坏迹象
- `exec_total_corr` 维持高位，`@25000=0.9203`、`@30000=0.9185`
- `prefix_drift_l1` 维持在 `21~22` 区间，`@30000` 略有上行但未失控
- `pause_support_cover_at_topk` 维持在 `0.93+`，说明 planner → projector 的主要 support 覆盖仍健康
- `pause_recall_drop_post_from_planner` 约 `0.04~0.05`，`@30000` 略抬到 `0.0531`，仍需盯紧
- `boundary recall` 持续约 `0.0018~0.0019`，说明 boundary 子集收益尚未体现出来

### 当前最重要风险
当前 Stage1 还不能因为全局指标健康就直接判定完成，主要卡点仍是：
- boundary pause 子集几乎没有抬起来
- boundary 相关收益还没有稳定反映到监控值上

---

## 4. 当前代码与训练的关系

当前仓库 HEAD 已经包含一批代码与文档修复，主要包括：
- 文档主线收敛
- commit controller / projector / metrics / losses 的若干修复
- boundary valid-only 指标与监控补强

但这些修改**尚未应用到当前正在运行的训练进程**。因此：
- 当前 run 仍可作为旧逻辑 baseline 观察
- 如果要验证新逻辑，必须新开 experiment 或重启 run

---

## 5. 当前下一步建议

1. 保持当前 run 继续跑到下一个 validation 点
2. 文档只保留 README + code/config guide + project status 三份
3. 本地代码整理后提交到 `autodl` 分支
4. 需要验证新 boundary 修复时，使用 `17500` warm-start 新开 experiment，不要污染当前 run

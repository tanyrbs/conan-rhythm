# AutoDL 当前快速操作手册（2026-04-07 UTC）

## 1. 先明确：现在不是哪个阶段

现在**不是**：

- 重做 train100 metadata / binary
- 重做 train360 metadata / binary
- 直接切 `student_kd`
- 直接切 `student_retimed`

现在**就是**：

- 启动 mixed `train100 + train360` 的 `teacher_offline` formal stage-1

## 2. 为什么现在可以直接这样做

因为当前已经有：

- `data/binary/libritts_train100_formal_rhythm_v5`
- `data/binary/libritts_train360_formal_trainset_rhythm_v5`

而且 v2 的失败点在代码 dry-run，不在数据产物。

## 3. 当前一条命令

```bash
cd /root/autodl-tmp/project-1/conan-rhythm
bash scripts/autodl_recovery_mixed100360_v3_trainonly.sh
```

## 4. 状态怎么看

```bash
cat logs/stage1_recovery_mixed100360_v3_trainonly.status
```

持续监控：

```bash
tail -f logs/stage1_recovery_mixed100360_v3_trainonly.status
```

## 5. 正常推进顺序应该是什么

如果脚本正常，应看到：

1. `RECOVERY_V3_TRAINONLY_START`
2. `TRAIN360_REUSE_EXISTING_ARTIFACTS`
3. `TRAIN360_PREFLIGHT_DONE`
4. `MIXED_PREFLIGHT_DONE`
5. `MIXED_REAL_SMOKE_DONE`
6. `FORMAL_STAGE1_RUN_START`

## 6. stage-1 结束后做什么

不是直接开 `student_kd`，也不是直接开 stage-3。

当前项目优先顺序：

1. 固化 teacher checkpoint
2. 准备 external-ref pair manifest / grouped batch
3. `student_ref_bootstrap`
4. `student_retimed`
5. `student_kd`（可选 baseline / ablation / 稳定性支线）

## 7. 为什么现在先不走 `student_kd`

因为当前最核心的问题不是“cached teacher 能不能蒸稳”，而是：

- 外部 `B` 是否真的被使用
- same-A / multi-B 是否会 collapse
- descriptor 语义是否真的跟着 `B` 变化

所以 teacher 之后当前更该优先的是：

- `student_ref_bootstrap`

## 8. 当前最关键的一句

> 现在真正要完成的是 mixed `teacher_offline` formal stage-1；只有这一步跑起来，后面的 `student_ref_bootstrap`、`student_retimed` 才有意义，而 `student_kd` 目前只是可选支线。

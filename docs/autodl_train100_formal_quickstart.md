# AutoDL 当前快速操作手册

更新时间：2026-04-06 UTC

## 1. 先看现在到哪一步了

### 已完成

- `train100` formal processed / binary 已完成
- `teacher_offline_train100_warmup` 已到 **20k**
- warmup 关键结论：
  - `exec_total_corr = 0.8907`
  - `pause_event_f1 = 0.5957`
  - `prefix_drift_l1 = 25.7781`
  - stage flags 全是 `1.0`
- export smoke 已通过

### 正在进行

- `train360` metadata 分片构建
- 当前大约 **64.5%**
- ETA 约：
  - **2026-04-06 18:03 UTC**

### 还没开始

- `train360` train-only binary
- mixed `train100|train360` preflight
- formal Stage-1 `teacher_offline_train100_360_stage1`

## 2. 当前最重要的文件

### warmup 证据

- `logs/stage1_warmup20k_formal100360_pipeline_20260406_075555.report.json`
- `artifacts/warmup20k_export_smoke/manifest.json`
- `artifacts/warmup_20k_evidence/`

### 当前运行中的 train360 metadata 进程

- launcher：
  - `logs/stage1_recovery_mixed100360_v2_20260406_091257.sh`
- status：
  - `logs/stage1_recovery_mixed100360_v2_20260406_091257.status`
- shard 日志：
  - `logs/prepare_train360_shards/shard_*.log`

### train360 metadata 完成后的接管脚本

- `logs/stage1_takeover_from_existing_train360_metadata_trainonly.sh`

## 3. 监控命令

### 看 train360 当前状态

```bash
cd /root/autodl-tmp/project-1/conan-rhythm
cat logs/stage1_recovery_mixed100360_v2_20260406_091257.status
tail -n 40 logs/live_recovery_monitor.log
```

### 看 shard 进度

```bash
cd /root/autodl-tmp/project-1/conan-rhythm
for f in logs/prepare_train360_shards/shard_*.log; do
  echo "### $f"
  tail -n 2 "$f"
done
```

### 看磁盘

```bash
df -h /root /root/autodl-tmp /mnt
du -sh data/binary/libritts_train100_formal_rhythm_v5
du -sh /root/autodl-tmp/data/LibriTTS/train-clean-360
```

## 4. 当前 teacher formal 的正确开法

当前 teacher formal 仍然应该这样理解：

- warmup `20k` 只是初始化
- formal Stage-1 要新开 exp
- 不要直接 resume 旧 warmup exp

核心原则：

- 新 exp_name
- `--reset`
- `load_ckpt='checkpoints/teacher_offline_train100_warmup/model_ckpt_steps_20000.ckpt'`
- `load_ckpt_strict=True`
- mixed train 用：
  - `train_sets='train100|train360'`

## 5. 当前 takeover 脚本是否要重写

结论：

- **teacher formal 这一步暂时不用重写**

原因：

- upstream 最新改动主要增强的是：
  - `student_retimed`
  - `student_ref_bootstrap`
- 当前正在执行的是：
  - `teacher_offline` 的 100+360 formal 准备

所以现有：

- `scripts/autodl_train_stage1.sh`
- `logs/stage1_takeover_from_existing_train360_metadata_trainonly.sh`

在 teacher formal 这条链上仍然可以继续用。

## 6. 但要注意一个现实坑：磁盘

当前磁盘：

- `/root/autodl-tmp` 空余约 **37G**
- overlay `/root`/`/mnt` 空余约 **12G**

批判性判断：

- train360 **train-only** 是正确的
- 但仅靠“不给 train360 单独建 valid/test”并不自动代表磁盘一定够
- 真正的大头仍然是：
  - raw `train-clean-360`
  - 新 train360 binary
  - 现有 train100 binary

因此在 metadata 完成后、真的开 `train360` binarize 前，要再检查一次：

- `/root/autodl-tmp` 可用空间如果还低于 **46~50G**
- 先不要硬开

优先动作：

1. 删 `dev-clean` / `test-clean`
2. 必要时把部分 `train100` binary 临时挪到 overlay `/root` 或 `/mnt`
3. 再开 train360 binarize

## 7. 当前和 upstream main 比，已经吸收了什么

已经吸收/保留的关键点：

- stage-3 acoustic scalar ramp
- stage-2.5 external-reference bootstrap
- external-reference 与 self-cache supervision conflict 的硬错误
- `rhythm_require_external_reference: true`
- reference self/external 监控指标
- 本地额外保留：
  - explicit `student_ref_bootstrap` stage 名称
  - streaming phase-nonretro chunk metrics

## 8. 下一步的正式顺序

### 现在

1. 等 train360 metadata 分片结束
2. takeover 做：
   - merge
   - train-only binarize
   - preflight
   - mixed preflight
   - mixed real smoke
3. 开 formal teacher Stage-1

### 之后

teacher 完成后，再走：

1. teacher export
2. rebuild student cache
3. `student_kd`
4. 可选 upper-bound：`student_ref_bootstrap`
5. `student_retimed`

## 9. 关于“先跳过 student_kd”

短期可以：

- 现在先专注 teacher

但长期不是：

- `student_kd` 不是永久删除

更准确说法是：

- **teacher 先行**
- `student_kd` 延后，不是取消

## 10. 后面如果要冲更高上限，应该怎么做

不要只做：

- `rhythm_cached_reference_policy=self -> sample_ref`

正确做法是：

- 用 runtime-only external-reference teacher 路线
- 也就是：
  - `egs/conan_emformer_rhythm_v2_student_ref_bootstrap.yaml`
  - 或
  - `egs/conan_emformer_rhythm_v2_student_pairwise_ref_runtime_teacher.yaml`

这是因为：

- 只换 ref，不换 target
- 会变成 external ref 条件去拟合 self cache target
- 容易把模型训成“忽略 reference”

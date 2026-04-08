# Conan Rhythm Branch

## 当前唯一维护主线

当前仓库只维护一条主线：

- **Stage1**：`teacher_offline`
- **数据**：`train100 + train360` mixed
- **lineage**：`v6 binary / cache`
- **pause 路径**：`split-head support + allocation`
- **启动语义**：`weight-only warm-start`
- **标准入口**：`scripts/autodl_train_stage1_mixed_v6_split_heads.sh`
- **标准配置**：`egs/conan_emformer_rhythm_v2_teacher_offline_train100_360_pause_split_heads_resume.yaml`
- **标准预热 checkpoint**：`checkpoints/bootstrap/model_ckpt_steps_17500.ckpt`

> 现在不要再把 `train360-only / legacy v5`、`pause_recall exact-resume`、`recovery_v3_trainonly` 视为当前主线。

---

## teacher-first 路线图

当前项目按下面顺序推进：

1. **Stage1**：`teacher_offline` 控制面拟合与 pause 结构修复
2. **Stage2**：teacher polish，与 export / stability / listening gate 绑定
3. **v2.5**：teacher-side external-reference / strong-variation 增强
4. **v3**：mixed v6 clean rerun + cache refresh + robustness / archive
5. teacher 线收口后，再进入：
   - `student_kd`
   - `student_ref_bootstrap`（可选）
   - `student_retimed`

---

## 当前标准训练语义

### 数据角色
- `data/binary/libritts_train100_formal_rhythm_v6`
  - mixed 训练的 `binary_data_dir`
  - valid / test contract 基座
- `data/binary/libritts_train360_formal_trainset_rhythm_v6`
  - mixed 训练的 train-only supplemental dataset

### warm-start 语义
- 当前主线是 **weight-only warm-start**
- 不是 **exact resume**
- bootstrap checkpoint 默认使用：
  - `checkpoints/bootstrap/model_ckpt_steps_17500.ckpt`

### 当前标准启动
```bash
cd /root/autodl-tmp/project/conan-rhythm
bash scripts/autodl_train_stage1_mixed_v6_split_heads.sh
```

---

## 当前标准产物

主线 stage1 统一约定：
- `MAX_UPDATES=80000`
- `VAL_CHECK_INTERVAL=5000`
- 最近 `3` 个 step ckpt
- `model_ckpt_best.pt`：整体最优（按 `val_loss`）
- `model_ckpt_pause_best.pt`：pause 最优（按 `rhythm_metric_pause_event_f1`）

---

## 文档入口（只保留这一套）

- **主线说明**：`README.md`
- **代码 / 配置说明**：`docs/code_config_guide.md`
- **当前项目状态 / 训练进度**：`docs/project_status.md`

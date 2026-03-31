# Rhythm V2 训练阶段建议

更新时间：2026-03-31

当前仓库已经具备把 Rhythm V2 分阶段训练起来的最小接口。

---

## Stage 0：结构热启动

目标：

- 先把 planner/projector 跑稳
- 先学会 source anchor 上的基本 timing

推荐配置：

- `rhythm_dataset_build_guidance_from_ref: true`
- `rhythm_dataset_build_teacher_from_ref: false`
- `lambda_rhythm_guidance > 0`
- `lambda_rhythm_distill = 0`

---

## Stage 1：reference-guided warm start

目标：

- 真正让模型看 `source + sampled reference`
- 学会 budget / redistribution 的基本映射

当前仓库已支持：

- dataset 在线构造 `rhythm_*_tgt`
- guidance 监督

---

## Stage 2：teacher/distill

目标：

- 用更强的离线教师替换 heuristic guidance ceiling
- 做 latency-matched surface distillation

当前仓库已预留字段：

- `rhythm_teacher_speech_exec_tgt`
- `rhythm_teacher_pause_exec_tgt`
- `rhythm_teacher_speech_budget_tgt`
- `rhythm_teacher_pause_budget_tgt`

训练侧 loss 已支持：

- `rhythm_distill`

当前 projector 还已支持：

- committed prefix freeze
- sparser pause allocation
- committed-prefix streaming extraction

推荐：

- 不蒸 raw hidden
- 优先蒸 projector 后 surface
- teacher/student 保持同一 public contract

---

## Stage 3：真实 streaming 验证

重点看：

- committed prefix 增长
- chunkwise mel 增量
- commit frontier 连续性
- backlog / clock_delta 是否失控

---

## 一句话总结

当前仓库最合理的训练路线是：

> heuristic warm start -> reference-guided training -> latency-matched teacher distillation -> streaming evaluation

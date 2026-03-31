# Rhythm V2 本地适配说明

更新时间：2026-03-31

本文档回答两个问题：

1. 当前 Rhythm V2 还可以怎么优化迁移。
2. 如果要适配到本地项目/其他 Conan 变体，应该改哪些层，不该改哪些层。

---

## 1. 迁移时优先保留什么

建议优先保留这 4 层：

1. `unit_frontend`
2. `reference_encoder`
3. `controller`
4. `projector`

原因：

- `unit_frontend` 决定 source anchor 是否 prefix-safe
- `reference_encoder` 决定 reference rhythm 是否显式、可复用
- `controller` 决定强节奏变化的“预算 + 重分配”主逻辑
- `projector` 决定 runtime 是否只有一个 timing authority

如果要迁移到别的本地项目，最先检查的不是 decoder，而是：

- 你的内容单位是不是单调、稳定、可去重
- 你的参考语音能不能拿到一条稳定的 progress-normalized rhythm trace

---

## 2. 适配到本地项目时的最小接口

### 输入侧最小要求

目标项目至少要能提供：

- `content` 或等价 token 序列
- `ref_mel`
- `content_lengths`

### 节奏模块输出侧最小要求

目标项目至少要能接收：

- 可选的 frame-level rendered content states
- 可选的新的 nonpadding mask

如果目标项目 decoder 已经天然工作在 unit 级而不是 frame 级，
那就不一定要启用 `renderer.py`，可以直接吃 projector 输出。

---

## 3. 哪些层最容易出兼容问题

### 3.1 内容单位层

如果目标项目的 token 不是 20ms HuBERT 类离散单位，而是：

- phone
- codec token
- semantic token with variable stride

那么只改 `unit_frontend.py`，不要改 planner/projector 合同。

### 3.2 reference 输入层

如果目标项目没有 mel reference，只有：

- style embedding
- codec latent
- SSL feature

那就只替换 `reference_encoder.py` 的输入适配层，仍然尽量输出：

- `ref_rhythm_stats`
- `ref_rhythm_trace`

不要把 style/timbre 整体重新塞回 planner 主合同。

### 3.3 decoder 长度接口

如果目标项目 decoder 只接受固定 source 长度，
那要么：

- 开启 `renderer.py` 做 frame-level render
要么：
- 重构 decoder 让它接受 projector 后时间轴

这里是集成层问题，不应倒逼 rhythm 主合同脏化。

---

## 4. 当前还可继续优化的迁移点

### P0：离线教师

当前仓库已经有：

- source cache
- reference stats/trace
- heuristic guidance targets

下一步最值的是接离线教师，而不是再堆更多 runtime head。

### P1：projector-aware distillation

蒸馏应优先放在 projector 后 surface：

- speech exec
- pause exec
- commit / clock consistency

而不是蒸 raw hidden。

### P2：增量执行

当前 streaming test 已恢复 chunkwise，
但 vocoder 仍偏“反复整段拼接再截尾”。

后续最值得做的是 committed-prefix 增量导出。

### P3：指标闭环

建议持续记录：

- `expand_ratio`
- `pause_share`
- `commit_ratio`
- `phase_ptr`
- `backlog`
- `clock_delta`

---

## 5. 迁移时不建议做的事

1. 不要重新回到 dense gap + ambiguity band 主线。
2. 不要把 `unit_type / unit_boundary_hint` 重新暴露成 maintained public input。
3. 不要让 decoder、budget 修正器、commit util 同时拥有 timing authority。
4. 不要一开始就蒸 full-context teacher 的不可达输出。

---

## 6. 建议的本地迁移顺序

1. 先接 `unit_frontend`
2. 再接 `reference_encoder`
3. 再接 `controller + projector`
4. 最后才接 `renderer` 与 decoder 时间轴改造

这能最大限度降低迁移复杂度。

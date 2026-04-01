# Rhythm Module Vision (2026-04-01)

## 1. Goal

This branch is now converging to a smaller and harder rhythm mainline for **strong rhythm transfer under streaming constraints**.

The main judgment is:

- do not hide rhythm inside a large style latent
- do not let multiple modules silently edit timing
- keep one explicit timing path from reference rhythm conditioning to execution

So the recommended path is:

```text
UnitFrontend
  -> RefRhythmDescriptor
  -> MonotonicRhythmScheduler
  -> StreamingRhythmProjector
  -> optional RhythmRenderer / decoder
```

There is one important internal sidecar:

- `source_boundary_cue`

It is not part of the maintained public contract.
It is built cheaply from source-side separator evidence and source duration shape, then consumed inside the scheduler.

---

## 2. Maintained public contract

### Inputs (4)

1. `content_units`
2. `dur_anchor_src`
3. `ref_rhythm_stats`
4. `ref_rhythm_trace`

### Outputs (4)

1. `speech_budget_win`
2. `pause_budget_win`
3. `dur_logratio_unit`
4. `pause_weight_unit`

### Losses (4)

1. `L_exec_speech`
2. `L_exec_pause`
3. `L_budget`
4. `L_guidance` / `L_distill`

`L_guidance` is the current warm-start surface.
`L_distill` is reserved for latency-matched teacher distillation, not naive full-context imitation.

---

## 3. Explicit descriptor

`modules/Conan/rhythm/reference_descriptor.py`

This file wraps the lower-level reference encoder and exposes a smaller rhythm-facing surface:

- `global_rate`
- `pause_ratio`
- `local_rate_trace`
- `boundary_trace`
- plus compatibility fields:
  - `ref_rhythm_stats`
  - `ref_rhythm_trace`

This keeps the external contract stable while making the rhythm semantics more explicit.

The design is deliberately aligned with the duration-modeling direction seen in explicit-rhythm VC work such as PromptVC and R-VC, instead of pushing rhythm back into a large style latent.

---

## 4. Single scheduler

`modules/Conan/rhythm/scheduler.py`

`MonotonicRhythmScheduler` is the current single timing planner surface.
Internally it still reuses:

- `WindowBudgetController`
- `UnitRedistributionHead`
- cheap `source_boundary_cue`

But externally the semantics are now cleaner:

```text
content/unit states + source anchor + explicit ref rhythm + streaming state
    -> timing plan
```

This is more appropriate for a strong-rhythm system than exposing many loosely coupled heads at the public surface.

---

## 5. Projector authority

`modules/Conan/rhythm/projector.py`

The projector remains the only hard execution authority.

Current implemented properties:

- committed-prefix freeze
- sparse pause allocation
- commit frontier tracking
- phase / backlog / clock state carry-over

This means the system is no longer treating the planner output as already-executed timing.
The projector is the place where timing becomes binding.

---

## 6. Hidden runtime state is still necessary

The public contract is intentionally small, but the runtime system still needs internal execution state:

- `phase_ptr`
- `backlog`
- `clock_delta`
- `commit_frontier`
- cached executed prefix (`previous_speech_exec`, `previous_pause_exec`)

These are **runtime hidden states**, not public contract pollution.

---

## 7. What is already in place

Already connected:

- unit dedup frontend
- explicit reference stats/trace
- descriptor wrapper
- monotonic scheduler wrapper
- cheap source-boundary sidecar
- optional rhythm-only staging that bypasses the heavier local style/prosody adaptor while keeping global timbre conditioning
- projector freeze / sparse pause / fixed-horizon trace sampling
- dataset cached-target preference
- teacher/distill field reservation
- chunkwise streaming evaluation
- committed-prefix mel increment extraction

---

## 8. What is still missing

The branch is structurally much cleaner now, but it is **not yet a fully proven strong-rhythm training closure**.

The biggest remaining gaps are:

1. **Offline teacher**
   - distillation interface exists
   - strong offline teacher is still missing

2. **Train / infer retiming consistency**
   - decoder side still needs stronger training on the retimed execution canvas

3. **Reference rhythm quality**
   - the current encoder is still a practical baseline, not the final strongest descriptor

4. **Evaluation**
   - need stronger metrics for pause placement, local-rate transfer, prefix stability, and long-utterance trace usage

5. **Offline supervision quality**
   - runtime heuristic targets are no longer the ideal mainline
   - cached/offline supervision should become the maintained path

---

## 9. Current recommendation

If the goal is only one thing ? **strong rhythm transfer in streaming VC** ? then the repository should continue to harden this path instead of expanding the style path again.

In short:

- keep Conan's streaming ears and mouth
- reduce rhythm to an explicit descriptor + stateful scheduler + single projector
- push teacher/distill and retimed decoder training as the next serious milestones

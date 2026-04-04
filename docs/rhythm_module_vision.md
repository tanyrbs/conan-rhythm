# Rhythm Module Vision (2026-04-01)

See also:

- `docs/rhythm_local_adaptation.md`
- `docs/rhythm_migration_plan.md`
- `docs/rhythm_training_stages.md`
- `docs/rhythm_supervision_policy.md`

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

### Planner-facing tensors (internal/public-debug, not the final binding contract)

1. `speech_budget_win`
2. `pause_budget_win`
3. `dur_logratio_unit`
4. `pause_weight_unit`

These four tensors are still useful for debugging and regression,
but they should now be treated as **planner surfaces**, not as the main
public execution contract.

### Binding execution contract (projector-owned)

1. `speech_exec`
2. `pause_exec` (`blank_exec` kept only as an internal/cache alias)
3. `commit_frontier`
4. `next_state`

### Losses (mainline + staged)

Mainline timing losses:

1. `L_exec_speech`
2. `L_exec_pause`
3. `L_budget`
4. `L_prefix_state` (`L_cumplan` kept as the compatibility alias)

Joint compact optimizer view (maintained):

5. `L_base`
6. `L_rhythm_exec` (macro: `L_exec_speech + L_exec_pause`)
7. `L_stream_state` (macro: budget + prefix-state guardrail; `L_cumplan` is the public compatibility alias)
8. `L_pitch` (optional, only when retimed pitch targets are ready)

Staged / optional branch losses:

9. `L_distill`
10. `L_distill_exec`
11. `L_distill_budget`
12. `L_distill_prefix`

Interpretation:

- `L_exec_*` is the main supervision surface because projector execution is the actual timing authority
- `L_budget` is a light streaming guardrail, not the main target
- `L_prefix_state` supervises cumulative prefix debt / backlog directly
- `L_base` is the outer acoustic closure once retimed decoder training is enabled
- `L_distill*` is reserved for explicit KD branch experiments, not default maintained training

Current objective priority in practice:

- schedule warm-start: `L_exec_speech + L_exec_pause + light L_budget + light L_prefix_state`
- joint retimed stage: compact `3+1` view (`L_base + L_rhythm_exec + L_stream_state + optional L_pitch`)
- maintained phase-2 uses teacher-main supervision + small shape-only KD; executed speech/pause + prefix/budget distill remains only for legacy/research KD branches
- `L_plan` and `L_guidance` remain available only as internal ablations/debug paths

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

Important contract note:

- scheduler tensors are useful to inspect and lightly regularize
- but the **maintained binding contract** should live one step later, at projector execution
- in other words: the planner may expose budgets and redistribution logits, but the system should report and compare `speech_exec / pause_exec / commit_frontier / next_state` as the real outcome

Important implementation note:

- `source_boundary_cue` is a **soft source-side prior**
- it should help commit safety / phrase safety
- it should not dominate pause placement over reference rhythm conditioning

---

## 5. Projector authority

`modules/Conan/rhythm/projector.py`

The projector remains the only hard execution authority.

Current implemented properties:

- committed-prefix freeze
- prefix-constrained feasible-budget lift
- sparse pause allocation
- commit frontier tracking
- anchor-progress phase / backlog / clock state carry-over

This means the system is no longer treating the planner output as already-executed timing.
The projector is the place where timing becomes binding.

Recommended maintained output semantics:

- `speech_exec`
- `pause_exec` (`blank_exec` remains an internal alias on the blank-slot graph)
- `commit_frontier`
- `next_state`

The older scheduler-level tensors should remain available for debug/regression, but they should not be described as the final external timing contract.

---

## 6. Hidden runtime state is still necessary

The public contract is intentionally small, but the runtime system still needs internal execution state:

- `phase_ptr`
- `backlog`
- `clock_delta`
- `commit_frontier`
- cached executed prefix (`previous_speech_exec`, `previous_pause_exec`)

These are **runtime hidden states**, not public contract pollution.

The scheduler should be understood as a **stateful timing module**, not just a small set of heads.
The most important internal semantics are:

- prefix debt / backlog
- trace cursor / phase pointer
- emitted total frames
- pending pause realization

State contract clarification:

- `phase_ptr` represents committed progress
- it should be monotonic under streaming updates
- visible-prefix growth without new commit should not move the phase backward

At the dataset / training-batch level, keep the exported rhythm fields layered:

- runtime-minimal contract
- stage-needed runtime targets
- streaming/offline sidecars only when needed
- debug/cache appendix only when explicitly requested

This prevents the sample schema from drifting back into a single undifferentiated sidecar blob.

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
- strict cached-only cache-contract validation for formal experiments

What the project should explicitly avoid:

- putting strong rhythm control back into the large style path
- treating decoder crop logic as a second hidden scheduler
- relying on runtime heuristic labels as the main formal supervision path

---

## 8. What is still missing

The branch is structurally much cleaner now, but it is **not yet a fully proven strong-rhythm training closure**.

The biggest remaining gaps are:

1. **Offline teacher evidence**
   - standalone offline-teacher / export / cache plumbing now exists
   - what is still missing is long-run training evidence and a stable asset-generation loop, not the structural branch itself

2. **Train / infer retiming consistency**
   - decoder side still needs stronger training on the retimed execution canvas

3. **Reference rhythm quality**
   - the current encoder is still a practical baseline, not the final strongest descriptor

4. **Evaluation**
   - need stronger metrics for pause placement, local-rate transfer, prefix stability, and long-utterance trace usage

5. **Offline supervision quality**
   - runtime heuristic targets are no longer the ideal mainline
   - cached/offline supervision should become the maintained path

6. **Reference mode split**
   - current path is still `static_ref_full`
   - `progressive_ref_stream` should be documented as future work, not implied by the current default

---

## 9. Current recommendation

If the goal is only one thing ? **strong rhythm transfer in streaming VC** ? then the repository should continue to harden this path instead of expanding the style path again.

In short:

- keep Conan's streaming ears and mouth
- reduce rhythm to an explicit descriptor + stateful scheduler + single projector
- push teacher/distill and retimed decoder training as the next serious milestones
- prefer `cached_only` for formal experiments once the cache has been regenerated
- treat runtime heuristic generation as a debug path, not a maintained public training contract

Current task focus:

- harden cache contracts
- train on projector outputs, not style-side rhythm proxies
- reduce train/infer mismatch
- improve streaming regression

Future expansion:

- stronger offline teacher
- dual-mode offline/online projector distillation
- richer rhythm-specific evaluation
- progressive streaming reference support

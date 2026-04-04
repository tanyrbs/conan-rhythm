# Rhythm Local Adaptation Notes (2026-04-04)

This note records what the repository is doing **right now** for local rhythm adaptation.

See also:

- `docs/rhythm_training_stages.md`
- `docs/rhythm_supervision_policy.md`
- `docs/rhythm_migration_plan.md`

## 1. Current adaptation strategy

The maintained branch does **not** treat rhythm as a generic style latent.

It adapts rhythm through:

1. explicit source unitization
2. explicit reference rhythm stats / trace
3. a stateful scheduler / projector path
4. projector execution and frame-plan rendering
5. optional retimed acoustic supervision on the executed canvas

In short:

```text
source content -> units
reference mel -> rhythm descriptor
scheduler/projector -> executed timing surface
frame plan / renderer -> acoustic canvas
decoder / vocoder -> realization
```

## 2. What is global vs local

### Global rhythm information

Carried by:

- `ref_rhythm_stats`
- low-rate cadence memory derived from `ref_rhythm_trace`

This captures:

- overall rate
- pause density
- cadence envelope
- phrase-final tendency

### Local rhythm information

Carried by:

- run-length source units
- source boundary evidence
- projector redistribution / feasibility state
- frame-plan alignment on the executed surface

This captures:

- where pauses can happen
- which units can stretch or compress
- how timing budget is redistributed locally
- how executed timing is projected onto acoustic supervision

## 3. Current engineering policy

For maintained experiments:

- prefer cache-backed supervision
- keep the cache contract explicit and versioned
- keep timing authority inside the projector / executed surface
- treat `minimal_v1` as the maintained formal base
- treat `cached_only` as a compatibility alias, not the preferred entrypoint name

For transitional / migration work:

- `prefer_cache` is still acceptable as a debug / compatibility mode
- legacy `schedule_only` and `dual_mode_kd` remain research branches only

## 4. Current limitations

The current repo still has these practical boundaries:

- structural support for the standalone learned offline teacher is present, but branch value still needs stronger empirical proof on real datasets
- `static_ref_full` is still the maintained reference mode
- `progressive_ref_stream` is future work
- final branch claims still depend on real-data preflight, export, cache rebuild, and long-run stability rather than smoke-only evidence
- preprocessing and binarization throughput on Windows remain behind Linux / WSL

## 5. Immediate next task

The immediate task is still **not** to add more style heads.

The immediate task is to keep hardening:

- cache contract correctness
- dataset / training throughput
- projector supervision
- retimed closure
- streaming evaluation and regression tests

That remains the shortest path to stronger rhythm transfer under streaming constraints.

# Rhythm Local Adaptation Notes (2026-04-01)

This note records what the repository is doing **right now** for local rhythm adaptation.

See also:

- `docs/rhythm_module_vision.md`
- `docs/rhythm_training_stages.md`
- `docs/rhythm_supervision_policy.md`

## 1. Current adaptation strategy

The current branch does **not** treat rhythm as a generic style latent.

It adapts rhythm through:

1. explicit source unitization
2. explicit reference rhythm stats / trace
3. a stateful scheduler / projector path
4. retimed execution rather than decoder-side hidden length writing

In short:

```text
source content -> units
reference mel -> rhythm descriptor
descriptor + source units -> projector schedule
projector schedule -> rendered time axis
decoder/vocoder -> acoustic realization
```

## 2. What is local vs global

### Global rhythm information

Carried by:

- `ref_rhythm_stats`
- low-rate cadence memory from `ref_rhythm_trace`

This captures:

- overall rate
- pause density
- cadence envelope
- phrase-final tendency

### Local rhythm information

Carried by:

- run-length source units
- source boundary evidence
- per-window redistribution inside the projector

This captures:

- where a pause can happen
- which units can be stretched or compressed
- how the budget is redistributed inside a local window

## 3. Current engineering policy

For formal experiments:

- prefer `cached_only`
- keep rhythm cache contract stable
- avoid runtime heuristic supervision as the mainline

For transitional work:

- `prefer_cache` is still acceptable while cache assets are being refreshed

## 4. Current limitations

The current repo still has these boundaries:

- cached rhythm targets are still surface-level, not a trained standalone offline teacher model
- `static_ref_full` is still the maintained reference mode
- `progressive_ref_stream` is future work
- decoder-side retimed training is staged, not the only active mode yet

## 5. Immediate next task

The immediate task is **not** to add more style heads.

The immediate task is to keep hardening:

- cache contract
- projector supervision
- retimed training
- streaming evaluation

That is the shortest path to strong rhythm transfer under streaming constraints.

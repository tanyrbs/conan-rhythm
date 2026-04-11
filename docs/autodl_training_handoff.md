# AutoDL Training Handoff (legacy v2 archive note)

This file is now **legacy v2 operational guidance only**.
It no longer defines the current branch architecture.

For the maintained current branch, use:

- `README.md`
- `docs/rhythm_migration_plan.md`
- `inference/README.md`

## What this file is still for

Use this file only when you intentionally resume or audit the older v2
teacher/export chain.

That includes:

- old `teacher_offline` experiments
- old v2 export pipelines
- old student KD / retimed handoff procedures
- old archive checkpoints that still depend on v2 semantics

## What this file is not for

Do **not** use this file as the authority for:

- current `rhythm_v3` architecture
- current unit-run / prompt-summary runtime
- current task-layer split
- current v3 public inputs / outputs / losses

## If you must touch legacy v2

Use the v2 configs in `egs/` explicitly, and treat them as archival or
compatibility-only experiments.

Practical rules:

- use a new `exp_name`
- prefer weight-only warm-starts from old checkpoints
- run strict preflight before any real launch
- do not mix legacy v2 explanations into current v3 docs

## Current branch reminder

The maintained current branch is:

- `rhythm_v3`
- default config: `egs/conan_emformer_rhythm_v3.yaml`
- default reading: unit-run stretch head + source-observed anchor + carry-only projector

Legacy v2 remains available, but it is not the maintained architecture story.

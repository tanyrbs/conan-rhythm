# Rhythm Migration Plan (2026-04-01)

This file tracks the current migration state of the rhythm branch.

See also:

- `docs/rhythm_module_vision.md`
- `docs/rhythm_training_stages.md`
- `docs/rhythm_supervision_policy.md`

## 1. What has already migrated

The project has already moved away from the old “rhythm mixed into style / decoder heuristics” path.

Current migrated blocks:

- `modules/Conan/rhythm/unit_frontend.py`
- `modules/Conan/rhythm/reference_descriptor.py`
- `modules/Conan/rhythm/reference_encoder.py`
- `modules/Conan/rhythm/scheduler.py`
- `modules/Conan/rhythm/projector.py`
- `modules/Conan/rhythm/renderer.py`
- `modules/Conan/rhythm/supervision.py`
- `modules/Conan/rhythm/module.py`
- `modules/Conan/rhythm/factory.py`

Task-side integration already exists in:

- `tasks/Conan/dataset.py`
- `tasks/Conan/Conan.py`
- `tasks/Conan/rhythm/metrics.py`
- `tasks/Conan/rhythm/streaming_eval.py`

## 2. Current stage

Current repository stage:

### Ready now

- cached rhythm supervision
- cached-only cache-contract validation
- stateful scheduler / projector path
- retimed mel target generation
- chunkwise streaming evaluation

### Partially ready

- retimed decoder training
- offline teacher surface distillation
- stronger rhythm-specific regression tests

### Not final yet

- standalone offline teacher model
- progressive streaming reference updates
- full production-grade rhythm-specific benchmark suite

## 3. Current task focus

The branch should currently focus on:

1. keeping timing authority inside the projector
2. hardening cache reproducibility
3. reducing train/infer mismatch with retimed targets
4. validating streaming no-rollback / carry / budget behavior

This means:

- do not expand the style path first
- do not add many new proxy losses first
- do not reintroduce runtime heuristic dependence as the mainline

## 4. Future expansion

After the current stage is stable, the next expansions should be:

### Expansion A: stronger offline teacher

- move from offline teacher surface to a stronger full-context teacher
- keep the same public execution contract
- distill schedule, not just acoustic output

### Expansion B: dual-mode projector training

- same projector contract
- offline/full-context branch as teacher
- streaming/chunkwise branch as student

### Expansion C: richer evaluation

- local-rate transfer consistency
- long-utterance trace utilization
- chunk continuity
- cold-start vs steady-state latency

### Expansion D: progressive reference mode

- keep `static_ref_full` as the current maintained mode
- add `progressive_ref_stream` only after the current cache / projector path is stable

## 5. Migration rule of thumb

If a new feature does not make one of these stronger:

- schedule quality
- cache reproducibility
- retimed training consistency
- streaming stability

then it is probably not the next thing to migrate.

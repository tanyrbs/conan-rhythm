# Rhythm Migration Plan (2026-04-04)

This file tracks the current maintained migration state of the rhythm branch.

See also:

- `docs/rhythm_training_stages.md`
- `docs/rhythm_supervision_policy.md`
- `docs/rhythm_train_runbook.md`
- `docs/rhythm_pretrain_audit_2026-04-04.md`

## 1. What is already migrated

The branch has already moved away from the old “rhythm mixed into style / decoder heuristics” path.

### Core rhythm modules now in the maintained path

- `modules/Conan/rhythm/unitizer.py`
- `modules/Conan/rhythm/unit_frontend.py`
- `modules/Conan/rhythm/reference_descriptor.py`
- `modules/Conan/rhythm/reference_encoder.py`
- `modules/Conan/rhythm/scheduler.py`
- `modules/Conan/rhythm/projector.py`
- `modules/Conan/rhythm/renderer.py`
- `modules/Conan/rhythm/frame_plan.py`
- `modules/Conan/rhythm/offline_teacher.py`
- `modules/Conan/rhythm/runtime_adapter.py`
- `modules/Conan/rhythm/policy.py`
- `modules/Conan/rhythm/stages.py`
- `modules/Conan/rhythm/surface_metadata.py`
- `modules/Conan/rhythm/supervision.py`
- `modules/Conan/rhythm/module.py`
- `modules/Conan/rhythm/factory.py`

### Task / data / validation integration already migrated

- `tasks/Conan/dataset.py`
- `tasks/Conan/Conan.py`
- `tasks/Conan/rhythm/dataset_contracts.py`
- `tasks/Conan/rhythm/dataset_target_builder.py`
- `tasks/Conan/rhythm/task_runtime_support.py`
- `tasks/Conan/rhythm/loss_routing.py`
- `tasks/Conan/rhythm/losses.py`
- `tasks/Conan/rhythm/targets.py`
- `tasks/Conan/rhythm/metrics.py`
- `tasks/Conan/rhythm/preflight_support.py`
- `tasks/Conan/rhythm/streaming_eval.py`

## 2. Current stage split

### Structurally completed

- cache-backed rhythm supervision
- cached-only contract validation
- stateful scheduler / projector path
- learned offline teacher stage (`teacher_offline`)
- cache-only student KD stage (`student_kd`)
- retimed acoustic training stage (`student_retimed`)
- chunkwise streaming evaluation
- local rhythm unit / loss / projector / preflight tests
- CI lane for compile + unit + smoke coverage

### Still needs empirical proof / real-data closure

- real dataset readiness instead of smoke-only structure checks
- teacher export -> re-binarize -> student run closure on maintained corpora
- long-run training stability
- stronger streaming latency / quality benchmarks
- richer regression coverage for runtime / renderer / cache interactions

### Still future work

- progressive streaming reference updates
- stronger benchmark suite for long-utterance continuity and latency ceilings
- any re-expansion of legacy runtime dual-mode branches as default behavior

## 3. Current blockers for formal training claims

These are the real blockers in a clean checkout:

1. checked-in default `data/binary/vc_6layer` is absent
2. bundled smoke caches are structural sanity assets, not maintained formal training data
3. student stages require teacher export plus cache rebuild before they are meaningful
4. `student_retimed` still needs real F0 side files when `with_f0: true`
5. Windows binarization throughput remains lower than Linux / WSL due worker constraints

## 4. Current task focus

The current branch should keep focusing on:

1. projector-centric timing authority
2. cache reproducibility and fail-fast validation
3. teacher-first target generation followed by student-only KD / retimed closure
4. retimed train/infer consistency
5. streaming no-rollback / carry / budget regression hardening
6. performance work in data loading, item reuse, and batch transfer efficiency

This means:

- do not expand the style path first
- do not reintroduce runtime heuristics as the maintained mainline
- do not treat legacy dual-mode branches as the default branch ceiling

## 5. Validation evidence available right now

Current local validation evidence on the maintained branch includes:

- rhythm unit test suite passing
- compile checks passing
- `scripts/smoke_test_rhythm_v2.py` passing
- `scripts/preflight_rhythm_v2.py --help` / CLI path alive
- CPU probe / preflight tooling checked into the repo
- GitHub Actions rhythm CI workflow for Ubuntu + Windows

This is enough to claim structural readiness, not enough to claim final strong-rhythm performance.

## 6. Near-term performance priorities

The highest-value performance headroom remains in:

- dynamic batch sampling without huge endless list materialization
- reducing repeated dataset item deserialization
- avoiding repeated H2D transfer for multi-optimizer training
- keeping training / binarization entrypoints off forced single-thread mode by default
- continuing to reduce Python-heavy frame-plan / retimed-target work

## 7. Migration rule of thumb

If a new change does not strengthen one of these:

- schedule quality
- cache reproducibility
- retimed training consistency
- streaming stability
- maintained-path throughput

then it is probably not the next thing to migrate.

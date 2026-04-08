# Rhythm Migration Plan (2026-04-08)

This file is the **concise architecture / migration note**.
Operational launch details live in `docs/autodl_training_handoff.md`.
Historical probe transcripts, large ablation menus, and outdated branch diaries
were intentionally removed.

## 1. Mainline decision

The maintained project path is now explicitly **teacher-first**:

1. train the offline teacher into a stable rhythm oracle;
2. verify that teacher through teacher-conditioned audio closure and export;
3. distill the student from those teacher assets;
4. close student acoustic behavior in `S3-retimed`.

The key judgment is architectural, not cosmetic:

- the teacher in this repo is a **non-causal offline teacher**
- it is naturally an oracle / asset producer / supervision source
- it is **not** naturally the final deployment model

## 2. Current implementation truth

The maintained implementation is best described as:

> **source-side boundary / commit evidence -> explicit speech/pause budget planner -> projector execution authority -> renderer**

Reference conditioning remains useful, but its role is narrower:

- global rhythm prior
- phrase-style prior
- optional compatibility / observability path

`phase_ptr`, local trace sampling, trace reliability, cold-start gating, and
active-tail sampling still exist in the codebase, but they should be understood
as **diagnostics / compatibility surfaces**, not as the maintained control
authority.

## 3. Maintained stage naming

To stop documentation drift, the active stage map is intentionally small:

| Stage | Meaning | Current implementation |
|---|---|---|
| `T1-surface` | stable offline teacher surface | `egs/conan_emformer_rhythm_v2_teacher_offline.yaml` |
| `T2-audio` | teacher-conditioned audio closure, audit, and export | maintained as a procedure centered on `scripts/export_rhythm_teacher_targets.py` |
| `S2-kd` | conservative teacher-conditioned student KD | `egs/conan_emformer_rhythm_v2_student_kd.yaml` |
| `S3-retimed` | student acoustic closure | `egs/conan_emformer_rhythm_v2_student_retimed.yaml` |

Anything outside this list should be treated as non-mainline, historical, or
experimental unless promoted explicitly later.

## 4. What changed from the old story

The active docs now commit to these decisions:

- **teacher-first, not teacher-as-deployment-model**
- **boundary / budget / execution authority first**, not phase-first storytelling
- **reference as prior**, not as a long continuous local script
- **cloud1 snapshot as archive**, not as the live design document

This is the critical cleanup: the repo already behaves more like a
boundary-first, teacher-first system than the older documentation admitted.
The docs now follow the implementation instead of preserving every historical
explanation.

## 5. Migration rule of thumb

Use the smallest migration class that matches the real change.

### A. Runtime-only change

- metrics
- diagnostics
- inference override
- execution-time routing only

**Action:** no retraining required.

### B. Sampling / routing semantic change

- active-tail normalization
- open-tail semantic alignment
- train/infer sparse-selection alignment

**Action:** old checkpoints remain usable, but short fine-tuning is the honest default.

### C. Public-contract change

- teacher export truth changes
- student-visible cache contract changes
- new planner head / new loss / new embedding changes the learned target surface

**Action:** treat it as a new experiment, rebuild export/cache, and use a fresh `exp_name`.

The practical question is simple:

> if the public teacher/student truth surface changed, it is not a "small patch"

## 6. Cloud1 preservation rule

The cloud AutoDL state is preserved in:

- `docs/cloud1_autodl_project_status_snapshot_2026-04-08.md`

That file is a frozen archive.
This local repository is the successor surface for new work.
For actual inheritance / warm-start procedure, use:

- `docs/autodl_training_handoff.md`

## 7. Current practical boundary

The remaining blockers are mostly asset-side:

- formal processed data and binary caches are external
- teacher export assets are external
- `S3-retimed` still requires matched retimed mel / F0 / UV side assets

`T2-audio` currently exists as a maintained **procedure**, not as a separate
checked-in audio-polish training YAML. That is intentional, not a missing
artifact.

## 8. Documentation rule

If a document section does not improve one of these, it should not stay in the
mainline docs:

- teacher asset quality and auditability
- student handoff clarity
- execution authority clarity
- asset reproducibility and inheritance safety

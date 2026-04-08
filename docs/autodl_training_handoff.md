# AutoDL Training Handoff (2026-04-08)

This file is the **single operational handoff document**.
Historical audit detail, large ablation menus, and outdated stage digressions
were intentionally removed.

## 1. Scope and authority

The maintained launch story is:

1. `T1-surface`: train the offline teacher into a stable rhythm oracle
2. `T2-audio`: verify the teacher through teacher-conditioned audio closure and export
3. `S2-kd`: distill the student from teacher assets
4. `S3-retimed`: close student acoustic behavior on retimed targets

The teacher is the offline oracle / asset producer.
The student is the deployment-facing line.

Non-mainline or experimental paths may still exist in `egs/`, but they are not
the default handoff chain.

## 2. Current local successor architecture summary

The stage chain below is a **training logistics chain**, not the full
architecture story.

The current local successor branch already assumes:

- source-side boundary / commit remains the main control entry
- duration / pause budgeting is the explicit control surface
- projector execution remains the binding authority
- reference is treated as global / phrase prior, not as a long continuous control script
- `phase_ptr`, trace reliability, cold-start gating, and active-tail sampling remain mainly diagnostics / compatibility controls

## 3. Cloud1 preservation / local-successor rule

The current cloud1 / AutoDL run is preserved as a frozen snapshot:

- `docs/cloud1_autodl_project_status_snapshot_2026-04-08.md`

That snapshot is the historical baseline. Do **not** overwrite its identity
from this local checkout.

This local repo is the successor surface for new experiments. When inheriting
cloud resources:

- use a **new `exp_name`**
- keep `load_ckpt_strict: false`
- treat cloud checkpoints as **weight-only warm-start**, not exact optimizer-state continuation
- for the train100+train360 cloud line, prefer:
  - `egs/conan_emformer_rhythm_v2_teacher_offline_train100_360_pause_recall_cloud1_handoff.yaml`

## 4. Required assets on AutoDL

Required runtime assets:

- cloned repo checkout
- Python 3.10 environment with `pip install -r requirements.txt`
- base Conan checkpoints you actually use
- real processed corpus
- real binary cache with non-empty `train` and `valid`
- teacher export directory for student stages
- retimed cache plus matched F0/UV side data for `S3-retimed`

Practical rule:

- the checked-in smoke assets are for structural verification only
- they are not the formal training assets for the maintained path

## 5. Minimal pre-launch checks

```bash
conda activate conan
python -m compileall -q modules tasks scripts tests utils data_gen
python -m unittest discover -s tests/rhythm -p "test_*.py"
python -u scripts/smoke_test_rhythm_v2.py
```

Use strict preflight before every real stage launch:

```bash
python scripts/preflight_rhythm_v2.py --config <stage_config> --binary_data_dir <binary_dir> --processed_data_dir <processed_dir> --splits train valid --inspect_items 4 --model_dry_run --strict_processed_data_dir
```

## 6. Operator gates before stage handoff

### 6.1 Teacher stabilization gate

Do **not** treat `T1-surface` as ready just because one validation scalar
improves. Before handing the teacher to the next stage, watch the whole
teacher-control set together:

- `pause_event_precision`
- `pause_event_recall`
- `pause_event_f1`
- `pause_event_recall_nonboundary`
- `pause_recall_drop_post_from_planner`
- `pause_support_cover_at_topk`
- `prefix_drift_l1`
- `exec_total_corr`
- `budget_projection_repair_ratio_mean`

Operationally, teacher steady-state means a **small rolling validation window**
is stable, not that one checkpoint spikes higher than the rest.

### 6.2 Teacher handoff checkpoint selection

Do not mechanically hand off `model_ckpt_best.pt` or the highest pause-F1
checkpoint.

Instead:

1. build a candidate pool
   - overall-best checkpoint
   - pause-oriented best checkpoint
   - recent step checkpoints
2. apply hard gates
   - no `nan/inf`
   - module-only semantics still intact
   - no unexpected acoustic/pitch leakage into the teacher stage
   - export/preflight path is complete
   - no clear late-window degradation trend
3. rank by handoff friendliness
   - recent checkpoint that still passes the hard gate
   - higher `pause_event_f1`
   - higher `exec_total_corr`
   - lower `prefix_drift_l1`
   - lower `pause_recall_drop_post_from_planner`

Selection rule:

- prefer the **most recent stable balanced checkpoint**
- do not optimize for one scalar in isolation
- judge the checkpoint by whether it is exportable, audible, and inheritable

### 6.3 Stage-2 takeover gate

`S2-kd` should start only when all of the following are true:

1. a teacher handoff checkpoint has been explicitly selected
2. that checkpoint passes the teacher steady-state gate
3. teacher targets have been exported for `train`, `valid`, and `test`
4. the rebuilt stage-2 binary passes strict preflight

Critical nuance:

- the maintained base config family already exposes streaming-prefix semantics during training
- stage 2 is still where the **remaining** runtime/prefix mismatch should be absorbed
- do not mutate the teacher export truth mid-handoff just to chase student-side behavior

The maintained reading stays conservative:

- keep `primary=teacher`
- keep the shape-style KD definition
- let the student absorb the remaining runtime mismatch on top of the fixed teacher export

### 6.4 Stage-3 takeover gate

`S3-retimed` should start only when all of the following are true:

1. stage 2 has a selected handoff checkpoint
2. the recent stage-2 validation window is stable
3. strict preflight passes
4. the rebuilt stage-3 binary really contains:
   - retimed mel targets
   - matched F0/UV side data
5. the stage does **not** need `use_pitch_embed=False` just to get through preflight

The maintained stage-3 baseline is cached-first.
Move to `hybrid` only after teacher handoff and stage-2 handoff are already stable.

## 7. Maintained launch order

### 7.1 `T1-surface` -> `teacher_offline`

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml --exp_name conan_rhythm_t1_surface --reset -hp "load_ckpt='checkpoints/<base_conan_ckpt>',binary_data_dir='data/binary/<teacher_binary>',processed_data_dir='data/processed/<dataset>'"
```

Intent:

- train a stable offline teacher surface
- keep it auditable
- keep it exportable
- keep teacher-side main-branch audit semantics available

### 7.2 `T2-audio` -> teacher-conditioned audio closure / export

The current repo does **not** require a separate formal teacher-audio-polish
YAML to express this step. `T2-audio` is maintained as the teacher-side closure
+ export procedure:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/export_rhythm_teacher_targets.py --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml --ckpt checkpoints/conan_rhythm_t1_surface --output_dir data/teacher_targets/conan_rhythm_t1_surface --binary_data_dir data/binary/<teacher_binary> --processed_data_dir data/processed/<dataset> --splits train valid test --device cuda --overwrite
```

Operational meaning of `T2-audio`:

- confirm the teacher can be audited / auditioned as the main branch
- export clean teacher assets for all required splits
- treat this export as the student supervision source
- freeze the export source only after the checkpoint passes the handoff rule above

### 7.3 Rebuild the student-facing cache

```bash
python -m data_gen.tts.runs.binarize --reset --config egs/conan_emformer_rhythm_v2_student_kd.yaml --exp_name bin_student_kd_<run_tag> -hp "processed_data_dir='data/processed/<dataset>',binary_data_dir='data/binary/<student_kd_binary>',rhythm_teacher_target_dir='data/teacher_targets/conan_rhythm_t1_surface'"
```

### 7.4 `S2-kd` -> `student_kd`

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/conan_emformer_rhythm_v2_student_kd.yaml --exp_name conan_rhythm_s2_kd --reset -hp "load_ckpt='checkpoints/conan_rhythm_t1_surface',binary_data_dir='data/binary/<student_kd_binary>',processed_data_dir='data/processed/<dataset>',rhythm_teacher_target_dir='data/teacher_targets/conan_rhythm_t1_surface'"
```

Maintained stage-2 intent:

- keep the teacher export truth fixed
- keep the student KD surface conservative
- let the student absorb the remaining streaming/prefix mismatch before stage 3 closes the acoustic loop

### 7.5 `S3-retimed` -> `student_retimed`

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/conan_emformer_rhythm_v2_student_retimed.yaml --exp_name conan_rhythm_s3_retimed --reset -hp "load_ckpt='checkpoints/conan_rhythm_s2_kd',binary_data_dir='data/binary/<student_retimed_binary>',processed_data_dir='data/processed/<dataset>',rhythm_teacher_target_dir='data/teacher_targets/conan_rhythm_t1_surface',rhythm_retimed_target_mode='cached'"
```

Hard requirement:

- `S3-retimed` needs retimed acoustic targets plus matched F0/UV assets
- do not launch it on the smoke binary and call that a formal pass
- keep the default cached-first baseline first
- move to `egs/conan_emformer_rhythm_v2_student_retimed_hybrid_ablation.yaml` only after the cached stage-3 path is stable

## 8. Stop conditions

Stop and inspect if any of the following happens:

- strict preflight only passes after weakening the maintained config
- `T1-surface` cannot emit a valid teacher-side main branch for audit/export
- teacher export does not cover `train`, `valid`, and `test`
- the teacher checkpoint was chosen by a single best scalar without passing the handoff gate
- `S2-kd` starts without a clean teacher export / rebuilt cache
- `S3-retimed` is missing matched retimed pitch / F0 assets
- you are about to reuse the cloud run's old `exp_name` as if it were an exact local continuation

## 9. First-day checklist

1. clone repo and create the `conan` environment
2. mount real processed data, binary cache, and checkpoints
3. run compileall + tests + maintained smoke
4. run strict preflight for `T1-surface`
5. launch `T1-surface`
6. run `T2-audio` export / closure checks
7. rebuild the student-facing cache
8. run strict preflight for `S2-kd`
9. launch `S2-kd`
10. prepare retimed cache + matched F0/UV assets
11. run strict preflight for `S3-retimed`
12. only then launch `S3-retimed`

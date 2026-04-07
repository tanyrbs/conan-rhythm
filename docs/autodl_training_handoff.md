# AutoDL Training Handoff (2026-04-07)

Validated local baseline before writing this handoff: `67bd85b` (`Add retimed pitch handoff regression coverage`).

This document is the practical launch handoff for moving the maintained rhythm path onto an AutoDL Linux GPU instance. It is intentionally operational: what to copy, what to run, what must pass, and what signals mean "stop".

## 1. Go / no-go summary

### Ready to launch once real assets are mounted

- `teacher_offline`
- `student_kd` (only after a real teacher export + rebuilt stage-2 binary)

### Do **not** bless yet from the shared smoke assets

- `student_retimed` with the default maintained config

Reason: the checked-in smoke `student_binary` still lacks real F0, and the formal maintained stage-3 config keeps `use_pitch_embed=true` / `with_f0=true`.

## 2. Fresh local checks rerun on this checkout

These were rerun locally before writing this handoff:

```bash
python -m compileall -q modules tasks scripts tests utils data_gen
python -m unittest discover -s tests/rhythm -p "test_*.py"
python -u scripts/smoke_test_rhythm_v2.py
python scripts/preflight_rhythm_v2.py \
  --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml \
  --binary_data_dir data/binary/libritts_single_smoke_rhythm_v4 \
  --processed_data_dir data/processed/libritts_local_real_smoke \
  --splits train \
  --inspect_items 2 \
  --model_dry_run \
  --strict_processed_data_dir
python scripts/preflight_rhythm_v2.py \
  --config egs/conan_emformer_rhythm_v2_student_kd.yaml \
  --binary_data_dir artifacts/rhythm_teacher_export_student_kd/5e1bc8ca5f/student_binary \
  --processed_data_dir data/processed/libritts_local_real_smoke \
  --splits train valid \
  --inspect_items 2 \
  --model_dry_run \
  --strict_processed_data_dir
python scripts/preflight_rhythm_v2.py \
  --config egs/conan_emformer_rhythm_v2_student_retimed.yaml \
  --binary_data_dir artifacts/rhythm_teacher_export_student_kd/5e1bc8ca5f/student_binary \
  --processed_data_dir data/processed/libritts_local_real_smoke \
  --splits train \
  --inspect_items 2 \
  --model_dry_run \
  --strict_processed_data_dir
```

Observed result:

- compileall: **passed**
- rhythm tests: **243 passed**
- maintained smoke test: **passed**
- `teacher_offline` strict preflight + dry-run: **passed**
- `student_kd` strict preflight + dry-run: **passed**
- default `student_retimed` strict preflight + dry-run: **failed as expected** on missing `f0`

The stage-3 smoke counter-check also still works only under a non-formal override:

```bash
python scripts/preflight_rhythm_v2.py \
  --config egs/conan_emformer_rhythm_v2_student_retimed.yaml \
  --hparams use_pitch_embed=False,binarization_args.with_f0=False \
  --binary_data_dir artifacts/rhythm_teacher_export_student_kd/5e1bc8ca5f/student_binary \
  --processed_data_dir data/processed/libritts_local_real_smoke \
  --splits train \
  --inspect_items 2 \
  --model_dry_run \
  --strict_processed_data_dir
```

That override passes structurally, but it is **smoke-only** and must not be treated as a formal stage-3 readiness signal.

## 3. What must exist on the AutoDL machine

Use Linux paths, but keep the same logical contract.

### Required runtime assets

| Surface | Must exist |
|---|---|
| repo | cloned checkout of this branch |
| conda env | Python 3.10 + `pip install -r requirements.txt` |
| base checkpoints | `checkpoints/Emformer`, `checkpoints/hifigan_vc`, and any path-specific optional ckpts you actually use |
| formal processed corpus | real `data/processed/<dataset>`; not the placeholder `data/processed/vc/example_metadata.json` |
| formal binary cache | real `data/binary/<dataset>` with non-empty `train` and `valid` splits |
| teacher export dir for stage-2/3 | `data/teacher_targets/<run_name>` |
| retimed stage-3 assets | rebuilt binary containing retimed mel targets **and** matched F0/UV side data |

### Asset rules that matter

- Formal training should use rhythm cache **v5**.
- The checked-in `libritts_single_smoke_rhythm_v4` is compatibility smoke only.
- The checked-in `artifacts/rhythm_teacher_export_student_kd/...` chain is smoke only because the teacher checkpoint is `bootstrap_random_init`.
- If you have older descriptor review exports or planner-facing cached sidecars
  from before the boundary-trace softening change, rebuild them; `boundary_trace`
  is now a soft strength trace while `boundary_ratio` remains the binary event-rate stat.
- Do not start `student_kd` until teacher export covers `train`, `valid`, and `test`.
- Do not start default `student_retimed` unless the rebuilt stage-3 binary really contains usable F0/UV for the retimed path.
- Even though `train_sets` is now checked more strictly, the first formal AutoDL baseline should still prefer one unified binary over concatenating multiple separately-binarized train roots.

## 4. AutoDL environment bootstrap

Recommended first-time setup on the cloud instance:

```bash
cd /root/autodl-tmp
git clone git@github.com:tanyrbs/conan-rhythm.git
cd conan-rhythm
conda create -n conan python=3.10 -y
conda activate conan
pip install -r requirements.txt
```

Then sync or mount your real data/checkpoint assets into the repo-relative paths you plan to use.

Recommended first launch posture:

- start with **single-GPU** training first
- keep maintained configs unchanged on the first long run
- avoid turning on extra research knobs (`context_match`, stage-2 algorithmic-distill ablations, etc.) before the default chain is clean
- use a **fresh `exp_name`** or `--reset` for fresh runs, so stale saved config does not silently override the current YAML
- the maintained stage configs intentionally pin `load_ckpt_strict: false` for cross-stage warm-start; still inspect startup logs so only expected Rhythm V2 / offline-teacher / pitch keys are missing
- maintained note:
  - the stage-3 acoustic warmup is now anchored to the actual start of
    `student_retimed`, so warm-starting from a high-step `student_kd`
    checkpoint no longer skips the intended stage-3 curriculum

## 5. Pre-launch checklist on AutoDL

Run these in order.

### 5.1 Code / environment sanity

```bash
conda activate conan
python -m compileall -q modules tasks scripts tests utils data_gen
python -m unittest discover -s tests/rhythm -p "test_*.py"
python -u scripts/smoke_test_rhythm_v2.py
```

Expected:

- compileall passes
- tests stay at least at the local baseline order of magnitude (currently 243 rhythm tests)

If you are building a LibriTTS processed metadata set for `train-clean-100 + train-clean-360`, the local metadata helper now supports repeated or comma-separated `--train_split` flags instead of forcing a single train split.
- smoke still reports healthy closure signals such as:
  - `metric exec total corr = 1.0`
  - `metric prefix drift l1 = 0.0`
  - very small `budget projection repair ratio`

### 5.2 Strict preflight before every stage

Always use `--strict_processed_data_dir` for formal cloud launches.

#### Stage 1: `teacher_offline`

```bash
python scripts/preflight_rhythm_v2.py \
  --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml \
  --binary_data_dir data/binary/<teacher_binary> \
  --processed_data_dir data/processed/<dataset> \
  --splits train valid \
  --inspect_items 4 \
  --model_dry_run \
  --strict_processed_data_dir
```

Pass criteria:

- non-empty `train` and `valid`
- no placeholder processed path warning/error
- dry-run succeeds
- because the maintained teacher config sets `rhythm_teacher_as_main: true`, this path is still allowed to emit `mel_out` for teacher audition/export semantics

#### Stage 2: `student_kd`

```bash
python scripts/preflight_rhythm_v2.py \
  --config egs/conan_emformer_rhythm_v2_student_kd.yaml \
  --binary_data_dir data/binary/<student_kd_binary> \
  --processed_data_dir data/processed/<dataset> \
  --hparams rhythm_teacher_target_dir='data/teacher_targets/<teacher_export>' \
  --splits train valid \
  --inspect_items 4 \
  --model_dry_run \
  --strict_processed_data_dir
```

Pass criteria:

- cached teacher fields present on both `train` and `valid`
- dry-run succeeds
- module-only objective is still the active stage contract

#### Stage 3: `student_retimed`

```bash
python scripts/preflight_rhythm_v2.py \
  --config egs/conan_emformer_rhythm_v2_student_retimed.yaml \
  --binary_data_dir data/binary/<student_retimed_binary> \
  --processed_data_dir data/processed/<dataset> \
  --hparams rhythm_teacher_target_dir='data/teacher_targets/<teacher_export>' \
  --splits train valid \
  --inspect_items 4 \
  --model_dry_run \
  --strict_processed_data_dir
```

Pass criteria:

- retimed mel target fields present
- real F0/UV present
- dry-run succeeds **without** `use_pitch_embed=False`

If this command only passes after disabling pitch embed, stop. That means your stage-3 asset contract is still smoke-grade, not formal.

## 6. Training commands to hand off

Use these as the maintained launch templates.

Warm-start rule:

- `teacher_offline` should normally warm-start from a base Conan checkpoint
- `student_kd` should normally warm-start from the finished `teacher_offline` checkpoint
- `student_retimed` should normally warm-start from the finished `student_kd` checkpoint
- for the **first formal stage-3 run**, prefer `rhythm_retimed_target_mode='cached'` as the safer debug baseline before you A/B `hybrid`

### 6.1 `teacher_offline`

```bash
conda activate conan
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
  --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml \
  --exp_name conan_rhythm_v2_teacher_offline \
  --reset \
  -hp "load_ckpt='checkpoints/<original_conan_ckpt_or_dir>',binary_data_dir='data/binary/<teacher_binary>',processed_data_dir='data/processed/<dataset>'"
```

What this stage is for:

- learn the offline teacher surfaces
- keep teacher as the main branch
- keep audio-export / audition semantics intact
- maintained semantics note:
  - the checked-in `teacher_offline` config is still a **cached-guidance
    bootstrap** (`cached_only + primary guidance`)
  - it is not yet the research variant where the offline teacher directly uses
    runtime algorithmic-teacher surfaces as the primary target
  - keep the external story aligned with the YAML unless you introduce a
    separate ablation config and validate it explicitly

### 6.2 Export teacher targets

```bash
conda activate conan
CUDA_VISIBLE_DEVICES=0 python scripts/export_rhythm_teacher_targets.py \
  --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml \
  --ckpt checkpoints/conan_rhythm_v2_teacher_offline \
  --output_dir data/teacher_targets/conan_rhythm_v2_teacher_offline \
  --binary_data_dir data/binary/<teacher_binary> \
  --processed_data_dir data/processed/<dataset> \
  --splits train valid test \
  --device cuda \
  --overwrite
```

Hard rule:

- export all of `train valid test`
- do not move to stage-2 on a partial export

### 6.3 Rebuild the stage-2 binary

```bash
conda activate conan
python -m data_gen.tts.runs.binarize \
  --reset \
  --config egs/conan_emformer_rhythm_v2_student_kd.yaml \
  --exp_name bin_student_kd_<run_tag> \
  -hp "processed_data_dir='data/processed/<dataset>',binary_data_dir='data/binary/<student_kd_binary>',rhythm_teacher_target_dir='data/teacher_targets/conan_rhythm_v2_teacher_offline'"
```

After rebuild, rerun stage-2 preflight before training.

### 6.4 `student_kd`

```bash
conda activate conan
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
  --config egs/conan_emformer_rhythm_v2_student_kd.yaml \
  --exp_name conan_rhythm_v2_student_kd \
  --reset \
  -hp "load_ckpt='checkpoints/conan_rhythm_v2_teacher_offline',binary_data_dir='data/binary/<student_kd_binary>',processed_data_dir='data/processed/<dataset>',rhythm_teacher_target_dir='data/teacher_targets/conan_rhythm_v2_teacher_offline'"
```

Maintained expectation:

- this is still a module-only stage
- the maintained KD path is effectively teacher-main + shape-only KD
- do not open the experimental context-match branch on the first formal run

### 6.5 Rebuild / verify the stage-3 binary

Before stage-3, the rebuilt binary must contain:

- teacher targets
- retimed mel targets
- matched F0/UV that stay valid for retimed supervision

If needed, rebuild the binary with the stage-3 config and the appropriate side assets mounted; then rerun strict preflight.

### 6.6 `student_retimed`

```bash
conda activate conan
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
  --config egs/conan_emformer_rhythm_v2_student_retimed.yaml \
  --exp_name conan_rhythm_v2_student_retimed \
  --reset \
  -hp "load_ckpt='checkpoints/conan_rhythm_v2_student_kd',binary_data_dir='data/binary/<student_retimed_binary>',processed_data_dir='data/processed/<dataset>',rhythm_teacher_target_dir='data/teacher_targets/conan_rhythm_v2_teacher_offline',rhythm_retimed_target_mode='cached'"
```

Optional upper-bound A/B after the maintained cached baseline is stable:

- `egs/conan_emformer_rhythm_v2_student_retimed_hybrid_ablation.yaml`
- keeps the same conservative stage-3 warm-start / ramp defaults
- switches `rhythm_retimed_target_mode` from `cached` to `hybrid`
- still waits until `rhythm_online_retimed_target_start_steps` before trusting
  online retimed targets

First formal stage-3 launch rules:

- keep the maintained config unchanged
- no need to start from `student_retimed_balanced`; the maintained stage-3 yaml now already carries the conservative ramp / EMA-balance defaults
- do **not** disable pitch embed just to get a run started
- if strict preflight fails on F0 or matched retimed pitch, fix assets first

## 7. What to watch during training

### 7.1 Global watch items

Watch these in logs / TensorBoard / probe summaries:

- `total_loss`
- `L_base`
- `L_rhythm_exec`
- `L_stream_state`
- `L_pitch`
- `grad_norm_before_clip`
- validation `val_loss`

### 7.2 Stage-1 / stage-2 watch items

For module-only stages (`teacher_offline`, `student_kd`):

- `L_base` should stay effectively off / zero-dominated
- acoustic objectives should not silently come back during validation
- if you see mel/pitch dominating these stages, stop and inspect stage semantics

Useful runtime flags if exported by probes/logs:

- `rhythm_metric_module_only_objective`
- `rhythm_metric_skip_acoustic_objective`
- `rhythm_metric_disable_acoustic_train_path`

Pause-specific interpretation:

- if `pause_event_precision` is clearly higher than `pause_event_recall`, the model is usually **too conservative about where to place pauses**
- that is more of a **pause support / event recall** problem than a pure `L_budget` problem
- a common companion pattern is:
  - `L_exec_pause` decreases slowly
  - `prefix_drift_l1` also decreases slowly because missed pauses keep the cumulative clock behind

The maintained code now exposes two separate pause views:

- `L_exec_pause_value`: the original pause-magnitude regression term
- `L_pause_event`: the auxiliary support-first event term

Recommended first response when recall is the clear bottleneck:

```yaml
rhythm_pause_event_weight: 0.20
rhythm_pause_event_threshold: 0.5
rhythm_pause_event_temperature: 0.20
rhythm_pause_event_pos_weight: 2.5
```

Why this is the preferred first move:

- it keeps the original pause amount regression intact
- it directly penalizes missed pause events (`FN`) instead of only matching pause magnitude
- it makes the run easier to interpret because "pause amount" and "pause support" are now separated in logs

What to watch after enabling it:

- `pause_event_precision`
- `pause_event_recall`
- `pause_event_f1`
- `L_exec_pause_value`
- `L_pause_event`
- `prefix_drift_l1`

If recall improves offline but streaming pause placement still looks conservative, the next lever is usually projector support capacity rather than more budget weight, e.g. `rhythm_projector_pause_topk_ratio` or stronger boundary-biased pause support.

### 7.3 Stage-3 watch items

These are the most important stage-3 observability signals:

- `rhythm_metric_pitch_supervision_disabled`
- `rhythm_metric_missing_retimed_pitch_target`
- `rhythm_metric_online_retimed_trace_gate_mean`
- `acoustic_target_length_mismatch_abs_before_align`
- `acoustic_target_resampled_to_output`
- `acoustic_target_trimmed_to_output`
- `rhythm_metric_acoustic_target_is_retimed`

Interpretation:

- formal stage-3 should **not** live in a state where `rhythm_metric_pitch_supervision_disabled=1`
- if `rhythm_metric_missing_retimed_pitch_target=1`, your retimed pitch assets are not ready
- if `rhythm_metric_online_retimed_trace_gate_mean` stays very low, the model is repeatedly telling you the local reference window is exhausted / unreliable; online retimed acoustic targets are being downweighted on purpose
- repeated large length mismatch / resample / trim activity means your retimed target contract needs inspection

### 7.4 Known stage-3 risk pattern

The latest 2000-step smoke probe still shows a bad imbalance pattern:

- `L_base.mean ~= 6.53`
- `L_rhythm_exec.mean ~= 0.00279`
- `L_stream_state.mean ~= 0.000746`
- `grad_norm_before_clip.mean ~= 158.86`
- `grad_norm_before_clip.max ~= 275.92`

So for the first real stage-3 run, actively watch whether:

- `L_base` overwhelms control losses for too long
- gradients stay clip-heavy for many thousands of steps
- pitch supervision is silently disabled
- the online trace gate collapses for a large fraction of training, which usually means the short-ref / longer-source regime is under-modeled or overrepresented

If those persist, stop and inspect before wasting a long AutoDL run.

## 8. Stop conditions

Stop the run and inspect assets/code if any of these happen:

- strict preflight only passes after weakening the maintained config
- module-only stages show meaningful acoustic/pitch objective activity
- stage-3 logs say pitch supervision is disabled or matched retimed pitch is missing
- stage-3 keeps resampling/trimming retimed targets on a large fraction of batches
- losses go `nan` / `inf`
- grad norm is persistently huge and clipping dominates progress
- you accidentally launched on smoke assets or smoke teacher export

## 9. Practical handoff notes for the next operator

- Keep the maintained chain narrow: `teacher_offline -> export -> student_kd -> retimed rebuild -> student_retimed`.
- Do not treat the smoke assets in this repo as formal training data.
- Keep `teacher_offline` as `teacher_as_main` if you need audio/audition outputs from the teacher stage.
- Use fresh `exp_name`s for new launches; reuse an old run only when you intentionally want to continue it.
- For the first AutoDL launch, prefer correctness over speed: single GPU, maintained config, strict preflight, no experimental flags.
- After every stage transition, rerun strict preflight before starting the next long job.

## 10. Minimal "first day on AutoDL" checklist

1. Clone repo and create `conan` env.
2. Mount/sync real processed data, binary cache, and checkpoints.
3. Run compileall + rhythm tests + maintained smoke.
4. Run strict teacher preflight.
5. Launch `teacher_offline`.
6. Export teacher targets for `train/valid/test`.
7. Rebuild stage-2 binary.
8. Run strict stage-2 preflight.
9. Launch `student_kd`.
10. Prepare stage-3 retimed/F0 assets.
11. Run strict stage-3 preflight with the default maintained config.
12. Only then launch `student_retimed`.

# rhythm_v3 Setup and Training Guide

This is the practical setup/training guide for the maintained **`rhythm_v3`** path.

If you are doing new work in this repository, start from:

- `egs/conan_emformer_rhythm_v3.yaml`
- `tasks/run.py`
- `tasks.Conan.Conan.ConanTask`
- `modules/Conan/rhythm_v3/`

For the architecture/spec reading, see `docs/rhythm_migration_plan.md`.
For validation/debug export and the retained five-figure review surface, see
`docs/rhythm_v3_validation_stack.md`.

---

## 1. What the maintained path is

The maintained V1 is now read as a **two-layer system**:

- **Layer 0: Stable Lattice Interface**  
  raw HuBERT/content units -> causal, debounce-stabilized, commit-safe run lattice
- **Layer 1: Minimal Retimer**  
  source-anchored relative retiming on top of that lattice

Operationally, the maintained default means:

- `rhythm_v3_backbone: prompt_summary`
- `rhythm_v3_warp_mode: none`
- `rhythm_v3_anchor_mode: source_observed`
- `rhythm_v3_minimal_v1_profile: true`
- `rhythm_v3_rate_mode: simple_global`
- `rhythm_v3_simple_global_stats: true`
- `rhythm_v3_emit_silence_runs: true`
- `rhythm_v3_use_log_base_rate: false`
- `rhythm_v3_use_reference_summary: false`
- `rhythm_v3_use_learned_residual_gate: false`
- `rhythm_v3_disable_learned_gate: true`
- `rhythm_v3_drop_edge_runs_for_g: 1`
- `rhythm_v3_require_same_text_paired_target: true`
- speech-only prompt global-rate estimation
- speech = coarse + local
- silence = clipped coarse-only follower
- carry rounding + prefix budget at execution time

The intended Layer-0 lattice contract is also conservative: before retiming,
the shared train/infer stabilizer should suppress short flicker bridges and
short fake silence islands when boundary evidence is weak, instead of asking
the duration writer to absorb that noise.

So this branch is **not** training a free-form pause planner. It is training a **source-anchored causal retimer** over a stabilized run lattice.

### 1.1 Recommended falsification profiles

The maintained yaml already exposes the switches needed for staged validation.
In practice, keep one base config and move between these four profiles with
small overrides:

| Profile | Purpose | Key overrides |
| --- | --- | --- |
| `A` | static / analytic falsification | `rhythm_v3_eval_mode=analytic`, `rhythm_v3_disable_local_residual=true`, `rhythm_v3_disable_coarse_bias=true`, `lambda_rhythm_pref=0.0`, `lambda_rhythm_cons=0.0`, `rhythm_v3_silence_coarse_weight=0.0` |
| `B` | coarse-only validation training | `rhythm_v3_eval_mode=coarse_only`, `rhythm_v3_disable_local_residual=true`, `rhythm_v3_disable_coarse_bias=false`, `lambda_rhythm_bias=0.20`, `lambda_rhythm_pref=0.0`, `lambda_rhythm_cons=0.0` |
| `C` | full learned sanity | `rhythm_v3_eval_mode=learned`, `rhythm_v3_disable_local_residual=false`, `rhythm_v3_disable_coarse_bias=false`, `lambda_rhythm_pref=0.0`, `lambda_rhythm_cons=0.0` |
| `D` | strict-causal prefix fine-tune | `rhythm_v3_eval_mode=learned`, `lambda_rhythm_pref=0.05`, `lambda_rhythm_cons=0.05`, `rhythm_v3_silence_coarse_weight=0.0` |

Keep `rhythm_v3_g_variant=raw_median` as the first-line baseline. Only move to
`weighted_median`, `trimmed_mean`, or `unit_norm` after the static `g` audit
shows a real reason.

--- 

## 2. Current project layout

The repo is easiest to understand as:

| Path | Role |
| --- | --- |
| `egs/` | experiment yaml configs |
| `tasks/run.py` | main train/test entrypoint |
| `tasks/Conan/Conan.py` | `ConanTask`, the training task class used by current configs |
| `tasks/Conan/rhythm/common/` | shared rhythm dataset/loss/target/runtime glue |
| `tasks/Conan/rhythm/duration_v3/` | maintained v3 task/data logic |
| `modules/Conan/rhythm_v3/` | maintained runtime implementation |
| `modules/Conan/rhythm/supervision.py` | source rhythm cache construction used by binarization/data paths |
| `data_gen/conan_binarizer.py` | Conan binary dataset builder |
| `data_gen/tts/runs/binarize.py` | binarization entrypoint |
| `inference/` | runtime/eval helpers |
| `tests/rhythm/` | contract/loss/runtime regression tests |

### 2.1 Training control flow

The current training control flow is:

1. `tasks/run.py`
2. `utils.commons.hparams.set_hparams()` loads config / overrides / saved experiment config
3. `task_cls` is resolved from yaml (`tasks.Conan.Conan.ConanTask` in current Conan configs)
4. `BaseTask.start()` creates `Trainer(...)`
5. `Trainer.fit(...)` runs training

### 2.2 Data/control flow for rhythm_v3

At a high level:

1. **processed metadata** describes wav paths + content tokens
2. **binarization** builds indexed datasets and rhythm caches
3. **dataset layer** loads:
   - source lattice / source anchor
   - prompt/reference unit conditioning
   - paired target projection targets
4. **`modules/Conan/rhythm_v3/`** runs the retimer during training/inference
5. **projector** turns predicted continuous stretch into committed integer run durations

The important contract detail is that train-time cache construction and
inference-time prompt/source preparation now share the same `rhythm_v3`
stable-lattice builder rather than separate legacy/v3 runizers.

For the maintained prompt-summary path, `prompt_speech_mask` is part of the
public conditioning contract rather than an internal optional hint.

---

## 3. What training actually expects

### 3.1 Processed-data requirements

For Conan binarization, your processed-data directory must provide at least:

- `metadata.json` **or** `metadata_vctk_librittsr_gt.json`
- `spker_set.json`
- the referenced wav files
- F0 `.npy` files if `with_f0: true` remains enabled in the config

### 3.2 Minimal metadata item schema

Each metadata item should contain at least:

- `item_name`
- `wav_fn`
- `hubert`

Common optional fields:

- `split` (`train` / `valid` / `test`, also accepts aliases like `dev`, `val`, `evaluation`)
- `duration`

Important details from the current implementation:

- `item_name` is used to derive speaker id as `item_name.split('_', 1)[0]`
- that speaker key **must** exist in `processed_data_dir/spker_set.json`
- if `split` is missing for all items, binarization falls back to `valid_prefixes` / `test_prefixes` or `build_summary.json`
- if only part of the metadata carries `split`, binarization raises an error instead of mixing split rules

### 3.3 F0 layout expected by `ConanBinarizer`

With the current default config, `ConanBinarizer` expects one F0 file per wav at:

- wav: `<dir>/<name>.wav`
- f0: `<dir>_f0/<name>_f0.npy`

So if your wav is:

- `data/raw/myset/train/spk1_0001.wav`

then the expected F0 path is:

- `data/raw/myset/train_f0/spk1_0001_f0.npy`

### 3.4 Paired-target supervision is required for canonical v3 training

This is the biggest training-contract detail to understand.

The maintained v3 config keeps:

- `rhythm_v3_allow_source_self_target_fallback: false`

So canonical v3 training does **not** silently use the source item itself as the duration target.

You need one of these:

1. **explicit `unit_duration_tgt` already stored in the binary sample**, or
2. **an external paired target item** that the dataset can project onto the source lattice

In practice, the normal way is to provide a **pair manifest** via:

- `rhythm_pair_manifest_path`

Without that, the dataset may still find a reference prompt item, but the paired target side will be missing and training will fail by design.

### 3.5 Pair manifest format

The loader accepts JSON / JSONL / YAML.

Two practical formats are supported.

#### Flat form

```json
[
  {
    "source_item_name": "spk1_train_0001",
    "ref_item_name": "spk1_train_0007",
    "target_item_name": "spk1_train_0015"
  }
]
```

#### Grouped form

```json
[
  {
    "source": "spk1_train_0001",
    "group_id": "spk1_case_01",
    "refs": [
      {
        "ref": "spk1_train_0007",
        "target": "spk1_train_0015",
        "pair_rank": 0
      },
      {
        "ref": "spk1_train_0010",
        "target": "spk1_train_0020",
        "pair_rank": 1
      }
    ]
  }
]
```

Semantics:

- `source_item_name`: acoustic/content source to be retimed
- `ref_item_name`: prompt/reference item used to build prompt rhythm conditioning
- `target_item_name`: paired target item whose run lattice is projected back onto the source lattice for supervision

Also note:

- current v3 defaults disallow same-text reference but allow same-text paired target
- pair-manifest grouping can be used for grouped batching and controlled A/B comparisons

For the maintained `rhythm_v3_minimal_v1_profile`, treat that as a hard data
contract:

- **reference prompt**: same-speaker / different-text
- **paired target supervision**: same-text projection target (`rhythm_v3_require_same_text_paired_target: true`), unless `unit_duration_tgt` is already explicitly cached in the sample

---

## 4. Recommended way to create a local experiment yaml

Do **not** edit `egs/conan_emformer_rhythm_v3.yaml` directly for every dataset.
Create a small derived yaml instead.

Example:

```yaml
# egs/local_rhythm_v3.yaml
base_config: egs/conan_emformer_rhythm_v3.yaml

processed_data_dir: data/processed/libritts_local
binary_data_dir: data/binary/libritts_local_rhythm_v3
metafile_path: data/processed/libritts_local/metadata.json

# F0 extraction helper
pe: rmvpe
pe_ckpt: checkpoints/rmvpe/model.pt

# Canonical paired-target training
rhythm_pair_manifest_path: data/processed/libritts_local/pairs.json
rhythm_pair_manifest_prefixes: train

# Optional: faster smoke run
# max_updates: 20000
# val_check_interval: 1000
```

Why this is safer:

- the base yaml in `egs/` stays as the branch reference
- dataset-specific paths live in one place
- `utils/extract_f0_rmvpe.py` can read `metafile_path` / `pe_ckpt` directly from the yaml
- you avoid mixing one experiment's paths with another's

---

## 5. Practical setup commands

Examples below use Windows-style `py -3`, matching the rest of this repo. Replace with `python` if preferred.

### 5.1 Install dependencies

```bash
py -3 -m pip install -r requirements.txt
```

If you use RMVPE F0 extraction, also make sure the optional RMVPE stack/checkpoint is available.

### 5.2 Build processed metadata (example helper for local LibriTTS)

If your corpus is a local LibriTTS tree, the repo already provides a helper:

```bash
py -3 scripts/build_libritts_local_processed_metadata.py ^
  --raw_root D:\datasets\LibriTTS ^
  --processed_data_dir data\processed\libritts_local ^
  --config egs\local_rhythm_v3.yaml ^
  --emformer_ckpt checkpoints\Emformer\model_ckpt_steps_700000.ckpt
```

That script writes, at minimum:

- `metadata_vctk_librittsr_gt.json`
- `metadata.json`
- `spker_set.json`
- `build_summary.json`

If you are using another corpus, generate the same artifacts yourself.

### 5.3 Extract F0

With `with_f0: true`, do this before binarization:

```bash
py -3 utils\extract_f0_rmvpe.py ^
  --config egs\local_rhythm_v3.yaml ^
  --batch-size 16 ^
  --max-tokens 120000
```

This writes `<wav_dir>_f0/*.npy` files that `ConanBinarizer` expects.

### 5.4 Binarize

```bash
py -3 -m data_gen.tts.runs.binarize ^
  --config egs\local_rhythm_v3.yaml
```

You can also attach an experiment name if you want a saved resolved config snapshot:

```bash
py -3 -m data_gen.tts.runs.binarize ^
  --config egs\local_rhythm_v3.yaml ^
  --exp_name local_rhythm_v3_data ^
  --reset
```

Binarization writes indexed datasets under `binary_data_dir`, including:

- `train.data` / `train.idx`
- `valid.data` / `valid.idx`
- `test.data` / `test.idx`
- `*_lengths.npy`
- `*_spk_ids.npy`

It also stores the rhythm cache fields that the v3 path consumes, such as:

- `content_units`
- `dur_anchor_src`
- `source_silence_mask`
- `sealed_mask`
- `boundary_confidence`
- `source_run_stability`
- `source_boundary_cue`
- `phrase_group_pos`
- `phrase_final_mask`

### 5.5 Train

```bash
py -3 tasks\run.py ^
  --config egs\local_rhythm_v3.yaml ^
  --exp_name local_rhythm_v3 ^
  --reset
```

This is the main maintained training entrypoint.

Checkpoint/log outputs go to:

- `checkpoints/local_rhythm_v3/config.yaml`
- `checkpoints/local_rhythm_v3/model_ckpt_steps_*.ckpt`
- `checkpoints/local_rhythm_v3/model_ckpt_best.pt`
- `checkpoints/local_rhythm_v3/tb_logs/`
- `checkpoints/local_rhythm_v3/terminal_logs/`

### 5.6 Resume training

If you rerun the same `exp_name`, the trainer auto-loads the last checkpoint in `checkpoints/<exp_name>/`.

```bash
py -3 tasks\run.py ^
  --config egs\local_rhythm_v3.yaml ^
  --exp_name local_rhythm_v3
```

### 5.7 Run task-level test/infer mode

`tasks/run.py --infer` runs the task's **test/evaluation path**, not the standalone streaming VC CLI.

```bash
py -3 tasks\run.py ^
  --config egs\local_rhythm_v3.yaml ^
  --exp_name local_rhythm_v3 ^
  --infer
```

For streaming VC runtime helpers, see `inference/README.md`.

---

## 6. Important config behavior: `--reset` matters

This is one of the most important gotchas in the repo.

`utils.commons.hparams.set_hparams()` will reuse:

- `checkpoints/<exp_name>/config.yaml`

and that saved config overrides the newly provided yaml unless you pass:

- `--reset`

So if you change your yaml and the change appears to be ignored, the first thing to check is whether you forgot `--reset`.

Also note:

- `--reset` refreshes the saved config snapshot
- it does **not** delete old checkpoints
- to start completely clean, use a new `exp_name` or `--remove`

---

## 7. When you must re-binarize

Re-binarize if you change anything that affects the cached source/target rhythm interface, especially:

- `rhythm_v3_emit_silence_runs`
- lattice stabilizer / debounce behavior
- `rhythm_tail_open_units`
- source-boundary / phrase sidecar generation
- anchor semantics that change `dur_anchor_src`

Why: the binary cache already stores rhythm-side fields derived from the lattice. If you only change code/config but keep old binary caches, training will still consume the old contract.

---

## 8. Common failure modes

### 8.1 "My config change does nothing"

Usually: same `exp_name`, missing `--reset`.

### 8.2 "Binarization says speaker is missing"

Check:

- `spker_set.json`
- `item_name`
- speaker prefix before the first underscore

They must agree.

### 8.3 "Training requires paired duration_v3 targets"

Usually: you forgot `rhythm_pair_manifest_path`, or your pair manifest did not resolve valid source/ref/target items after filtering.

### 8.4 "Explicit silence-run frontend requires source_silence_mask"

Usually: you enabled `rhythm_v3_emit_silence_runs: true` but are still using old cached binaries. Re-binarize.

### 8.5 "Missing f0 file"

With current Conan defaults, the binarizer expects `<wav_dir>_f0/<name>_f0.npy`. Generate F0 first or disable `with_f0` in a controlled experiment.

### 8.6 `--validate` is not a standalone eval-only mode

In the current codebase, `--infer` is the explicit task test/eval mode. `--validate` is not wired as a separate validate-only workflow.

---

## 9. Recommended smoke checks before a long run

### 9.1 Run targeted rhythm tests

```bash
py -3 -m pytest -q ^
  tests/rhythm/test_rhythm_v3_losses.py ^
  tests/rhythm/test_rhythm_v3_runtime.py ^
  tests/rhythm/test_cache_contracts.py ^
  tests/rhythm/test_task_config_v3.py
```

### 9.2 Do a short training smoke run

```bash
py -3 tasks\run.py ^
  --config egs\local_rhythm_v3.yaml ^
  --exp_name smoke_rhythm_v3 ^
  --reset ^
  -hp "max_updates=1000,val_check_interval=200,max_tokens=8000,max_sentences=4"
```

This is the fastest way to catch:

- pair-manifest mistakes
- missing cached fields
- dataset shape drift
- checkpoint/config path issues

### 9.3 Export one falsification bundle before scaling up

Once a smoke run or eval pass can emit debug bundles, prefer exporting the
review surface before starting a larger training schedule:

```bash
py -3 scripts\rhythm_v3_debug_records.py ^
  --input artifacts\rhythm_debug.pt ^
  --output artifacts\rhythm_v3_summary.csv ^
  --review-dir artifacts\rhythm_v3_review ^
  --g-variant raw_median ^
  --drop-edge-runs 1
```

This keeps the current workflow aligned with the falsification-first order:

1. static `g` audit
2. analytic monotonicity
3. coarse-only sanity
4. learned sanity
5. prefix fine-tune

If you want the narrower per-gate tables directly, read them from the
`--review-dir` bundle that the same command writes:

- `gate_ref_crop_table.csv`
- `gate_monotonicity_table.csv`
- `gate_prefix_silence_table.csv`
- `gate_mode_ladder_table.csv`

This keeps the default CLI surface to one maintained export command instead of
multiple gate-specific wrappers.

For the strongest Gate-0 / cross-text readout, prefer a debug bundle exported
from the maintained train/eval path with pair metadata still attached
(`pair_id`, prompt ids, same-text flags, `lexical_mismatch`, `ref_len_sec`,
`speech_ratio`). Pure inference bundles can still be summarized, but some slice
plots will only provide partial evidence.

---

## 10. One-line training recipe

If you only want the shortest correct sequence:

1. prepare `metadata.json` + `spker_set.json`
2. prepare `<wav_dir>_f0/*.npy` if `with_f0: true`
3. create a derived yaml from `egs/conan_emformer_rhythm_v3.yaml`
4. set `processed_data_dir`, `binary_data_dir`, `metafile_path`, and usually `rhythm_pair_manifest_path`
5. binarize
6. train with `tasks/run.py --config ... --exp_name ... --reset`
7. re-binarize whenever the lattice/cache contract changes

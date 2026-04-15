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
For the latest local Gate rerun and current stop/go verdict, see
`docs/rhythm_v3_local_status_2026-04-15.md`.

---

## 0. Current local status

Latest local rerun date:

- `2026-04-15`

Current verdict:

- latest local strongest-contract Gate 0: pass
- latest local strongest-contract Gate 1: pass
- Gate2-online local `src_gap` candidate: rerun and reviewed, no material aggregate improvement
- Gate3 local candidate: wiring-fixed and locally advanced to a checkpointed run
- official training: blocked
- prefix fine-tune: blocked

The current repo now carries two distinct gate truths and they must not be
collapsed into one:

- official training gate:
  `egs/overrides/rhythm_v3_gate_status.json`
  This remains blocked on the maintained online contract
  `weighted_median + ema + first_speech`.
- latest local strongest-contract candidate:
  `egs/overrides/rhythm_v3_gate_status_local_candidate_20260414.json`
  This records a local zero-train `Gate0/1` pass for
  `weighted_median + exact_global_family + target_as_ref`.
- latest local online-training candidate:
  `tmp/gate2_candidate_20260415_s75_srcgap/review/gate_status.json`
  plus the local Gate3 work directory
  `checkpoints/rhythm_v3_gate3_candidate_20260415_s126_srcgapfix1/`
  These are local candidate diagnostics only; they are **not** official
  unblock artifacts.

Official training is still blocked because:

- Gate 1-online was not re-recorded under the official gate JSON on the
  maintained online contract
- the local Gate2-online `src_gap` candidate did not materially improve the
  aggregate gate picture
- local Gate3 candidate work has not been converted into an official reviewed
  Gate3 pass
- `exact_global_family` is currently an upper-bound validation contract, not
  the maintained strict online runtime default
- the dominant failure signature still looks execution-side:
  `preproj` can show signal while `exec` often collapses to ties/flat buckets

So the right reading is:

- the old local `Gate0/1 fail` story is no longer current
- the newest upper-bound exact-family candidate is alive
- local online candidates have advanced the diagnosis, but not cleared the
  official training gate
- official online training still stays blocked until the maintained online
  contract is rerun through Gate 1-online, Gate 2, and Gate 3 under the same
  fingerprint

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
- `rhythm_v3_prompt_domain_mode: meaningful_reference`
- `rhythm_v3_prompt_require_clean_support: false`
- `rhythm_v3_g_variant: weighted_median`
- `rhythm_v3_prompt_g_variant: weighted_median`
- `rhythm_v3_src_g_variant: weighted_median`
- `rhythm_v3_prompt_g_drop_edge_runs: 0`
- `rhythm_v3_src_g_drop_edge_runs: 1`
- `rhythm_v3_min_boundary_confidence_for_g: 0.5`
- `rhythm_v3_src_prefix_stat_mode: ema`
- `rhythm_v3_src_rate_init_mode: first_speech`
- `rhythm_v3_min_prompt_ref_len_sec: 3.0`
- `rhythm_v3_max_prompt_ref_len_sec: 8.0`
- `rhythm_v3_require_same_text_paired_target: true`
- `rhythm_v3_use_continuous_alignment: true`
- `rhythm_v3_alignment_mode: continuous_viterbi_v1`
- `rhythm_v3_detach_global_term_in_local_head: true`
- `rhythm_v3_freeze_src_rate_init: true`
- `rhythm_v3_gate_quality_strict: true`
- `rhythm_v3_required_gate_status_json: egs/overrides/rhythm_v3_gate_status.json`
- `rhythm_v3_strict_eval_invalid_g: true`
- `rhythm_v3_eval_mode: learned`
- duration-weighted speech-ratio gating end-to-end for prompt-domain checks
- prompt-side weighted global-rate estimation with a broader
  `meaningful_reference` policy by default
- strict 3-8s prompt-domain claims remain available as
  `rhythm_v3_prompt_domain_mode=minimal_strict`, but they are no longer the
  maintained online mainline assumption
- speech = coarse + local
- silence = clipped coarse-only follower
- carry rounding + prefix budget at execution time

The intended Layer-0 lattice contract is also conservative: before retiming,
the shared train/infer stabilizer should suppress short flicker bridges and
short fake silence islands when boundary evidence is weak, instead of asking
the duration writer to absorb that noise.

So this branch is **not** training a free-form pause planner. It is training a **source-anchored causal retimer** over a stabilized run lattice.
The maintained runtime default is therefore
`weighted_median + ema + first_speech`; the falsification ladder should only
override `eval_mode` on that same online surface unless you are explicitly
running the stronger `Gate1-upper` exact-family check.
For local quick experimentation in this workspace, the checked-in
`egs/local_arctic_rhythm_v3_quick.yaml` now reads dataset I/O from
`D:/conan_data/...` instead of the earlier `C:/project/...` paths. That is a
local quick-config operational change only; it does not redefine the repo-wide
default data layout.
Minimal-V1 training now also fail-fast checks the prompt-side domain sidecars
(`prompt_speech_mask`, `prompt_closed_mask`, `prompt_boundary_confidence`,
`prompt_ref_len_sec`) before entering the model forward path.

### 1.1 Recommended falsification profiles

The maintained yaml already exposes the switches needed for staged validation.
In practice, keep one maintained online config and move between these profiles
with small overrides:

- maintained online contract:
  `egs/overrides/rhythm_v3_online_weighted_streaming.yaml`
- stronger upper-bound validation contract:
  `egs/overrides/rhythm_v3_local_weighted_exact.yaml`

| Profile | Purpose | Key overrides |
| --- | --- | --- |
| `A-online` | Gate1-online on the maintained runtime contract | `egs/overrides/rhythm_v3_gate1_analytic.yaml` |
| `A-upper` | Gate1-upper on the stronger exact-family validation contract | `egs/overrides/rhythm_v3_gate1_upper_exact.yaml` |
| `B` | coarse-only validation training | `rhythm_v3_eval_mode=coarse_only`, `rhythm_v3_disable_local_residual=true`, `rhythm_v3_disable_coarse_bias=false`, `lambda_rhythm_bias=0.20`, `lambda_rhythm_pref=0.0`, `lambda_rhythm_cons=0.0` |
| `C` | Gate 3 local residual falsification | `rhythm_v3_eval_mode=learned`, `rhythm_v3_disable_local_residual=false`, `rhythm_v3_disable_coarse_bias=false`, `rhythm_v3_detach_global_term_in_local_head=true`, `lambda_rhythm_pref=0.0`, `lambda_rhythm_cons=0.0` |
| `D` | no-detach ablation | same as `C`, but `rhythm_v3_detach_global_term_in_local_head=false`; treat this as an explicit comparison run rather than the maintained minimal-V1 contract |
| `E` | strict-causal prefix fine-tune | `rhythm_v3_eval_mode=learned`, `rhythm_v3_detach_global_term_in_local_head=true`, `lambda_rhythm_pref=0.05`, `lambda_rhythm_cons=0.05`, `rhythm_v3_silence_coarse_weight=0.0` |

Keep `weighted_median` as the checked-in maintained online baseline, but do not
confuse the maintained online contract with the stronger exact-family upper
bound. Current local zero-train evidence now points to
`weighted_median + exact_global_family + target_as_ref` as the strongest local
upper-bound validation contract, while the maintained runtime default is now
`weighted_median + ema + first_speech`. `trimmed_mean` remains a nearby local
comparator rather than the promoted winner. `softclean_wmed` /
`softclean_wtmean` still remain diagnostic-only because they are not the
promoted contract. The
maintained strict-claim contract forbids
`rhythm_v3_g_variant=unit_norm`, so keep
`rhythm_v3_minimal_v1_profile=true` but set
`rhythm_v3_strict_minimal_claim_profile=false` before running any `unit_norm`
overlay checks, even when you already have a reproducible prior bundle.
For `unit_norm`, also wire a reproducible prior bundle through
`rhythm_v3_unit_prior_path`; the maintained repo now ships
`scripts/build_unit_log_prior.py` so `unit_norm` is no longer just a consumer
interface with no official producer. The maintained prior bundle now carries
policy + frontend provenance metadata in addition to the prior values
themselves.

One practical note: the maintained config file still ships with
`rhythm_v3_eval_mode=learned` because that is the runtime default surface. For
actual falsification work, do not start there. Override into profile `A`
first, then `B`, then `C`. Only use `D` if you are explicitly testing whether
local residual quality depends on letting the local head backprop through the
global/coarse term.

One more local-candidate note matters for Gate3: under
`rhythm_v3_strict_minimal_claim_profile=false`, local candidate work may set
`rhythm_v3_disable_learned_gate=false` to reopen the learned-path training
surface, but minimal-V1 runtime should still **not** reinterpret that as
`use_learned_residual_gate=true`. The recent Gate3 unblock in this workspace
was a config/runtime wiring fix along that line, not a claim that Gate3 has
passed.

Current local evidence also says `src_gap` is not the dominant bottleneck by
itself. The next debugging priority is projector/clipping/budget/headroom and
exec flattening, not simply increasing model freedom.

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
public conditioning contract rather than an internal optional hint. The
maintained prompt policy is now `meaningful_reference`: prompt references still
need usable speech support, but the old 3-8s strict slice is no longer the
default online claim. That stricter slice survives as
`rhythm_v3_prompt_domain_mode=minimal_strict` for claim-tight validation.
`prompt_closed_mask`, `prompt_boundary_confidence`, and `prompt_ref_len_sec`
are still exported and audited, but the maintained prompt path now tolerates a
broader reference set than the old fail-closed clean-support contract. By
contrast, the source side stays stricter because online chunk continuation must
remain state-sufficient. The generic summary-pooling path still keeps a lenient
all-silence zero-summary fallback for diagnostics. The maintained mainline also
keeps `rhythm_prompt_dropout=0.0` and `rhythm_prompt_truncation=0.0`; if you
want prompt augmentation, treat it as an explicit ablation.

### 2.3 Canonical modules vs compatibility shims

For the maintained minimal-V1 surface, keep these file roles explicit:

- canonical prompt/global cue path:
  `modules/Conan/rhythm_v3/global_condition.py`
- canonical minimal writer path:
  `modules/Conan/rhythm_v3/minimal_writer.py`
- canonical causal run encoder:
  `modules/Conan/rhythm_v3/run_encoder.py`
- canonical runtime assembly / projector wiring:
  `modules/Conan/rhythm_v3/module.py`,
  `modules/Conan/rhythm_v3/projector.py`,
  `modules/Conan/rhythm_v3/runtime_adapter.py`
- generic / legacy prompt-summary container:
  `modules/Conan/rhythm_v3/summary_memory.py`
- compatibility-only shims that should not become the new mainline surface:
  `modules/Conan/rhythm_v3/minimal_head.py`,
  `modules/Conan/rhythm_v3/role_memory.py`

That split matters for maintenance. New minimal-V1 logic should land in the
canonical files above rather than drifting back into the legacy wrapper names
unless the code is truly shared by both surfaces.

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

When that target comes from the maintained paired-projection path,
`unit_duration_proj_raw_tgt` is the preferred explicit alias for the raw
projection surface. Training still decomposes supervision into
`global_shift_tgt`, `coarse_logstretch_tgt`, `local_residual_tgt`, and the
clipped coarse-derived silence target instead of treating the raw projection
surface as the final supervision object.

If you cache `unit_duration_tgt` for the maintained continuous path, also cache
`unit_duration_proj_raw_tgt`, `unit_alignment_mode_id_tgt`,
`unit_alignment_kind_tgt`, `unit_alignment_source_tgt`, and
`unit_alignment_version_tgt`, plus
`unit_alignment_source_cache_signature_tgt`,
`unit_alignment_target_cache_signature_tgt`, and
`unit_alignment_sidecar_signature_tgt` so later review/export can distinguish
`continuous_precomputed` from `continuous_viterbi_v1` instead of treating the
sample as opaque legacy supervision.

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
- when same-text reference is disallowed, source/reference items must also carry comparable text signatures so the runtime can prove they are different-text instead of assuming it
- pair-manifest grouping can be used for grouped batching and controlled A/B comparisons

For the maintained `rhythm_v3_minimal_v1_profile`, treat that as a hard data
contract:

- **reference prompt**: same-speaker / different-text
- **paired target supervision**: same-text projection target (`rhythm_v3_require_same_text_paired_target: true`), unless `unit_duration_tgt` is already explicitly cached in the sample

`rhythm_v3_use_continuous_alignment: true` now supports
`continuous_precomputed` and the built-in offline `continuous_viterbi_v1`
source-run / target-frame aligner. If neither explicit continuous provenance
nor the required frame-state sidecars are attached to the paired target, the
maintained dataset path fails fast instead of silently falling back to discrete
projection.

Likewise, treat gate completeness as a contract rather than a plotting detail:

- missing `analytic`, `coarse_only`, or `learned` means the falsification
  ladder is incomplete
- missing complete `slow / mid / fast` triplets means Gate 1 is incomplete
- missing `source_only` / `random_ref` / `shuffled_ref` style controls means
  you only have correlation evidence, not a strong control claim
- a bundle only counts as a control result when real-reference rows beat the
  negative controls on monotonicity / transfer metrics; mere presence is not
  enough
- large positive `analytic_same_text_gap` is now a Gate 1 failure, not just a
  review note, because it means same-text is outperforming cross-text by an
  unsafe margin

When the contract list has passed for Gates 0–2, run the optional
`egs/overrides/rhythm_v3_gate3_learned.yaml` overlay (or the alias
`egs/overrides/rhythm_v3_gate3_local_residual.yaml`) so Gate 3 can falsify the
learned local residual on the same maintained `rhythm_v3_eval_mode=learned`
surface. This stage keeps the coarse path intact, re-enables the learned path
(`rhythm_v3_disable_learned_gate=false`), and asks whether local residuals add
speech-side value without stealing coarse control or leaking into silence before
you move on to prefix fine-tuning.

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
  tests/rhythm/test_continuous_alignment_projection.py ^
  tests/rhythm/test_rhythm_v3_losses.py ^
  tests/rhythm/test_rhythm_v3_runtime.py ^
  tests/rhythm/test_cache_contracts.py ^
  tests/rhythm/test_task_config_v3.py
```

### 9.2 Do a short training smoke run

The old standalone `scripts/smoke_test_rhythm_v3.py` wrapper has been retired.
For the maintained path, this short task-level run plus the targeted entrypoint
tests above are now the supported structural smoke checks.

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
  --g-variant weighted_median ^
  --drop-edge-runs 1
```

`--review-dir` now implies strict gate enforcement in the maintained path. Use
`--allow-partial-gates` only when exporting an explicitly partial audit bundle.
The maintained training config also treats strict gate as a startup contract:
`rhythm_v3_gate_quality_strict=true` now requires
`rhythm_v3_required_gate_status_json` and refuses training when the recorded
gate bundle does not pass. The strict gate now also fails on low
`alignment_mean_local_confidence_speech`,
`alignment_mean_coarse_confidence_speech`, and low
`alignment_local_margin_p10`, so the startup contract and the review script use
the same quality floor.

The checked-in base config intentionally points at the official blocked gate:

- `egs/overrides/rhythm_v3_gate_status.json`

Use the local candidate JSON only as a machine-readable summary of the latest
zero-train strongest-contract evidence. It is not the official training
unblock artifact. The base config now also requires `gate2_pass=true` before
official learned training can start, so refreshing Gate 0 / Gate 1 alone is
not enough to unblock training.

This keeps the current workflow aligned with the falsification-first order:

1. static `g` audit
2. analytic monotonicity
3. coarse-only falsification
4. local residual falsification
5. prefix fine-tune

Stop the run and reconsider the statistic/interface if any of these show up
early on native-native sanity:

- `delta_g` barely explains `c_star`
- same-text looks good but cross-text collapses
- analytic mode cannot produce slow < mid < fast control
- coarse-only only helps by sharply worsening silence leakage or prefix drift

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
When reading prefix/runtime audit, keep `projector_boundary_hit_rate` separate
from `projector_boundary_decay_rate`: the first counts boundary events, the
second counts only the subset where decay was actually applied. Also read the
new projector telemetry as the companion pre/post surface:

- `projector_preclamp_exec`: continuous proposal before rounding/budget clamp
- `projector_clamp_mass`: absolute correction applied by the projector after
  rounded preclamp execution
- `projector_rounding_regret`: gap between committed discrete execution and the
  continuous proposal

### 9.4 Closure checkpoints before calling the branch stable

Treat these as the practical implementation checkpoints for the maintained
minimal-V1 line:

1. contract checkpoint:
   prompt/reference metadata proves a meaningful speech-dominant reference,
   paired-target provenance stays continuous, and same-text /
   different-text rules are explicit rather than implicit; use
   `minimal_strict` only when you explicitly need the older 3-8s claim slice
2. writer checkpoint:
   `rhythm_v3_detach_global_term_in_local_head=true` remains the maintained
   default, `rhythm_v3_freeze_src_rate_init=true` remains the maintained
   default, frozen `src_rate_init` is stored as module state rather than a
   trainable parameter, and `learned + no_detach` stays an ablation
3. `g` checkpoint:
   runtime and audit both read the same speech-only / closed / boundary-clean
   support semantics instead of separate debug-only heuristics
4. projector checkpoint:
   pre/post execution telemetry is exported, so improvements can be attributed
   to the writer or the projector instead of being conflated
5. gate checkpoint:
   Gate 0, Gate 1, Gate 2, and Gate 3 all run on the same maintained bundle, and
   negative controls must lose rather than merely exist
6. test harness checkpoint:
   rhythm tests run under `pytest`, and repo-local temporary paths are used for
   file-writing tests on Windows-restricted hosts to avoid flaky tmpdir
   failures

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

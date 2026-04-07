# Rhythm Migration Plan (2026-04-07)

This remains the maintained migration/audit note under `docs/`. Together with `README.md` and `docs/autodl_training_handoff.md`, it defines the current training path, the April 4-5 audit outcome, and the cloud-launch handoff.

## 1. Maintained path and code boundaries

AutoDL handoff note: if you are about to launch the real cloud run, read `docs/autodl_training_handoff.md` first; this file stays focused on migration status and audit conclusions.

The maintained rhythm path remains:

1. `teacher_offline`
2. export teacher targets / rebuild student cache
3. `student_kd`
4. optional upper-bound extension: `student_ref_bootstrap`
5. prepare retimed cache + F0 side files
6. `student_retimed`

The default maintained claims in this checkout still refer to the shorter
`teacher_offline -> student_kd -> student_retimed` chain. On April 6, 2026,
the recommended practical upper-bound path became:

1. `teacher_offline`
2. export teacher targets / rebuild student cache
3. `student_kd`
4. `student_ref_bootstrap`
5. `student_retimed`

That stage-2.5 branch is experimental, but it fixes the most important
remaining train/infer mismatch in the current repo: the acoustic path already
uses external same-speaker references, while the formal rhythm supervision had
still been largely self-conditioned.

Critical code ownership is intentionally narrow:

- timing / feasibility / projector logic: `modules/Conan/rhythm/`
- stage contracts, targets, losses, metrics, preflight: `tasks/Conan/rhythm/`
- maintained validation entrypoints:
  - `scripts/smoke_test_rhythm_v2.py`
  - `scripts/preflight_rhythm_v2.py`
  - `scripts/cpu_probe_rhythm_train.py`
  - `scripts/integration_teacher_export_student_kd.py`
  - `scripts/plot_rhythm_diagnostics.py`

The new descriptor plotting utility is deliberately aligned with the maintained
cache contract instead of a simplified external view:

- reads or derives from `ref_rhythm_stats[6]` / `ref_rhythm_trace[5]`
- exposes the compact planner-facing slice
  - `global_rate`
  - `pause_ratio`
  - `local_rate_trace`
  - `boundary_trace`
- planner-facing boundary semantics:
  - `boundary_trace` is now the **soft boundary-strength trace**
  - `boundary_ratio` remains the **binary event-rate stat**
- supports direct audio, feature bundles, and `descriptors.json::sample_id=...`
  style bundle selectors
- for audio or mel-backed inputs, also renders raw repo-aligned proxy signals:
  - `energy_z`
  - `delta`
  - `pause_mask`
  - `voiced`
  - `local_rate_raw`
  - `boundary_strength`
  - `boundary_events`
- bundle-only inspections can keep it light with `--disable_audio_backfill`
  when raw time-axis reconstruction is unnecessary
- corpus review can now emit a richer human-inspection pack:
  - `*_global_dashboard.png`
  - `*_progress_cards.png`
  - regenerated per-sample panels via `--single_output_dir`
  - copied review audio via `--export_audio_dir`

This should be treated as a **qualitative review surface**, not as the final
paper-grade sufficiency proof for the descriptor or source-compression design.

## 2. What the 6-agent training-prep audit actually covered

The April 4-5 audit was split across six concrete surfaces:

- parameter / contract defaults
- inference / runtime edge cases
- training losses / gradients / probe behavior
- data / cache / preflight assets
- docs / stage guidance
- optimizer scope / test coverage

That audit was used to update code, not just write commentary.

## 3. High-value fixes already absorbed into the local mainline

The maintained branch now includes these corrective changes:

- explicit zero-confidence stays hard-off for component KD and retimed confidence weighting
- factor intervention syncs compact edits back into `ref_rhythm_stats` / `ref_rhythm_trace` and drops stale sidecars
- budget supervision and feasibility-debt accounting are cleaner under repair-heavy cases
- online retimed acoustic targets now inherit sample confidence and projector-repair gating instead of being implicitly trusted at weight `1.0`
- online retimed acoustic targets now also inherit local trace-reliability gating, so exhausted / replayed prompt tails are downweighted instead of being trusted like clean local evidence
- module-only stages now keep train/valid objectives aligned instead of reintroducing acoustic/pitch losses only during validation
- `teacher_offline` validation now uses the real offline-teacher branch instead of silently falling back to the student/runtime path
- `teacher_offline` preflight dry-run now distinguishes:
  - `teacher_only` / module-only runtime, where `mel_out` is optional
  - `teacher_as_main` runtime, where `mel_out` is still required for audition / export semantics
- preflight orchestration and rhythm diagnostics are split into smaller units instead of monoliths
- preflight can now escalate missing or placeholder processed paths with `--strict_processed_data_dir`
- the local deprecated TorchScript helper in `modules/Conan/diff/net.py` is gone
- the acoustic runtime no longer touches pitch-embed branches when `use_pitch_embed=false`
- flow components in `modules/Conan/Conan.py` now lazy-load `torchdyn`, so non-flow rhythm checks do not fail at import time
- RMVPE / torchaudio loading is more lazy, so optional dependency failures happen only on the paths that really need them
- rhythm-only imports no longer require the English text frontend or tensorboard at import time
- inference helpers tolerate `f0_denorm_pred=None` instead of indexing into a missing tensor
- Conan / Emformer export paths also tolerate missing source or predicted F0 instead of indexing blindly
- weighted retimed acoustic losses now normalize by the full broadcasted weight mass instead of frame count only
- retimed blank-frame inference no longer copies a single argmin mel frame:
  - pause fill now uses a small bottom-k low-energy pool average, which is less likely to leak voiced color into cached / online retimed targets
- retimed mel supervision now disables pitch loss when matched `retimed_f0_tgt` / `retimed_uv_tgt` are missing, unless source-axis fallback is explicitly opted in
- retimed mel / weight alignment now keeps `retimed_f0_tgt`, `retimed_uv_tgt`, and `tgt_nonpadding` on the same time axis as `mel_out`, with linear resize for `retimed_f0_tgt`, nearest / binary-preserving resize for `retimed_uv_tgt`, and trim-mode parity with the aligned mel target
- integration export now covers `train/valid/test` by default for the teacher->student KD smoke chain
- stage-validation defaults now match runtime defaults for:
  - `rhythm_enable_learned_offline_teacher`
  - `rhythm_disable_pitch_loss_when_retimed`
- frame-plan construction now uses masked sum-preserving rounding for speech slots, blank slots, and `dur_anchor_src`, so integerization no longer leaks or drops total frame mass on fractional edge cases
- projector outer hot paths are more vectorized in:
  - `_allocate_pause_budget`
  - `_project_pause_impl`
  - `_compute_commit_frontier`
  - `_advance_state`
- strict-mainline projector defaults now match the real maintained intent:
  - pause selection defaults to `sparse`, so `pause_topk_ratio` annealing really reaches the sparse projector path
  - `use_boundary_commit_guard` stays enabled by default
  - `build_render_plan` stays enabled by default
- stage-specific optimizer scope no longer silently falls back to full-model training:
  - empty collectors now fail fast
  - post-freeze zero-trainable states now fail fast
  - foreign / stale params from the wrong model instance now fail fast
- stage-1 pause-recall repair now has a more explicit ablation ladder instead of only "increase boundary bias":
  - `pause_recall_run_a_soft_teacher_selection`: isolate soft sparse selection on the active teacher path
  - `pause_recall_run_b_lower_boundary`: de-emphasize late planner/projector/loss-side boundary ownership
  - `pause_recall_run_c_event_threshold`: soften the pause-event definition
  - `pause_recall_next`: combine the recommended teacher-soft + lower-boundary + warmer-gate settings
  - `pause_recall_structural`: opt-in support/allocation split plus run-length / breath-debt pause features
- the structural pause-recall overlay adds explicit planner observability and loss surfacing for the new path:
  - planner heads: `pause_support_prob_unit`, `pause_allocation_weight_unit`
  - planner features: `pause_run_length_unit`, `pause_breath_debt_unit`
  - loss / reporting: `rhythm_pause_allocation`, `L_pause_allocation`
  - diagnostics: pre-vs-post projector recall, top-k support coverage, and FN boundary-quartile shares
- maintained stage configs now pin `load_ckpt_strict: false` so cross-stage warm-start behaves like partial restore instead of exact-architecture resume:
  - base Conan -> `teacher_offline`
  - `teacher_offline` -> `student_kd`
  - `student_kd` -> `student_retimed`
- non-strict checkpoint load now reports missing / unexpected key counts with a preview instead of only logging shape mismatches
- `content_lengths` is now propagated explicitly through the rhythm dataset collater, runtime forward kwargs, and streaming evaluation instead of quietly reusing `mel_lengths`
- teacher target export now accepts either `rhythm_teacher_as_main` or `rhythm_teacher_only_stage` runtime semantics, while still rejecting shadow `rhythm_offline_execution` on the export path
- the local LibriTTS metadata helper now supports repeated or comma-separated `--train_split` values, so `train-clean-100 + train-clean-360` can be built in one processed metadata pass
- a conservative group-level EMA rhythm-loss balancer now exists behind an explicit opt-in flag:
  - `rhythm_loss_balance_mode: ema_group`
  - maintained default stays `none`
- the maintained stage-3 config now ships a more conservative acoustic curriculum:
  - `rhythm_stage3_acoustic_weight_start: 0.10`
  - `rhythm_stage3_acoustic_ramp_steps: 20000`
  - stage-3 warmup is anchored to the **actual start of stage-3**, not the restored absolute `global_step`
  - bounded EMA group balancing is enabled in the maintained stage-3 yaml
- context-matched KD gating now exists as an opt-in stage-2 research path:
  - `rhythm_enable_distill_context_match`
  - `rhythm_distill_context_floor`
  - `rhythm_distill_context_power`
  - `rhythm_distill_context_open_run_penalty`
- new experimental configs were added without changing the maintained default chain:
  - `egs/conan_emformer_rhythm_v2_student_kd_context_match.yaml`
  - `egs/conan_emformer_rhythm_v2_student_kd_algorithmic_distill_ablation.yaml`
  - `egs/conan_emformer_rhythm_v2_student_retimed_balanced.yaml`
- a new experimental external-reference bootstrap stage is now available:
  - `egs/conan_emformer_rhythm_v2_student_pairwise_ref_runtime_teacher.yaml`
  - `egs/conan_emformer_rhythm_v2_student_ref_bootstrap.yaml`
- external-reference rhythm supervision is now protected against the most dangerous configuration mismatch:
  - `sample_ref` / external rhythm policy can no longer silently pair with `cached_only` or `prefer_cache`
  - that mismatch now raises a contract error instead of only warning
- runtime-only teacher stages no longer falsely require cached teacher fields during cache-contract / preflight inspection
- stage-2.5 can now fail fast when a same-speaker pool collapses to a singleton and would otherwise silently fall back to self reference:
  - `rhythm_require_external_reference: true`
- runtime batches and metrics now expose whether rhythm supervision really used an external reference:
  - sample field: `rhythm_reference_is_self`
  - metrics: `rhythm_metric_reference_self_rate`, `rhythm_metric_reference_external_rate`
- fixed one-to-many pair-manifest bootstrap is now supported:
  - `rhythm_pair_manifest_path`
  - `rhythm_pair_manifest_prefixes`
  - `rhythm_pair_manifest_group_batches`
  - grouped same-`A` entries stay contiguous in dataset ordering, so batch construction is much more likely to preserve the intended one-to-many bundle
- explicit `A|A` identity anchors can now coexist with the external-reference fail-fast guard:
  - `rhythm_allow_identity_pairs: true`
  - sample / metric field: `rhythm_pair_is_identity`
- stage-2.5 can now add supervision that explicitly punishes "ignore B" collapse instead of relying on algorithmic teacher imitation alone:
  - `lambda_rhythm_ref_descriptor_stats`
  - `lambda_rhythm_ref_descriptor_trace`
  - `lambda_rhythm_ref_group_contrastive`
- predicted descriptor boundary traces used by descriptor-consistency / bootstrap supervision are now execution-derived:
  - they use executed blank mass as a boundary proxy instead of replaying `planner.boundary_score_unit`
  - this avoids leaking reference/source conditioning directly into the predicted descriptor branch
- same-`A` group contrastive supervision is now gap-aware instead of pure thresholded:
  - descriptor pairs that are clearly farther apart in target space receive stronger contrastive pressure than near-threshold pairs
- descriptor-consistency is now implemented against the compact maintained reference contract:
  - `global_rate`
  - `pause_ratio`
  - `local_rate_trace`
  - `boundary_trace`
- planner-facing boundary traces are now soft rather than binary:
  - `ref_rhythm_trace[..., boundary_trace]` stores soft boundary strength
  - `ref_rhythm_stats[..., boundary_ratio]` still stores the binary event rate
  - if you exported descriptor bundles / sidecars before this change and rely on planner-facing traces, rebuild them
- short-reference / long-stream robustness is now implemented as an **opt-in runtime profile**, not a silent rewrite of the maintained utterance-bounded mainline:
  - keep the full/global teacher unchanged
  - runtime/student side adds explicit `trace_reliability` gating instead of value-level blending of local trace with slow summary
  - local trace sampling can be switched to anchor-aware spacing (`dur_anchor_src` aware) instead of flat unit-count spacing
  - when trace evidence is clearly exhausted, planner slow-memory retrieval is preferred before plain summary fallback, and phrase-final slow-memory cells are downweighted
  - config keys:
    - `rhythm_trace_reliability_enable`
    - `rhythm_trace_exhaustion_gap_start`
    - `rhythm_trace_exhaustion_gap_end`
    - `rhythm_trace_exhaustion_local_floor`
    - `rhythm_trace_exhaustion_boundary_floor`
    - `rhythm_trace_exhaustion_reuse_full_count`
    - `rhythm_trace_exhaustion_final_cell_suppress`
    - `rhythm_trace_anchor_aware_sampling`
  - helper override yaml: `egs/conan_emformer_rhythm_v2_long_stream_short_ref_overrides.yaml`
  - exception for the external-reference bootstrap path:
    - `student_pairwise_ref_runtime_teacher` / `student_ref_bootstrap` now enable
      `rhythm_emit_reference_sidecar`, `rhythm_trace_reliability_enable`, and
      `rhythm_trace_anchor_aware_sampling` by default because that regime is
      much more exposed to short-ref / longer-source mismatch than the
      utterance-bounded maintained chain

## 4. Validation actually run on this checkout

The following were run locally in the `conda` `conan` environment:

- `conda run -n conan python -m compileall -q modules tasks scripts tests utils data_gen`
- `conda run -n conan python -m unittest discover -s tests/rhythm -p "test_*.py"`
- `conda run -n conan python -u scripts/smoke_test_rhythm_v2.py`
- teacher-offline smoke preflight / dry-run on `train`
- teacher-offline smoke preflight / dry-run on `train valid` to confirm the known empty-split failure
- teacher export -> student KD integration smoke
- 1-step `student_retimed` CPU probe with default `use_pitch_embed=true`
- 1-step `student_retimed` CPU probe with `--hparams use_pitch_embed=False`
- 2-step `student_retimed` CPU probe after probe-observability wiring to confirm runtime summaries are exported
- 2000-step CPU probe for `teacher_offline`
- 2000-step CPU probe for `student_kd`
- 2000-step CPU smoke probe for `student_retimed`
- focused April 6, 2026 regression rerun for the new external-reference bootstrap path:
  - `tests/rhythm/test_cache_contracts.py`
  - `tests/rhythm/test_dataset_target_builder.py`
  - `tests/rhythm/test_policy_contract_and_loss_routing.py`
  - `tests/rhythm/test_stage_warmstart_defaults.py`
  - `tests/rhythm/test_reference_bootstrap_runtime.py`
  - `tests/rhythm/test_metrics_masking.py`
  - `tests/rhythm/test_reference_sidecar.py`
  - `tests/rhythm/test_pair_manifest_sampler.py`
  - `tests/rhythm/test_reference_regularization.py`
- focused April 7, 2026 regression rerun for the new stage-3 / projector /
  descriptor-contract fixes:
  - `tests/rhythm/test_factory_defaults.py`
  - `tests/rhythm/test_reference_encoder.py`
  - `tests/rhythm/test_stage_warmstart_defaults.py`
  - `tests/rhythm/test_task_runtime_support.py`
  - `tests/rhythm/test_trace_reliability_contract.py`
  - `tests/rhythm/test_weighted_acoustic_losses.py`
  - `tests/rhythm/test_reference_sidecar.py`
  - `tests/rhythm/test_reference_regularization.py`
  - `tests/rhythm/test_target_builder.py`
  - `tests/rhythm/test_projector_invariants.py`

Observed result summary:

- unit coverage: **243 rhythm tests passed**
- latest compile/unit/smoke rerun after the teacher-offline preflight fix, online-retimed confidence/repair gating, stage-freeze fail-fast guards, train-set contract hardening, and warm-start missing/unexpected-key observability: **passed**
- teacher-offline probe: healthy loss descent and low gradient pressure
- student-KD probe: healthy and fast on smoke student binary
- student-retimed smoke: structurally runnable, but still high-risk because `L_base` dominates and gradients are large / clip-heavy
- fresh stage-3 spot-check: default `student_retimed` probe fails immediately on missing `f0` in the smoke `student_binary`
- fresh stage-3 spot-check with `use_pitch_embed=False`: runs, but still shows large one-step gradient pressure (`grad_norm_before_clip ~= 274`)
- CPU probe now also exports runtime observability summaries for:
  - `rhythm_metric_disable_acoustic_train_path`
  - `rhythm_metric_module_only_objective`
  - `rhythm_metric_skip_acoustic_objective`
  - `rhythm_metric_pitch_supervision_disabled`
  - `rhythm_metric_missing_retimed_pitch_target`
  - pre-align retimed-target mismatch / resample / trim signals
- focused external-reference bootstrap regression rerun on April 6, 2026:
  - **57 tests passed**
  - validated stage-2.5 config defaults, contract hard-errors, pair-manifest expansion, identity-anchor allowance, descriptor regularization helpers, runtime fail-fast against accidental self fallback, reference self/external observability metrics, and trace-exhaustion fallback wiring
- focused stage-3 / projector / descriptor-contract regression rerun on April 7, 2026:
  - **63 tests passed**
  - validated strict-mainline projector defaults, stage-3 acoustic-ramp defaults, online retimed trace-gating, soft boundary-trace encoding, and the `_sample_trace_pair(...) -> forward()/forward_teacher()` contract

The newest rerun extended regression coverage specifically around:

- preserved-sum frame-plan rounding
- projector invariants after hot-path vectorization
- target-builder context-match gating / dedupe semantics
- conservative rhythm-loss balancing
- module-only train/valid objective alignment
- teacher-offline validation routing
- teacher-offline preflight semantics for `teacher_only` vs `teacher_as_main`
- optional dependency guards for text frontend / tensorboard imports
- optional dependency guards for `torchdyn` on non-flow rhythm paths
- weighted acoustic-loss normalization over broadcasted weight mass
- retimed pitch fallback suppression, missing-target detection, post-align length parity with `mel_out`, and the `attach_acoustic_target_bundle(...) -> add_pitch_loss(...)` no-mismatch handoff
- online retimed target confidence / repair gating
- stage-specific parameter freeze guards and foreign-param detection
- validation / smoke observability for skipped acoustic objectives and missing matched retimed pitch targets

Important asset-level caveats from the same audit:

- `data/binary/libritts_single_smoke_rhythm_v4` is train-only in practice:
  - `valid.data` is empty
  - `test.data` is empty
- the checked-in LibriTTS smoke binary is rhythm-cache **v4 compatibility smoke**, not maintained v5 training data
- the current teacher->student KD integration artifact under `artifacts/rhythm_teacher_export_student_kd/...` is smoke-only because its teacher ckpt mode is `bootstrap_random_init`
- the generated stage-2 smoke `student_binary` already contains learned-offline teacher + retimed targets, but it still does **not** include F0
- formal stage-3 still needs real F0 side files; the smoke probe used `--hparams use_pitch_embed=False` only to avoid claiming a false formal pass

Additional training-prep caution:

- `train_sets` is now better guarded, but it is still not the preferred first-formal-run path
- the safer baseline remains: merge raw data first, then preprocess / binarize once into a single `binary_data_dir`
- multi-binary concatenation is best treated as an advanced path, not the default maintained launch recipe
- if you do choose multi-binary training, preflight now checks extra train-side indexed sidecars plus shared JSON / known condition maps before the run starts

## 4.1 Training-prep audit notes that matter before you launch

- `scripts/preflight_rhythm_v2.py` is only a thin wrapper; the real validation lives in `tasks/Conan/rhythm/preflight_support.py`
- preflight is **binary-cache-first**, not a full processed-corpus validator:
  - it strongly checks indexed cache fields / contracts
  - by default it still only lightly checks `processed_data_dir`
  - for formal readiness runs, `--strict_processed_data_dir` now escalates missing or placeholder processed paths into hard errors
- `teacher_offline` preflight now intentionally treats runtime semantics, not just stage name, as the source of truth:
  - `teacher_only` / module-only dry-runs may omit `mel_out`
  - `teacher_as_main` dry-runs must still emit `mel_out`, because that path is also used for teacher audition / export
- `scripts/cpu_probe_rhythm_train.py` is a throughput / gradient probe, not a full cache-contract validator
- `scripts/integration_teacher_export_student_kd.py` exports `train/valid/test`, but the post-export smoke assertions are still centered on `train/valid`
- on the checked-in LibriTTS smoke corpus, integration split inference relies on the generated `build_summary.json`

## 5. Training-readiness conclusion by stage

### `teacher_offline`

- code path: **ready**
- smoke training path: **ready**
- formal run from this checkout: **blocked by real dataset assets**
- maintained semantics note:
  - the checked-in stage-1 config is a **cached-guidance bootstrap** for the
    offline teacher branch
  - it keeps `rhythm_dataset_target_mode: cached_only`,
    `rhythm_primary_target_surface: guidance`, and
    `rhythm_dataset_build_teacher_from_ref: false`
  - so the honest story is "offline teacher branch first learns on the cached
    guidance floor", not "stage-1 already makes the offline teacher directly
    learn from runtime algorithmic-teacher targets"
- research note:
  - a true runtime algorithmic-teacher-supervised offline-teacher stage remains
    a separate upper-bound / paper-facing ablation, not the maintained default
- recall-repair note:
  - if stage-1 recall remains low while `L_budget` / repair metrics stay healthy,
    prefer the shipped recall ladder above over "increase boundary bias again"
  - the structural overlay is intentionally opt-in and keeps non-strict
    warm-start because older checkpoints will not contain the new pause-support
    split heads / breath-feature projection parameters

### `student_kd`

- code path: **ready**
- teacher-export integration chain: **ready as smoke**
- formal run from this checkout: **blocked by missing real learned teacher export / rebuilt cache**
- experimental branch available: `conan_emformer_rhythm_v2_student_kd_context_match.yaml`
- extra ceiling-only ablation branch available:
  - `conan_emformer_rhythm_v2_student_kd_algorithmic_distill_ablation.yaml`
  - keeps the primary teacher unchanged but re-enables very small algorithmic budget/prefix KD weights from a different distill surface
- maintained KD reminder: the real maintained signal is `teacher-main + shape-only KD`; exec/budget/prefix/allocation KD stay disabled in the default stage-2 config

### `student_ref_bootstrap`

- code path: **ready as experimental stage-2.5**
- formal run from this checkout: **still blocked by real dataset assets**
- purpose:
  - switch rhythm supervision from self-conditioned cached surfaces to external same-speaker runtime teacher targets
  - reduce the train/infer mismatch before `student_retimed`
- important runtime rule:
  - default config sets `rhythm_require_external_reference: true`
  - if speaker filtering leaves only one item for a speaker, the run now fails fast instead of silently falling back to self
- one-to-many bootstrap rule:
  - if you want real pairwise bootstrap instead of random same-speaker refs, provide a fixed `rhythm_pair_manifest_path`
  - include `A|A` plus descriptor-diverse `A|B_*` entries in the manifest
  - keep `rhythm_pair_manifest_group_batches: true` so grouped batch construction sees the one-to-many bundle
- important monitoring rule:
  - track `rhythm_metric_reference_self_rate`
  - healthy formal stage-2.5 runs should keep self-rate near `0`
- short-ref / longer-source safety rule:
  - the checked-in bootstrap config now already enables:
    - `rhythm_emit_reference_sidecar`
    - `rhythm_trace_reliability_enable`
    - `rhythm_trace_anchor_aware_sampling`
  - so the default stage-2.5 path does not keep replaying a tiny prompt tail as if it were fresh local evidence
- anti-collapse rule:
  - stage-2.5 no longer has to rely only on `L_algo`
  - descriptor-consistency and same-`A` group contrastive loss are now available so the model is explicitly pushed to preserve `A` while still responding to `B`
  - current implementation note:
    - group contrastive now scales with target descriptor gap, so clearly different `B` references are emphasized more than near-threshold pairs
- identity-anchor monitoring rule:
  - `A|A` anchors remain allowed, but keep watching `rhythm_metric_pair_identity_rate`
  - if identity pairs become too frequent, they can dilute the external-reference signal even without any explicit loss bug

### `student_retimed`

- code path: **closer to ready**
- smoke path: **runs**
- formal run from this checkout: **not ready to bless**
- maintained stage-3 base now already absorbs the conservative acoustic-ramp /
  EMA-balance defaults that were previously only carried as an experimental
  branch:
  - `rhythm_stage3_acoustic_weight_start: 0.10`
  - `rhythm_stage3_acoustic_ramp_steps: 20000`
  - `rhythm_loss_balance_mode: ema_group`
  - explicit sparse / guarded / render-plan projector defaults
- `conan_emformer_rhythm_v2_student_retimed_balanced.yaml` remains as a
  compatibility alias / research handle, but it is no longer the only place
  carrying the conservative stage-3 balancer defaults
- an extra upper-bound A/B config now exists when you want to probe the small
  train/test gap between cached retimed targets and the current student
  execution:
  - `conan_emformer_rhythm_v2_student_retimed_hybrid_ablation.yaml`
  - keeps the maintained cached-first warmup, then switches to `hybrid`
    retimed targets after `rhythm_online_retimed_target_start_steps`

Why stage-3 is still guarded:

- smoke requires `use_pitch_embed=False`
- the smoke binary already has retimed targets, but default stage-3 still fails because it lacks F0
- retimed mel no longer silently falls back to source-axis pitch; missing matched retimed pitch now disables pitch supervision first, and aligned retimed pitch tracks are kept on the same time axis as `mel_out`
- maintained stage-3 now explicitly counters the worst early acoustic-domination pattern with:
  - a lower acoustic-loss start
  - a longer acoustic ramp
  - anchored warm-start ramp semantics
  - bounded EMA group balancing
  - sparse / guarded / render-plan projector defaults
- `L_base` dominates the current smoke objective
- `grad_norm_before_clip` stays very high for long stretches
- the real F0 / retimed asset contract is not present in this shared checkout

Concrete 2000-step smoke numbers from `artifacts/probe/student_retimed_cpu_probe_2000_smoke.json`:

- `L_base.mean ~= 6.5258`
- `L_rhythm_exec.mean ~= 0.00279`
- `L_stream_state.mean ~= 0.000746`
- `L_pitch.mean = 0.0` because that smoke probe had to force `use_pitch_embed=False`
- `grad_norm_before_clip.mean ~= 158.86`
- `grad_norm_before_clip.max ~= 275.92`

That is roughly:

- `L_base / L_rhythm_exec ~= 2342x`
- `L_base / L_stream_state ~= 8745x`

So the current stage-3 smoke result is not "mildly imbalanced"; it is still overwhelmingly acoustic-dominated.

The new probe-observability wiring was also validated on a fresh 2-step `student_retimed` CPU probe:

- `rhythm_metric_pitch_supervision_disabled = 1.0`
- `rhythm_metric_skip_acoustic_objective = 0.0`
- `rhythm_metric_acoustic_target_is_retimed = 1.0`

So the probe can now distinguish "pitch was intentionally disabled because matched retimed pitch is unavailable" from "the whole acoustic objective got skipped".

## 6. Remaining blockers before formal training

This shared checkout is still missing the real maintained training assets:

- `data/binary/vc_6layer/{train,valid}.{data,idx}`
- a real `data/processed/vc`
- exported `data/teacher_targets/...`
- dedicated formal retimed cache / side files for `student_retimed`
- F0 side files when `with_f0=true`

So the current state is:

- code readiness: **yes**
- smoke/probe readiness in `conda` env: **yes**
- formal maintained training readiness on this checkout: **no**

The main blocker is still assets, not the control-loss hot path.

For the experimental stage-2.5 branch, there is one extra blocker beyond the
usual dataset assets:

- same-speaker train splits must still contain at least two usable items per speaker after filtering, otherwise `rhythm_require_external_reference: true` will stop the run as intended

## 6.1 Gradient / control audit notes

- maintained optimization is still centered on the composite rhythm losses:
  - execution
  - budget
  - prefix-state / carry
- many logged subterms are reporting surfaces after scaling, not independent optimizer-driving objectives
- projector / frame-plan behavior is intentionally discrete:
  - rounding
  - top-k / thresholding
  - monotonic commit frontiers
  - detached prefix reuse
- so retimed acoustic targets do **not** backprop through frame-plan/control decisions; the explicit rhythm losses remain the control-learning path

## 7. Remaining engineering gaps / future directions after this audit

The audit found a few remaining gaps that were intentionally not over-corrected in the same round:

- optimizer-scope coverage is now guarded against empty / foreign collector output, but it still benefits from more direct end-to-end param-group tests under real checkpoints
- this repo still does not have a checked-in `teacher_pairwise_joint_upper_bound` config; that remains a next-stage research branch when you want to audition the real teacher main branch and establish an audible upper bound
- the real ceiling path is still a pairwise offline-cache / export path, not just runtime teacher synthesis:
  - `teacher(A | B)`
  - `retimed mel(A | B)`
  - `retimed f0/uv(A | B)`
- student-retimed needs a cleaner formal stage-3 integration smoke once real F0 assets exist
- `teacher_pairwise_refine` is still the recommended next ceiling step after runtime-only bootstrap:
  - export fixed `(A,B)` pairwise offline surfaces from an EMA bootstrap teacher
  - then shift supervision from pure algorithmic oracle imitation toward pair-cache + descriptor + ranking losses
- a full runtime-matched teacher export pipeline is still future work:
  - do not treat privileged full-teacher internals as the long-term student export truth
  - eventually export student-visible/runtime-shaped control targets explicitly
- some legacy / research-stage paths still exist outside the maintained mainline, even though the maintained docs now stop centering them
- projector still keeps the row-wise bounded-simplex core as the conservative speech-path solver; only the outer hot paths were vectorized in this round
- loss balancing and context-matched KD are deliberately opt-in research knobs, not new maintained defaults
- the current repo now has a much better qualitative review surface for
  descriptor plots and linked audio, but it still lacks three important
  evidence blocks if the goal is a stronger paper-level claim:
  1. source-compression sufficiency
     - beyond smoke / invariants, test whether the compressed unit sequence
       preserves enough information relative to the uncompressed token/source
       sequence for target-rhythm prediction
  2. descriptor causal sufficiency
     - intervention monotonicity:
       - `global_rate` should mainly move speech budget / pacing
       - `pause_ratio` should mainly move pause share
       - `boundary_trace` should mainly move pause / boundary placement
     - ablation specificity:
       - removing one factor should selectively damage the matching capability
         most tied to that factor
     - leakage analysis:
       - check whether a factor is silently carrying another factor's control
         burden
  3. descriptor-to-annotation correspondence
     - correlation with human pause / boundary labels
     - correlation with forced-alignment pause durations
     - agreement with speaking-rate annotation or rating

The current branch should therefore be described as having:

- stronger local diagnostics
- stronger qualitative review
- better future-ablation scaffolding

but **not yet** a complete causal / annotation-backed descriptor proof.

These are follow-up / next-stage tasks, not blockers for the current minimal
high-value landing. The current branch intentionally stops at the runtime-only
stage-2.5 bootstrap plus robustness fixes, instead of trying to land the whole
ceiling stack in one round.

## 8. Training rule of thumb

Keep maintenance effort focused on changes that improve one of these four things:

- projector trust / feasible execution
- cache reproducibility / fail-fast validation
- retimed supervision consistency
- streaming stability / prefix consistency

If a proposed change only exists in another branch's research path and does not have a real local analogue, do not port it by default.

For this checkout specifically, the current priority order is:

1. keep the maintained `teacher_offline -> student_kd -> student_retimed` path healthy
2. use `student_ref_bootstrap` when you want a practical upper-bound extension for external-reference rhythm learning
3. treat `teacher_pairwise_refine` as the next-stage ceiling step after runtime-only bootstrap, not as part of the current minimum rollout
4. add a future `teacher_pairwise_joint_upper_bound` branch only as a separate upper-bound proof path, not as a replacement for the clean teacher-surface stage
5. treat pairwise offline cache / export plus full runtime-matched teacher export as the real long-term ceiling project

from __future__ import annotations

import numpy as np
import torch

from modules.Conan.rhythm.policy import is_duration_operator_mode
from modules.Conan.rhythm.supervision import build_source_rhythm_cache
from tasks.Conan.rhythm.duration_v3.targets import build_pseudo_source_duration_context
from tasks.Conan.rhythm.duration_v3.task_config import is_duration_v3_prompt_summary_backbone


def _as_int64_1d(value) -> np.ndarray:
    return np.asarray(value, dtype=np.int64).reshape(-1)


def _as_float32_1d(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _resolve_run_silence_mask(*, size: int, silence_mask=None) -> np.ndarray:
    if silence_mask is None:
        return np.zeros((size,), dtype=np.float32)
    return _as_float32_1d(silence_mask)[:size]


def _expand_run_sequence(
    *,
    units: np.ndarray,
    durations: np.ndarray,
    silence_mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    token_ids: list[int] = []
    token_run_index: list[int] = []
    token_silence: list[int] = []
    for run_idx, (unit_id, duration, silence_flag) in enumerate(zip(units.tolist(), durations.tolist(), silence_mask.tolist())):
        if valid_mask is not None and (run_idx >= len(valid_mask) or float(valid_mask[run_idx]) <= 0.5):
            continue
        count = int(max(0, round(float(duration))))
        if count <= 0:
            continue
        token_ids.extend([int(unit_id)] * count)
        token_run_index.extend([int(run_idx)] * count)
        token_silence.extend([1 if float(silence_flag) > 0.5 else 0] * count)
    return (
        np.asarray(token_ids, dtype=np.int64),
        np.asarray(token_run_index, dtype=np.int64),
        np.asarray(token_silence, dtype=np.int64),
    )


def _align_target_tokens_to_source(
    *,
    source_tokens: np.ndarray,
    source_silence: np.ndarray,
    target_tokens: np.ndarray,
    target_silence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_source = int(source_tokens.shape[0])
    num_target = int(target_tokens.shape[0])
    if num_source <= 0 or num_target <= 0:
        raise RuntimeError("paired projection requires non-empty expanded source and target token streams.")

    local_cost = np.zeros((num_source, num_target), dtype=np.float32)
    gap_penalty = np.float32(0.10)
    for src_idx in range(num_source):
        src_token = int(source_tokens[src_idx])
        src_is_sil = bool(source_silence[src_idx] > 0)
        src_pos = float(src_idx + 1) / float(max(1, num_source))
        for tgt_idx in range(num_target):
            tgt_token = int(target_tokens[tgt_idx])
            tgt_is_sil = bool(target_silence[tgt_idx] > 0)
            tgt_pos = float(tgt_idx + 1) / float(max(1, num_target))
            if src_token == tgt_token and src_is_sil == tgt_is_sil:
                base = 0.0
            elif src_is_sil != tgt_is_sil:
                base = 1.60
            elif src_token == tgt_token:
                base = 0.15
            else:
                base = 0.90
            local_cost[src_idx, tgt_idx] = np.float32(base + (0.15 * abs(src_pos - tgt_pos)))

    dp = np.full((num_source + 1, num_target + 1), np.inf, dtype=np.float32)
    back = np.zeros((num_source + 1, num_target + 1), dtype=np.uint8)
    dp[0, 0] = 0.0
    for src_idx in range(1, num_source + 1):
        dp[src_idx, 0] = dp[src_idx - 1, 0] + 1.0
        back[src_idx, 0] = 1
    for tgt_idx in range(1, num_target + 1):
        dp[0, tgt_idx] = dp[0, tgt_idx - 1] + 1.0
        back[0, tgt_idx] = 2

    for src_idx in range(1, num_source + 1):
        for tgt_idx in range(1, num_target + 1):
            cost = local_cost[src_idx - 1, tgt_idx - 1]
            diag = dp[src_idx - 1, tgt_idx - 1]
            up = dp[src_idx - 1, tgt_idx] + gap_penalty
            left = dp[src_idx, tgt_idx - 1] + gap_penalty
            if diag <= up and diag <= left:
                dp[src_idx, tgt_idx] = cost + diag
                back[src_idx, tgt_idx] = 0
            elif up <= left:
                dp[src_idx, tgt_idx] = cost + up
                back[src_idx, tgt_idx] = 1
            else:
                dp[src_idx, tgt_idx] = cost + left
                back[src_idx, tgt_idx] = 2

    assigned_source = np.zeros((num_target,), dtype=np.int64)
    assigned_cost = np.zeros((num_target,), dtype=np.float32)
    src_idx = num_source
    tgt_idx = num_target
    while src_idx > 0 or tgt_idx > 0:
        move = int(back[src_idx, tgt_idx])
        if move == 0:
            assigned_source[tgt_idx - 1] = max(0, src_idx - 1)
            assigned_cost[tgt_idx - 1] = local_cost[src_idx - 1, tgt_idx - 1]
            src_idx -= 1
            tgt_idx -= 1
        elif move == 1:
            src_idx -= 1
        else:
            assigned_source[tgt_idx - 1] = max(0, src_idx - 1)
            assigned_cost[tgt_idx - 1] = local_cost[max(0, src_idx - 1), tgt_idx - 1]
            tgt_idx -= 1
    return assigned_source, assigned_cost


def _project_target_runs_onto_source(
    *,
    source_units: np.ndarray,
    source_durations: np.ndarray,
    source_silence_mask: np.ndarray,
    target_units: np.ndarray,
    target_durations: np.ndarray,
    target_valid_mask: np.ndarray,
    target_speech_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_silence_mask = _resolve_run_silence_mask(size=len(source_units), silence_mask=source_silence_mask)
    target_silence_mask = ((target_valid_mask > 0.5) & ~(target_speech_mask > 0.5)).astype(np.float32)
    src_tokens, src_token_to_run, src_token_silence = _expand_run_sequence(
        units=source_units,
        durations=source_durations,
        silence_mask=source_silence_mask,
    )
    tgt_tokens, _, tgt_token_silence = _expand_run_sequence(
        units=target_units,
        durations=target_durations,
        silence_mask=target_silence_mask,
        valid_mask=target_valid_mask,
    )
    if src_tokens.size <= 0:
        raise RuntimeError("paired projection requires a non-empty source run lattice.")
    if tgt_tokens.size <= 0:
        raise RuntimeError("paired projection requires a non-empty paired target prompt lattice.")

    assigned_source, assigned_cost = _align_target_tokens_to_source(
        source_tokens=src_tokens,
        source_silence=src_token_silence,
        target_tokens=tgt_tokens,
        target_silence=tgt_token_silence,
    )

    num_source_runs = int(source_units.shape[0])
    projected = np.zeros((num_source_runs,), dtype=np.float32)
    aligned_target = np.zeros((num_source_runs,), dtype=np.float32)
    exact_match = np.zeros((num_source_runs,), dtype=np.float32)
    cost_mass = np.zeros((num_source_runs,), dtype=np.float32)
    source_support = np.bincount(src_token_to_run, minlength=num_source_runs).astype(np.float32)

    for tgt_idx, src_token_idx in enumerate(assigned_source.tolist()):
        safe_src_token_idx = int(max(0, min(src_token_idx, int(src_tokens.shape[0]) - 1)))
        run_idx = int(src_token_to_run[safe_src_token_idx])
        projected[run_idx] += 1.0
        aligned_target[run_idx] += 1.0
        if int(src_tokens[safe_src_token_idx]) == int(tgt_tokens[tgt_idx]) and int(src_token_silence[safe_src_token_idx]) == int(tgt_token_silence[tgt_idx]):
            exact_match[run_idx] += 1.0
        cost_mass[run_idx] += float(assigned_cost[tgt_idx])

    coverage = np.divide(
        aligned_target,
        np.clip(source_support, 1.0, None),
        out=np.zeros_like(aligned_target),
        where=source_support > 0.0,
    )
    match_rate = np.divide(
        exact_match,
        np.clip(aligned_target, 1.0, None),
        out=np.zeros_like(exact_match),
        where=aligned_target > 0.0,
    )
    mean_cost = np.divide(
        cost_mass,
        np.clip(aligned_target, 1.0, None),
        out=np.zeros_like(cost_mass),
        where=aligned_target > 0.0,
    )
    confidence = np.clip(
        0.20 + (0.40 * coverage) + (0.25 * match_rate) + (0.15 * np.exp(-mean_cost)),
        0.05,
        1.0,
    ).astype(np.float32)

    source_is_silence = source_silence_mask > 0.5
    projected[source_is_silence] = source_durations[source_is_silence].astype(np.float32)
    confidence[source_is_silence] = np.maximum(confidence[source_is_silence], 1.0)

    speech_zero = (~source_is_silence) & (projected <= 0.0)
    projected[speech_zero] = 1.0
    confidence[speech_zero] = np.minimum(confidence[speech_zero], 0.10)
    return projected.astype(np.float32), confidence.astype(np.float32)


class DurationV3DatasetMixin:
    def _use_duration_v3_dataset_contract(self) -> bool:
        return bool(
            self.hparams.get("rhythm_enable_v3", False)
            or is_duration_operator_mode(self.hparams.get("rhythm_mode", ""))
        )

    def _should_emit_duration_v3_silence_runs(self) -> bool:
        if not self._use_duration_v3_dataset_contract():
            return False
        return bool(self.hparams.get("rhythm_v3_emit_silence_runs", True))

    def _should_perturb_duration_v3_source_context(self) -> bool:
        if not self._use_duration_v3_dataset_contract():
            return False
        if str(self.prefix).lower() != "train":
            return False
        if not is_duration_v3_prompt_summary_backbone(self.hparams.get("rhythm_v3_backbone", "global_only")):
            return False
        if str(self.hparams.get("rhythm_v3_anchor_mode", "baseline") or "baseline").strip().lower() != "source_observed":
            return False
        return self._is_enabled_flag(
            self.hparams.get(
                "pseudo_source_duration_perturbation",
                self.hparams.get("rhythm_pseudo_source_duration_perturbation", False),
            )
        )

    def _maybe_build_duration_v3_training_source_cache(self, source_cache: dict, *, item_name: str) -> dict:
        if not self._should_perturb_duration_v3_source_context():
            return source_cache
        source_duration = np.asarray(source_cache["dur_anchor_src"], dtype=np.float32).reshape(1, -1)
        unit_mask = (source_duration > 0.0).astype(np.float32)
        sep_hint = np.asarray(
            source_cache.get("sep_hint", np.zeros_like(source_duration, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(1, -1)
        generator = self._make_item_generator(item_name=item_name, salt="pseudo_source_duration")
        perturbed = build_pseudo_source_duration_context(
            torch.from_numpy(source_duration),
            torch.from_numpy(unit_mask),
            torch.from_numpy(sep_hint),
            global_scale_range=(
                float(self.hparams.get("rhythm_pseudo_source_global_scale_min", 0.85)),
                float(self.hparams.get("rhythm_pseudo_source_global_scale_max", 1.15)),
            ),
            local_span_prob=float(self.hparams.get("rhythm_pseudo_source_local_span_prob", 0.20)),
            local_span_scale=(
                float(self.hparams.get("rhythm_pseudo_source_local_scale_min", 0.70)),
                float(self.hparams.get("rhythm_pseudo_source_local_scale_max", 1.30)),
            ),
            mask_prob=float(self.hparams.get("rhythm_pseudo_source_mask_prob", 0.10)),
            flatten_boundary_prob=float(self.hparams.get("rhythm_pseudo_source_flatten_boundary_prob", 0.15)),
            generator=generator,
        ).squeeze(0).cpu().numpy().astype(np.float32)
        runtime_cache = self._copy_numpy_fields(source_cache)
        runtime_cache["dur_anchor_src"] = perturbed
        return runtime_cache

    def _maybe_augment_prompt_unit_conditioning(self, conditioning: dict, *, item_name: str) -> dict:
        if str(self.prefix).lower() != "train":
            return conditioning
        dropout = float(self.hparams.get("rhythm_prompt_dropout", 0.0) or 0.0)
        truncation = float(self.hparams.get("rhythm_prompt_truncation", 0.0) or 0.0)
        if dropout <= 0.0 and truncation <= 0.0:
            return conditioning
        prompt_valid_mask = np.asarray(conditioning.get("prompt_valid_mask", conditioning.get("prompt_unit_mask")), dtype=np.float32).copy()
        prompt_speech_mask = np.asarray(conditioning.get("prompt_speech_mask", prompt_valid_mask), dtype=np.float32).copy()
        if prompt_valid_mask.size <= 0:
            return conditioning
        valid = torch.from_numpy(prompt_valid_mask).reshape(1, -1) > 0.5
        if not bool(valid.any().item()):
            return conditioning
        keep = valid.clone()
        generator = self._make_item_generator(item_name=item_name, salt="prompt_duration_memory")
        if truncation > 0.0:
            valid_indices = torch.nonzero(valid[0], as_tuple=False).reshape(-1)
            valid_count = int(valid_indices.numel())
            if valid_count > 0:
                if truncation < 1.0:
                    min_keep = max(1, int(np.ceil(valid_count * truncation)))
                    max_keep = valid_count
                    if min_keep >= max_keep:
                        keep_units = max_keep
                    else:
                        sampled = torch.randint(min_keep, max_keep + 1, (1,), generator=generator, device=valid.device)
                        keep_units = int(sampled.item())
                else:
                    keep_units = max(1, min(valid_count, int(round(truncation))))
                cutoff_index = int(valid_indices[keep_units - 1].item())
                keep[:, cutoff_index + 1 :] = False
        if dropout > 0.0:
            drop = torch.rand(keep.shape, generator=generator, device=keep.device) < float(max(0.0, min(1.0, dropout)))
            keep = keep & ~drop
        if not bool(keep.any().item()):
            first_valid = int(torch.nonzero(valid[0], as_tuple=False)[0].item())
            keep[:, first_valid] = True
        keep_np = keep.float().reshape(-1).cpu().numpy().astype(np.float32)
        augmented = self._copy_numpy_fields(conditioning)
        augmented["prompt_unit_mask"] = keep_np
        augmented["prompt_valid_mask"] = keep_np
        augmented["prompt_speech_mask"] = prompt_speech_mask * keep_np
        for key in ("prompt_duration_obs", "prompt_source_boundary_cue", "prompt_phrase_group_pos", "prompt_phrase_final_mask"):
            if key in augmented:
                augmented[key] = np.asarray(augmented[key], dtype=np.float32) * keep_np
        return augmented

    def _build_reference_prompt_unit_conditioning(self, prompt_item, *, target_mode: str):
        if prompt_item is None:
            return {}
        source_cache = None
        item_name = str(prompt_item.get("item_name", "<prompt-item>")) if isinstance(prompt_item, dict) else "<prompt-item>"
        explicit_silence = self._should_emit_duration_v3_silence_runs()
        has_cached_prompt_source = all(key in prompt_item for key in ("content_units", "dur_anchor_src"))
        has_prompt_silence = "source_silence_mask" in prompt_item
        if has_cached_prompt_source and (not explicit_silence or has_prompt_silence):
            source_cache = {
                "content_units": prompt_item["content_units"],
                "dur_anchor_src": prompt_item["dur_anchor_src"],
            }
            if has_prompt_silence:
                source_cache["source_silence_mask"] = prompt_item["source_silence_mask"]
            if "sep_hint" in prompt_item:
                source_cache["sep_hint"] = prompt_item["sep_hint"]
            for extra_key in ("source_boundary_cue", "phrase_group_pos", "phrase_final_mask"):
                if extra_key in prompt_item:
                    source_cache[extra_key] = prompt_item[extra_key]
        elif target_mode != "cached_only" and "hubert" in prompt_item:
            source_cache = build_source_rhythm_cache(
                np.asarray(prompt_item["hubert"]),
                silent_token=self.hparams.get("silent_token", 57),
                separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
                tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
                emit_silence_runs=explicit_silence,
                phrase_boundary_threshold=float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
            )
        elif has_cached_prompt_source and target_mode != "cached_only":
            source_cache = {
                "content_units": prompt_item["content_units"],
                "dur_anchor_src": prompt_item["dur_anchor_src"],
            }
            if "sep_hint" in prompt_item:
                source_cache["sep_hint"] = prompt_item["sep_hint"]
            for extra_key in ("source_boundary_cue", "phrase_group_pos", "phrase_final_mask"):
                if extra_key in prompt_item:
                    source_cache[extra_key] = prompt_item[extra_key]
        elif target_mode == "cached_only" and explicit_silence and has_cached_prompt_source and not has_prompt_silence:
            raise RuntimeError(
                "rhythm_v3 cached_only with explicit silence-run frontend requires source_silence_mask in prompt/reference caches. "
                "Re-binarize with explicit silence runs enabled."
            )
        if source_cache is None:
            return {}
        prompt_content_units = np.asarray(source_cache["content_units"], dtype=np.int64)
        prompt_duration_obs = np.asarray(source_cache["dur_anchor_src"], dtype=np.float32)
        prompt_valid_mask = (prompt_duration_obs > 0).astype(np.float32)
        prompt_silence_mask = _resolve_run_silence_mask(
            size=prompt_duration_obs.shape[0],
            silence_mask=source_cache.get("source_silence_mask"),
        )
        prompt_speech_mask = prompt_valid_mask * (1.0 - prompt_silence_mask.clip(0.0, 1.0))
        return self._maybe_augment_prompt_unit_conditioning(
            {
                "prompt_content_units": prompt_content_units,
                "prompt_duration_obs": prompt_duration_obs,
                "prompt_unit_mask": prompt_valid_mask,
                "prompt_valid_mask": prompt_valid_mask,
                "prompt_speech_mask": prompt_speech_mask,
                "prompt_source_boundary_cue": np.asarray(
                    source_cache.get("source_boundary_cue", np.zeros_like(prompt_duration_obs)),
                    dtype=np.float32,
                ),
                "prompt_phrase_group_pos": np.asarray(
                    source_cache.get("phrase_group_pos", np.zeros_like(prompt_duration_obs)),
                    dtype=np.float32,
                ),
                "prompt_phrase_final_mask": np.asarray(
                    source_cache.get("phrase_final_mask", np.zeros_like(prompt_duration_obs)),
                    dtype=np.float32,
                ),
            },
            item_name=item_name,
        )

    @staticmethod
    def _resolve_paired_target_projection_inputs(
        paired_target_conditioning: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if not isinstance(paired_target_conditioning, dict):
            return None
        target_units = paired_target_conditioning.get("paired_target_content_units")
        target_duration = paired_target_conditioning.get("paired_target_duration_obs")
        target_valid = paired_target_conditioning.get("paired_target_valid_mask")
        target_speech = paired_target_conditioning.get("paired_target_speech_mask")
        if any(value is None for value in (target_units, target_duration, target_valid, target_speech)):
            return None
        return (
            _as_int64_1d(target_units),
            _as_float32_1d(target_duration),
            _as_float32_1d(target_valid),
            _as_float32_1d(target_speech),
        )

    def _build_paired_target_projection_conditioning(self, paired_target_item, *, target_mode: str):
        if paired_target_item is None:
            return {}
        source_cache = None
        item_name = (
            str(paired_target_item.get("item_name", "<paired-target-item>"))
            if isinstance(paired_target_item, dict)
            else "<paired-target-item>"
        )
        explicit_silence = self._should_emit_duration_v3_silence_runs()
        has_cached_target_source = (
            isinstance(paired_target_item, dict)
            and all(key in paired_target_item for key in ("content_units", "dur_anchor_src"))
        )
        has_target_silence = isinstance(paired_target_item, dict) and "source_silence_mask" in paired_target_item
        if has_cached_target_source and (not explicit_silence or has_target_silence):
            source_cache = {
                "content_units": paired_target_item["content_units"],
                "dur_anchor_src": paired_target_item["dur_anchor_src"],
            }
            if has_target_silence:
                source_cache["source_silence_mask"] = paired_target_item["source_silence_mask"]
        elif target_mode != "cached_only" and isinstance(paired_target_item, dict) and "hubert" in paired_target_item:
            source_cache = build_source_rhythm_cache(
                np.asarray(paired_target_item["hubert"]),
                silent_token=self.hparams.get("silent_token", 57),
                separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
                tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
                emit_silence_runs=explicit_silence,
                phrase_boundary_threshold=float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
            )
        elif has_cached_target_source and target_mode != "cached_only":
            source_cache = {
                "content_units": paired_target_item["content_units"],
                "dur_anchor_src": paired_target_item["dur_anchor_src"],
            }
        elif target_mode == "cached_only" and explicit_silence and has_cached_target_source and not has_target_silence:
            raise RuntimeError(
                "rhythm_v3 cached_only paired-target supervision with explicit silence-run frontend requires "
                "source_silence_mask in paired-target caches. Re-binarize with explicit silence runs enabled."
            )
        if source_cache is None:
            return {}
        paired_target_content_units = np.asarray(source_cache["content_units"], dtype=np.int64)
        paired_target_duration_obs = np.asarray(source_cache["dur_anchor_src"], dtype=np.float32)
        paired_target_valid_mask = (paired_target_duration_obs > 0).astype(np.float32)
        paired_target_silence_mask = _resolve_run_silence_mask(
            size=paired_target_duration_obs.shape[0],
            silence_mask=source_cache.get("source_silence_mask"),
        )
        paired_target_speech_mask = paired_target_valid_mask * (1.0 - paired_target_silence_mask.clip(0.0, 1.0))
        return {
            "paired_target_content_units": paired_target_content_units,
            "paired_target_duration_obs": paired_target_duration_obs,
            "paired_target_valid_mask": paired_target_valid_mask,
            "paired_target_speech_mask": paired_target_speech_mask,
            "paired_target_item_name": np.asarray([item_name], dtype=object),
        }

    def _build_paired_duration_v3_targets(self, *, item, source_cache, paired_target_conditioning):
        projection_inputs = self._resolve_paired_target_projection_inputs(paired_target_conditioning)
        if projection_inputs is None:
            return None
        item_name = str(item.get("item_name", "<unknown-item>")) if isinstance(item, dict) else "<unknown-item>"
        source_units = _as_int64_1d(source_cache["content_units"])
        source_duration = _as_float32_1d(source_cache["dur_anchor_src"])
        source_silence = _resolve_run_silence_mask(
            size=source_duration.shape[0],
            silence_mask=source_cache.get("source_silence_mask"),
        )
        target_units, target_duration, target_valid, target_speech = projection_inputs
        try:
            projected, confidence = _project_target_runs_onto_source(
                source_units=source_units,
                source_durations=source_duration,
                source_silence_mask=source_silence,
                target_units=target_units,
                target_durations=target_duration,
                target_valid_mask=target_valid,
                target_speech_mask=target_speech,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to build paired duration_v3 targets for {item_name}: {exc}") from exc
        if projected.shape[0] != source_duration.shape[0] or confidence.shape[0] != source_duration.shape[0]:
            raise RuntimeError(
                f"Paired duration_v3 projection length mismatch for {item_name}: "
                f"source={source_duration.shape[0]}, projected={projected.shape[0]}, confidence={confidence.shape[0]}"
            )
        return {
            "unit_duration_tgt": projected.astype(np.float32),
            "unit_confidence_tgt": confidence.astype(np.float32),
        }

    def _merge_duration_v3_rhythm_targets(self, item, source_cache, paired_target_conditioning, sample):
        if isinstance(sample, dict) and "unit_duration_tgt" in sample:
            out = {
                "unit_duration_tgt": np.asarray(sample["unit_duration_tgt"], dtype=np.float32),
            }
            if "unit_confidence_tgt" in sample:
                out["unit_confidence_tgt"] = np.asarray(sample["unit_confidence_tgt"], dtype=np.float32)
            return out

        source_item_name = str(item.get("item_name", "<unknown-item>")) if isinstance(item, dict) else "<unknown-item>"
        if isinstance(paired_target_conditioning, dict):
            paired_target_item_name = paired_target_conditioning.get("paired_target_item_name")
            if paired_target_item_name is not None:
                paired_target_item_name = str(np.asarray(paired_target_item_name).reshape(-1)[0])
                if (
                    paired_target_item_name == source_item_name
                    and not self._is_enabled_flag(self.hparams.get("rhythm_v3_allow_source_self_target_fallback", False))
                ):
                    raise RuntimeError(
                        "Canonical duration_v3 training requires an external paired target item or explicit unit_duration_tgt. "
                        "Self paired-target projection is disabled unless rhythm_v3_allow_source_self_target_fallback=true."
                    )

        paired = self._build_paired_duration_v3_targets(
            item=item,
            source_cache=source_cache,
            paired_target_conditioning=paired_target_conditioning,
        )
        if paired is not None:
            return paired

        if self._is_enabled_flag(self.hparams.get("rhythm_v3_allow_source_self_target_fallback", False)):
            target = np.asarray(source_cache["dur_anchor_src"], dtype=np.float32)
            return {
                "unit_duration_tgt": target,
                "unit_confidence_tgt": np.ones_like(target, dtype=np.float32),
            }

        raise RuntimeError(
            "Canonical duration_v3 training now requires paired run projection/alignment targets. "
            "Provide unit_duration_tgt explicitly or supply an external paired-target run lattice that can be projected onto the source lattice."
        )


__all__ = ["DurationV3DatasetMixin"]

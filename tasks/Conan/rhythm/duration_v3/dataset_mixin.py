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
    silence = _as_float32_1d(silence_mask)
    if silence.shape[0] == size:
        return silence
    out = np.zeros((size,), dtype=np.float32)
    limit = min(size, silence.shape[0])
    out[:limit] = silence[:limit]
    return out


def _filter_valid_runs(
    *,
    units: np.ndarray,
    durations: np.ndarray,
    silence_mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keep = np.asarray(durations, dtype=np.float32).reshape(-1) > 0.0
    if valid_mask is not None:
        keep = keep & (_as_float32_1d(valid_mask) > 0.5)
    indices = np.nonzero(keep)[0].astype(np.int64)
    return (
        _as_int64_1d(units)[indices],
        _as_float32_1d(durations)[indices],
        _as_float32_1d(silence_mask)[indices],
        indices,
    )


def _align_target_runs_to_source(
    *,
    source_units: np.ndarray,
    source_durations: np.ndarray,
    source_silence: np.ndarray,
    target_units: np.ndarray,
    target_durations: np.ndarray,
    target_silence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_source = int(source_units.shape[0])
    num_target = int(target_units.shape[0])
    if num_source <= 0 or num_target <= 0:
        raise RuntimeError("paired projection requires non-empty source and target run lattices.")

    gap_penalty = np.float32(0.40)
    src_pos = (np.arange(num_source, dtype=np.float32) + 1.0) / float(max(1, num_source))
    tgt_pos = (np.arange(num_target, dtype=np.float32) + 1.0) / float(max(1, num_target))
    token_match = source_units[:, None] == target_units[None, :]
    sil_match = (source_silence[:, None] > 0.5) == (target_silence[None, :] > 0.5)
    log_duration_delta = np.abs(
        np.log(source_durations[:, None].clip(min=1.0e-4))
        - np.log(target_durations[None, :].clip(min=1.0e-4))
    ).astype(np.float32)
    local_cost = np.where(
        token_match & sil_match,
        0.0,
        np.where(
            ~sil_match,
            1.60,
            np.where(token_match, 0.15, 0.90),
        ),
    ).astype(np.float32)
    local_cost += (0.20 * log_duration_delta).astype(np.float32)
    local_cost += (0.15 * np.abs(src_pos[:, None] - tgt_pos[None, :])).astype(np.float32)

    back = np.zeros((num_source + 1, num_target + 1), dtype=np.uint8)
    dp_prev = np.full((num_target + 1,), np.inf, dtype=np.float32)
    dp_curr = np.full((num_target + 1,), np.inf, dtype=np.float32)
    dp_prev[0] = 0.0
    for tgt_idx in range(1, num_target + 1):
        dp_prev[tgt_idx] = dp_prev[tgt_idx - 1] + gap_penalty
        back[0, tgt_idx] = 2

    band = int(max(8, round(0.15 * max(num_source, num_target))))
    for src_idx in range(1, num_source + 1):
        dp_curr.fill(np.inf)
        dp_curr[0] = dp_prev[0] + gap_penalty
        back[src_idx, 0] = 1
        center = int(round((src_idx - 1) * num_target / float(max(1, num_source))))
        left = max(1, center - band)
        right = min(num_target + 1, center + band + 1)
        for tgt_idx in range(left, right):
            cost = local_cost[src_idx - 1, tgt_idx - 1]
            diag = dp_prev[tgt_idx - 1]
            up = dp_prev[tgt_idx] + gap_penalty
            left_cost = dp_curr[tgt_idx - 1] + gap_penalty
            if diag <= up and diag <= left_cost:
                dp_curr[tgt_idx] = cost + diag
                back[src_idx, tgt_idx] = 0
            elif up <= left_cost:
                dp_curr[tgt_idx] = cost + up
                back[src_idx, tgt_idx] = 1
            else:
                dp_curr[tgt_idx] = cost + left_cost
                back[src_idx, tgt_idx] = 2
        dp_prev, dp_curr = dp_curr, dp_prev

    assigned_source = np.full((num_target,), -1, dtype=np.int64)
    assigned_cost = np.full((num_target,), gap_penalty, dtype=np.float32)
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
            if src_idx > 0:
                assigned_source[tgt_idx - 1] = src_idx - 1
                assigned_cost[tgt_idx - 1] = local_cost[src_idx - 1, tgt_idx - 1]
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
 ) -> dict[str, np.ndarray]:
    source_silence_mask = _resolve_run_silence_mask(size=len(source_units), silence_mask=source_silence_mask)
    target_silence_mask = ((target_valid_mask > 0.5) & ~(target_speech_mask > 0.5)).astype(np.float32)
    (
        src_units_valid,
        src_durations_valid,
        src_silence_valid,
        src_run_index,
    ) = _filter_valid_runs(
        units=source_units,
        durations=source_durations,
        silence_mask=source_silence_mask,
    )
    (
        tgt_units_valid,
        tgt_durations_valid,
        tgt_silence_valid,
        _,
    ) = _filter_valid_runs(
        units=target_units,
        durations=target_durations,
        silence_mask=target_silence_mask,
        valid_mask=target_valid_mask,
    )
    if src_units_valid.size <= 0:
        raise RuntimeError("paired projection requires a non-empty source run lattice.")
    if tgt_units_valid.size <= 0:
        raise RuntimeError("paired projection requires a non-empty paired target prompt lattice.")

    assigned_source, assigned_cost = _align_target_runs_to_source(
        source_units=src_units_valid,
        source_durations=src_durations_valid,
        source_silence=src_silence_valid,
        target_units=tgt_units_valid,
        target_durations=tgt_durations_valid,
        target_silence=tgt_silence_valid,
    )

    num_source_runs = int(source_units.shape[0])
    projected = np.zeros((num_source_runs,), dtype=np.float32)
    aligned_target = np.zeros((num_source_runs,), dtype=np.float32)
    exact_match = np.zeros((num_source_runs,), dtype=np.float32)
    cost_mass = np.zeros((num_source_runs,), dtype=np.float32)
    source_support = np.zeros((num_source_runs,), dtype=np.float32)
    source_support[src_run_index] = 1.0

    for tgt_idx, src_token_idx in enumerate(assigned_source.tolist()):
        if src_token_idx < 0 or src_token_idx >= int(src_run_index.shape[0]):
            continue
        safe_src_token_idx = int(src_token_idx)
        run_idx = int(src_run_index[safe_src_token_idx])
        projected[run_idx] += float(tgt_durations_valid[tgt_idx])
        aligned_target[run_idx] += 1.0
        if (
            int(src_units_valid[safe_src_token_idx]) == int(tgt_units_valid[tgt_idx])
            and int(src_silence_valid[safe_src_token_idx] > 0.5) == int(tgt_silence_valid[tgt_idx] > 0.5)
        ):
            exact_match[run_idx] += 1.0
        cost_mass[run_idx] += float(assigned_cost[tgt_idx])

    coverage = np.divide(
        aligned_target,
        np.clip(source_support, 1.0, None),
        out=np.zeros_like(aligned_target),
        where=source_support > 0.0,
    )
    coverage = np.clip(coverage, 0.0, 1.0).astype(np.float32)
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
    mass_agree = np.divide(
        np.minimum(projected, source_durations.astype(np.float32)),
        np.maximum(np.maximum(projected, source_durations.astype(np.float32)), 1.0),
        out=np.zeros_like(projected),
        where=(projected > 0.0) | (source_durations.astype(np.float32) > 0.0),
    )
    confidence_coarse = np.clip(
        (0.70 * np.sqrt(np.clip(mass_agree, 0.0, 1.0))) + (0.15 * coverage) + (0.15 * np.exp(-0.5 * mean_cost)),
        0.0,
        1.0,
    ).astype(np.float32)
    confidence_local = np.clip(
        (0.55 * mass_agree) + (0.30 * match_rate) + (0.15 * np.exp(-mean_cost)),
        0.0,
        1.0,
    ).astype(np.float32)

    source_is_silence = source_silence_mask > 0.5
    projected[source_is_silence] = np.maximum(projected[source_is_silence], 1.0).astype(np.float32)
    confidence_local[source_is_silence] = 0.0
    confidence_coarse[source_is_silence] = 0.0

    speech_zero = (~source_is_silence) & (projected <= 0.0)
    projected[speech_zero] = source_durations[speech_zero].astype(np.float32)
    confidence_local[speech_zero] = 0.0
    confidence_coarse[speech_zero] = np.minimum(confidence_coarse[speech_zero], 0.20).astype(np.float32)
    return {
        "projected": projected.astype(np.float32),
        "confidence_local": confidence_local.astype(np.float32),
        "confidence_coarse": confidence_coarse.astype(np.float32),
        "coverage": coverage.astype(np.float32),
        "match_rate": match_rate.astype(np.float32),
        "mean_cost": mean_cost.astype(np.float32),
    }


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
        source_silence = np.asarray(
            source_cache.get("source_silence_mask", np.zeros_like(source_duration, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(1, -1)
        generator = self._make_item_generator(item_name=item_name, salt="pseudo_source_duration")
        perturbed = build_pseudo_source_duration_context(
            torch.from_numpy(source_duration),
            torch.from_numpy(unit_mask),
            torch.from_numpy(sep_hint),
            silence_mask=torch.from_numpy(source_silence),
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
        speech_valid = (torch.from_numpy(prompt_speech_mask).reshape(1, -1) > 0.5) & valid
        if not bool((keep & speech_valid).any().item()) and bool(speech_valid.any().item()):
            first_speech = int(torch.nonzero(speech_valid[0], as_tuple=False)[0].item())
            keep[:, first_speech] = True
        if not bool(keep.any().item()):
            first_valid = int(torch.nonzero(valid[0], as_tuple=False)[0].item())
            keep[:, first_valid] = True
        keep_np = keep.float().reshape(-1).cpu().numpy().astype(np.float32)
        augmented = self._copy_numpy_fields(conditioning)
        augmented["prompt_unit_mask"] = keep_np
        augmented["prompt_valid_mask"] = keep_np
        augmented["prompt_speech_mask"] = prompt_speech_mask * keep_np
        for key in (
            "prompt_duration_obs",
            "prompt_unit_anchor_base",
            "prompt_log_base",
            "prompt_source_boundary_cue",
            "prompt_phrase_group_pos",
            "prompt_phrase_final_mask",
        ):
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
            for extra_key in ("unit_anchor_base", "unit_rate_log_base"):
                if extra_key in prompt_item:
                    source_cache[extra_key] = prompt_item[extra_key]
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
                debounce_min_run_frames=int(self.hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
                phrase_boundary_threshold=float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
            )
        elif has_cached_prompt_source and target_mode != "cached_only":
            if explicit_silence:
                if "hubert" in prompt_item:
                    source_cache = build_source_rhythm_cache(
                        np.asarray(prompt_item["hubert"]),
                        silent_token=self.hparams.get("silent_token", 57),
                        separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
                        tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
                        emit_silence_runs=True,
                        debounce_min_run_frames=int(self.hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
                        phrase_boundary_threshold=float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
                    )
                else:
                    raise RuntimeError(
                        "explicit silence-run frontend requires source_silence_mask in prompt/reference caches. "
                        "Rebuild the cache or provide raw hubert."
                    )
            else:
                source_cache = {
                    "content_units": prompt_item["content_units"],
                    "dur_anchor_src": prompt_item["dur_anchor_src"],
                }
                for extra_key in ("unit_anchor_base", "unit_rate_log_base"):
                    if extra_key in prompt_item:
                        source_cache[extra_key] = prompt_item[extra_key]
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
        conditioning = {
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
        }
        if "unit_anchor_base" in source_cache:
            conditioning["prompt_unit_anchor_base"] = np.asarray(source_cache["unit_anchor_base"], dtype=np.float32)
        if "unit_rate_log_base" in source_cache:
            conditioning["prompt_log_base"] = np.asarray(source_cache["unit_rate_log_base"], dtype=np.float32)
        return self._maybe_augment_prompt_unit_conditioning(
            conditioning,
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

    def _build_paired_target_projection_conditioning(self, paired_target_item, *, target_mode: str, source_item=None):
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
                debounce_min_run_frames=int(self.hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
                phrase_boundary_threshold=float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
            )
        elif has_cached_target_source and target_mode != "cached_only":
            if explicit_silence:
                if isinstance(paired_target_item, dict) and "hubert" in paired_target_item:
                    source_cache = build_source_rhythm_cache(
                        np.asarray(paired_target_item["hubert"]),
                        silent_token=self.hparams.get("silent_token", 57),
                        separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
                        tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
                        emit_silence_runs=True,
                        debounce_min_run_frames=int(self.hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
                        phrase_boundary_threshold=float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
                    )
                else:
                    raise RuntimeError(
                        "explicit silence-run frontend requires source_silence_mask in cached paired-target sources. "
                        "Rebuild the cache or provide raw hubert."
                    )
            else:
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
            "paired_target_text_signature": np.asarray(
                [self._rhythm_text_signature(paired_target_item if isinstance(paired_target_item, dict) else None)],
                dtype=object,
            ),
            "source_text_signature": np.asarray(
                [self._rhythm_text_signature(source_item if isinstance(source_item, dict) else None)],
                dtype=object,
            ),
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
            projection = _project_target_runs_onto_source(
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
        projected = np.asarray(projection["projected"], dtype=np.float32)
        confidence_local = np.asarray(projection["confidence_local"], dtype=np.float32)
        confidence_coarse = np.asarray(projection["confidence_coarse"], dtype=np.float32)
        coverage = np.asarray(projection["coverage"], dtype=np.float32)
        match_rate = np.asarray(projection["match_rate"], dtype=np.float32)
        mean_cost = np.asarray(projection["mean_cost"], dtype=np.float32)
        if (
            projected.shape[0] != source_duration.shape[0]
            or confidence_local.shape[0] != source_duration.shape[0]
            or confidence_coarse.shape[0] != source_duration.shape[0]
        ):
            raise RuntimeError(
                f"Paired duration_v3 projection length mismatch for {item_name}: "
                f"source={source_duration.shape[0]}, projected={projected.shape[0]}, "
                f"confidence_local={confidence_local.shape[0]}, confidence_coarse={confidence_coarse.shape[0]}"
            )
        return {
            "unit_duration_tgt": projected.astype(np.float32),
            "unit_confidence_local_tgt": confidence_local.astype(np.float32),
            "unit_confidence_coarse_tgt": confidence_coarse.astype(np.float32),
            "unit_confidence_tgt": confidence_coarse.astype(np.float32),
            "unit_alignment_coverage_tgt": coverage.astype(np.float32),
            "unit_alignment_match_tgt": match_rate.astype(np.float32),
            "unit_alignment_cost_tgt": mean_cost.astype(np.float32),
        }

    def _merge_duration_v3_rhythm_targets(self, item, source_cache, paired_target_conditioning, sample):
        if isinstance(sample, dict) and "unit_duration_tgt" in sample:
            out = {
                "unit_duration_tgt": np.asarray(sample["unit_duration_tgt"], dtype=np.float32),
            }
            if "unit_confidence_local_tgt" in sample:
                out["unit_confidence_local_tgt"] = np.asarray(sample["unit_confidence_local_tgt"], dtype=np.float32)
            if "unit_confidence_coarse_tgt" in sample:
                out["unit_confidence_coarse_tgt"] = np.asarray(sample["unit_confidence_coarse_tgt"], dtype=np.float32)
            if "unit_confidence_tgt" in sample:
                out["unit_confidence_tgt"] = np.asarray(sample["unit_confidence_tgt"], dtype=np.float32)
            if "unit_confidence_tgt" in out:
                out.setdefault("unit_confidence_local_tgt", out["unit_confidence_tgt"])
                out.setdefault("unit_confidence_coarse_tgt", out["unit_confidence_tgt"])
            elif "unit_confidence_coarse_tgt" in out:
                out["unit_confidence_tgt"] = np.asarray(out["unit_confidence_coarse_tgt"], dtype=np.float32)
                out.setdefault("unit_confidence_local_tgt", out["unit_confidence_coarse_tgt"])
            for key in (
                "unit_alignment_coverage_tgt",
                "unit_alignment_match_tgt",
                "unit_alignment_cost_tgt",
            ):
                if key in sample:
                    out[key] = np.asarray(sample[key], dtype=np.float32)
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
            if self._disallow_same_text_paired_target():
                paired_target_text_signature = paired_target_conditioning.get("paired_target_text_signature")
                source_text_signature = paired_target_conditioning.get("source_text_signature")
                if paired_target_text_signature is not None and source_text_signature is not None:
                    paired_sig = np.asarray(paired_target_text_signature, dtype=object).reshape(-1)[0]
                    source_sig = np.asarray(source_text_signature, dtype=object).reshape(-1)[0]
                    if paired_sig is not None and source_sig is not None and paired_sig == source_sig:
                        raise RuntimeError(
                            "rhythm_v3 paired-target projection forbids same-text targets by default. "
                            "Provide a different-text paired target or set rhythm_v3_disallow_same_text_paired_target=false."
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
                "unit_confidence_local_tgt": np.ones_like(target, dtype=np.float32),
                "unit_confidence_coarse_tgt": np.ones_like(target, dtype=np.float32),
                "unit_confidence_tgt": np.ones_like(target, dtype=np.float32),
                "unit_alignment_coverage_tgt": np.ones_like(target, dtype=np.float32),
                "unit_alignment_match_tgt": np.ones_like(target, dtype=np.float32),
                "unit_alignment_cost_tgt": np.zeros_like(target, dtype=np.float32),
            }

        raise RuntimeError(
            "Canonical duration_v3 training now requires paired run projection/alignment targets. "
            "Provide unit_duration_tgt explicitly or supply an external paired-target run lattice that can be projected onto the source lattice."
        )


__all__ = ["DurationV3DatasetMixin"]

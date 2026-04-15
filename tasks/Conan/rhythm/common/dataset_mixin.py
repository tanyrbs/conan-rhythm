from __future__ import annotations

from collections.abc import Mapping
import hashlib

import numpy as np
import torch
from utils.commons.dataset_utils import collate_1d_or_2d

from modules.Conan.rhythm.policy import build_rhythm_hparams_policy
from modules.Conan.rhythm.supervision import RHYTHM_CACHE_VERSION
from modules.Conan.rhythm_v3.source_cache import (
    DURATION_V3_CACHE_META_KEY,
    build_source_rhythm_cache_v3 as build_source_rhythm_cache,
    estimate_boundary_confidence,
    estimate_run_stability,
)
from tasks.Conan.rhythm.dataset_contracts import RhythmDatasetCacheContract
from tasks.Conan.rhythm.dataset_errors import RhythmDatasetPrefilterDrop
from tasks.Conan.rhythm.dataset_sample_builder import RhythmDatasetSampleAssembler
from tasks.Conan.rhythm.dataset_target_builder import RhythmDatasetTargetBuilder


class CommonRhythmDatasetMixin:
    @staticmethod
    def _restore_debug_sidecar(
        full_side: dict[str, np.ndarray],
        key: str,
        fallback: np.ndarray,
    ) -> np.ndarray:
        value = full_side.get(key)
        if value is None:
            return fallback.copy()
        restored = np.asarray(value, dtype=fallback.dtype).reshape(-1)
        if restored.shape != fallback.shape:
            return fallback.copy()
        return restored.copy()

    def _sample_prefilter_retry_index(self, current_index: int) -> int:
        dataset_len = int(len(self))
        if dataset_len <= 1:
            return int(current_index)
        candidate = int(np.random.randint(0, dataset_len))
        if candidate == int(current_index):
            candidate = (candidate + 1) % dataset_len
        return candidate

    @staticmethod
    def _rhythm_text_signature(item) -> tuple | str | None:
        if not isinstance(item, dict):
            return None
        for key in ("ph_token", "txt_token", "txt_tokens", "word_token", "word_tokens"):
            value = item.get(key)
            if value is None:
                continue
            arr = np.asarray(value).reshape(-1)
            if arr.size > 0:
                return (key, tuple(arr.tolist()))
        for key in ("ph", "txt", "word", "words"):
            value = item.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return (key, text)
        return None

    def _same_rhythm_text(self, item_a, item_b) -> bool:
        sig_a = self._rhythm_text_signature(item_a)
        sig_b = self._rhythm_text_signature(item_b)
        return sig_a is not None and sig_b is not None and sig_a == sig_b

    def _disallow_same_text_reference(self) -> bool:
        return bool(
            self.hparams.get(
                "rhythm_v3_disallow_same_text_reference",
                self.hparams.get("rhythm_disallow_same_text_reference", True),
            )
        )

    def _disallow_same_text_paired_target(self) -> bool:
        if self._is_enabled_flag(self.hparams.get("rhythm_v3_minimal_v1_profile", False)):
            return False
        return bool(
            self.hparams.get(
                "rhythm_v3_disallow_same_text_paired_target",
                self.hparams.get(
                    "rhythm_disallow_same_text_paired_target",
                    False,
                ),
            )
        )

    def _require_same_text_paired_target(self) -> bool:
        return bool(
            self.hparams.get(
                "rhythm_v3_require_same_text_paired_target",
                self._is_enabled_flag(self.hparams.get("rhythm_v3_minimal_v1_profile", False)),
            )
        )

    def _should_emit_explicit_silence_runs(self) -> bool:
        resolver = getattr(self, "_should_emit_duration_v3_silence_runs", None)
        if callable(resolver):
            return bool(resolver())
        return False

    def _require_minimal_v1_prompt_sidecars(self, item, *, item_name: str) -> None:
        if not self._is_enabled_flag(self.hparams.get("rhythm_v3_minimal_v1_profile", False)):
            return
        if not isinstance(item, dict):
            raise RuntimeError(f"{item_name}: prompt/reference item must be a dict under minimal_v1_profile")
        missing: list[str] = []
        if item.get("prompt_speech_mask") is None and item.get("source_silence_mask") is None:
            missing.append("prompt_speech_mask/source_silence_mask")
        if item.get("prompt_duration_obs") is None and item.get("dur_anchor_src") is None:
            missing.append("prompt_duration_obs/dur_anchor_src")
        if item.get("prompt_unit_mask") is None and item.get("prompt_valid_mask") is None:
            can_derive_mask = item.get("prompt_duration_obs") is not None or item.get("dur_anchor_src") is not None
            if not can_derive_mask:
                missing.append("prompt_unit_mask/prompt_valid_mask")
        if missing:
            raise RuntimeError(f"{item_name}: missing minimal_v1 prompt sidecars {missing}")

    def _rhythm_policy(self):
        policy = getattr(self, "_cached_rhythm_policy", None)
        if policy is None:
            policy = build_rhythm_hparams_policy(self.hparams)
            self._cached_rhythm_policy = policy
        return policy

    def _rhythm_cache_contract(self) -> RhythmDatasetCacheContract:
        contract = getattr(self, "_cached_rhythm_cache_contract", None)
        if contract is None:
            contract = RhythmDatasetCacheContract(self)
            self._cached_rhythm_cache_contract = contract
        return contract

    def _rhythm_sample_assembler(self) -> RhythmDatasetSampleAssembler:
        assembler = getattr(self, "_cached_rhythm_sample_assembler", None)
        if assembler is None:
            assembler = RhythmDatasetSampleAssembler(self)
            self._cached_rhythm_sample_assembler = assembler
        return assembler

    def _rhythm_target_builder(self) -> RhythmDatasetTargetBuilder:
        builder = getattr(self, "_cached_rhythm_target_builder", None)
        if builder is None:
            builder = RhythmDatasetTargetBuilder(self)
            self._cached_rhythm_target_builder = builder
        return builder

    def _resolve_primary_target_surface(self) -> str:
        return self._rhythm_policy().primary_target_surface

    def _resolve_rhythm_stage(self) -> str:
        return self._rhythm_policy().stage

    def _resolve_distill_surface(self) -> str:
        return self._rhythm_policy().distill_surface

    def _resolve_rhythm_target_mode(self) -> str:
        return self._rhythm_policy().target_mode

    def _resolve_teacher_target_source(self) -> str:
        return self._rhythm_policy().teacher_target_source

    def _resolve_expected_teacher_surface_name(self) -> str:
        return self._rhythm_policy().teacher_surface_name

    def _resolve_expected_teacher_target_source_id(self) -> int:
        return self._rhythm_policy().teacher_target_source_id

    def _should_sample_streaming_prefix(self) -> bool:
        return (
            self.prefix == "train"
            and bool(self.hparams.get("rhythm_streaming_prefix_train", False))
        )

    def _should_export_rhythm_debug_sidecars(self) -> bool:
        return bool(self.hparams.get("rhythm_export_debug_sidecars", False))

    def _should_export_runtime_phrase_bank_sidecars(self) -> bool:
        return bool(
            self.hparams.get("rhythm_runtime_phrase_bank_enable", False)
            or self._should_export_rhythm_debug_sidecars()
        )

    def _should_export_rhythm_cache_audit(self) -> bool:
        return bool(self.hparams.get("rhythm_export_cache_audit_to_sample", False))

    def _should_export_streaming_offline_sidecars(self) -> bool:
        return self._rhythm_policy().exports_streaming_offline_sidecars()

    def _should_export_offline_teacher_aux(self) -> bool:
        return self._rhythm_policy().should_export_offline_teacher_aux()

    def _should_export_streaming_prefix_meta(self) -> bool:
        return self._should_sample_streaming_prefix()

    def _should_export_runtime_retimed_targets(self) -> bool:
        return self._rhythm_policy().should_export_runtime_retimed_targets(split=self.prefix)

    def _resolve_runtime_target_export_keys(self) -> tuple[str, ...]:
        policy = self._rhythm_policy()
        primary_surface = policy.primary_target_surface
        distill_surface = policy.distill_surface
        lambda_guidance = float(self.hparams.get("lambda_rhythm_guidance", 0.0))
        lambda_distill = float(self.hparams.get("lambda_rhythm_distill", 0.0))
        distill_exec_weight = float(self.hparams.get("rhythm_distill_exec_weight", 1.0))
        distill_budget_weight = float(self.hparams.get("rhythm_distill_budget_weight", 0.5))
        distill_allocation_weight = float(self.hparams.get("rhythm_distill_allocation_weight", 0.5))
        distill_prefix_weight = float(self.hparams.get("rhythm_distill_prefix_weight", 0.25))
        distill_speech_shape_weight = float(self.hparams.get("rhythm_distill_speech_shape_weight", 0.0))
        distill_pause_shape_weight = float(self.hparams.get("rhythm_distill_pause_shape_weight", 0.0))
        require_retimed_cache = policy.require_retimed_cache
        retimed_source = str(self.hparams.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower()
        export_retimed_targets = self._should_export_runtime_retimed_targets()

        keys = list(self._RHYTHM_RUNTIME_TARGET_CORE_KEYS)
        if lambda_guidance > 0.0:
            keys.extend(self._RHYTHM_RUNTIME_GUIDANCE_KEYS)

        need_teacher_core = (
            primary_surface == "teacher"
            or policy.require_cached_teacher
            or (lambda_distill > 0.0 and distill_surface == "cache")
            or ((export_retimed_targets or require_retimed_cache) and retimed_source == "teacher")
        )
        if need_teacher_core:
            keys.extend(self._RHYTHM_RUNTIME_TEACHER_CORE_KEYS)
        if lambda_distill > 0.0 and distill_surface == "cache":
            if distill_exec_weight > 0.0:
                keys.extend(("rhythm_teacher_confidence_exec",))
            if distill_budget_weight > 0.0:
                keys.extend(("rhythm_teacher_confidence_budget",))
            if distill_prefix_weight > 0.0:
                keys.extend(("rhythm_teacher_confidence_prefix",))
            if distill_allocation_weight > 0.0:
                keys.extend(("rhythm_teacher_confidence_allocation",))
            if distill_speech_shape_weight > 0.0 or distill_pause_shape_weight > 0.0:
                keys.extend(("rhythm_teacher_confidence_shape",))

        need_teacher_prefix = lambda_distill > 0.0 and distill_surface == "cache"
        if need_teacher_prefix and distill_allocation_weight > 0.0:
            keys.extend(self._RHYTHM_RUNTIME_TEACHER_ALLOCATION_KEYS)
        if need_teacher_prefix and distill_prefix_weight > 0.0:
            keys.extend(self._RHYTHM_RUNTIME_TEACHER_PREFIX_KEYS)

        need_teacher_budget = (
            primary_surface == "teacher"
            or policy.require_cached_teacher
            or (lambda_distill > 0.0 and distill_surface == "cache" and distill_budget_weight > 0.0)
        )
        if not need_teacher_budget:
            keys = [key for key in keys if key not in {"rhythm_teacher_speech_budget_tgt", "rhythm_teacher_pause_budget_tgt"}]

        if export_retimed_targets:
            keys.extend(self._RHYTHM_RUNTIME_RETIMED_KEYS)
        return tuple(dict.fromkeys(keys))

    @staticmethod
    def _build_optional_collate_spec() -> dict[str, tuple[str, float | int | None]]:
        return {
            "content_units": ("long", 0),
            "dur_anchor_src": ("float", 0.0),
            "unit_anchor_base": ("float", 0.0),
            "unit_rate_log_base": ("float", 0.0),
            "open_run_mask": ("long", 0),
            "sealed_mask": ("float", 0.0),
            "sep_hint": ("long", 0),
            "boundary_confidence": ("float", 0.0),
            "source_silence_mask": ("float", 0.0),
            "source_run_stability": ("float", 0.0),
            "source_boundary_cue": ("float", 0.0),
            "phrase_group_index": ("long", 0),
            "phrase_group_pos": ("float", 0.0),
            "phrase_final_mask": ("float", 0.0),
            "rhythm_v3_cache_meta": ("object", None),
            "rhythm_offline_content_units": ("long", 0),
            "rhythm_offline_dur_anchor_src": ("long", 0),
            "rhythm_offline_source_silence_mask": ("float", 0.0),
            "rhythm_offline_source_run_stability": ("float", 0.0),
            "rhythm_offline_open_run_mask": ("long", 0),
            "rhythm_offline_sealed_mask": ("float", 0.0),
            "rhythm_offline_sep_hint": ("long", 0),
            "rhythm_offline_boundary_confidence": ("float", 0.0),
            "rhythm_offline_source_boundary_cue": ("float", 0.0),
            "rhythm_offline_phrase_group_index": ("long", 0),
            "rhythm_offline_phrase_group_pos": ("float", 0.0),
            "rhythm_offline_phrase_final_mask": ("float", 0.0),
            "rhythm_offline_rhythm_v3_cache_meta": ("object", None),
            "ref_rhythm_stats": ("float", 0.0),
            "ref_rhythm_trace": ("float", 0.0),
            "prompt_content_units": ("long", 0),
            "prompt_duration_obs": ("float", 0.0),
            "prompt_unit_mask": ("float", 0.0),
            "prompt_valid_mask": ("float", 0.0),
            "prompt_speech_mask": ("float", 0.0),
            "prompt_closed_mask": ("float", 0.0),
            "prompt_closed_mask_present": ("float", 0.0),
            "prompt_boundary_confidence": ("float", 0.0),
            "prompt_boundary_confidence_present": ("float", 0.0),
            "prompt_ref_len_sec": ("float", 0.0),
            "prompt_ref_len_present": ("float", 0.0),
            "prompt_speech_ratio_scalar": ("float", 0.0),
            "prompt_global_weight": ("float", 0.0),
            "prompt_global_weight_present": ("float", 0.0),
            "prompt_unit_log_prior": ("float", 0.0),
            "prompt_unit_log_prior_present": ("float", 0.0),
            "prompt_unit_prior_vocab_size": ("long", 0),
            "g_trim_ratio": ("float", 0.0),
            "prompt_unit_anchor_base": ("float", 0.0),
            "prompt_log_base": ("float", 0.0),
            "prompt_source_boundary_cue": ("float", 0.0),
            "prompt_phrase_group_pos": ("float", 0.0),
            "prompt_phrase_final_mask": ("float", 0.0),
            "unit_duration_tgt": ("float", 0.0),
            "unit_duration_proj_raw_tgt": ("float", 0.0),
            "unit_confidence_tgt": ("float", 0.0),
            "unit_confidence_local_tgt": ("float", 0.0),
            "unit_confidence_coarse_tgt": ("float", 0.0),
            "unit_annotation_weight_tgt": ("float", 0.0),
            "unit_alignment_coverage_tgt": ("float", 0.0),
            "unit_alignment_match_tgt": ("float", 0.0),
            "unit_alignment_cost_tgt": ("float", 0.0),
            "unit_alignment_unmatched_speech_ratio_tgt": ("float", 0.0),
            "unit_alignment_mean_local_confidence_speech_tgt": ("float", 0.0),
            "unit_alignment_mean_coarse_confidence_speech_tgt": ("float", 0.0),
            "unit_alignment_projection_hard_bad_tgt": ("float", 0.0),
            "unit_alignment_projection_hard_bad_source_tgt": ("float", 0.0),
            "unit_alignment_projection_violation_count_tgt": ("float", 0.0),
            "unit_alignment_projection_fallback_ratio_tgt": ("float", 0.0),
            "unit_alignment_global_path_cost_tgt": ("float", 0.0),
            "unit_alignment_global_path_mean_cost_tgt": ("float", 0.0),
            "unit_alignment_projection_gate_reason_tgt": ("object", None),
            "unit_alignment_projection_health_bucket_tgt": ("object", None),
            "unit_alignment_mode_id_tgt": ("long", 0),
            "unit_alignment_kind_tgt": ("object", None),
            "unit_alignment_source_tgt": ("object", None),
            "unit_alignment_version_tgt": ("object", None),
            "alignment_source": ("object", None),
            "alignment_version": ("object", None),
            "ref_phrase_trace": ("float", 0.0),
            "planner_ref_phrase_trace": ("float", 0.0),
            "ref_phrase_valid": ("float", 0.0),
            "ref_phrase_lengths": ("long", 0),
            "ref_phrase_starts": ("long", 0),
            "ref_phrase_ends": ("long", 0),
            "ref_phrase_boundary_strength": ("float", 0.0),
            "ref_phrase_stats": ("float", 0.0),
            "slow_rhythm_memory": ("float", 0.0),
            "slow_rhythm_summary": ("float", 0.0),
            "planner_slow_rhythm_memory": ("float", 0.0),
            "planner_slow_rhythm_summary": ("float", 0.0),
            "selector_meta_indices": ("long", 0),
            "selector_meta_scores": ("float", 0.0),
            "selector_meta_starts": ("long", 0),
            "selector_meta_ends": ("long", 0),
            "rhythm_cache_version": ("long", 0),
            "rhythm_unit_hop_ms": ("long", 0),
            "rhythm_trace_hop_ms": ("long", 0),
            "rhythm_trace_bins": ("long", 0),
            "rhythm_trace_horizon": ("float", 0.0),
            "rhythm_slow_topk": ("long", 0),
            "rhythm_selector_cell_size": ("long", 0),
            "rhythm_source_phrase_threshold": ("float", 0.0),
            "rhythm_reference_mode_id": ("long", 0),
            "rhythm_target_confidence": ("float", 0.0),
            "rhythm_guidance_confidence": ("float", 0.0),
            "rhythm_teacher_confidence": ("float", 0.0),
            "rhythm_teacher_confidence_exec": ("float", 0.0),
            "rhythm_teacher_confidence_budget": ("float", 0.0),
            "rhythm_teacher_confidence_prefix": ("float", 0.0),
            "rhythm_teacher_confidence_allocation": ("float", 0.0),
            "rhythm_teacher_confidence_shape": ("float", 0.0),
            "rhythm_teacher_target_source_id": ("long", 0),
            "rhythm_retimed_target_source_id": ("long", 0),
            "rhythm_retimed_target_confidence": ("float", 0.0),
            "rhythm_speech_exec_tgt": ("float", 0.0),
            "rhythm_pause_exec_tgt": ("float", 0.0),
            "rhythm_speech_budget_tgt": ("float", 0.0),
            "rhythm_pause_budget_tgt": ("float", 0.0),
            "rhythm_guidance_speech_tgt": ("float", 0.0),
            "rhythm_guidance_pause_tgt": ("float", 0.0),
            "rhythm_teacher_speech_exec_tgt": ("float", 0.0),
            "rhythm_teacher_pause_exec_tgt": ("float", 0.0),
            "rhythm_teacher_speech_budget_tgt": ("float", 0.0),
            "rhythm_teacher_pause_budget_tgt": ("float", 0.0),
            "rhythm_teacher_allocation_tgt": ("float", 0.0),
            "rhythm_teacher_prefix_clock_tgt": ("float", 0.0),
            "rhythm_teacher_prefix_backlog_tgt": ("float", 0.0),
            "rhythm_retimed_mel_tgt": ("float", 0.0),
            "rhythm_retimed_mel_len": ("long", 0),
            "rhythm_retimed_frame_weight": ("float", 0.0),
            "rhythm_stream_prefix_ratio": ("float", 0.0),
            "rhythm_stream_visible_units": ("float", 0.0),
            "rhythm_stream_full_units": ("float", 0.0),
            "rhythm_reference_is_self": ("float", 0.0),
            "rhythm_pair_group_id": ("long", 0),
            "rhythm_pair_rank": ("long", 0),
            "rhythm_pair_is_identity": ("float", 0.0),
            "rhythm_offline_teacher_speech_exec_tgt": ("float", 0.0),
            "rhythm_offline_teacher_pause_exec_tgt": ("float", 0.0),
            "rhythm_offline_teacher_speech_budget_tgt": ("float", 0.0),
            "rhythm_offline_teacher_pause_budget_tgt": ("float", 0.0),
            "rhythm_offline_teacher_confidence": ("float", 0.0),
        }

    def _resolve_optional_sample_keys(self) -> tuple[str, ...]:
        if self._use_duration_v3_dataset_contract():
            keys = list(
                self._RHYTHM_RUNTIME_MINIMAL_KEYS
                + self._RHYTHM_RUNTIME_REFERENCE_META_KEYS
            )
        else:
            keys = list(
                self._RHYTHM_RUNTIME_MINIMAL_KEYS
                + self._RHYTHM_REF_CACHE_KEYS
                + self._RHYTHM_RUNTIME_REFERENCE_META_KEYS
                + self._resolve_runtime_target_export_keys()
            )
        if self._should_export_streaming_prefix_meta():
            keys.extend(self._RHYTHM_STREAMING_PREFIX_META_KEYS)
        if self._should_export_streaming_offline_sidecars():
            keys.extend(self._RHYTHM_STREAMING_OFFLINE_SOURCE_KEYS)
            if self._should_export_offline_teacher_aux():
                keys.extend(self._RHYTHM_STREAMING_OFFLINE_TEACHER_AUX_KEYS)
        if self._should_export_runtime_phrase_bank_sidecars():
            keys.extend(self._RHYTHM_REF_PHRASE_CACHE_KEYS)
        if self._should_export_rhythm_debug_sidecars():
            keys.extend(self._RHYTHM_DEBUG_SIDECAR_KEYS)
        if self._should_export_rhythm_cache_audit():
            keys.extend(self._RHYTHM_CACHE_AUDIT_KEYS)
        return tuple(dict.fromkeys(keys))

    def _select_streaming_visible_tokens(self, visible_tokens, *, item_name: str):
        full_tokens = np.asarray(visible_tokens).reshape(-1)
        full_len = int(full_tokens.shape[0])
        if not self._should_sample_streaming_prefix() or full_len <= 1:
            return full_tokens
        min_ratio = float(self.hparams.get("rhythm_streaming_prefix_min_ratio", 0.5))
        max_ratio = float(self.hparams.get("rhythm_streaming_prefix_max_ratio", 0.9))
        min_ratio = max(0.05, min(0.99, min_ratio))
        max_ratio = max(min_ratio, min(0.99, max_ratio))
        min_tokens = max(1, int(self.hparams.get("rhythm_streaming_prefix_min_tokens", 8)))
        multiple = max(1, int(self.hparams.get("rhythm_streaming_prefix_multiple", 1)))
        max_prefix = max(min(full_len - 1, full_len), 1)
        if max_prefix <= min_tokens:
            return full_tokens
        if bool(self.hparams.get("rhythm_streaming_prefix_deterministic", True)):
            key = f"{item_name}|{self.prefix}|{int(self.hparams.get('seed', 0))}"
            digest = hashlib.md5(key.encode("utf-8")).hexdigest()
            sample_u = int(digest[:8], 16) / float(0xFFFFFFFF)
        else:
            sample_u = float(np.random.uniform(0.0, 1.0))
        ratio = min_ratio + (max_ratio - min_ratio) * sample_u
        target_len = int(round(full_len * ratio))
        target_len = max(min_tokens, min(full_len - 1, target_len))
        if multiple > 1:
            target_len = max(min_tokens, (target_len // multiple) * multiple)
        target_len = max(min_tokens, min(full_len - 1, target_len))
        if target_len >= full_len:
            return full_tokens
        return full_tokens[:target_len]

    @staticmethod
    def _is_enabled_flag(value) -> bool:
        if value is None:
            return False
        if isinstance(value, (int, float, np.floating, np.integer)):
            return float(value) > 0.0
        return bool(value)

    def _make_item_generator(self, *, item_name: str, salt: str) -> torch.Generator | None:
        deterministic = bool(
            self.hparams.get(
                "rhythm_augmentation_deterministic",
                self.hparams.get("rhythm_streaming_prefix_deterministic", True),
            )
        )
        if not deterministic:
            return None
        seed = int(self.hparams.get("seed", 0) or 0)
        digest = hashlib.md5(f"{item_name}|{self.prefix}|{salt}|{seed}".encode("utf-8")).hexdigest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(digest[:8], 16))
        return generator

    @staticmethod
    def _copy_numpy_fields(payload: dict) -> dict:
        copied = {}
        for key, value in payload.items():
            array = np.asarray(value)
            copied[key] = array.copy() if isinstance(array, np.ndarray) else value
        return copied

    @staticmethod
    def _coerce_optional_local_item_id(value) -> int | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() <= 0:
                return None
            value = value.reshape(-1)[0].item()
        elif isinstance(value, np.ndarray):
            if value.size <= 0:
                return None
            scalar = value.reshape(-1)[0]
            value = scalar.item() if hasattr(scalar, "item") else scalar
        elif isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
        local_idx = int(value)
        return local_idx if local_idx >= 0 else None

    @staticmethod
    def _clone_cached_raw_item(item):
        return dict(item) if isinstance(item, dict) else item

    def _get_raw_item_cached(self, local_idx: int):
        local_idx = int(local_idx)
        cache = getattr(self, "_rhythm_raw_item_cache", None)
        if cache is None:
            cache = {}
            self._rhythm_raw_item_cache = cache
        item = cache.get(local_idx)
        if item is None:
            item = self._get_item(local_idx)
            cache[local_idx] = item
        return self._clone_cached_raw_item(item)

    def _require_source_local_item_id(self, sample: dict) -> int:
        local_idx = self._coerce_optional_local_item_id(sample.get("item_id"))
        if local_idx is None:
            raise RuntimeError(
                "Rhythm dataset received no _raw_item from the upstream sample and no explicit item_id. "
                "Refusing to fall back to the dataset index because pair-manifest/filter/remap datasets can "
                "make the ordinal index differ from the source local item id."
            )
        return local_idx

    @staticmethod
    def _prefix_source_cache(cache: dict, *, prefix: str) -> dict:
        return {f"{prefix}{key}": value for key, value in cache.items()}

    @staticmethod
    def _coerce_content_sequence(content) -> list[int]:
        if isinstance(content, str):
            return [int(float(x)) for x in content.split()]
        arr = np.asarray(content).reshape(-1)
        return [int(x) for x in arr.tolist()]

    def _adapt_source_cache_to_visible_prefix(self, *, item, visible_tokens) -> dict:
        cache = {key: np.asarray(item[key]) for key in self._RHYTHM_SOURCE_CACHE_KEYS if key in item}
        if len(cache) != len(self._RHYTHM_SOURCE_CACHE_KEYS):
            missing = [key for key in self._RHYTHM_SOURCE_CACHE_KEYS if key not in cache]
            raise RuntimeError(
                f"Rhythm cached_only requires full source cache before prefix adaptation. Missing keys in "
                f"{item.get('item_name', '<unknown-item>')}: {missing}"
            )
        remaining = int(np.asarray(visible_tokens).reshape(-1).shape[0])
        full_units = np.asarray(cache["content_units"]).reshape(-1)
        full_durations = np.asarray(cache["dur_anchor_src"]).reshape(-1)
        full_sep = np.asarray(cache["sep_hint"]).reshape(-1)
        full_silence = (
            np.asarray(item["source_silence_mask"], dtype=np.float32).reshape(-1)
            if "source_silence_mask" in item
            else None
        )
        out_units = []
        out_durations = []
        out_silence = []
        out_sep = []
        silence_iter = full_silence if full_silence is not None else np.zeros_like(full_durations, dtype=np.float32)
        for unit_id, duration, sep_flag, silence_flag in zip(full_units, full_durations, full_sep, silence_iter):
            if remaining <= 0:
                break
            take = int(min(int(duration), remaining))
            out_units.append(int(unit_id))
            out_durations.append(take)
            out_silence.append(float(silence_flag))
            out_sep.append(int(sep_flag) if take == int(duration) else 0)
            remaining -= take
        if remaining > 0:
            raise RuntimeError(
                f"Visible prefix in {item.get('item_name', '<unknown-item>')} exceeds cached source duration. "
                "Re-binarize the dataset."
            )
        visible_total = int(np.asarray(visible_tokens).reshape(-1).shape[0])
        full_total = int(np.asarray(full_durations).reshape(-1).sum())
        is_truncated_prefix = visible_total < full_total
        if is_truncated_prefix and len(out_sep) > 0:
            out_sep[-1] = 0
        open_run_mask = np.zeros((len(out_units),), dtype=np.int64)
        sealed_mask = np.ones((len(out_units),), dtype=np.int64)
        if is_truncated_prefix and len(out_units) > 0:
            tail_open_units = int(self.hparams.get("rhythm_prefix_tail_open_units", 1) or 0)
            tail_open_units = max(1, min(len(out_units), tail_open_units))
            open_run_mask[-tail_open_units:] = 1
            sealed_mask[-tail_open_units:] = 0
        boundary_confidence = np.asarray(
            estimate_boundary_confidence(out_durations, out_sep, open_run_mask.tolist()),
            dtype=np.float32,
        )
        run_stability = np.asarray(
            estimate_run_stability(
                out_durations,
                out_silence,
                open_run_mask.tolist(),
                sep_hint=out_sep,
                boundary_confidence=boundary_confidence.tolist(),
                min_speech_frames=int(self.hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
                min_silence_frames=int(self.hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
            ),
            dtype=np.float32,
        )
        adapted = {
            "content_units": np.asarray(out_units, dtype=np.int64),
            "dur_anchor_src": np.asarray(out_durations, dtype=np.int64),
            "open_run_mask": open_run_mask,
            "sealed_mask": sealed_mask,
            "sep_hint": np.asarray(out_sep, dtype=np.int64),
            "boundary_confidence": boundary_confidence,
            "source_run_stability": run_stability,
        }
        full_side = {key: np.asarray(item[key]) for key in self._RHYTHM_SOURCE_DEBUG_CACHE_KEYS if key in item}
        adapted["source_boundary_cue"] = self._restore_debug_sidecar(
            full_side,
            "source_boundary_cue",
            boundary_confidence.astype(np.float32, copy=False),
        )
        cache_meta = item.get(DURATION_V3_CACHE_META_KEY) if isinstance(item, Mapping) else None
        if isinstance(cache_meta, Mapping):
            adapted[DURATION_V3_CACHE_META_KEY] = dict(cache_meta)
        if full_silence is not None:
            adapted["source_silence_mask"] = np.asarray(out_silence, dtype=np.float32)
        if self._should_export_rhythm_debug_sidecars():
            default_phrase_group_index = np.zeros_like(adapted["content_units"], dtype=np.int64)
            default_phrase_group_pos = np.zeros_like(boundary_confidence, dtype=np.float32)
            default_phrase_final_mask = np.zeros_like(boundary_confidence, dtype=np.float32)
            adapted.update({
                "phrase_group_index": self._restore_debug_sidecar(
                    full_side,
                    "phrase_group_index",
                    default_phrase_group_index,
                ),
                "phrase_group_pos": self._restore_debug_sidecar(
                    full_side,
                    "phrase_group_pos",
                    default_phrase_group_pos,
                ),
                "phrase_final_mask": self._restore_debug_sidecar(
                    full_side,
                    "phrase_final_mask",
                    default_phrase_final_mask,
                ),
            })
        return adapted

    def _expected_rhythm_cache_version(self) -> int:
        return int(self.hparams.get("rhythm_cache_version", RHYTHM_CACHE_VERSION))

    def _materialize_rhythm_cache_compat(self, item, *, item_name: str):
        return self._rhythm_cache_contract().materialize_rhythm_cache_compat(
            item,
            item_name=item_name,
        )

    @staticmethod
    def _extract_scalar(value):
        return RhythmDatasetCacheContract.extract_scalar(value)

    @staticmethod
    def _missing_keys(item, keys):
        return RhythmDatasetCacheContract.missing_keys(item, keys)

    def _require_cached_keys(self, *, item, keys, item_name: str, reason: str):
        self._rhythm_cache_contract().require_cached_keys(
            item=item,
            keys=keys,
            item_name=item_name,
            reason=reason,
        )

    def _validate_rhythm_cache_version(self, item, *, item_name: str):
        self._rhythm_cache_contract().validate_rhythm_cache_version(item, item_name=item_name)

    def _validate_rhythm_cache_contract(self, item, *, item_name: str):
        self._rhythm_cache_contract().validate_rhythm_cache_contract(item, item_name=item_name)

    def _required_cached_target_keys(self):
        return self._rhythm_cache_contract().required_cached_target_keys()

    def _validate_retimed_cache_contract(self, item, *, item_name: str):
        self._rhythm_cache_contract().validate_retimed_cache_contract(item, item_name=item_name)

    def _validate_source_cache_shapes(self, cache, *, item_name: str):
        self._rhythm_cache_contract().validate_source_cache_shapes(cache, item_name=item_name)

    def _validate_reference_conditioning_shapes(self, conditioning, *, item_name: str):
        self._rhythm_cache_contract().validate_reference_conditioning_shapes(
            conditioning,
            item_name=item_name,
        )

    def _validate_target_shapes(self, targets, *, item_name: str, expected_units: int):
        self._rhythm_cache_contract().validate_target_shapes(
            targets,
            item_name=item_name,
            expected_units=expected_units,
        )

    def _should_use_self_rhythm_reference(self, item, *, target_mode: str) -> bool:
        policy = str(self.hparams.get("rhythm_cached_reference_policy", "self") or "self").strip().lower()
        if policy in {"sample_ref", "paired", "external"}:
            return False
        has_cached = all(key in item for key in self._RHYTHM_REF_CACHE_KEYS) and any(
            key in item for key in ("rhythm_speech_exec_tgt", "rhythm_pause_exec_tgt")
        )
        return target_mode != "runtime_only" and has_cached

    def _adapt_cached_targets_to_prefix(self, *, item, cached_targets, source_cache, sample):
        return self._rhythm_target_builder().adapt_cached_targets_to_prefix(
            item=item,
            cached_targets=cached_targets,
            source_cache=source_cache,
            sample=sample,
        )

    def _get_source_rhythm_cache(self, item, visible_tokens, *, target_mode: str):
        cache_keys = self._RHYTHM_SOURCE_CACHE_KEYS
        full_tokens = np.asarray(item["hubert"])
        visible_tokens = np.asarray(visible_tokens)
        explicit_silence = self._should_emit_explicit_silence_runs()
        has_source_silence_cache = "source_silence_mask" in item
        if all(key in item for key in cache_keys) and (not explicit_silence or has_source_silence_cache):
            cache = {key: item[key] for key in cache_keys}
            cache_meta = item.get(DURATION_V3_CACHE_META_KEY) if isinstance(item, Mapping) else None
            if isinstance(cache_meta, Mapping):
                cache[DURATION_V3_CACHE_META_KEY] = dict(cache_meta)
            if has_source_silence_cache:
                cache["source_silence_mask"] = item["source_silence_mask"]
            if "source_run_stability" in item:
                cache["source_run_stability"] = item["source_run_stability"]
            if "source_boundary_cue" in item:
                cache["source_boundary_cue"] = item["source_boundary_cue"]
            for extra_key in ("phrase_group_index", "phrase_group_pos", "phrase_final_mask"):
                if extra_key in item:
                    cache[extra_key] = item[extra_key]
            self._validate_source_cache_shapes(
                cache,
                item_name=str(item.get("item_name", "<unknown-item>")),
            )
            if target_mode == "cached_only":
                item_name = str(item.get("item_name", "<unknown-item>"))
                self._validate_rhythm_cache_contract(item, item_name=item_name)
                self._require_cached_keys(
                    item=item,
                    keys=cache_keys,
                    item_name=item_name,
                    reason="source rhythm cache",
                )
                self._validate_source_cache_shapes(cache, item_name=item_name)
            if int(full_tokens.shape[0]) == int(visible_tokens.shape[0]):
                return cache
        if target_mode == "cached_only":
            if explicit_silence and not has_source_silence_cache:
                raise RuntimeError(
                    "rhythm_v3 cached_only with explicit silence-run frontend requires source_silence_mask in the cached source bundle. "
                    "Re-binarize with explicit silence runs enabled."
                )
            return self._adapt_source_cache_to_visible_prefix(item=item, visible_tokens=visible_tokens)
        return build_source_rhythm_cache(
            visible_tokens,
            silent_token=self.hparams.get("silent_token", 57),
            separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
            tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
            emit_silence_runs=explicit_silence,
            debounce_min_run_frames=int(self.hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
            phrase_boundary_threshold=float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
            mel=(
                None
                if item.get("mel") is None
                else np.asarray(item["mel"])[: int(np.asarray(visible_tokens).reshape(-1).shape[0])]
            ),
        )

    def _resolve_reference_rhythm_item(self, *, sample, item, target_mode: str):
        ref_local = self._coerce_optional_local_item_id(sample.get("ref_item_id"))
        if ref_local is None or self._should_use_self_rhythm_reference(item, target_mode=target_mode):
            sample.pop("_raw_ref_item", None)
            return None
        raw_ref_item = sample.pop("_raw_ref_item", None)
        if raw_ref_item is None:
            raw_ref_item = self._get_raw_item_cached(ref_local)
        return self._materialize_rhythm_cache_compat(
            raw_ref_item,
            item_name=str(raw_ref_item.get("item_name", "<unknown-ref-item>")),
        )

    def _resolve_paired_target_rhythm_item(self, *, sample, item, target_mode: str):
        del item, target_mode
        paired_target_local = self._coerce_optional_local_item_id(sample.get("paired_target_item_id"))
        if paired_target_local is None:
            sample.pop("_raw_paired_target_item", None)
            return None
        raw_paired_target_item = sample.pop("_raw_paired_target_item", None)
        if raw_paired_target_item is None:
            raw_paired_target_item = self._get_raw_item_cached(paired_target_local)
        return self._materialize_rhythm_cache_compat(
            raw_paired_target_item,
            item_name=str(raw_paired_target_item.get("item_name", "<unknown-paired-target-item>")),
        )

    def _get_reference_rhythm_conditioning(self, ref_item, sample, *, target_mode: str, item=None):
        if (
            self._use_duration_v3_dataset_contract()
            and self._disallow_same_text_reference()
            and item is not None
            and ref_item is None
        ):
            raise RuntimeError(
                "rhythm_v3 prompt conditioning resolved to self-reference, which is disabled by default for cross-text training. "
                "Provide an external different-text reference or set rhythm_v3_disallow_same_text_reference=false."
            )
        if (
            self._use_duration_v3_dataset_contract()
            and self._disallow_same_text_reference()
            and item is not None
            and ref_item is not None
        ):
            item_sig = self._rhythm_text_signature(item)
            ref_sig = self._rhythm_text_signature(ref_item)
            if item_sig is None or ref_sig is None:
                raise RuntimeError(
                    "rhythm_v3 reference conditioning requires comparable source/reference text signatures "
                    "when same-text references are disabled. Provide explicit text tokens/text strings "
                    "or set rhythm_v3_disallow_same_text_reference=false."
                )
            if item_sig == ref_sig:
                raise RuntimeError(
                    "rhythm_v3 reference conditioning forbids same-text reference items by default. "
                    "Provide a different-text reference or set rhythm_v3_disallow_same_text_reference=false."
                )
        prompt_source = ref_item if ref_item is not None else item
        if self._use_duration_v3_dataset_contract():
            self._require_minimal_v1_prompt_sidecars(
                prompt_source,
                item_name=(
                    str(prompt_source.get("item_name", "<unknown-prompt-item>"))
                    if isinstance(prompt_source, dict)
                    else "<unknown-prompt-item>"
                ),
            )
        prompt_conditioning = self._build_reference_prompt_unit_conditioning(
            prompt_source,
            target_mode=target_mode,
        )
        if not self._use_duration_v3_dataset_contract():
            raise RuntimeError("This repository is sealed to the rhythm_v3 V1 dataset path.")
        return prompt_conditioning

    def _build_paired_target_rhythm_conditioning(self, paired_target_item, sample, *, target_mode: str, item=None):
        del sample
        if not self._use_duration_v3_dataset_contract():
            return {}
        builder = getattr(self, "_build_paired_target_projection_conditioning", None)
        if not callable(builder):
            return {}
        return builder(
            paired_target_item,
            target_mode=target_mode,
            source_item=item,
        )

    def _merge_rhythm_targets(self, item, source_cache, ref_conditioning, paired_target_conditioning, sample):
        if not self._use_duration_v3_dataset_contract():
            raise RuntimeError("This repository is sealed to the rhythm_v3 V1 dataset path.")
        return self._merge_duration_v3_rhythm_targets(item, source_cache, paired_target_conditioning, sample)

    def __getitem__(self, index):
        max_prefilter_attempts = 0
        resolve_attempts = getattr(self, "_alignment_prefilter_max_attempts", None)
        if callable(resolve_attempts):
            max_prefilter_attempts = max(0, int(resolve_attempts() or 0))
        candidate_index = int(index)
        attempts = 0
        while True:
            try:
                sample = super().__getitem__(candidate_index)
                raw_item = sample.pop("_raw_item", None)
                if raw_item is None:
                    raw_item = self._get_raw_item_cached(self._require_source_local_item_id(sample))
                item_name = str(raw_item.get("item_name", "<unknown-item>"))
                item = self._materialize_rhythm_cache_compat(raw_item, item_name=item_name)
                target_mode = self._resolve_rhythm_target_mode()
                ref_item = self._resolve_reference_rhythm_item(
                    sample=sample,
                    item=item,
                    target_mode=target_mode,
                )
                return self._rhythm_sample_assembler().assemble(
                    sample=sample,
                    item=item,
                    ref_item=ref_item,
                    item_name=item_name,
                )
            except RhythmDatasetPrefilterDrop as exc:
                if attempts >= max_prefilter_attempts:
                    raise RuntimeError(
                        "Failed to fetch rhythm sample after bounded bad-alignment prefilter retries: "
                        f"start_index={int(index)}, attempts={attempts + 1}, last_error={exc}"
                    ) from exc
                next_index = self._sample_prefilter_retry_index(candidate_index)
                if next_index == candidate_index:
                    raise RuntimeError(
                        "Bad-alignment prefilter could not select an alternate dataset index for retry."
                    ) from exc
                attempts += 1
                candidate_index = next_index

    def _collate_optional_rhythm_fields(self, *, batch: dict, samples) -> None:
        optional_collate = self._build_optional_collate_spec()
        optional_keys = self._resolve_optional_sample_keys()
        for key in optional_keys:
            if key not in optional_collate:
                continue
            dtype_name, pad_value = optional_collate[key]
            present_count = sum(1 for sample in samples if key in sample)
            if 0 < present_count < len(samples):
                raise RuntimeError(
                    f"Rhythm batch contains partial optional field '{key}' "
                    f"({present_count}/{len(samples)} samples). Re-binarize or fix the cache contract."
                )
            if present_count != len(samples):
                continue
            if dtype_name == "object":
                batch[key] = [sample[key] for sample in samples]
                continue
            value = collate_1d_or_2d([sample[key] for sample in samples], pad_value)
            batch[key] = value.long() if dtype_name == "long" else value.float()

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super().collater(samples)
        batch["content"] = collate_1d_or_2d([sample["content"] for sample in samples], 0).long()
        batch["content_lengths"] = torch.tensor(
            [sample["content"].shape[0] for sample in samples],
            dtype=torch.long,
        )
        self._collate_optional_rhythm_fields(batch=batch, samples=samples)
        return batch


__all__ = ["CommonRhythmDatasetMixin"]

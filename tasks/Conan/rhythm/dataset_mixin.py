"""Rhythm-specific Conan dataset logic extracted from tasks/Conan/dataset.py.

This mixin keeps the public ConanDataset entrypoint thin while preserving the
existing runtime/cache contract.
"""


import hashlib
import torch
from utils.commons.dataset_utils import collate_1d_or_2d
import numpy as np
from tasks.Conan.rhythm.dataset_contracts import RhythmDatasetCacheContract
from tasks.Conan.rhythm.dataset_sample_builder import RhythmDatasetSampleAssembler
from tasks.Conan.rhythm.dataset_target_builder import RhythmDatasetTargetBuilder
from modules.Conan.rhythm.supervision import (
    RHYTHM_CACHE_VERSION,
    build_reference_rhythm_conditioning,
    build_source_rhythm_cache,
)
from modules.Conan.rhythm.policy import build_rhythm_hparams_policy
from modules.Conan.rhythm.unitizer import estimate_boundary_confidence

class RhythmConanDatasetMixin:
    _RHYTHM_SOURCE_CACHE_KEYS = (
        "content_units",
        "dur_anchor_src",
        "open_run_mask",
        "sealed_mask",
        "sep_hint",
        "boundary_confidence",
    )
    _RHYTHM_SOURCE_DEBUG_CACHE_KEYS = (
        "source_boundary_cue",
        "phrase_group_index",
        "phrase_group_pos",
        "phrase_final_mask",
    )
    _RHYTHM_REF_CACHE_KEYS = (
        "ref_rhythm_stats",
        "ref_rhythm_trace",
    )
    _RHYTHM_REF_DEBUG_CACHE_KEYS = (
        "slow_rhythm_memory",
        "slow_rhythm_summary",
        "selector_meta_indices",
        "selector_meta_scores",
        "selector_meta_starts",
        "selector_meta_ends",
    )
    _RHYTHM_REF_PLANNER_DEBUG_CACHE_KEYS = (
        "planner_slow_rhythm_memory",
        "planner_slow_rhythm_summary",
    )
    _RHYTHM_TARGET_KEYS = (
        "rhythm_speech_exec_tgt",
        "rhythm_pause_exec_tgt",
        "rhythm_blank_exec_tgt",
        "rhythm_speech_budget_tgt",
        "rhythm_pause_budget_tgt",
        "rhythm_blank_budget_tgt",
        "rhythm_guidance_speech_tgt",
        "rhythm_guidance_pause_tgt",
        "rhythm_guidance_blank_tgt",
        "rhythm_teacher_speech_exec_tgt",
        "rhythm_teacher_pause_exec_tgt",
        "rhythm_teacher_blank_exec_tgt",
        "rhythm_teacher_speech_budget_tgt",
        "rhythm_teacher_pause_budget_tgt",
        "rhythm_teacher_blank_budget_tgt",
        "rhythm_teacher_allocation_tgt",
        "rhythm_teacher_prefix_clock_tgt",
        "rhythm_teacher_prefix_backlog_tgt",
    )
    _RHYTHM_BLANK_COMPAT_TARGET_KEYS = (
        "rhythm_blank_exec_tgt",
        "rhythm_blank_budget_tgt",
        "rhythm_guidance_blank_tgt",
        "rhythm_teacher_blank_exec_tgt",
        "rhythm_teacher_blank_budget_tgt",
    )
    _RHYTHM_META_KEYS = (
        "rhythm_cache_version",
        "rhythm_unit_hop_ms",
        "rhythm_trace_hop_ms",
        "rhythm_trace_bins",
        "rhythm_trace_horizon",
        "rhythm_slow_topk",
        "rhythm_selector_cell_size",
        "rhythm_source_phrase_threshold",
        "rhythm_reference_mode_id",
        "rhythm_target_confidence",
        "rhythm_guidance_confidence",
        "rhythm_teacher_confidence",
        "rhythm_teacher_confidence_exec",
        "rhythm_teacher_confidence_budget",
        "rhythm_teacher_confidence_prefix",
        "rhythm_teacher_confidence_allocation",
        "rhythm_teacher_confidence_shape",
        "rhythm_teacher_target_source_id",
        "rhythm_retimed_target_source_id",
        "rhythm_retimed_target_confidence",
    )
    # Keep the batch schema layered:
    #   1) runtime-minimal contract: maintained timing path
    #   2) runtime targets: maintained supervision surfaces
    #   3) optional streaming sidecars: only when a stage actually needs them
    #   4) debug/cache audit appendices: opt-in only
    _RHYTHM_RUNTIME_MINIMAL_KEYS = (
        "content_units",
        "dur_anchor_src",
        "open_run_mask",
        "sealed_mask",
        "sep_hint",
        "boundary_confidence",
        "ref_rhythm_stats",
        "ref_rhythm_trace",
    )
    _RHYTHM_STREAMING_PREFIX_META_KEYS = (
        "rhythm_stream_prefix_ratio",
        "rhythm_stream_visible_units",
        "rhythm_stream_full_units",
    )
    _RHYTHM_STREAMING_OFFLINE_SOURCE_KEYS = (
        "rhythm_offline_content_units",
        "rhythm_offline_dur_anchor_src",
        "rhythm_offline_open_run_mask",
        "rhythm_offline_sealed_mask",
        "rhythm_offline_sep_hint",
        "rhythm_offline_boundary_confidence",
    )
    _RHYTHM_STREAMING_OFFLINE_TEACHER_AUX_KEYS = (
        "rhythm_offline_teacher_speech_exec_tgt",
        "rhythm_offline_teacher_pause_exec_tgt",
        "rhythm_offline_teacher_speech_budget_tgt",
        "rhythm_offline_teacher_pause_budget_tgt",
        "rhythm_offline_teacher_confidence",
    )
    _RHYTHM_DEBUG_SIDECAR_KEYS = (
        "source_boundary_cue",
        "phrase_group_index",
        "phrase_group_pos",
        "phrase_final_mask",
        "rhythm_offline_source_boundary_cue",
        "rhythm_offline_phrase_group_index",
        "rhythm_offline_phrase_group_pos",
        "rhythm_offline_phrase_final_mask",
        "slow_rhythm_memory",
        "slow_rhythm_summary",
        "planner_slow_rhythm_memory",
        "planner_slow_rhythm_summary",
        "selector_meta_indices",
        "selector_meta_scores",
        "selector_meta_starts",
        "selector_meta_ends",
    )
    # Public/runtime batch contract prefers pause-* naming. Keep blank-* only as
    # cache/backward-compat aliases inside cached target validation / adaptation.
    _RHYTHM_RUNTIME_TARGET_CORE_KEYS = (
        "rhythm_speech_exec_tgt",
        "rhythm_pause_exec_tgt",
        "rhythm_speech_budget_tgt",
        "rhythm_pause_budget_tgt",
        "rhythm_target_confidence",
    )
    _RHYTHM_RUNTIME_GUIDANCE_KEYS = (
        "rhythm_guidance_speech_tgt",
        "rhythm_guidance_pause_tgt",
        "rhythm_guidance_confidence",
    )
    _RHYTHM_RUNTIME_TEACHER_CORE_KEYS = (
        "rhythm_teacher_speech_exec_tgt",
        "rhythm_teacher_pause_exec_tgt",
        "rhythm_teacher_speech_budget_tgt",
        "rhythm_teacher_pause_budget_tgt",
        "rhythm_teacher_confidence",
    )
    _RHYTHM_RUNTIME_TEACHER_CONFIDENCE_COMPONENT_KEYS = (
        "rhythm_teacher_confidence_exec",
        "rhythm_teacher_confidence_budget",
        "rhythm_teacher_confidence_prefix",
        "rhythm_teacher_confidence_allocation",
        "rhythm_teacher_confidence_shape",
    )
    _RHYTHM_RUNTIME_TEACHER_ALLOCATION_KEYS = (
        "rhythm_teacher_allocation_tgt",
    )
    _RHYTHM_RUNTIME_TEACHER_PREFIX_KEYS = (
        "rhythm_teacher_prefix_clock_tgt",
        "rhythm_teacher_prefix_backlog_tgt",
    )
    _RHYTHM_RUNTIME_RETIMED_KEYS = (
        "rhythm_retimed_mel_tgt",
        "rhythm_retimed_mel_len",
        "rhythm_retimed_frame_weight",
        "rhythm_retimed_target_confidence",
    )
    _RHYTHM_CACHE_AUDIT_KEYS = _RHYTHM_META_KEYS

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
    def _build_optional_collate_spec() -> dict[str, tuple[str, float | int]]:
        return {
            "content_units": ("long", 0),
            "dur_anchor_src": ("long", 0),
            "open_run_mask": ("long", 0),
            "sealed_mask": ("float", 0.0),
            "sep_hint": ("long", 0),
            "boundary_confidence": ("float", 0.0),
            "source_boundary_cue": ("float", 0.0),
            "phrase_group_index": ("long", 0),
            "phrase_group_pos": ("float", 0.0),
            "phrase_final_mask": ("float", 0.0),
            "rhythm_offline_content_units": ("long", 0),
            "rhythm_offline_dur_anchor_src": ("long", 0),
            "rhythm_offline_open_run_mask": ("long", 0),
            "rhythm_offline_sealed_mask": ("float", 0.0),
            "rhythm_offline_sep_hint": ("long", 0),
            "rhythm_offline_boundary_confidence": ("float", 0.0),
            "rhythm_offline_source_boundary_cue": ("float", 0.0),
            "rhythm_offline_phrase_group_index": ("long", 0),
            "rhythm_offline_phrase_group_pos": ("float", 0.0),
            "rhythm_offline_phrase_final_mask": ("float", 0.0),
            "ref_rhythm_stats": ("float", 0.0),
            "ref_rhythm_trace": ("float", 0.0),
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
            "rhythm_offline_teacher_speech_exec_tgt": ("float", 0.0),
            "rhythm_offline_teacher_pause_exec_tgt": ("float", 0.0),
            "rhythm_offline_teacher_speech_budget_tgt": ("float", 0.0),
            "rhythm_offline_teacher_pause_budget_tgt": ("float", 0.0),
            "rhythm_offline_teacher_confidence": ("float", 0.0),
        }

    def _resolve_optional_sample_keys(self) -> tuple[str, ...]:
        keys = list(self._RHYTHM_RUNTIME_MINIMAL_KEYS + self._resolve_runtime_target_export_keys())
        if self._should_export_streaming_prefix_meta():
            keys.extend(self._RHYTHM_STREAMING_PREFIX_META_KEYS)
        if self._should_export_streaming_offline_sidecars():
            keys.extend(self._RHYTHM_STREAMING_OFFLINE_SOURCE_KEYS)
            if self._should_export_offline_teacher_aux():
                keys.extend(self._RHYTHM_STREAMING_OFFLINE_TEACHER_AUX_KEYS)
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
    def _prefix_source_cache(cache: dict, *, prefix: str) -> dict:
        return {f"{prefix}{key}": value for key, value in cache.items()}

    @staticmethod
    def _build_offline_teacher_aux_fields(item, *, offline_units: int) -> dict:
        speech_key = "rhythm_teacher_speech_exec_tgt"
        pause_key = "rhythm_teacher_pause_exec_tgt" if "rhythm_teacher_pause_exec_tgt" in item else "rhythm_teacher_blank_exec_tgt"
        if speech_key not in item or pause_key not in item:
            return {}
        speech = np.asarray(item[speech_key]).reshape(-1).astype(np.float32)
        pause = np.asarray(item[pause_key]).reshape(-1).astype(np.float32)
        teacher_units = min(int(offline_units), int(speech.shape[0]), int(pause.shape[0]))
        if teacher_units <= 0:
            return {}
        speech = speech[:teacher_units]
        pause = pause[:teacher_units]
        fields = {
            "rhythm_offline_teacher_speech_exec_tgt": speech,
            "rhythm_offline_teacher_pause_exec_tgt": pause,
            "rhythm_offline_teacher_speech_budget_tgt": np.asarray([float(speech.sum())], dtype=np.float32),
            "rhythm_offline_teacher_pause_budget_tgt": np.asarray([float(pause.sum())], dtype=np.float32),
        }
        if "rhythm_teacher_confidence" in item:
            fields["rhythm_offline_teacher_confidence"] = np.asarray(item["rhythm_teacher_confidence"]).reshape(-1)[:1].astype(np.float32)
        return fields

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
        out_units = []
        out_durations = []
        out_sep = []
        for unit_id, duration, sep_flag in zip(full_units, full_durations, full_sep):
            if remaining <= 0:
                break
            take = int(min(int(duration), remaining))
            out_units.append(int(unit_id))
            out_durations.append(take)
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
        adapted = {
            "content_units": np.asarray(out_units, dtype=np.int64),
            "dur_anchor_src": np.asarray(out_durations, dtype=np.int64),
            "open_run_mask": open_run_mask,
            "sealed_mask": sealed_mask,
            "sep_hint": np.asarray(out_sep, dtype=np.int64),
            "boundary_confidence": boundary_confidence,
        }
        if self._should_export_rhythm_debug_sidecars():
            full_side = {key: np.asarray(item[key]) for key in self._RHYTHM_SOURCE_DEBUG_CACHE_KEYS if key in item}
            adapted.update({
                "source_boundary_cue": boundary_confidence.copy() if "source_boundary_cue" in full_side else boundary_confidence.copy(),
                "phrase_group_index": np.zeros_like(adapted["content_units"], dtype=np.int64),
                "phrase_group_pos": np.zeros_like(boundary_confidence, dtype=np.float32),
                "phrase_final_mask": np.zeros_like(boundary_confidence, dtype=np.float32),
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
        if all(key in item for key in cache_keys):
            cache = {key: item[key] for key in cache_keys}
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
            return self._adapt_source_cache_to_visible_prefix(item=item, visible_tokens=visible_tokens)
        return build_source_rhythm_cache(
            visible_tokens,
            silent_token=self.hparams.get("silent_token", 57),
            separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
            tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
            phrase_boundary_threshold=float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
        )

    def _get_reference_rhythm_conditioning(self, ref_item, sample, *, target_mode: str):
        cache_keys = self._RHYTHM_REF_CACHE_KEYS
        if ref_item is not None and all(key in ref_item for key in cache_keys):
            conditioning = {key: ref_item[key] for key in cache_keys}
            for debug_key in self._RHYTHM_REF_DEBUG_CACHE_KEYS + self._RHYTHM_REF_PLANNER_DEBUG_CACHE_KEYS:
                if debug_key in ref_item:
                    conditioning[debug_key] = ref_item[debug_key]
            if target_mode == "cached_only":
                item_name = str(ref_item.get("item_name", "<unknown-ref-item>"))
                self._validate_rhythm_cache_contract(ref_item, item_name=item_name)
                self._require_cached_keys(
                    item=ref_item,
                    keys=cache_keys,
                    item_name=item_name,
                    reason="reference rhythm cache",
                )
            self._validate_reference_conditioning_shapes(
                conditioning,
                item_name=str(ref_item.get("item_name", "<unknown-ref-item>")),
            )
            return conditioning
        if target_mode == "cached_only":
            item_name = str(ref_item.get("item_name", "<unknown-ref-item>")) if ref_item is not None else "<missing-ref-item>"
            if ref_item is None:
                raise RuntimeError("Rhythm cached_only requires cached reference conditioning, but ref_item is missing.")
            raise RuntimeError(
                f"Rhythm cached_only requires cached reference conditioning in {item_name}. Re-binarize the dataset."
            )
        conditioning = build_reference_rhythm_conditioning(
            sample["ref_mel"],
            trace_bins=int(self.hparams.get("rhythm_trace_bins", 24)),
            trace_horizon=float(self.hparams.get("rhythm_trace_horizon", 0.35)),
            smooth_kernel=int(self.hparams.get("rhythm_trace_smooth_kernel", 5)),
            slow_topk=int(self.hparams.get("rhythm_slow_topk", 6)),
            selector_cell_size=int(self.hparams.get("rhythm_selector_cell_size", 3)),
        )
        self._validate_reference_conditioning_shapes(conditioning, item_name="<runtime-ref-conditioning>")
        return conditioning

    def _build_runtime_rhythm_targets(self, source_cache, ref_conditioning):
        return self._rhythm_target_builder().build_runtime_rhythm_targets(source_cache, ref_conditioning)

    def _merge_rhythm_targets(self, item, source_cache, ref_conditioning, sample):
        return self._rhythm_target_builder().merge_rhythm_targets(
            item=item,
            source_cache=source_cache,
            ref_conditioning=ref_conditioning,
            sample=sample,
        )

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        raw_item = self._get_item(index)
        item_name = str(raw_item.get("item_name", "<unknown-item>"))
        item = self._materialize_rhythm_cache_compat(raw_item, item_name=item_name)
        ref_item = None
        if "ref_item_id" in sample:
            raw_ref_item = self._get_item(int(sample["ref_item_id"]))
            ref_item = self._materialize_rhythm_cache_compat(
                raw_ref_item,
                item_name=str(raw_ref_item.get("item_name", "<unknown-ref-item>")),
            )
        return self._rhythm_sample_assembler().assemble(
            sample=sample,
            item=item,
            ref_item=ref_item,
            item_name=item_name,
        )
    
    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super().collater(samples)
        content= collate_1d_or_2d([s['content'] for s in samples], 0).long()
        batch['content'] = content

        optional_collate = self._build_optional_collate_spec()
        optional_keys = self._resolve_optional_sample_keys()
        for key in optional_keys:
            if key not in optional_collate:
                continue
            dtype_name, pad_value = optional_collate[key]
            if all(key in s for s in samples):
                value = collate_1d_or_2d([s[key] for s in samples], pad_value)
                if dtype_name == "long":
                    value = value.long()
                else:
                    value = value.float()
                batch[key] = value
        # notes = collate_1d_or_2d([s['note'] for s in samples], 0.0)
        # note_durs = collate_1d_or_2d([s['note_dur'] for s in samples], 0.0)
        # note_types = collate_1d_or_2d([s['note_type'] for s in samples], 0.0)
        # batch["notes"], batch["note_durs"], batch["note_types"] = notes, note_durs, note_types

        # mix = collate_1d_or_2d([s['mix'] for s in samples], 0.0)
        # falsetto = collate_1d_or_2d([s['falsetto'] for s in samples], 0.0)
        # breathy = collate_1d_or_2d([s['breathy'] for s in samples], 0.0)
        # batch['mix'],batch['falsetto'],batch['breathy']=mix,falsetto,breathy

        # bubble = collate_1d_or_2d([s['bubble'] for s in samples], 0.0)
        # strong = collate_1d_or_2d([s['strong'] for s in samples], 0.0)
        # weak = collate_1d_or_2d([s['weak'] for s in samples], 0.0)
        # batch['bubble'],batch['strong'],batch['weak']=bubble,strong,weak

        # pharyngeal = collate_1d_or_2d([s['pharyngeal'] for s in samples], 0.0)
        # vibrato = collate_1d_or_2d([s['vibrato'] for s in samples], 0.0)
        # glissando = collate_1d_or_2d([s['glissando'] for s in samples], 0.0)
        # batch['pharyngeal'],batch['vibrato'],batch['glissando'] = pharyngeal,vibrato,glissando    

        return batch

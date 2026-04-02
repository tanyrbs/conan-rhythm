
from tasks.tts.dataset_utils import FastSpeechDataset
import hashlib
import torch
from utils.commons.dataset_utils import collate_1d_or_2d
import numpy as np
from modules.Conan.rhythm.supervision import (
    RHYTHM_CACHE_VERSION,
    RHYTHM_GUIDANCE_SURFACE_NAME,
    RHYTHM_REFERENCE_MODE_STATIC_REF_FULL,
    RHYTHM_RETIMED_SOURCE_GUIDANCE,
    RHYTHM_RETIMED_SOURCE_TEACHER,
    RHYTHM_TEACHER_SURFACE_NAME,
    RHYTHM_TRACE_HOP_MS,
    RHYTHM_UNIT_HOP_MS,
    build_reference_guided_targets,
    build_reference_rhythm_conditioning,
    build_reference_teacher_targets,
    build_retimed_mel_target,
    build_source_rhythm_cache,
    normalize_teacher_target_source,
    resolve_teacher_surface_name,
    resolve_teacher_target_source_id,
    with_blank_aliases,
)
from modules.Conan.rhythm.unitizer import estimate_boundary_confidence

class ConanDataset(FastSpeechDataset):
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

    def _resolve_primary_target_surface(self) -> str:
        surface = str(self.hparams.get("rhythm_primary_target_surface", "guidance") or "guidance").strip().lower()
        aliases = {
            "cache_teacher": "teacher",
            "offline": "teacher",
            "offline_teacher": "teacher",
            "teacher_surface": "teacher",
            "guidance_surface": "guidance",
            "self": "guidance",
        }
        resolved = aliases.get(surface, surface)
        if resolved not in {"guidance", "teacher"}:
            raise ValueError(f"Unsupported rhythm_primary_target_surface: {surface}")
        return resolved

    def _resolve_distill_surface(self) -> str:
        surface = str(self.hparams.get("rhythm_distill_surface", "auto") or "auto").strip().lower()
        aliases = {
            "off": "none",
            "disable": "none",
            "disabled": "none",
            "false": "none",
            "cache_teacher": "cache",
            "cached_teacher": "cache",
            "full_context": "offline",
            "shared_offline": "offline",
            "algo": "algorithmic",
            "teacher": "cache",
        }
        resolved = aliases.get(surface, surface)
        if resolved not in {"auto", "none", "cache", "offline", "algorithmic"}:
            raise ValueError(f"Unsupported rhythm_distill_surface: {surface}")
        return resolved

    def _resolve_rhythm_target_mode(self) -> str:
        mode = str(self.hparams.get("rhythm_dataset_target_mode", "prefer_cache") or "prefer_cache").strip().lower()
        aliases = {
            "auto": "prefer_cache",
            "offline": "cached_only",
            "offline_only": "cached_only",
            "never": "cached_only",
            "runtime": "runtime_only",
            "always": "runtime_only",
        }
        return aliases.get(mode, mode)

    def _resolve_teacher_target_source(self) -> str:
        return normalize_teacher_target_source(self.hparams.get("rhythm_teacher_target_source", "algorithmic"))

    def _resolve_expected_teacher_surface_name(self) -> str:
        return resolve_teacher_surface_name(self._resolve_teacher_target_source())

    def _resolve_expected_teacher_target_source_id(self) -> int:
        return resolve_teacher_target_source_id(self._resolve_teacher_target_source())

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
        return bool(self.hparams.get("rhythm_enable_dual_mode_teacher", False)) or float(
            self.hparams.get("lambda_rhythm_teacher_aux", 0.0) or 0.0
        ) > 0.0

    def _should_export_offline_teacher_aux(self) -> bool:
        return (
            bool(self.hparams.get("rhythm_enable_dual_mode_teacher", False))
            and bool(self.hparams.get("rhythm_enable_learned_offline_teacher", True))
        ) or float(self.hparams.get("lambda_rhythm_teacher_aux", 0.0) or 0.0) > 0.0

    def _should_export_streaming_prefix_meta(self) -> bool:
        return self._should_sample_streaming_prefix()

    def _should_export_runtime_retimed_targets(self) -> bool:
        if bool(self.hparams.get("rhythm_require_retimed_cache", False)):
            return True
        if not bool(self.hparams.get("rhythm_use_retimed_target_if_available", False)):
            return False
        if self.prefix == "train":
            return bool(self.hparams.get("rhythm_apply_train_override", False))
        if self.prefix in {"valid", "dev"}:
            return bool(self.hparams.get("rhythm_apply_valid_override", False))
        # test/infer does not need acoustic supervision targets by default.
        return False

    def _resolve_runtime_target_export_keys(self) -> tuple[str, ...]:
        primary_surface = self._resolve_primary_target_surface()
        distill_surface = self._resolve_distill_surface()
        lambda_guidance = float(self.hparams.get("lambda_rhythm_guidance", 0.0))
        lambda_distill = float(self.hparams.get("lambda_rhythm_distill", 0.0))
        distill_budget_weight = float(self.hparams.get("rhythm_distill_budget_weight", 0.5))
        distill_allocation_weight = float(self.hparams.get("rhythm_distill_allocation_weight", 0.5))
        distill_prefix_weight = float(self.hparams.get("rhythm_distill_prefix_weight", 0.25))
        require_retimed_cache = bool(self.hparams.get("rhythm_require_retimed_cache", False))
        retimed_source = str(self.hparams.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower()
        export_retimed_targets = self._should_export_runtime_retimed_targets()

        keys = list(self._RHYTHM_RUNTIME_TARGET_CORE_KEYS)
        if lambda_guidance > 0.0:
            keys.extend(self._RHYTHM_RUNTIME_GUIDANCE_KEYS)

        need_teacher_core = (
            primary_surface == "teacher"
            or bool(self.hparams.get("rhythm_require_cached_teacher", False))
            or (lambda_distill > 0.0 and distill_surface == "cache")
            or ((export_retimed_targets or require_retimed_cache) and retimed_source == "teacher")
        )
        if need_teacher_core:
            keys.extend(self._RHYTHM_RUNTIME_TEACHER_CORE_KEYS)

        need_teacher_prefix = lambda_distill > 0.0 and distill_surface == "cache"
        if need_teacher_prefix and distill_allocation_weight > 0.0:
            keys.extend(self._RHYTHM_RUNTIME_TEACHER_ALLOCATION_KEYS)
        if need_teacher_prefix and distill_prefix_weight > 0.0:
            keys.extend(self._RHYTHM_RUNTIME_TEACHER_PREFIX_KEYS)

        need_teacher_budget = (
            primary_surface == "teacher"
            or bool(self.hparams.get("rhythm_require_cached_teacher", False))
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

    @staticmethod
    def _extract_scalar(value):
        arr = np.asarray(value)
        if arr.size <= 0:
            raise RuntimeError("Rhythm cache metadata contains an empty scalar field.")
        return arr.reshape(-1)[0]

    @staticmethod
    def _missing_keys(item, keys):
        return [key for key in keys if key not in item]

    def _require_cached_keys(self, *, item, keys, item_name: str, reason: str):
        missing = self._missing_keys(item, keys)
        if missing:
            raise RuntimeError(
                f"Rhythm cached_only requires {reason} in {item_name}, missing keys: {missing}"
            )

    def _validate_rhythm_cache_version(self, item, *, item_name: str):
        if "rhythm_cache_version" not in item:
            raise RuntimeError(
                f"Rhythm cached_only requires rhythm_cache_version in {item_name}. Re-binarize the dataset."
            )
        found = int(self._extract_scalar(item["rhythm_cache_version"]))
        expected = self._expected_rhythm_cache_version()
        if found != expected:
            raise RuntimeError(
                f"Rhythm cache version mismatch in {item_name}: found={found}, expected={expected}. Re-binarize the dataset."
            )

    def _validate_rhythm_cache_contract(self, item, *, item_name: str):
        self._validate_rhythm_cache_version(item, item_name=item_name)
        expected_numeric = {
            "rhythm_unit_hop_ms": int(self.hparams.get("rhythm_unit_hop_ms", RHYTHM_UNIT_HOP_MS)),
            "rhythm_trace_hop_ms": int(self.hparams.get("rhythm_trace_hop_ms", RHYTHM_TRACE_HOP_MS)),
            "rhythm_trace_bins": int(self.hparams.get("rhythm_trace_bins", 24)),
            "rhythm_trace_horizon": float(self.hparams.get("rhythm_trace_horizon", 0.35)),
            "rhythm_slow_topk": int(self.hparams.get("rhythm_slow_topk", 6)),
            "rhythm_selector_cell_size": int(self.hparams.get("rhythm_selector_cell_size", 3)),
            "rhythm_source_phrase_threshold": float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
            "rhythm_reference_mode_id": int(
                self.hparams.get("rhythm_reference_mode_id", RHYTHM_REFERENCE_MODE_STATIC_REF_FULL)
            ),
        }
        for key, expected in expected_numeric.items():
            if key not in item:
                raise RuntimeError(
                    f"Rhythm cached_only requires {key} in {item_name}. Re-binarize the dataset."
                )
            found = self._extract_scalar(item[key])
            if isinstance(expected, float):
                if abs(float(found) - expected) > 1e-5:
                    raise RuntimeError(
                        f"Rhythm cache contract mismatch in {item_name} for {key}: "
                        f"found={float(found):.6f}, expected={expected:.6f}. Re-binarize the dataset."
                    )
            else:
                if int(found) != expected:
                    raise RuntimeError(
                        f"Rhythm cache contract mismatch in {item_name} for {key}: "
                        f"found={int(found)}, expected={expected}. Re-binarize the dataset."
                    )
        if "rhythm_guidance_surface_name" not in item:
            raise RuntimeError(
                f"Rhythm cached_only requires rhythm_guidance_surface_name in {item_name}. Re-binarize the dataset."
            )
        guidance_name = str(self._extract_scalar(item["rhythm_guidance_surface_name"]))
        if guidance_name != RHYTHM_GUIDANCE_SURFACE_NAME:
            raise RuntimeError(
                f"Rhythm guidance surface mismatch in {item_name}: "
                f"found={guidance_name}, expected={RHYTHM_GUIDANCE_SURFACE_NAME}. Re-binarize the dataset."
            )

    def _required_cached_target_keys(self):
        primary_surface = self._resolve_primary_target_surface()
        distill_surface = self._resolve_distill_surface()
        keys = [
            "rhythm_cache_version",
            "rhythm_unit_hop_ms",
            "rhythm_trace_hop_ms",
            "rhythm_trace_bins",
            "rhythm_trace_horizon",
            "rhythm_slow_topk",
            "rhythm_selector_cell_size",
            "rhythm_source_phrase_threshold",
            "rhythm_reference_mode_id",
            "rhythm_guidance_surface_name",
        ]
        need_guidance = primary_surface == "guidance" or float(self.hparams.get("lambda_rhythm_guidance", 0.0)) > 0
        if need_guidance:
            keys.extend([
                "rhythm_speech_exec_tgt",
                "rhythm_pause_exec_tgt",
                "rhythm_speech_budget_tgt",
                "rhythm_pause_budget_tgt",
                "rhythm_target_confidence",
                "rhythm_guidance_confidence",
            ])
        if float(self.hparams.get("lambda_rhythm_guidance", 0.0)) > 0:
            keys.extend([
                "rhythm_guidance_speech_tgt",
                "rhythm_guidance_pause_tgt",
            ])
        need_teacher = (
            primary_surface == "teacher"
            or bool(self.hparams.get("rhythm_require_cached_teacher", False))
            or (
                float(self.hparams.get("lambda_rhythm_distill", 0.0)) > 0
                and distill_surface == "cache"
            )
            or (
                bool(self.hparams.get("rhythm_use_retimed_target_if_available", False))
                and str(self.hparams.get("rhythm_binarize_retimed_mel_source", "guidance")).strip().lower() == "teacher"
            )
        )
        if need_teacher:
            keys.extend([
                "rhythm_teacher_speech_exec_tgt",
                "rhythm_teacher_pause_exec_tgt",
                "rhythm_teacher_speech_budget_tgt",
                "rhythm_teacher_pause_budget_tgt",
                "rhythm_teacher_allocation_tgt",
                "rhythm_teacher_prefix_clock_tgt",
                "rhythm_teacher_prefix_backlog_tgt",
                "rhythm_teacher_confidence",
                "rhythm_teacher_target_source_id",
                "rhythm_teacher_surface_name",
            ])
        if bool(self.hparams.get("rhythm_require_retimed_cache", False)):
            keys.extend([
                "rhythm_retimed_mel_tgt",
                "rhythm_retimed_mel_len",
                "rhythm_retimed_frame_weight",
                "rhythm_retimed_target_source_id",
                "rhythm_retimed_target_confidence",
                "rhythm_retimed_target_surface_name",
            ])
        return tuple(dict.fromkeys(keys))

    def _validate_retimed_cache_contract(self, item, *, item_name: str):
        if "rhythm_retimed_target_source_id" not in item:
            raise RuntimeError(
                f"Rhythm cached_only requires rhythm_retimed_target_source_id in {item_name}. Re-binarize the dataset."
            )
        found_source_id = int(self._extract_scalar(item["rhythm_retimed_target_source_id"]))
        expected_source = str(self.hparams.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower()
        expected_source_id = RHYTHM_RETIMED_SOURCE_TEACHER if expected_source == "teacher" else RHYTHM_RETIMED_SOURCE_GUIDANCE
        if found_source_id != expected_source_id:
            raise RuntimeError(
                f"Rhythm retimed cache source mismatch in {item_name}: "
                f"found={found_source_id}, expected={expected_source_id}. Re-binarize the dataset."
            )
        if "rhythm_retimed_target_surface_name" not in item:
            raise RuntimeError(
                f"Rhythm cached_only requires rhythm_retimed_target_surface_name in {item_name}. Re-binarize the dataset."
            )
        found_surface = str(self._extract_scalar(item["rhythm_retimed_target_surface_name"]))
        expected_surface = self._resolve_expected_teacher_surface_name() if expected_source_id == RHYTHM_RETIMED_SOURCE_TEACHER else RHYTHM_GUIDANCE_SURFACE_NAME
        if found_surface != expected_surface:
            raise RuntimeError(
                f"Rhythm retimed surface mismatch in {item_name}: "
                f"found={found_surface}, expected={expected_surface}. Re-binarize the dataset."
            )

    def _validate_source_cache_shapes(self, cache, *, item_name: str):
        lengths = {key: int(np.asarray(cache[key]).reshape(-1).shape[0]) for key in self._RHYTHM_SOURCE_CACHE_KEYS}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(
                f"Rhythm source cache shape mismatch in {item_name}: {lengths}. Re-binarize the dataset."
            )

    def _validate_reference_conditioning_shapes(self, conditioning, *, item_name: str):
        stats = np.asarray(conditioning["ref_rhythm_stats"])
        trace = np.asarray(conditioning["ref_rhythm_trace"])
        stats_dim = int(self.hparams.get("rhythm_stats_dim", 6))
        trace_bins = int(self.hparams.get("rhythm_trace_bins", 24))
        trace_dim = int(self.hparams.get("rhythm_trace_dim", 5))
        if stats.reshape(-1).shape[0] != stats_dim:
            raise RuntimeError(
                f"Rhythm stats shape mismatch in {item_name}: found={tuple(stats.shape)}, expected_last_dim={stats_dim}."
            )
        if trace.ndim != 2 or trace.shape[0] != trace_bins or trace.shape[1] != trace_dim:
            raise RuntimeError(
                f"Rhythm trace shape mismatch in {item_name}: found={tuple(trace.shape)}, expected=({trace_bins}, {trace_dim})."
            )
        sidecar_keys = self._RHYTHM_REF_DEBUG_CACHE_KEYS
        if not all(key in conditioning for key in sidecar_keys):
            return
        slow_memory = np.asarray(conditioning["slow_rhythm_memory"])
        slow_summary = np.asarray(conditioning["slow_rhythm_summary"])
        selector_indices = np.asarray(conditioning["selector_meta_indices"]).reshape(-1)
        selector_scores = np.asarray(conditioning["selector_meta_scores"]).reshape(-1)
        selector_starts = np.asarray(conditioning["selector_meta_starts"]).reshape(-1)
        selector_ends = np.asarray(conditioning["selector_meta_ends"]).reshape(-1)
        slow_topk = int(self.hparams.get("rhythm_slow_topk", 6))
        if slow_memory.ndim != 2 or slow_memory.shape[1] != trace_dim:
            raise RuntimeError(
                f"Slow rhythm memory shape mismatch in {item_name}: found={tuple(slow_memory.shape)}, expected=(*,{trace_dim})."
            )
        if slow_memory.shape[0] > slow_topk:
            raise RuntimeError(
                f"Slow rhythm memory count mismatch in {item_name}: found={slow_memory.shape[0]}, expected<= {slow_topk}."
            )
        if slow_summary.reshape(-1).shape[0] != trace_dim:
            raise RuntimeError(
                f"Slow rhythm summary shape mismatch in {item_name}: found={tuple(slow_summary.shape)}, expected_last_dim={trace_dim}."
            )
        selector_len = int(slow_memory.shape[0])
        selector_lengths = {
            "selector_meta_indices": int(selector_indices.shape[0]),
            "selector_meta_scores": int(selector_scores.shape[0]),
            "selector_meta_starts": int(selector_starts.shape[0]),
            "selector_meta_ends": int(selector_ends.shape[0]),
        }
        if len(set(selector_lengths.values()) | {selector_len}) != 1:
            raise RuntimeError(
                f"Selector metadata shape mismatch in {item_name}: slow_memory={selector_len}, meta={selector_lengths}."
            )
        if selector_len > 0:
            if selector_indices.min() < 0 or selector_indices.max() >= trace_bins:
                raise RuntimeError(
                    f"Selector indices out of range in {item_name}: min={selector_indices.min()}, max={selector_indices.max()}, trace_bins={trace_bins}."
                )
            if selector_starts.min() < 0 or selector_ends.max() >= trace_bins or np.any(selector_starts > selector_ends):
                raise RuntimeError(
                    f"Selector cell spans invalid in {item_name}: starts={selector_starts.tolist()}, ends={selector_ends.tolist()}, trace_bins={trace_bins}."
                )

    def _validate_target_shapes(self, targets, *, item_name: str, expected_units: int):
        unit_keys = [
            "rhythm_speech_exec_tgt",
            "rhythm_pause_exec_tgt",
            "rhythm_blank_exec_tgt",
            "rhythm_guidance_speech_tgt",
            "rhythm_guidance_pause_tgt",
            "rhythm_guidance_blank_tgt",
            "rhythm_teacher_speech_exec_tgt",
            "rhythm_teacher_pause_exec_tgt",
            "rhythm_teacher_blank_exec_tgt",
            "rhythm_teacher_allocation_tgt",
            "rhythm_teacher_prefix_clock_tgt",
            "rhythm_teacher_prefix_backlog_tgt",
        ]
        for key in unit_keys:
            if key not in targets:
                continue
            found = int(np.asarray(targets[key]).reshape(-1).shape[0])
            if found != expected_units:
                raise RuntimeError(
                    f"Rhythm target length mismatch in {item_name} for {key}: "
                    f"found={found}, expected={expected_units}."
                )
        budget_keys = [
            "rhythm_speech_budget_tgt",
            "rhythm_pause_budget_tgt",
            "rhythm_teacher_speech_budget_tgt",
            "rhythm_teacher_pause_budget_tgt",
        ]
        for key in budget_keys:
            if key not in targets:
                continue
            found = int(np.asarray(targets[key]).reshape(-1).shape[0])
            if found != 1:
                raise RuntimeError(
                    f"Rhythm budget target shape mismatch in {item_name} for {key}: expected scalar/[1], found={tuple(np.asarray(targets[key]).shape)}."
                )
        if "rhythm_retimed_mel_tgt" in targets or "rhythm_retimed_mel_len" in targets or "rhythm_retimed_frame_weight" in targets:
            required = ("rhythm_retimed_mel_tgt", "rhythm_retimed_mel_len", "rhythm_retimed_frame_weight")
            missing = [key for key in required if key not in targets]
            if missing:
                raise RuntimeError(
                    f"Rhythm retimed cache incomplete in {item_name}, missing={missing}. Re-binarize the dataset."
                )
            mel = np.asarray(targets["rhythm_retimed_mel_tgt"])
            frame_weight = np.asarray(targets["rhythm_retimed_frame_weight"]).reshape(-1)
            mel_len = int(self._extract_scalar(targets["rhythm_retimed_mel_len"]))
            if mel.ndim != 2:
                raise RuntimeError(
                    f"Rhythm retimed mel target must be rank-2 in {item_name}, found shape={tuple(mel.shape)}."
                )
            if mel.shape[0] != mel_len or frame_weight.shape[0] != mel_len:
                raise RuntimeError(
                    f"Rhythm retimed cache length mismatch in {item_name}: mel_len={mel_len}, "
                    f"mel_frames={mel.shape[0]}, weight_frames={frame_weight.shape[0]}."
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
        visible_units = int(np.asarray(source_cache["dur_anchor_src"]).reshape(-1).shape[0])
        full_units = int(np.asarray(item["dur_anchor_src"]).reshape(-1).shape[0]) if "dur_anchor_src" in item else visible_units
        if visible_units >= full_units:
            return with_blank_aliases(dict(cached_targets))
        adapted = with_blank_aliases(dict(cached_targets))
        unit_keys = [
            "rhythm_speech_exec_tgt",
            "rhythm_pause_exec_tgt",
            "rhythm_blank_exec_tgt",
            "rhythm_guidance_speech_tgt",
            "rhythm_guidance_pause_tgt",
            "rhythm_guidance_blank_tgt",
            "rhythm_teacher_speech_exec_tgt",
            "rhythm_teacher_pause_exec_tgt",
            "rhythm_teacher_blank_exec_tgt",
            "rhythm_teacher_allocation_tgt",
            "rhythm_teacher_prefix_clock_tgt",
            "rhythm_teacher_prefix_backlog_tgt",
        ]
        for key in unit_keys:
            if key in adapted:
                adapted[key] = np.asarray(adapted[key]).reshape(-1)[:visible_units].astype(np.float32)
        if "rhythm_teacher_allocation_tgt" in adapted:
            alloc = np.asarray(adapted["rhythm_teacher_allocation_tgt"]).reshape(-1).astype(np.float32)
            denom = float(np.maximum(alloc.sum(), 1e-6))
            adapted["rhythm_teacher_allocation_tgt"] = alloc / denom
        if "rhythm_speech_exec_tgt" in adapted:
            adapted["rhythm_speech_budget_tgt"] = np.asarray(
                [float(np.asarray(adapted["rhythm_speech_exec_tgt"]).sum())], dtype=np.float32
            )
        if "rhythm_pause_exec_tgt" in adapted:
            adapted["rhythm_pause_budget_tgt"] = np.asarray(
                [float(np.asarray(adapted["rhythm_pause_exec_tgt"]).sum())], dtype=np.float32
            )
            adapted["rhythm_blank_budget_tgt"] = adapted["rhythm_pause_budget_tgt"].copy()
        if "rhythm_teacher_speech_exec_tgt" in adapted:
            adapted["rhythm_teacher_speech_budget_tgt"] = np.asarray(
                [float(np.asarray(adapted["rhythm_teacher_speech_exec_tgt"]).sum())], dtype=np.float32
            )
        if "rhythm_teacher_pause_exec_tgt" in adapted:
            adapted["rhythm_teacher_pause_budget_tgt"] = np.asarray(
                [float(np.asarray(adapted["rhythm_teacher_pause_exec_tgt"]).sum())], dtype=np.float32
            )
            adapted["rhythm_teacher_blank_budget_tgt"] = adapted["rhythm_teacher_pause_budget_tgt"].copy()
        if "rhythm_retimed_mel_tgt" in adapted:
            source_id = int(np.asarray(adapted.get("rhythm_retimed_target_source_id", [RHYTHM_RETIMED_SOURCE_GUIDANCE])).reshape(-1)[0])
            if source_id == RHYTHM_RETIMED_SOURCE_TEACHER and "rhythm_teacher_speech_exec_tgt" in adapted:
                speech_key = "rhythm_teacher_speech_exec_tgt"
                pause_key = "rhythm_teacher_pause_exec_tgt"
            else:
                speech_key = "rhythm_speech_exec_tgt"
                pause_key = "rhythm_pause_exec_tgt"
            adapted.update(
                build_retimed_mel_target(
                    mel=sample["mel"].cpu().numpy(),
                    dur_anchor_src=source_cache["dur_anchor_src"],
                    speech_exec_tgt=adapted[speech_key],
                    pause_exec_tgt=adapted[pause_key],
                    unit_mask=(np.asarray(source_cache["dur_anchor_src"]) > 0).astype(np.float32),
                    pause_frame_weight=float(self.hparams.get("rhythm_retimed_pause_frame_weight", 0.20)),
                    stretch_weight_min=float(self.hparams.get("rhythm_retimed_stretch_weight_min", 0.35)),
                )
            )
        return with_blank_aliases(adapted)

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
        unit_mask = (np.asarray(source_cache["dur_anchor_src"]) > 0).astype(np.float32)
        shared_kwargs = dict(
            dur_anchor_src=source_cache["dur_anchor_src"],
            unit_mask=unit_mask,
            ref_rhythm_stats=ref_conditioning["ref_rhythm_stats"],
            ref_rhythm_trace=ref_conditioning["ref_rhythm_trace"],
            rate_scale_min=float(self.hparams.get("rhythm_guidance_rate_scale_min", 0.60)),
            rate_scale_max=float(self.hparams.get("rhythm_guidance_rate_scale_max", 1.80)),
            local_rate_strength=float(self.hparams.get("rhythm_guidance_local_rate_strength", 0.35)),
            segment_bias_strength=float(self.hparams.get("rhythm_guidance_segment_bias_strength", 0.25)),
            pause_strength=float(self.hparams.get("rhythm_guidance_pause_strength", 1.00)),
            boundary_strength=float(self.hparams.get("rhythm_guidance_boundary_strength", 1.25)),
            pause_budget_ratio_cap=float(self.hparams.get("rhythm_guidance_pause_budget_ratio_cap", 0.75)),
        )
        teacher_kwargs = dict(
            dur_anchor_src=source_cache["dur_anchor_src"],
            unit_mask=unit_mask,
            source_boundary_cue=source_cache.get("source_boundary_cue", source_cache.get("boundary_confidence")),
            ref_rhythm_stats=ref_conditioning["ref_rhythm_stats"],
            ref_rhythm_trace=ref_conditioning["ref_rhythm_trace"],
            rate_scale_min=float(self.hparams.get("rhythm_teacher_rate_scale_min", 0.55)),
            rate_scale_max=float(self.hparams.get("rhythm_teacher_rate_scale_max", 1.95)),
            local_rate_strength=float(self.hparams.get("rhythm_teacher_local_rate_strength", 0.45)),
            segment_bias_strength=float(self.hparams.get("rhythm_teacher_segment_bias_strength", 0.30)),
            pause_strength=float(self.hparams.get("rhythm_teacher_pause_strength", 1.10)),
            boundary_strength=float(self.hparams.get("rhythm_teacher_boundary_strength", 1.50)),
            pause_budget_ratio_cap=float(self.hparams.get("rhythm_teacher_pause_budget_ratio_cap", 0.80)),
            speech_smooth_kernel=int(self.hparams.get("rhythm_teacher_speech_smooth_kernel", 3)),
            pause_topk_ratio=float(self.hparams.get("rhythm_teacher_pause_topk_ratio", 0.30)),
        )
        targets = {}
        primary_surface = self._resolve_primary_target_surface()
        distill_surface = self._resolve_distill_surface()
        teacher_target_source = self._resolve_teacher_target_source()
        need_guidance = primary_surface == "guidance" or float(self.hparams.get("lambda_rhythm_guidance", 0.0)) > 0
        need_teacher = (
            primary_surface == "teacher"
            or bool(self.hparams.get("rhythm_require_cached_teacher", False))
            or (float(self.hparams.get("lambda_rhythm_distill", 0.0)) > 0 and distill_surface == "algorithmic")
            or (
                bool(self.hparams.get("rhythm_use_retimed_target_if_available", False))
                and str(self.hparams.get("rhythm_binarize_retimed_mel_source", "guidance")).strip().lower() == "teacher"
            )
        )
        if self.hparams.get("rhythm_dataset_build_guidance_from_ref", True) or need_guidance:
            targets.update(build_reference_guided_targets(**shared_kwargs))
        if self.hparams.get("rhythm_dataset_build_teacher_from_ref", False) or need_teacher:
            if teacher_target_source != "algorithmic":
                if need_teacher:
                    raise RuntimeError(
                        "Rhythm runtime teacher synthesis only supports rhythm_teacher_target_source=algorithmic. "
                        "Use cached_only with precomputed learned_offline teacher surfaces."
                    )
            else:
                targets.update(build_reference_teacher_targets(**teacher_kwargs))
        return targets

    def _merge_rhythm_targets(self, item, source_cache, ref_conditioning, sample):
        target_mode = self._resolve_rhythm_target_mode()
        cached_targets = {key: item[key] for key in self._RHYTHM_TARGET_KEYS if key in item}
        cached_targets.update({key: item[key] for key in self._RHYTHM_META_KEYS if key in item})
        if target_mode == "cached_only":
            item_name = str(item.get("item_name", "<unknown-item>"))
            self._validate_rhythm_cache_contract(item, item_name=item_name)
            self._require_cached_keys(
                item=item,
                keys=self._required_cached_target_keys(),
                item_name=item_name,
                reason="rhythm targets/meta cache",
            )
            if "rhythm_teacher_surface_name" in item:
                teacher_name = str(self._extract_scalar(item["rhythm_teacher_surface_name"]))
                expected_teacher_name = self._resolve_expected_teacher_surface_name()
                if teacher_name != expected_teacher_name:
                    raise RuntimeError(
                        f"Rhythm teacher surface mismatch in {item_name}: "
                        f"found={teacher_name}, expected={expected_teacher_name}. Re-binarize the dataset."
                    )
            if "rhythm_teacher_target_source_id" in item:
                found_source_id = int(self._extract_scalar(item["rhythm_teacher_target_source_id"]))
                expected_source_id = self._resolve_expected_teacher_target_source_id()
                if found_source_id != expected_source_id:
                    raise RuntimeError(
                        f"Rhythm teacher target source mismatch in {item_name}: "
                        f"found={found_source_id}, expected={expected_source_id}. Re-binarize the dataset."
                    )
            if bool(self.hparams.get("rhythm_require_retimed_cache", False)):
                self._validate_retimed_cache_contract(item, item_name=item_name)
            cached_targets = self._adapt_cached_targets_to_prefix(
                item=item,
                cached_targets=cached_targets,
                source_cache=source_cache,
                sample=sample,
            )
            self._validate_target_shapes(
                cached_targets,
                item_name=item_name,
                expected_units=int(np.asarray(source_cache["dur_anchor_src"]).reshape(-1).shape[0]),
            )
            return cached_targets

        runtime_targets = self._build_runtime_rhythm_targets(source_cache, ref_conditioning)
        if target_mode == "runtime_only":
            return runtime_targets

        merged = dict(cached_targets)
        for key, value in runtime_targets.items():
            merged.setdefault(key, value)
        if "rhythm_speech_exec_tgt" in merged:
            merged = self._adapt_cached_targets_to_prefix(
                item=item,
                cached_targets=merged,
                source_cache=source_cache,
                sample=sample,
            )
            self._validate_target_shapes(
                merged,
                item_name=str(item.get("item_name", "<unknown-item>")),
                expected_units=int(np.asarray(source_cache["dur_anchor_src"]).reshape(-1).shape[0]),
            )
        return merged

    def __getitem__(self, index):
        hparams=self.hparams
        sample = super(ConanDataset, self).__getitem__(index)
        item = self._get_item(index)
        ref_item = None
        if "ref_item_id" in sample:
            ref_item = self._get_item(int(sample["ref_item_id"]))
        
        # if isinstance(item['hubert'], str):
        #     # Convert string to numeric array
        #     content = [float(x) for x in item['hubert'].split()]
        #     sample["content"] = torch.LongTensor(content)
        # else:
            # Already a numeric array case
        visible_len = int(sample["mel"].shape[0])
        item_name = str(item.get("item_name", "<unknown-item>"))
        full_content = self._coerce_content_sequence(item["hubert"])
        full_content_len = len(full_content)
        if bool(hparams.get("rhythm_enable_v2", False)) and bool(
            hparams.get("rhythm_strict_content_mel_contract", True)
        ):
            tolerance = int(hparams.get("rhythm_content_mel_tolerance", 0) or 0)
            if abs(full_content_len - visible_len) > tolerance:
                raise RuntimeError(
                    f"Rhythm content/mel contract violated for {item_name}: hubert_len={full_content_len}, "
                    f"mel_len={visible_len}, tolerance={tolerance}. Re-binarize or align upstream hop settings."
                )
        if full_content_len < visible_len:
            raise RuntimeError(
                f"Rhythm content sequence shorter than mel for {item_name}: hubert_len={full_content_len}, "
                f"mel_len={visible_len}."
            )
        content_visible = full_content[:visible_len]
        if len(content_visible) != visible_len:
            raise RuntimeError(
                f"Visible content prefix mismatch for {item_name}: visible_tokens={len(content_visible)}, "
                f"mel_len={visible_len}."
            )
        sample["content"] = torch.LongTensor(content_visible)
        target_mode = self._resolve_rhythm_target_mode()
        full_visible_tokens = np.asarray(content_visible, dtype=np.int64)
        stream_visible_tokens = self._select_streaming_visible_tokens(
            full_visible_tokens,
            item_name=item_name,
        )

        optional_rhythm_keys = self._resolve_optional_sample_keys()
        rhythm_runtime_fields = {}
        source_cache = self._get_source_rhythm_cache(item, stream_visible_tokens, target_mode=target_mode)
        stream_units = int(np.asarray(source_cache["dur_anchor_src"]).reshape(-1).shape[0])
        offline_units = stream_units
        if int(stream_visible_tokens.shape[0]) < int(full_visible_tokens.shape[0]):
            if self._should_export_streaming_offline_sidecars():
                offline_source_cache = self._get_source_rhythm_cache(item, full_visible_tokens, target_mode=target_mode)
                rhythm_runtime_fields.update(self._prefix_source_cache(offline_source_cache, prefix="rhythm_offline_"))
                offline_units = int(np.asarray(offline_source_cache["dur_anchor_src"]).reshape(-1).shape[0])
                if self._should_export_offline_teacher_aux():
                    rhythm_runtime_fields.update(
                        self._build_offline_teacher_aux_fields(
                            item,
                            offline_units=offline_units,
                        )
                    )
            else:
                offline_units = int(np.asarray(self._get_source_rhythm_cache(item, full_visible_tokens, target_mode=target_mode)["dur_anchor_src"]).reshape(-1).shape[0])
        if self._should_export_streaming_prefix_meta():
            rhythm_runtime_fields["rhythm_stream_visible_units"] = np.asarray([stream_units], dtype=np.float32)
            rhythm_runtime_fields["rhythm_stream_full_units"] = np.asarray([offline_units], dtype=np.float32)
            rhythm_runtime_fields["rhythm_stream_prefix_ratio"] = np.asarray(
                [float(stream_units) / float(max(offline_units, 1))],
                dtype=np.float32,
            )
        rhythm_ref_item = item if self._should_use_self_rhythm_reference(item, target_mode=target_mode) else ref_item
        ref_conditioning = self._get_reference_rhythm_conditioning(rhythm_ref_item, sample, target_mode=target_mode)
        rhythm_runtime_fields.update(source_cache)
        rhythm_runtime_fields.update(ref_conditioning)
        rhythm_runtime_fields.update(
            self._merge_rhythm_targets(
                item,
                source_cache,
                ref_conditioning,
                sample,
            )
        )

        for key in optional_rhythm_keys:
            if key in rhythm_runtime_fields:
                value = rhythm_runtime_fields[key]
            elif key in item:
                value = item[key]
            else:
                continue
            if key in {"ref_rhythm_trace", "slow_rhythm_memory"}:
                sample[key] = torch.tensor(value, dtype=torch.float32)
            elif key in {
                "source_boundary_cue",
                "phrase_group_pos",
                "phrase_final_mask",
                "slow_rhythm_summary",
                "selector_meta_scores",
                "rhythm_teacher_allocation_tgt",
                "rhythm_teacher_prefix_clock_tgt",
                "rhythm_teacher_prefix_backlog_tgt",
                "rhythm_offline_source_boundary_cue",
                "rhythm_offline_phrase_group_pos",
                "rhythm_offline_phrase_final_mask",
                "rhythm_stream_prefix_ratio",
                "rhythm_stream_visible_units",
                "rhythm_stream_full_units",
            } or "stats" in key or "budget" in key:
                sample[key] = torch.tensor(value, dtype=torch.float32)
            elif key in {"sealed_mask", "boundary_confidence", "rhythm_offline_sealed_mask", "rhythm_offline_boundary_confidence"}:
                sample[key] = torch.tensor(value, dtype=torch.float32)
            elif key in {"rhythm_target_confidence", "rhythm_guidance_confidence", "rhythm_teacher_confidence", "rhythm_offline_teacher_confidence"}:
                sample[key] = torch.tensor(value, dtype=torch.float32)
            elif key in {"rhythm_retimed_target_confidence", "rhythm_trace_horizon", "rhythm_source_phrase_threshold"}:
                sample[key] = torch.tensor(value, dtype=torch.float32)
            elif key in {
                "phrase_group_index",
                "rhythm_offline_content_units",
                "rhythm_offline_dur_anchor_src",
                "rhythm_offline_open_run_mask",
                "rhythm_offline_sep_hint",
                "rhythm_offline_phrase_group_index",
                "selector_meta_indices",
                "selector_meta_starts",
                "selector_meta_ends",
                "rhythm_cache_version",
                "rhythm_unit_hop_ms",
                "rhythm_trace_hop_ms",
                "rhythm_trace_bins",
                "rhythm_slow_topk",
                "rhythm_selector_cell_size",
                "rhythm_reference_mode_id",
                "rhythm_teacher_target_source_id",
                "rhythm_retimed_target_source_id",
            }:
                sample[key] = torch.tensor(value, dtype=torch.long)
            elif key == "rhythm_retimed_mel_tgt":
                sample[key] = torch.tensor(value, dtype=torch.float32)
            elif key == "rhythm_retimed_mel_len":
                sample[key] = torch.tensor(value, dtype=torch.long)
            elif key == "rhythm_retimed_frame_weight":
                sample[key] = torch.tensor(value, dtype=torch.float32)
            else:
                sample[key] = torch.tensor(value)
        sample.pop("ref_item_id", None)

        # sample['content'] = torch.LongTensor(item['hubert'])
        # note = torch.LongTensor(item['ep_pitches'][:hparams['max_input_tokens']])
        # note_dur = torch.FloatTensor(item['ep_notedurs'][:hparams['max_input_tokens']])
        # note_type = torch.LongTensor(item['ep_types'][:hparams['max_input_tokens']])
        # sample["note"], sample["note_dur"], sample["note_type"] = note, note_dur, note_type

        # for key in ['mix_tech','falsetto_tech','breathy_tech','bubble_tech','strong_tech','weak_tech','pharyngeal_tech','vibrato_tech','glissando_tech']:
        #     if key not in item:
        #         item[key] = [2] * len(item['ph'])

        # mix = torch.LongTensor(item['mix_tech'][:hparams['max_input_tokens']])
        # falsetto= torch.LongTensor(item['falsetto_tech'][:hparams['max_input_tokens']])
        # breathy = torch.LongTensor(item['breathy_tech'][:hparams['max_input_tokens']])
        # sample['mix'],sample['falsetto'],sample['breathy']=mix,falsetto,breathy

        # bubble = torch.LongTensor(item['bubble_tech'][:hparams['max_input_tokens']])
        # strong = torch.LongTensor(item['strong_tech'][:hparams['max_input_tokens']])
        # weak = torch.LongTensor(item['weak_tech'][:hparams['max_input_tokens']])
        # sample['bubble'],sample['strong'],sample['weak']=bubble,strong,weak

        # pharyngeal = torch.LongTensor(item['pharyngeal_tech'][:hparams['max_input_tokens']])
        # vibrato = torch.LongTensor(item['vibrato_tech'][:hparams['max_input_tokens']])
        # glissando = torch.LongTensor(item['glissando_tech'][:hparams['max_input_tokens']])
        # sample['pharyngeal'],sample['vibrato'],sample['glissando'] = pharyngeal,vibrato,glissando
        
        return sample
    
    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(ConanDataset, self).collater(samples)
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

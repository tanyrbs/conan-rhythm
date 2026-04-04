from __future__ import annotations

import numpy as np

from modules.Conan.rhythm.supervision import (
    RHYTHM_CACHE_VERSION,
    RHYTHM_GUIDANCE_SURFACE_NAME,
    RHYTHM_REFERENCE_MODE_STATIC_REF_FULL,
    RHYTHM_RETIMED_SOURCE_GUIDANCE,
    RHYTHM_RETIMED_SOURCE_TEACHER,
    RHYTHM_TRACE_HOP_MS,
    RHYTHM_UNIT_HOP_MS,
    compatible_rhythm_cache_versions,
    is_rhythm_cache_version_compatible,
    materialize_rhythm_cache_compat_fields,
)


class RhythmDatasetCacheContract:
    def __init__(self, owner):
        self.owner = owner

    @property
    def hparams(self):
        return self.owner.hparams

    def expected_rhythm_cache_version(self) -> int:
        return int(self.hparams.get("rhythm_cache_version", RHYTHM_CACHE_VERSION))

    def materialize_rhythm_cache_compat(self, item, *, item_name: str):
        adapted = materialize_rhythm_cache_compat_fields(item)
        if adapted is None or "rhythm_cache_version" not in adapted:
            return adapted
        found = int(self.extract_scalar(adapted["rhythm_cache_version"]))
        expected = self.expected_rhythm_cache_version()
        if found == expected or not is_rhythm_cache_version_compatible(found, expected):
            return adapted
        warned = getattr(self.owner, "_rhythm_cache_compat_warned", None)
        if warned is None:
            warned = set()
            self.owner._rhythm_cache_compat_warned = warned
        warn_key = (found, expected)
        if warn_key not in warned:
            print(
                f"[rhythm-cache-compat] using compatible cached rhythm metadata "
                f"version {found} for expected v{expected}; item={item_name}. "
                "Re-binarizing to the maintained cache version is still recommended."
            )
            warned.add(warn_key)
        return adapted

    @staticmethod
    def extract_scalar(value):
        arr = np.asarray(value)
        if arr.size <= 0:
            raise RuntimeError("Rhythm cache metadata contains an empty scalar field.")
        if arr.size != 1:
            raise RuntimeError(
                f"Rhythm cache metadata expected scalar/[1], found shape={tuple(arr.shape)}."
            )
        return arr.reshape(-1)[0]

    @staticmethod
    def missing_keys(item, keys):
        return [key for key in keys if key not in item]

    def require_cached_keys(self, *, item, keys, item_name: str, reason: str):
        missing = self.missing_keys(item, keys)
        if missing:
            raise RuntimeError(
                f"Rhythm cached_only requires {reason} in {item_name}, missing keys: {missing}"
            )

    def validate_rhythm_cache_version(self, item, *, item_name: str):
        if "rhythm_cache_version" not in item:
            raise RuntimeError(
                f"Rhythm cached_only requires rhythm_cache_version in {item_name}. Re-binarize the dataset."
            )
        found = int(self.extract_scalar(item["rhythm_cache_version"]))
        expected = self.expected_rhythm_cache_version()
        if not is_rhythm_cache_version_compatible(found, expected):
            compatible = compatible_rhythm_cache_versions(expected)
            raise RuntimeError(
                f"Rhythm cache version mismatch in {item_name}: found={found}, expected one of {compatible}. "
                "Re-binarize the dataset."
            )

    def validate_rhythm_cache_contract(self, item, *, item_name: str):
        self.validate_rhythm_cache_version(item, item_name=item_name)
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
            found = self.extract_scalar(item[key])
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
        guidance_name = str(self.extract_scalar(item["rhythm_guidance_surface_name"]))
        if guidance_name != RHYTHM_GUIDANCE_SURFACE_NAME:
            raise RuntimeError(
                f"Rhythm guidance surface mismatch in {item_name}: "
                f"found={guidance_name}, expected={RHYTHM_GUIDANCE_SURFACE_NAME}. Re-binarize the dataset."
            )

    def required_cached_target_keys(self):
        primary_surface = self.owner._resolve_primary_target_surface()
        distill_surface = self.owner._resolve_distill_surface()
        lambda_distill = float(self.hparams.get("lambda_rhythm_distill", 0.0))
        distill_budget_weight = float(self.hparams.get("rhythm_distill_budget_weight", 0.5))
        distill_allocation_weight = float(self.hparams.get("rhythm_distill_allocation_weight", 0.5))
        distill_prefix_weight = float(self.hparams.get("rhythm_distill_prefix_weight", 0.25))
        distill_speech_shape_weight = float(self.hparams.get("rhythm_distill_speech_shape_weight", 0.0))
        distill_pause_shape_weight = float(self.hparams.get("rhythm_distill_pause_shape_weight", 0.0))
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
            keys.extend(
                [
                    "rhythm_speech_exec_tgt",
                    "rhythm_pause_exec_tgt",
                    "rhythm_speech_budget_tgt",
                    "rhythm_pause_budget_tgt",
                    "rhythm_target_confidence",
                    "rhythm_guidance_confidence",
                ]
            )
        if float(self.hparams.get("lambda_rhythm_guidance", 0.0)) > 0:
            keys.extend(
                [
                    "rhythm_guidance_speech_tgt",
                    "rhythm_guidance_pause_tgt",
                ]
            )
        need_teacher = (
            primary_surface == "teacher"
            or bool(self.hparams.get("rhythm_require_cached_teacher", False))
            or (
                lambda_distill > 0
                and distill_surface == "cache"
            )
            or (
                bool(self.hparams.get("rhythm_use_retimed_target_if_available", False))
                and str(self.hparams.get("rhythm_binarize_retimed_mel_source", "guidance")).strip().lower() == "teacher"
            )
        )
        if need_teacher:
            keys.extend(
                [
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
                ]
            )
        if lambda_distill > 0.0 and distill_surface == "cache":
            keys.append("rhythm_teacher_confidence_exec")
            if distill_budget_weight > 0.0:
                keys.append("rhythm_teacher_confidence_budget")
            if distill_prefix_weight > 0.0:
                keys.append("rhythm_teacher_confidence_prefix")
            if distill_allocation_weight > 0.0:
                keys.append("rhythm_teacher_confidence_allocation")
            if distill_speech_shape_weight > 0.0 or distill_pause_shape_weight > 0.0:
                keys.append("rhythm_teacher_confidence_shape")
        if (
            bool(self.hparams.get("rhythm_require_retimed_cache", False))
            or bool(self.hparams.get("rhythm_apply_train_override", False))
            or bool(self.hparams.get("rhythm_apply_valid_override", False))
        ):
            keys.extend(
                [
                    "rhythm_retimed_mel_tgt",
                    "rhythm_retimed_mel_len",
                    "rhythm_retimed_frame_weight",
                    "rhythm_retimed_target_source_id",
                    "rhythm_retimed_target_confidence",
                    "rhythm_retimed_target_surface_name",
                ]
            )
        return tuple(dict.fromkeys(keys))

    def validate_retimed_cache_contract(self, item, *, item_name: str):
        if "rhythm_retimed_target_source_id" not in item:
            raise RuntimeError(
                f"Rhythm cached_only requires rhythm_retimed_target_source_id in {item_name}. Re-binarize the dataset."
            )
        found_source_id = int(self.extract_scalar(item["rhythm_retimed_target_source_id"]))
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
        found_surface = str(self.extract_scalar(item["rhythm_retimed_target_surface_name"]))
        expected_surface = (
            self.owner._resolve_expected_teacher_surface_name()
            if expected_source_id == RHYTHM_RETIMED_SOURCE_TEACHER
            else RHYTHM_GUIDANCE_SURFACE_NAME
        )
        if found_surface != expected_surface:
            raise RuntimeError(
                f"Rhythm retimed surface mismatch in {item_name}: "
                f"found={found_surface}, expected={expected_surface}. Re-binarize the dataset."
            )

    def validate_source_cache_shapes(self, cache, *, item_name: str):
        lengths = {
            key: int(np.asarray(cache[key]).reshape(-1).shape[0])
            for key in self.owner._RHYTHM_SOURCE_CACHE_KEYS
        }
        if len(set(lengths.values())) != 1:
            raise RuntimeError(
                f"Rhythm source cache shape mismatch in {item_name}: {lengths}. Re-binarize the dataset."
            )

    def validate_reference_conditioning_shapes(self, conditioning, *, item_name: str):
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
        slow_topk = int(self.hparams.get("rhythm_slow_topk", 6))

        def _validate_slow_sidecar(memory_key: str, summary_key: str, *, label: str) -> int:
            slow_memory = np.asarray(conditioning[memory_key])
            slow_summary = np.asarray(conditioning[summary_key])
            if slow_memory.ndim != 2 or slow_memory.shape[1] != trace_dim:
                raise RuntimeError(
                    f"{label} memory shape mismatch in {item_name}: found={tuple(slow_memory.shape)}, expected=(*,{trace_dim})."
                )
            if slow_memory.shape[0] > slow_topk:
                raise RuntimeError(
                    f"{label} memory count mismatch in {item_name}: found={slow_memory.shape[0]}, expected<= {slow_topk}."
                )
            if slow_summary.reshape(-1).shape[0] != trace_dim:
                raise RuntimeError(
                    f"{label} summary shape mismatch in {item_name}: found={tuple(slow_summary.shape)}, expected_last_dim={trace_dim}."
                )
            return int(slow_memory.shape[0])

        def _require_complete_optional_group(group, *, label: str) -> bool:
            present = [key for key in group if key in conditioning]
            if present and len(present) != len(group):
                missing = [key for key in group if key not in conditioning]
                raise RuntimeError(
                    f"{label} sidecar incomplete in {item_name}: present={present}, missing={missing}."
                )
            return len(present) == len(group)

        base_sidecar_keys = self.owner._RHYTHM_REF_DEBUG_CACHE_KEYS
        if _require_complete_optional_group(base_sidecar_keys, label="Reference"):
            selector_len = _validate_slow_sidecar(
                "slow_rhythm_memory",
                "slow_rhythm_summary",
                label="Slow rhythm",
            )
            selector_indices = np.asarray(conditioning["selector_meta_indices"]).reshape(-1)
            selector_scores = np.asarray(conditioning["selector_meta_scores"]).reshape(-1)
            selector_starts = np.asarray(conditioning["selector_meta_starts"]).reshape(-1)
            selector_ends = np.asarray(conditioning["selector_meta_ends"]).reshape(-1)
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

        planner_sidecar_keys = getattr(self.owner, "_RHYTHM_REF_PLANNER_DEBUG_CACHE_KEYS", ())
        if planner_sidecar_keys and _require_complete_optional_group(planner_sidecar_keys, label="Planner reference"):
            _validate_slow_sidecar(
                "planner_slow_rhythm_memory",
                "planner_slow_rhythm_summary",
                label="Planner slow rhythm",
            )

    def validate_target_shapes(self, targets, *, item_name: str, expected_units: int):
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
        for key in (
            "rhythm_speech_budget_tgt",
            "rhythm_pause_budget_tgt",
            "rhythm_teacher_speech_budget_tgt",
            "rhythm_teacher_pause_budget_tgt",
        ):
            if key not in targets:
                continue
            found = int(np.asarray(targets[key]).reshape(-1).shape[0])
            if found != 1:
                raise RuntimeError(
                    f"Rhythm budget target shape mismatch in {item_name} for {key}: expected scalar/[1], found={tuple(np.asarray(targets[key]).shape)}."
                )
        if "rhythm_retimed_mel_tgt" not in targets and "rhythm_retimed_mel_len" not in targets and "rhythm_retimed_frame_weight" not in targets:
            return
        required = ("rhythm_retimed_mel_tgt", "rhythm_retimed_mel_len", "rhythm_retimed_frame_weight")
        missing = [key for key in required if key not in targets]
        if missing:
            raise RuntimeError(
                f"Rhythm retimed cache incomplete in {item_name}, missing={missing}. Re-binarize the dataset."
            )
        mel = np.asarray(targets["rhythm_retimed_mel_tgt"])
        frame_weight = np.asarray(targets["rhythm_retimed_frame_weight"]).reshape(-1)
        mel_len = int(self.extract_scalar(targets["rhythm_retimed_mel_len"]))
        if mel.ndim != 2:
            raise RuntimeError(
                f"Rhythm retimed mel target must be rank-2 in {item_name}, found shape={tuple(mel.shape)}."
            )
        if mel.shape[0] != mel_len or frame_weight.shape[0] != mel_len:
            raise RuntimeError(
                f"Rhythm retimed cache length mismatch in {item_name}: mel_len={mel_len}, "
                f"mel_frames={mel.shape[0]}, weight_frames={frame_weight.shape[0]}."
            )

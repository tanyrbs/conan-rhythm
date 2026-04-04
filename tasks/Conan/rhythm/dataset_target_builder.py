from __future__ import annotations

import numpy as np

from modules.Conan.rhythm.prefix_state import build_prefix_state_from_exec_numpy
from modules.Conan.rhythm.supervision import (
    RHYTHM_RETIMED_SOURCE_GUIDANCE,
    RHYTHM_RETIMED_SOURCE_TEACHER,
    build_reference_guided_targets,
    build_reference_teacher_targets,
    build_retimed_mel_target,
    with_blank_aliases,
)


class RhythmDatasetTargetBuilder:
    _RHYTHM_PREFIX_UNIT_TARGET_KEYS = (
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
    )
    _RHYTHM_PREFIX_TAIL_RESCALE_KEYS = (
        "rhythm_speech_exec_tgt",
        "rhythm_pause_exec_tgt",
        "rhythm_blank_exec_tgt",
        "rhythm_guidance_speech_tgt",
        "rhythm_guidance_pause_tgt",
        "rhythm_guidance_blank_tgt",
        "rhythm_teacher_speech_exec_tgt",
        "rhythm_teacher_pause_exec_tgt",
        "rhythm_teacher_blank_exec_tgt",
    )
    _RHYTHM_PREFIX_BUDGET_REDUCTION_PAIRS = (
        ("rhythm_speech_exec_tgt", "rhythm_speech_budget_tgt"),
        ("rhythm_pause_exec_tgt", "rhythm_pause_budget_tgt"),
        ("rhythm_teacher_speech_exec_tgt", "rhythm_teacher_speech_budget_tgt"),
        ("rhythm_teacher_pause_exec_tgt", "rhythm_teacher_pause_budget_tgt"),
    )

    def __init__(self, owner) -> None:
        self.owner = owner

    @property
    def hparams(self):
        return self.owner.hparams

    @staticmethod
    def _expected_units_from_source_cache(source_cache) -> int:
        return int(np.asarray(source_cache["dur_anchor_src"]).reshape(-1).shape[0])

    def _collect_cached_target_payload(self, item) -> dict:
        payload = {key: item[key] for key in self.owner._RHYTHM_TARGET_KEYS if key in item}
        payload.update({key: item[key] for key in self.owner._RHYTHM_META_KEYS if key in item})
        return payload

    def _validate_cached_teacher_identity(self, item, *, item_name: str) -> None:
        if "rhythm_teacher_surface_name" in item:
            teacher_name = str(self.owner._extract_scalar(item["rhythm_teacher_surface_name"]))
            expected_teacher_name = self.owner._resolve_expected_teacher_surface_name()
            if teacher_name != expected_teacher_name:
                raise RuntimeError(
                    f"Rhythm teacher surface mismatch in {item_name}: "
                    f"found={teacher_name}, expected={expected_teacher_name}. Re-binarize the dataset."
                )
        if "rhythm_teacher_target_source_id" in item:
            found_source_id = int(self.owner._extract_scalar(item["rhythm_teacher_target_source_id"]))
            expected_source_id = self.owner._resolve_expected_teacher_target_source_id()
            if found_source_id != expected_source_id:
                raise RuntimeError(
                    f"Rhythm teacher target source mismatch in {item_name}: "
                    f"found={found_source_id}, expected={expected_source_id}. Re-binarize the dataset."
                )

    @staticmethod
    def _resolve_prefix_alignment(item, source_cache) -> dict[str, np.ndarray | int | bool]:
        visible_anchor_raw = np.asarray(source_cache["dur_anchor_src"]).reshape(-1)
        visible_units = int(visible_anchor_raw.shape[0])
        if "dur_anchor_src" in item:
            full_anchor_raw = np.asarray(item["dur_anchor_src"]).reshape(-1)
        else:
            full_anchor_raw = visible_anchor_raw.copy()
        full_units = int(full_anchor_raw.shape[0])
        full_prefix_anchor_raw = full_anchor_raw[:visible_units]
        is_truncated = visible_units < full_units
        if not is_truncated and full_prefix_anchor_raw.shape[0] == visible_units:
            is_truncated = not np.allclose(
                visible_anchor_raw,
                full_prefix_anchor_raw,
                atol=1e-5,
                rtol=1e-5,
            )
        return {
            "visible_units": visible_units,
            "visible_anchor": visible_anchor_raw.astype(np.float32),
            "full_prefix_anchor": full_prefix_anchor_raw.astype(np.float32),
            "is_truncated": bool(is_truncated),
        }

    @classmethod
    def _slice_prefix_unit_targets(cls, adapted, *, visible_units: int) -> None:
        for key in cls._RHYTHM_PREFIX_UNIT_TARGET_KEYS:
            if key in adapted:
                adapted[key] = np.asarray(adapted[key]).reshape(-1)[:visible_units].astype(np.float32)

    @classmethod
    def _apply_prefix_tail_ratio(cls, adapted, *, tail_ratio: float) -> None:
        tail_ratio = float(np.clip(tail_ratio, 0.0, 1.0))
        if tail_ratio >= 0.999999:
            return
        for key in cls._RHYTHM_PREFIX_TAIL_RESCALE_KEYS:
            if key in adapted:
                arr = np.asarray(adapted[key]).reshape(-1).astype(np.float32)
                if arr.shape[0] > 0:
                    arr[-1] *= float(tail_ratio)
                adapted[key] = arr

    @classmethod
    def _refresh_prefix_budget_targets(cls, adapted) -> None:
        for exec_key, budget_key in cls._RHYTHM_PREFIX_BUDGET_REDUCTION_PAIRS:
            if exec_key in adapted:
                adapted[budget_key] = np.asarray([float(np.asarray(adapted[exec_key]).sum())], dtype=np.float32)
        if "rhythm_pause_budget_tgt" in adapted:
            adapted["rhythm_blank_budget_tgt"] = adapted["rhythm_pause_budget_tgt"].copy()
        if "rhythm_teacher_pause_budget_tgt" in adapted:
            adapted["rhythm_teacher_blank_budget_tgt"] = adapted["rhythm_teacher_pause_budget_tgt"].copy()

    @staticmethod
    def _refresh_prefix_teacher_targets(adapted, *, visible_anchor: np.ndarray) -> None:
        if "rhythm_teacher_allocation_tgt" in adapted:
            if "rhythm_teacher_speech_exec_tgt" in adapted and "rhythm_teacher_pause_exec_tgt" in adapted:
                alloc = (
                    np.asarray(adapted["rhythm_teacher_speech_exec_tgt"]).reshape(-1).astype(np.float32)
                    + np.asarray(adapted["rhythm_teacher_pause_exec_tgt"]).reshape(-1).astype(np.float32)
                )
            else:
                alloc = np.asarray(adapted["rhythm_teacher_allocation_tgt"]).reshape(-1).astype(np.float32)
            alloc = alloc * (visible_anchor > 0).astype(np.float32)
            denom = float(np.maximum(alloc.sum(), 1e-6))
            adapted["rhythm_teacher_allocation_tgt"] = alloc / denom
        if "rhythm_teacher_speech_exec_tgt" in adapted and "rhythm_teacher_pause_exec_tgt" in adapted:
            prefix_clock, prefix_backlog = build_prefix_state_from_exec_numpy(
                speech_exec=adapted["rhythm_teacher_speech_exec_tgt"],
                pause_exec=adapted["rhythm_teacher_pause_exec_tgt"],
                dur_anchor_src=visible_anchor,
                unit_mask=(visible_anchor > 0).astype(np.float32),
            )
            adapted["rhythm_teacher_prefix_clock_tgt"] = prefix_clock.astype(np.float32)
            adapted["rhythm_teacher_prefix_backlog_tgt"] = prefix_backlog.astype(np.float32)

    def _refresh_prefix_retimed_target(self, adapted, *, source_cache, sample) -> None:
        source_id = int(
            np.asarray(adapted.get("rhythm_retimed_target_source_id", [RHYTHM_RETIMED_SOURCE_GUIDANCE])).reshape(-1)[0]
        )
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

    def adapt_cached_targets_to_prefix(self, *, item, cached_targets, source_cache, sample):
        alignment = self._resolve_prefix_alignment(item, source_cache)
        if not alignment["is_truncated"]:
            return with_blank_aliases(dict(cached_targets))
        adapted = with_blank_aliases(dict(cached_targets))
        visible_units = int(alignment["visible_units"])
        visible_anchor = np.asarray(alignment["visible_anchor"]).reshape(-1).astype(np.float32)
        full_prefix_anchor = np.asarray(alignment["full_prefix_anchor"]).reshape(-1).astype(np.float32)
        self._slice_prefix_unit_targets(adapted, visible_units=visible_units)
        tail_ratio = 1.0
        if visible_units > 0 and full_prefix_anchor.shape[0] >= visible_units:
            tail_ratio = float(visible_anchor[-1] / max(float(full_prefix_anchor[-1]), 1e-6))
            tail_ratio = float(np.clip(tail_ratio, 0.0, 1.0))
        self._apply_prefix_tail_ratio(adapted, tail_ratio=tail_ratio)
        self._refresh_prefix_budget_targets(adapted)
        self._refresh_prefix_teacher_targets(adapted, visible_anchor=visible_anchor)
        if "rhythm_retimed_mel_tgt" in adapted:
            self._refresh_prefix_retimed_target(
                adapted,
                source_cache=source_cache,
                sample=sample,
            )
        return with_blank_aliases(adapted)

    def _resolve_runtime_target_generation_flags(self) -> dict[str, bool | str]:
        primary_surface = self.owner._resolve_primary_target_surface()
        distill_surface = self.owner._resolve_distill_surface()
        retimed_source = str(
            self.hparams.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance"
        ).strip().lower()
        return {
            "teacher_target_source": self.owner._resolve_teacher_target_source(),
            "need_guidance": (
                primary_surface == "guidance"
                or float(self.hparams.get("lambda_rhythm_guidance", 0.0)) > 0.0
            ),
            "need_teacher": (
                primary_surface == "teacher"
                or bool(self.hparams.get("rhythm_require_cached_teacher", False))
                or (
                    float(self.hparams.get("lambda_rhythm_distill", 0.0)) > 0.0
                    and distill_surface == "algorithmic"
                )
                or (
                    bool(self.hparams.get("rhythm_use_retimed_target_if_available", False))
                    and retimed_source == "teacher"
                )
            ),
        }

    def _build_runtime_guidance_kwargs(self, source_cache, ref_conditioning, *, unit_mask):
        return {
            "dur_anchor_src": source_cache["dur_anchor_src"],
            "unit_mask": unit_mask,
            "ref_rhythm_stats": ref_conditioning["ref_rhythm_stats"],
            "ref_rhythm_trace": ref_conditioning["ref_rhythm_trace"],
            "rate_scale_min": float(self.hparams.get("rhythm_guidance_rate_scale_min", 0.60)),
            "rate_scale_max": float(self.hparams.get("rhythm_guidance_rate_scale_max", 1.80)),
            "local_rate_strength": float(self.hparams.get("rhythm_guidance_local_rate_strength", 0.35)),
            "segment_bias_strength": float(self.hparams.get("rhythm_guidance_segment_bias_strength", 0.25)),
            "pause_strength": float(self.hparams.get("rhythm_guidance_pause_strength", 1.00)),
            "boundary_strength": float(self.hparams.get("rhythm_guidance_boundary_strength", 1.25)),
            "pause_budget_ratio_cap": float(self.hparams.get("rhythm_guidance_pause_budget_ratio_cap", 0.75)),
        }

    def _build_runtime_teacher_kwargs(self, source_cache, ref_conditioning, *, unit_mask):
        return {
            "dur_anchor_src": source_cache["dur_anchor_src"],
            "unit_mask": unit_mask,
            "source_boundary_cue": source_cache.get("source_boundary_cue", source_cache.get("boundary_confidence")),
            "ref_rhythm_stats": ref_conditioning["ref_rhythm_stats"],
            "ref_rhythm_trace": ref_conditioning["ref_rhythm_trace"],
            "rate_scale_min": float(self.hparams.get("rhythm_teacher_rate_scale_min", 0.55)),
            "rate_scale_max": float(self.hparams.get("rhythm_teacher_rate_scale_max", 1.95)),
            "local_rate_strength": float(self.hparams.get("rhythm_teacher_local_rate_strength", 0.45)),
            "segment_bias_strength": float(self.hparams.get("rhythm_teacher_segment_bias_strength", 0.30)),
            "pause_strength": float(self.hparams.get("rhythm_teacher_pause_strength", 1.10)),
            "boundary_strength": float(self.hparams.get("rhythm_teacher_boundary_strength", 1.50)),
            "pause_budget_ratio_cap": float(self.hparams.get("rhythm_teacher_pause_budget_ratio_cap", 0.80)),
            "speech_smooth_kernel": int(self.hparams.get("rhythm_teacher_speech_smooth_kernel", 3)),
            "pause_topk_ratio": float(self.hparams.get("rhythm_teacher_pause_topk_ratio", 0.30)),
        }

    def build_runtime_rhythm_targets(self, source_cache, ref_conditioning):
        unit_mask = (np.asarray(source_cache["dur_anchor_src"]) > 0).astype(np.float32)
        flags = self._resolve_runtime_target_generation_flags()
        targets = {}
        if self.hparams.get("rhythm_dataset_build_guidance_from_ref", True) or flags["need_guidance"]:
            targets.update(
                build_reference_guided_targets(
                    **self._build_runtime_guidance_kwargs(
                        source_cache,
                        ref_conditioning,
                        unit_mask=unit_mask,
                    )
                )
            )
        if self.hparams.get("rhythm_dataset_build_teacher_from_ref", False) or flags["need_teacher"]:
            if flags["teacher_target_source"] != "algorithmic":
                if flags["need_teacher"]:
                    raise RuntimeError(
                        "Rhythm runtime teacher synthesis only supports rhythm_teacher_target_source=algorithmic. "
                        "Use cached_only with precomputed learned_offline teacher surfaces."
                    )
            else:
                targets.update(
                    build_reference_teacher_targets(
                        **self._build_runtime_teacher_kwargs(
                            source_cache,
                            ref_conditioning,
                            unit_mask=unit_mask,
                        )
                    )
                )
        return targets

    def merge_rhythm_targets(self, *, item, source_cache, ref_conditioning, sample):
        target_mode = self.owner._resolve_rhythm_target_mode()
        cached_targets = self._collect_cached_target_payload(item)
        if target_mode == "cached_only":
            item_name = str(item.get("item_name", "<unknown-item>"))
            self.owner._validate_rhythm_cache_contract(item, item_name=item_name)
            self.owner._require_cached_keys(
                item=item,
                keys=self.owner._required_cached_target_keys(),
                item_name=item_name,
                reason="rhythm targets/meta cache",
            )
            self._validate_cached_teacher_identity(item, item_name=item_name)
            if bool(self.hparams.get("rhythm_require_retimed_cache", False)):
                self.owner._validate_retimed_cache_contract(item, item_name=item_name)
            cached_targets = self.adapt_cached_targets_to_prefix(
                item=item,
                cached_targets=cached_targets,
                source_cache=source_cache,
                sample=sample,
            )
            self.owner._validate_target_shapes(
                cached_targets,
                item_name=item_name,
                expected_units=self._expected_units_from_source_cache(source_cache),
            )
            return cached_targets

        runtime_targets = self.build_runtime_rhythm_targets(source_cache, ref_conditioning)
        if target_mode == "runtime_only":
            return runtime_targets

        merged = dict(cached_targets)
        for key, value in runtime_targets.items():
            merged.setdefault(key, value)
        if "rhythm_speech_exec_tgt" in merged:
            merged = self.adapt_cached_targets_to_prefix(
                item=item,
                cached_targets=merged,
                source_cache=source_cache,
                sample=sample,
            )
            self.owner._validate_target_shapes(
                merged,
                item_name=str(item.get("item_name", "<unknown-item>")),
                expected_units=self._expected_units_from_source_cache(source_cache),
            )
        return merged


__all__ = ["RhythmDatasetTargetBuilder"]

"""Compatibility shell for the split Conan rhythm dataset mixins."""

from __future__ import annotations

from tasks.Conan.rhythm.common.dataset_mixin import CommonRhythmDatasetMixin
from tasks.Conan.rhythm.duration_v3.dataset_mixin import DurationV3DatasetMixin
from tasks.Conan.rhythm.rhythm_v2.dataset_mixin import RhythmV2DatasetMixin


class RhythmConanDatasetMixin(DurationV3DatasetMixin, RhythmV2DatasetMixin, CommonRhythmDatasetMixin):
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
    _RHYTHM_REF_PROMPT_UNIT_KEYS = (
        "prompt_content_units",
        "prompt_duration_obs",
        "prompt_unit_mask",
        "prompt_valid_mask",
        "prompt_speech_mask",
        "prompt_global_weight",
        "prompt_unit_log_prior",
        "prompt_unit_anchor_base",
        "prompt_log_base",
        "prompt_source_boundary_cue",
        "prompt_phrase_group_pos",
        "prompt_phrase_final_mask",
    )
    _RHYTHM_REF_PROMPT_SOURCE_KEYS = (
        "content_units",
        "dur_anchor_src",
        "source_silence_mask",
    )
    _RHYTHM_REF_PHRASE_CACHE_KEYS = (
        "ref_phrase_trace",
        "planner_ref_phrase_trace",
        "ref_phrase_valid",
        "ref_phrase_lengths",
        "ref_phrase_starts",
        "ref_phrase_ends",
        "ref_phrase_boundary_strength",
        "ref_phrase_stats",
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
        "unit_anchor_base",
        "unit_rate_log_base",
        "open_run_mask",
        "sealed_mask",
        "sep_hint",
        "boundary_confidence",
        "source_silence_mask",
        "source_run_stability",
        "prompt_content_units",
        "prompt_duration_obs",
        "prompt_unit_mask",
        "prompt_valid_mask",
        "prompt_speech_mask",
        "prompt_global_weight",
        "prompt_unit_log_prior",
        "prompt_unit_anchor_base",
        "prompt_log_base",
        "prompt_source_boundary_cue",
        "prompt_phrase_group_pos",
        "prompt_phrase_final_mask",
        "unit_duration_tgt",
        "unit_duration_proj_raw_tgt",
        "unit_confidence_tgt",
        "unit_confidence_local_tgt",
        "unit_confidence_coarse_tgt",
        "unit_alignment_coverage_tgt",
        "unit_alignment_match_tgt",
        "unit_alignment_cost_tgt",
        "unit_alignment_mode_id_tgt",
        "unit_alignment_kind_tgt",
        "unit_alignment_source_tgt",
        "unit_alignment_version_tgt",
    )
    _RHYTHM_STREAMING_PREFIX_META_KEYS = (
        "rhythm_stream_prefix_ratio",
        "rhythm_stream_visible_units",
        "rhythm_stream_full_units",
    )
    _RHYTHM_RUNTIME_REFERENCE_META_KEYS = (
        "rhythm_reference_is_self",
        "rhythm_pair_group_id",
        "rhythm_pair_rank",
        "rhythm_pair_is_identity",
        "alignment_source",
        "alignment_version",
    )
    _RHYTHM_STREAMING_OFFLINE_SOURCE_KEYS = (
        "rhythm_offline_content_units",
        "rhythm_offline_dur_anchor_src",
        "rhythm_offline_source_silence_mask",
        "rhythm_offline_source_run_stability",
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
        "ref_phrase_trace",
        "planner_ref_phrase_trace",
        "ref_phrase_valid",
        "ref_phrase_lengths",
        "ref_phrase_starts",
        "ref_phrase_ends",
        "ref_phrase_boundary_strength",
        "ref_phrase_stats",
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


__all__ = ["RhythmConanDatasetMixin"]

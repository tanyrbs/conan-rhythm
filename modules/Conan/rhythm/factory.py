from __future__ import annotations

from .module import StreamingRhythmModule
from .offline_teacher import OfflineTeacherConfig
from .policy import (
    resolve_runtime_offline_teacher_enable as resolve_runtime_offline_teacher_enable_from_policy,
)
from .projector import ProjectorConfig
from .teacher import AlgorithmicTeacherConfig


def resolve_content_vocab_size(hparams) -> int:
    for key in ("content_vocab_size", "content_num_units", "content_num_embeddings", "content_embedding_dim"):
        value = hparams.get(key, None)
        if value is not None:
            return int(value)
    return 102


def build_projector_config_from_hparams(hparams) -> ProjectorConfig:
    pause_selection_mode = str(
        hparams.get(
            'rhythm_projector_pause_selection_mode',
            # Maintained mainline stages also rely on pause-topk annealing.
            # If strict_mainline silently falls back to "simple", the resolved
            # top-k schedule is logged but never actually affects the projector.
            'sparse',
        )
        or 'sparse'
    ).strip().lower()
    if pause_selection_mode not in {'simple', 'sparse'}:
        raise ValueError(f'Unsupported rhythm_projector_pause_selection_mode: {pause_selection_mode}')
    return ProjectorConfig(
        min_speech_frames=float(hparams.get('rhythm_projector_min_speech_frames', 1.0)),
        max_speech_expand=float(hparams.get('rhythm_projector_max_speech_expand', 3.0)),
        tail_hold_units=int(hparams.get('rhythm_projector_tail_hold_units', 2)),
        boundary_commit_threshold=float(hparams.get('rhythm_projector_boundary_commit_threshold', 0.45)),
        pause_topk_ratio=float(hparams.get('rhythm_projector_pause_topk_ratio', 0.35)),
        pause_min_boundary_weight=float(hparams.get('rhythm_projector_pause_min_boundary_weight', 0.10)),
        pause_boundary_bias_weight=float(hparams.get('rhythm_projector_pause_boundary_bias_weight', 0.15)),
        pause_train_soft=bool(hparams.get('rhythm_projector_pause_train_soft', True)),
        pause_soft_temperature=float(hparams.get('rhythm_projector_pause_soft_temperature', 0.12)),
        pause_selection_mode=pause_selection_mode,
        use_boundary_commit_guard=bool(
            # Streaming prefix release is safer with the guard on by default.
            hparams.get('rhythm_projector_use_boundary_commit_guard', True)
        ),
        build_render_plan=bool(
            # Keep render-plan construction on by default so retimed-target and
            # observability paths stay aligned with the execution authority.
            hparams.get('rhythm_projector_build_render_plan', True)
        ),
    )


def resolve_runtime_offline_teacher_enable(hparams, *, stage: str | None = None, config_path: str | None = None) -> bool:
    return resolve_runtime_offline_teacher_enable_from_policy(
        hparams,
        stage=stage,
        config_path=config_path,
    )


def _resolve_runtime_offline_teacher_enable(hparams) -> bool:
    # Backward-compatible private alias for older call sites.
    return resolve_runtime_offline_teacher_enable(hparams)


def _trace_exhaustion_fallback_requested(hparams) -> bool:
    return bool(
        hparams.get(
            'rhythm_trace_reliability_enable',
            hparams.get('rhythm_enable_trace_exhaustion_fallback', False),
        )
    )


def _should_auto_enable_reference_sidecar(hparams) -> bool:
    cached_reference_policy = str(hparams.get('rhythm_cached_reference_policy', '') or '').strip().lower()
    descriptor_bootstrap_active = any(
        float(hparams.get(key, 0.0) or 0.0) > 0.0
        for key in (
            'lambda_rhythm_ref_descriptor_stats',
            'lambda_rhythm_ref_descriptor_trace',
            'lambda_rhythm_ref_group_contrastive',
        )
    )
    return (
        bool(hparams.get('rhythm_require_external_reference', False))
        or cached_reference_policy == 'sample_ref'
        or descriptor_bootstrap_active
        or _trace_exhaustion_fallback_requested(hparams)
    )


def build_offline_teacher_config_from_hparams(hparams) -> OfflineTeacherConfig:
    phrase_kernels = hparams.get('rhythm_offline_teacher_phrase_kernels', (3, 7))
    if isinstance(phrase_kernels, int):
        phrase_kernels = (int(phrase_kernels),)
    return OfflineTeacherConfig(
        num_blocks=int(hparams.get('rhythm_offline_teacher_num_blocks', 6)),
        kernel_size=int(hparams.get('rhythm_offline_teacher_kernel_size', 5)),
        dilations=tuple(int(x) for x in hparams.get('rhythm_offline_teacher_dilations', (1, 2, 4, 8, 2, 1))),
        phrase_kernel_sizes=tuple(int(x) for x in phrase_kernels),
        global_gate_scale=float(hparams.get('rhythm_offline_teacher_global_gate_scale', 0.12)),
        confidence_agreement_weight=float(hparams.get('rhythm_offline_teacher_confidence_agreement_weight', 0.25)),
        confidence_floor=float(hparams.get('rhythm_offline_teacher_confidence_floor', 0.05)),
        confidence_ceiling=float(hparams.get('rhythm_offline_teacher_confidence_ceiling', 1.0)),
        max_total_logratio=float(hparams.get('rhythm_max_total_logratio', 0.8)),
        max_unit_logratio=float(hparams.get('rhythm_max_unit_logratio', 0.6)),
        pause_share_max=float(hparams.get('rhythm_pause_share_max', 0.45)),
        pause_share_residual_max=float(hparams.get('rhythm_pause_share_residual_max', 0.12)),
        boundary_feature_scale=float(hparams.get('rhythm_boundary_feature_scale', 0.35)),
        boundary_source_cue_weight=float(hparams.get('rhythm_boundary_source_cue_weight', 0.65)),
        pause_source_boundary_weight=float(hparams.get('rhythm_pause_source_boundary_weight', 0.20)),
        min_speech_frames=float(hparams.get('rhythm_projector_min_speech_frames', 1.0)),
    )


def build_streaming_rhythm_module_from_hparams(hparams) -> StreamingRhythmModule:
    num_units = resolve_content_vocab_size(hparams)
    emit_reference_sidecar = hparams.get(
        'rhythm_emit_reference_sidecar',
        hparams.get('rhythm_reference_emit_sidecar', None),
    )
    if emit_reference_sidecar is not None:
        emit_reference_sidecar = bool(emit_reference_sidecar)
    else:
        # Keep the maintained utterance-bounded default lean, but do not require
        # every external-reference / descriptor-bootstrap config to remember the
        # sidecar knob manually. Those regimes are the ones that benefit most
        # from slow-memory summaries and trace-exhaustion fallback.
        emit_reference_sidecar = _should_auto_enable_reference_sidecar(hparams)
    return StreamingRhythmModule(
        num_units=num_units,
        hidden_size=int(hparams.get('rhythm_hidden_size', hparams.get('hidden_size', 256))),
        trace_bins=int(hparams.get('rhythm_trace_bins', 24)),
        # These configure the public cached descriptor contract.
        # Planner-internal dims are fixed and validated inside StreamingRhythmModule.
        stats_dim=int(hparams.get('rhythm_stats_dim', 6)),
        trace_dim=int(hparams.get('rhythm_trace_dim', 5)),
        trace_horizon=float(hparams.get('rhythm_trace_horizon', 0.35)),
        slow_topk=int(hparams.get('rhythm_slow_topk', 6)),
        selector_cell_size=int(hparams.get('rhythm_selector_cell_size', 3)),
        trace_smooth_kernel=int(hparams.get('rhythm_trace_smooth_kernel', 5)),
        maintained_stats_trace_only=bool(hparams.get('rhythm_maintained_stats_trace_only', True)),
        emit_reference_sidecar=emit_reference_sidecar,
        max_total_logratio=float(hparams.get('rhythm_max_total_logratio', 0.8)),
        max_unit_logratio=float(hparams.get('rhythm_max_unit_logratio', 0.6)),
        pause_share_max=float(hparams.get('rhythm_pause_share_max', 0.45)),
        pause_share_residual_max=float(hparams.get('rhythm_pause_share_residual_max', 0.12)),
        boundary_feature_scale=float(hparams.get('rhythm_boundary_feature_scale', 0.35)),
        boundary_source_cue_weight=float(hparams.get('rhythm_boundary_source_cue_weight', 0.65)),
        pause_source_boundary_weight=float(hparams.get('rhythm_pause_source_boundary_weight', 0.20)),
        min_speech_frames=float(hparams.get('rhythm_projector_min_speech_frames', 1.0)),
        trace_reliability_enable=_trace_exhaustion_fallback_requested(hparams),
        trace_exhaustion_gap_start=float(hparams.get('rhythm_trace_exhaustion_gap_start', 0.08)),
        trace_exhaustion_gap_end=float(hparams.get('rhythm_trace_exhaustion_gap_end', 0.22)),
        trace_exhaustion_local_floor=float(hparams.get('rhythm_trace_exhaustion_local_floor', 0.20)),
        trace_exhaustion_boundary_floor=float(hparams.get('rhythm_trace_exhaustion_boundary_floor', 0.05)),
        trace_exhaustion_reuse_full_count=int(hparams.get('rhythm_trace_exhaustion_reuse_full_count', 3)),
        trace_exhaustion_final_cell_suppress=float(
            hparams.get('rhythm_trace_exhaustion_final_cell_suppress', 0.65)
        ),
        trace_anchor_aware_sampling=bool(hparams.get('rhythm_trace_anchor_aware_sampling', False)),
        projector_config=build_projector_config_from_hparams(hparams),
        enable_learned_offline_teacher=resolve_runtime_offline_teacher_enable(hparams),
        offline_teacher_config=build_offline_teacher_config_from_hparams(hparams),
        teacher_config=AlgorithmicTeacherConfig(
            rate_scale_min=float(hparams.get('rhythm_teacher_rate_scale_min', 0.55)),
            rate_scale_max=float(hparams.get('rhythm_teacher_rate_scale_max', 1.95)),
            local_rate_strength=float(hparams.get('rhythm_teacher_local_rate_strength', 0.45)),
            segment_bias_strength=float(hparams.get('rhythm_teacher_segment_bias_strength', 0.30)),
            pause_strength=float(hparams.get('rhythm_teacher_pause_strength', 1.10)),
            boundary_strength=float(hparams.get('rhythm_teacher_boundary_strength', 1.50)),
            source_boundary_pause_weight=float(hparams.get('rhythm_teacher_source_boundary_pause_weight', 0.35)),
            source_boundary_prior_clip=float(hparams.get('rhythm_teacher_source_boundary_prior_clip', 1.50)),
            source_boundary_gate_floor=float(hparams.get('rhythm_teacher_source_boundary_gate_floor', 0.05)),
            source_boundary_gate_ceiling=float(hparams.get('rhythm_teacher_source_boundary_gate_ceiling', 0.55)),
            source_boundary_agreement_center=float(hparams.get('rhythm_teacher_source_boundary_agreement_center', 0.15)),
            source_boundary_agreement_scale=float(hparams.get('rhythm_teacher_source_boundary_agreement_scale', 4.0)),
            pause_budget_ratio_cap=float(hparams.get('rhythm_teacher_pause_budget_ratio_cap', 0.80)),
            speech_smooth_kernel=int(hparams.get('rhythm_teacher_speech_smooth_kernel', 3)),
            pause_topk_ratio=float(hparams.get('rhythm_teacher_pause_topk_ratio', 0.30)),
            phrase_final_bonus=float(hparams.get('rhythm_teacher_phrase_final_bonus', 0.20)),
            confidence_bonus=float(hparams.get('rhythm_teacher_confidence_bonus', 0.05)),
            anchor_aware_trace_sampling=bool(hparams.get('rhythm_trace_anchor_aware_sampling', False)),
        ),
    )

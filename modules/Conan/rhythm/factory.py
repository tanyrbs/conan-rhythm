from __future__ import annotations

from .module import StreamingRhythmModule
from .offline_teacher import OfflineTeacherConfig
from .projector import ProjectorConfig
from .teacher import AlgorithmicTeacherConfig


def build_projector_config_from_hparams(hparams) -> ProjectorConfig:
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
    )


def _resolve_runtime_offline_teacher_enable(hparams) -> bool:
    # Explicit runtime override always wins for experiments/debug.
    explicit_runtime = hparams.get('rhythm_runtime_enable_learned_offline_teacher', None)
    if explicit_runtime is not None:
        return bool(explicit_runtime)

    # Dual-mode branch requires the learned offline teacher runtime path.
    if bool(hparams.get('rhythm_enable_dual_mode_teacher', False)):
        return True

    # Maintained defaults: do not keep runtime teacher alive in schedule-only / non-KD paths.
    schedule_only = bool(hparams.get('rhythm_schedule_only_stage', False))
    if schedule_only:
        return False

    legacy_enable = bool(hparams.get('rhythm_enable_learned_offline_teacher', False))
    if not legacy_enable:
        return False

    lambda_distill = float(hparams.get('lambda_rhythm_distill', 0.0) or 0.0)
    distill_surface = str(hparams.get('rhythm_distill_surface', 'none') or 'none').strip().lower()
    offline_distill_surface = distill_surface in {'offline', 'full_context', 'shared_offline'}
    return lambda_distill > 0.0 and offline_distill_surface


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
        pause_trace_weight=float(hparams.get('rhythm_offline_teacher_pause_trace_weight', 0.30)),
        boundary_trace_weight=float(hparams.get('rhythm_offline_teacher_boundary_trace_weight', 0.30)),
        confidence_agreement_weight=float(hparams.get('rhythm_offline_teacher_confidence_agreement_weight', 0.25)),
        confidence_floor=float(hparams.get('rhythm_offline_teacher_confidence_floor', 0.05)),
        confidence_ceiling=float(hparams.get('rhythm_offline_teacher_confidence_ceiling', 1.0)),
        max_total_logratio=float(hparams.get('rhythm_max_total_logratio', 0.8)),
        max_unit_logratio=float(hparams.get('rhythm_max_unit_logratio', 0.6)),
        pause_share_max=float(hparams.get('rhythm_pause_share_max', 0.45)),
        boundary_feature_scale=float(hparams.get('rhythm_boundary_feature_scale', 0.35)),
        boundary_source_cue_weight=float(hparams.get('rhythm_boundary_source_cue_weight', 0.20)),
        pause_boundary_latent_weight=float(hparams.get('rhythm_pause_boundary_latent_weight', 0.25)),
        pause_source_boundary_weight=float(hparams.get('rhythm_pause_source_boundary_weight', 0.10)),
        min_speech_frames=float(hparams.get('rhythm_projector_min_speech_frames', 1.0)),
    )


def build_streaming_rhythm_module_from_hparams(hparams) -> StreamingRhythmModule:
    num_units = int(
        hparams.get(
            'content_vocab_size',
            hparams.get('content_num_units', hparams.get('content_num_embeddings', 102)),
        )
    )
    return StreamingRhythmModule(
        num_units=num_units,
        hidden_size=int(hparams.get('rhythm_hidden_size', hparams.get('hidden_size', 256))),
        trace_bins=int(hparams.get('rhythm_trace_bins', 24)),
        stats_dim=int(hparams.get('rhythm_stats_dim', 6)),
        trace_dim=int(hparams.get('rhythm_trace_dim', 5)),
        trace_horizon=float(hparams.get('rhythm_trace_horizon', 0.35)),
        slow_topk=int(hparams.get('rhythm_slow_topk', 6)),
        selector_cell_size=int(hparams.get('rhythm_selector_cell_size', 3)),
        trace_smooth_kernel=int(hparams.get('rhythm_trace_smooth_kernel', 5)),
        max_total_logratio=float(hparams.get('rhythm_max_total_logratio', 0.8)),
        max_unit_logratio=float(hparams.get('rhythm_max_unit_logratio', 0.6)),
        pause_share_max=float(hparams.get('rhythm_pause_share_max', 0.45)),
        boundary_feature_scale=float(hparams.get('rhythm_boundary_feature_scale', 0.35)),
        boundary_source_cue_weight=float(hparams.get('rhythm_boundary_source_cue_weight', 0.20)),
        pause_boundary_latent_weight=float(hparams.get('rhythm_pause_boundary_latent_weight', 0.25)),
        pause_source_boundary_weight=float(hparams.get('rhythm_pause_source_boundary_weight', 0.10)),
        projector_config=build_projector_config_from_hparams(hparams),
        enable_learned_offline_teacher=_resolve_runtime_offline_teacher_enable(hparams),
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
        ),
    )

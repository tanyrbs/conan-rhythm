from __future__ import annotations

from .module import StreamingRhythmModule
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


def build_streaming_rhythm_module_from_hparams(hparams) -> StreamingRhythmModule:
    return StreamingRhythmModule(
        num_units=int(hparams.get('content_embedding_dim', 102)),
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
        boundary_source_cue_weight=float(hparams.get('rhythm_boundary_source_cue_weight', 0.35)),
        pause_boundary_latent_weight=float(hparams.get('rhythm_pause_boundary_latent_weight', 0.35)),
        pause_source_boundary_weight=float(hparams.get('rhythm_pause_source_boundary_weight', 0.20)),
        projector_config=build_projector_config_from_hparams(hparams),
        teacher_config=AlgorithmicTeacherConfig(
            rate_scale_min=float(hparams.get('rhythm_teacher_rate_scale_min', 0.55)),
            rate_scale_max=float(hparams.get('rhythm_teacher_rate_scale_max', 1.95)),
            local_rate_strength=float(hparams.get('rhythm_teacher_local_rate_strength', 0.45)),
            segment_bias_strength=float(hparams.get('rhythm_teacher_segment_bias_strength', 0.30)),
            pause_strength=float(hparams.get('rhythm_teacher_pause_strength', 1.10)),
            boundary_strength=float(hparams.get('rhythm_teacher_boundary_strength', 1.50)),
            source_boundary_pause_weight=float(hparams.get('rhythm_teacher_source_boundary_pause_weight', 0.35)),
            pause_budget_ratio_cap=float(hparams.get('rhythm_teacher_pause_budget_ratio_cap', 0.80)),
            speech_smooth_kernel=int(hparams.get('rhythm_teacher_speech_smooth_kernel', 3)),
            pause_topk_ratio=float(hparams.get('rhythm_teacher_pause_topk_ratio', 0.30)),
            phrase_final_bonus=float(hparams.get('rhythm_teacher_phrase_final_bonus', 0.20)),
            confidence_bonus=float(hparams.get('rhythm_teacher_confidence_bonus', 0.05)),
        ),
    )

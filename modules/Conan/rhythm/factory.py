from __future__ import annotations

from .module import StreamingRhythmModule
from .projector import ProjectorConfig


def build_projector_config_from_hparams(hparams) -> ProjectorConfig:
    return ProjectorConfig(
        min_speech_frames=float(hparams.get('rhythm_projector_min_speech_frames', 1.0)),
        max_speech_expand=float(hparams.get('rhythm_projector_max_speech_expand', 3.0)),
        tail_hold_units=int(hparams.get('rhythm_projector_tail_hold_units', 2)),
        boundary_commit_threshold=float(hparams.get('rhythm_projector_boundary_commit_threshold', 0.45)),
        pause_topk_ratio=float(hparams.get('rhythm_projector_pause_topk_ratio', 0.35)),
        pause_min_boundary_weight=float(hparams.get('rhythm_projector_pause_min_boundary_weight', 0.10)),
    )


def build_streaming_rhythm_module_from_hparams(hparams) -> StreamingRhythmModule:
    return StreamingRhythmModule(
        num_units=int(hparams.get('content_embedding_dim', 102)),
        hidden_size=int(hparams.get('rhythm_hidden_size', hparams.get('hidden_size', 256))),
        trace_bins=int(hparams.get('rhythm_trace_bins', 24)),
        stats_dim=int(hparams.get('rhythm_stats_dim', 6)),
        trace_dim=int(hparams.get('rhythm_trace_dim', 5)),
        trace_horizon=float(hparams.get('rhythm_trace_horizon', 0.35)),
        trace_smooth_kernel=int(hparams.get('rhythm_trace_smooth_kernel', 5)),
        max_total_logratio=float(hparams.get('rhythm_max_total_logratio', 0.8)),
        max_unit_logratio=float(hparams.get('rhythm_max_unit_logratio', 0.6)),
        pause_share_max=float(hparams.get('rhythm_pause_share_max', 0.45)),
        projector_config=build_projector_config_from_hparams(hparams),
    )

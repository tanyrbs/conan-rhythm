from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "RhythmLossTargets": (".losses", "RhythmLossTargets"),
    "build_rhythm_loss_dict": (".losses", "build_rhythm_loss_dict"),
    "build_rhythm_metric_dict": (".metrics", "build_rhythm_metric_dict"),
    "build_streaming_chunk_metrics": (".metrics", "build_streaming_chunk_metrics"),
    "DistillConfidenceBundle": (".targets", "DistillConfidenceBundle"),
    "RhythmTargetBuildConfig": (".targets", "RhythmTargetBuildConfig"),
    "build_identity_rhythm_loss_targets": (".targets", "build_identity_rhythm_loss_targets"),
    "build_rhythm_loss_targets_from_sample": (".targets", "build_rhythm_loss_targets_from_sample"),
    "resolve_rhythm_sample_keys": (".targets", "resolve_rhythm_sample_keys"),
    "scale_rhythm_loss_terms": (".targets", "scale_rhythm_loss_terms"),
    "compute_committed_mel_length": (".streaming_commit", "compute_committed_mel_length"),
    "extract_incremental_committed_mel": (".streaming_commit", "extract_incremental_committed_mel"),
    "StreamingEvalResult": (".streaming_eval", "StreamingEvalResult"),
    "run_chunkwise_streaming_inference": (".streaming_eval", "run_chunkwise_streaming_inference"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)

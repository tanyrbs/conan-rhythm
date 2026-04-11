from .losses import RhythmLossTargets, build_rhythm_loss_dict, build_rhythm_v2_loss_dict
from .metrics import build_rhythm_metric_dict, build_rhythm_metric_sections
from .runtime_modes import build_legacy_v2_ref_conditioning, collect_legacy_planner_runtime_outputs
from .targets import (
    DistillConfidenceBundle,
    RhythmSampleKeyBundle,
    RhythmTargetBuildConfig,
    build_identity_rhythm_loss_targets,
    build_rhythm_loss_targets_from_sample,
    resolve_rhythm_sample_keys,
    scale_rhythm_loss_terms,
)
from .task_config import validate_rhythm_v2_training_hparams
from .task_runtime_support import RhythmV2TaskRuntimeSupport
from importlib import import_module


def __getattr__(name: str):
    if name == "RhythmV2DatasetMixin":
        module = import_module(".dataset_mixin", __name__)
        value = getattr(module, name)
    elif name == "RhythmV2TaskMixin":
        module = import_module(".task_mixin", __name__)
        value = getattr(module, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value

__all__ = [
    "DistillConfidenceBundle",
    "RhythmLossTargets",
    "RhythmSampleKeyBundle",
    "RhythmTargetBuildConfig",
    "RhythmV2TaskRuntimeSupport",
    "RhythmV2DatasetMixin",
    "RhythmV2TaskMixin",
    "build_identity_rhythm_loss_targets",
    "build_legacy_v2_ref_conditioning",
    "build_rhythm_loss_dict",
    "build_rhythm_loss_targets_from_sample",
    "build_rhythm_metric_dict",
    "build_rhythm_metric_sections",
    "build_rhythm_v2_loss_dict",
    "collect_legacy_planner_runtime_outputs",
    "resolve_rhythm_sample_keys",
    "scale_rhythm_loss_terms",
    "validate_rhythm_v2_training_hparams",
]

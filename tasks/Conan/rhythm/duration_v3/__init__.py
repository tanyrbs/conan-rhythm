from .losses import DurationV3LossTargets, build_duration_v3_loss_dict, build_rhythm_loss_dict
from .metrics import build_duration_v3_metric_sections, build_rhythm_metric_dict, build_rhythm_metric_sections
from .runtime_modes import build_duration_v3_ref_conditioning
from .targets import (
    DurationV3TargetBuildConfig,
    build_duration_v3_loss_targets,
    build_pseudo_source_duration_context,
)
from .task_config import (
    is_duration_v3_prompt_summary_backbone,
    normalize_duration_v3_backbone_mode,
    validate_duration_v3_training_hparams,
)
from .task_runtime_support import DurationV3TaskRuntimeSupport, DurationV3TaskRuntimeSupportMixin
from importlib import import_module


def __getattr__(name: str):
    if name == "DurationV3DatasetMixin":
        module = import_module(".dataset_mixin", __name__)
        value = getattr(module, name)
    elif name == "DurationV3TaskMixin":
        module = import_module(".task_mixin", __name__)
        value = getattr(module, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value

__all__ = [
    "DurationV3LossTargets",
    "DurationV3TargetBuildConfig",
    "DurationV3TaskRuntimeSupport",
    "DurationV3TaskRuntimeSupportMixin",
    "DurationV3DatasetMixin",
    "DurationV3TaskMixin",
    "build_duration_v3_loss_dict",
    "build_duration_v3_loss_targets",
    "build_duration_v3_metric_sections",
    "build_duration_v3_ref_conditioning",
    "build_pseudo_source_duration_context",
    "build_rhythm_loss_dict",
    "build_rhythm_metric_dict",
    "build_rhythm_metric_sections",
    "is_duration_v3_prompt_summary_backbone",
    "normalize_duration_v3_backbone_mode",
    "validate_duration_v3_training_hparams",
]

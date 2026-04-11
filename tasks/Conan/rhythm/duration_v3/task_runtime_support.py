from __future__ import annotations

from ..common.task_runtime_support import CommonTaskRuntimeSupport
from utils.commons.hparams import hparams

from .targets import DurationV3TargetBuildConfig


class DurationV3TaskRuntimeSupportMixin:
    def build_duration_v3_target_build_config(self) -> DurationV3TargetBuildConfig:
        lambda_summary = hparams.get("lambda_rhythm_summary", None)
        lambda_mem = hparams.get("lambda_rhythm_mem", None)
        lambda_op = hparams.get("lambda_rhythm_op", 0.25)
        if lambda_summary is not None:
            lambda_op = lambda_summary
        elif lambda_mem is not None:
            lambda_op = lambda_mem
        return DurationV3TargetBuildConfig(
            lambda_dur=max(0.0, float(hparams.get("lambda_rhythm_dur", 1.0) or 1.0)),
            lambda_op=max(0.0, float(lambda_op or 0.0)),
            lambda_pref=max(0.0, float(hparams.get("lambda_rhythm_pref", 0.20) or 0.0)),
            lambda_base=max(0.0, float(hparams.get("lambda_rhythm_base", 0.0) or 0.0)),
            lambda_cons=max(0.0, float(hparams.get("lambda_rhythm_cons", 0.0) or 0.0)),
            lambda_zero=max(0.0, float(hparams.get("lambda_rhythm_zero", 0.05) or 0.0)),
            lambda_ortho=max(0.0, float(hparams.get("lambda_rhythm_ortho", 0.0) or 0.0)),
            strict_target_alignment=bool(hparams.get("rhythm_v3_strict_target_alignment", True)),
            anchor_mode=str(hparams.get("rhythm_v3_anchor_mode", "baseline") or "baseline").strip().lower(),
            baseline_target_mode=str(
                hparams.get("rhythm_v3_baseline_target_mode", "deglobalized") or "deglobalized"
            ).strip().lower(),
            baseline_train_mode=str(
                hparams.get("rhythm_v3_baseline_train_mode", "joint") or "joint"
            ).strip().lower(),
        )


class DurationV3TaskRuntimeSupport(DurationV3TaskRuntimeSupportMixin, CommonTaskRuntimeSupport):
    pass


__all__ = [
    "DurationV3TaskRuntimeSupport",
    "DurationV3TaskRuntimeSupportMixin",
]

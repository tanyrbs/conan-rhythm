from __future__ import annotations

from ..common.task_runtime_support import CommonTaskRuntimeSupport
from utils.commons.hparams import hparams

from .targets import DurationV3TargetBuildConfig
from .task_config import is_duration_v3_prompt_summary_backbone, resolve_duration_v3_rate_mode


def _is_enabled_flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class DurationV3TaskRuntimeSupportMixin:
    def build_duration_v3_target_build_config(self) -> DurationV3TargetBuildConfig:
        lambda_summary = hparams.get("lambda_rhythm_summary", None)
        lambda_mem = hparams.get("lambda_rhythm_mem", None)
        lambda_op = hparams.get("lambda_rhythm_op", 0.25)
        if is_duration_v3_prompt_summary_backbone(hparams.get("rhythm_v3_backbone", "global_only")):
            lambda_op = 0.0
        if lambda_summary is not None:
            lambda_op = lambda_summary
        elif lambda_mem is not None:
            lambda_op = lambda_mem
        default_lambda_bias = 0.20 if is_duration_v3_prompt_summary_backbone(hparams.get("rhythm_v3_backbone", "global_only")) else 0.0
        minimal_v1_profile = _is_enabled_flag(hparams.get("rhythm_v3_minimal_v1_profile", False))
        rate_mode = resolve_duration_v3_rate_mode(hparams)
        simple_global_stats = rate_mode == "simple_global" or _is_enabled_flag(
            hparams.get("rhythm_v3_simple_global_stats", minimal_v1_profile)
        )
        use_log_base_rate = bool(hparams.get("rhythm_v3_use_log_base_rate", False))
        if rate_mode == "simple_global" or simple_global_stats:
            use_log_base_rate = False
        return DurationV3TargetBuildConfig(
            lambda_dur=max(0.0, float(hparams.get("lambda_rhythm_dur", 1.0) or 1.0)),
            lambda_op=max(0.0, float(lambda_op or 0.0)),
            lambda_pref=max(0.0, float(hparams.get("lambda_rhythm_pref", 0.20) or 0.0)),
            lambda_bias=max(0.0, float(hparams.get("lambda_rhythm_bias", default_lambda_bias) or 0.0)),
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
            silence_coarse_weight=(
                0.0
                if minimal_v1_profile
                else max(0.0, float(hparams.get("rhythm_v3_silence_coarse_weight", 0.25) or 0.0))
            ),
            silence_logstretch_max=max(0.01, float(hparams.get("rhythm_v3_silence_max_logstretch", 0.35) or 0.35)),
            local_rate_decay=float(hparams.get("rhythm_v3_local_rate_decay", 0.95) or 0.95),
            analytic_gap_clip=max(0.0, float(hparams.get("rhythm_v3_analytic_gap_clip", 0.35) or 0.0)),
            silence_short_gap_scale=float(hparams.get("rhythm_v3_short_gap_silence_scale", 0.35) or 0.35),
            use_log_base_rate=use_log_base_rate,
            simple_global_stats=simple_global_stats,
            rate_mode=rate_mode,
            minimal_v1_profile=minimal_v1_profile,
        )


class DurationV3TaskRuntimeSupport(DurationV3TaskRuntimeSupportMixin, CommonTaskRuntimeSupport):
    pass


__all__ = [
    "DurationV3TaskRuntimeSupport",
    "DurationV3TaskRuntimeSupportMixin",
]

from __future__ import annotations

from ..common.task_runtime_support import CommonTaskRuntimeSupport
from ..config_contract_rules.compat import (
    resolve_duplicate_primary_distill_dedupe_flag as _resolve_duplicate_primary_distill_dedupe_flag,
)
from utils.commons.hparams import hparams

from .targets import RhythmTargetBuildConfig


class RhythmV2TaskRuntimeSupport(CommonTaskRuntimeSupport):
    _OFFLINE_CONFIDENCE_COMPONENTS = (
        ("rhythm_offline_confidence", "overall", None),
        ("rhythm_offline_confidence_exec", "exec", None),
        ("rhythm_offline_confidence_budget", "budget", None),
        ("rhythm_offline_confidence_prefix", "prefix", None),
        ("rhythm_offline_confidence_allocation", "allocation", None),
        ("rhythm_offline_confidence_shape", "shape", "exec"),
    )

    def build_rhythm_target_build_config(self) -> RhythmTargetBuildConfig:
        def _nonnegative_hparam(name: str, default: float) -> float:
            return max(0.0, float(hparams.get(name, default) or default))

        plan_local_weight, plan_cum_weight = self.owner._resolve_rhythm_plan_weights()
        return RhythmTargetBuildConfig(
            primary_target_surface=self.owner._resolve_rhythm_primary_target_surface(),
            distill_surface=self.owner._resolve_rhythm_distill_surface(),
            lambda_guidance=float(hparams.get("lambda_rhythm_guidance", 0.0) or 0.0),
            lambda_distill=float(hparams.get("lambda_rhythm_distill", 0.0) or 0.0),
            distill_exec_weight=float(hparams.get("rhythm_distill_exec_weight", 1.0)),
            distill_budget_weight=float(hparams.get("rhythm_distill_budget_weight", 0.5)),
            distill_allocation_weight=float(hparams.get("rhythm_distill_allocation_weight", 0.5)),
            distill_prefix_weight=float(hparams.get("rhythm_distill_prefix_weight", 0.25)),
            distill_speech_shape_weight=float(hparams.get("rhythm_distill_speech_shape_weight", 0.0)),
            distill_pause_shape_weight=float(hparams.get("rhythm_distill_pause_shape_weight", 0.0)),
            plan_local_weight=plan_local_weight,
            plan_cum_weight=plan_cum_weight,
            unit_logratio_weight=_nonnegative_hparam("rhythm_unit_logratio_weight", 0.0),
            srmdp_role_consistency_weight=_nonnegative_hparam("rhythm_srmdp_role_consistency_weight", 0.0),
            srmdp_notimeline_weight=_nonnegative_hparam("rhythm_srmdp_notimeline_weight", 0.0),
            srmdp_memory_role_weight=_nonnegative_hparam("rhythm_srmdp_memory_role_weight", 0.0),
            pause_boundary_weight=self.owner._resolve_rhythm_pause_boundary_weight(),
            budget_raw_weight=float(hparams.get("rhythm_budget_raw_weight", 1.0)),
            budget_exec_weight=float(hparams.get("rhythm_budget_exec_weight", 0.25)),
            feasible_debt_weight=float(hparams.get("rhythm_feasible_debt_weight", 0.05)),
            plan_segment_shape_weight=_nonnegative_hparam("rhythm_plan_segment_shape_weight", 0.0),
            plan_pause_release_weight=_nonnegative_hparam("rhythm_plan_pause_release_weight", 0.0),
            pause_event_weight=_nonnegative_hparam("rhythm_pause_event_weight", 0.0),
            pause_support_weight=_nonnegative_hparam("rhythm_pause_support_weight", 0.0),
            pause_allocation_weight=_nonnegative_hparam("rhythm_pause_allocation_weight", 0.0),
            pause_event_threshold=_nonnegative_hparam("rhythm_pause_event_threshold", 0.5),
            pause_event_temperature=max(1e-4, _nonnegative_hparam("rhythm_pause_event_temperature", 0.25)),
            pause_event_pos_weight=max(1.0, _nonnegative_hparam("rhythm_pause_event_pos_weight", 2.0)),
            dedupe_primary_teacher_cache_distill=_resolve_duplicate_primary_distill_dedupe_flag(hparams),
            enable_distill_context_match=bool(
                hparams.get("rhythm_enable_distill_context_match", False)
            ),
            distill_context_floor=float(hparams.get("rhythm_distill_context_floor", 0.35)),
            distill_context_power=float(hparams.get("rhythm_distill_context_power", 1.0)),
            distill_context_open_run_penalty=float(
                hparams.get("rhythm_distill_context_open_run_penalty", 0.50)
            ),
        )

    def build_offline_confidence_outputs(self, confidence) -> dict:
        outputs = {}
        for output_key, confidence_key, fallback_key in self._OFFLINE_CONFIDENCE_COMPONENTS:
            value = None
            if isinstance(confidence, dict):
                value = confidence.get(confidence_key)
                if value is None and fallback_key is not None:
                    value = confidence.get(fallback_key)
            outputs[output_key] = value
        return outputs

    def collect_runtime_offline_source_cache(self, sample, *, infer: bool):
        if getattr(self.owner.model, "rhythm_enable_v3", False):
            return None
        if self.owner._use_runtime_dual_mode_teacher() and not bool(infer):
            return self.owner._collect_rhythm_source_cache(sample, prefix="rhythm_offline_")
        return None


__all__ = ["RhythmV2TaskRuntimeSupport"]

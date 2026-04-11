from __future__ import annotations

import torch

from ..budget_repair import compute_budget_projection_repair_stats


def build_legacy_v2_ref_conditioning(sample, *, explicit=None):
    if explicit is not None:
        return explicit
    ref_stats = sample.get("ref_rhythm_stats")
    ref_trace = sample.get("ref_rhythm_trace")
    if ref_stats is None or ref_trace is None:
        return None
    conditioning = {
        "ref_rhythm_stats": ref_stats,
        "ref_rhythm_trace": ref_trace,
    }
    for extra_key in (
        "global_rate",
        "pause_ratio",
        "local_rate_trace",
        "boundary_trace",
        "planner_ref_stats",
        "planner_ref_trace",
        "slow_rhythm_memory",
        "slow_rhythm_summary",
        "planner_slow_rhythm_memory",
        "planner_slow_rhythm_summary",
        "selector_meta_indices",
        "selector_meta_scores",
        "selector_meta_starts",
        "selector_meta_ends",
        "ref_phrase_trace",
        "planner_ref_phrase_trace",
        "ref_phrase_valid",
        "ref_phrase_lengths",
        "ref_phrase_starts",
        "ref_phrase_ends",
        "ref_phrase_boundary_strength",
        "ref_phrase_stats",
    ):
        extra_value = sample.get(extra_key)
        if extra_value is not None:
            conditioning[extra_key] = extra_value
    return conditioning


def collect_legacy_planner_runtime_outputs(rhythm_execution) -> dict[str, torch.Tensor]:
    runtime_outputs = {}
    if rhythm_execution is None or getattr(rhythm_execution, "planner", None) is None:
        return runtime_outputs
    planner = rhythm_execution.planner
    for attr_name, output_key in (
        ("raw_speech_budget_win", "raw_speech_budget_win"),
        ("raw_pause_budget_win", "raw_pause_budget_win"),
        ("effective_speech_budget_win", "effective_speech_budget_win"),
        ("effective_pause_budget_win", "effective_pause_budget_win"),
        ("pause_topk_ratio", "rhythm_projector_pause_topk_ratio"),
        ("pause_soft_selection_active", "rhythm_projector_pause_soft_selection_active"),
        ("projector_force_full_commit", "rhythm_projector_force_full_commit"),
        ("pause_selection_mode_id", "rhythm_projector_pause_selection_mode_id"),
    ):
        attr_value = getattr(planner, attr_name, None)
        if attr_value is not None:
            runtime_outputs[output_key] = attr_value
    for attr_name in (
        "feasible_speech_budget_delta",
        "feasible_pause_budget_delta",
        "feasible_total_budget_delta",
        "pause_support_prob_unit",
        "pause_allocation_weight_unit",
        "pause_support_logit_unit",
        "pause_run_length_unit",
        "pause_breath_debt_unit",
        "trace_reliability",
        "local_trace_path_weight",
        "boundary_trace_path_weight",
        "trace_phase_gap_runtime",
        "trace_phase_gap_anchor",
        "trace_coverage_alpha",
        "trace_blend",
        "trace_phrase_blend",
        "trace_global_blend",
        "trace_tail_alpha",
        "trace_gap_alpha",
        "trace_reuse_alpha",
        "trace_tail_reuse_count",
        "ref_phrase_index",
        "commit_confidence",
        "planned_commit_frontier",
    ):
        attr_value = getattr(planner, attr_name, None)
        if attr_value is not None:
            runtime_outputs[attr_name] = attr_value
    return runtime_outputs


def _apply_online_retimed_repair_gate(frame_weight, model_out):
    if frame_weight is None or model_out is None:
        return frame_weight
    execution = model_out.get("rhythm_execution")
    planner = getattr(execution, "planner", None) if execution is not None else None
    if planner is None:
        return frame_weight
    repair_stats = compute_budget_projection_repair_stats(planner)
    effective_total = (
        repair_stats.effective_speech_budget + repair_stats.effective_pause_budget
    ).clamp_min(0.0)
    repair_mass = repair_stats.repair_mass.clamp_min(0.0)
    repair_gate = (effective_total / (effective_total + repair_mass).clamp_min(1e-6)).clamp_(0.0, 1.0)
    repair_gate = repair_gate.to(device=frame_weight.device, dtype=frame_weight.dtype)
    while repair_gate.dim() < frame_weight.dim():
        repair_gate = repair_gate.unsqueeze(-1)
    return frame_weight * repair_gate


def _apply_online_retimed_trace_reliability_gate(frame_weight, model_out):
    if frame_weight is None or model_out is None:
        return frame_weight
    execution = model_out.get("rhythm_execution")
    planner = getattr(execution, "planner", None) if execution is not None else None
    if planner is None:
        return frame_weight
    reliability_gate = getattr(planner, "local_trace_path_weight", None)
    if reliability_gate is None:
        reliability_gate = getattr(planner, "trace_reliability", None)
    if reliability_gate is None:
        return frame_weight
    reliability_gate = reliability_gate.float().to(device=frame_weight.device, dtype=frame_weight.dtype)
    while reliability_gate.dim() < frame_weight.dim():
        reliability_gate = reliability_gate.unsqueeze(-1)
    model_out["rhythm_online_retimed_trace_gate"] = reliability_gate.detach()
    return frame_weight * reliability_gate


__all__ = [
    "build_legacy_v2_ref_conditioning",
    "collect_legacy_planner_runtime_outputs",
]

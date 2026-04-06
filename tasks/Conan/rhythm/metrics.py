from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from modules.Conan.rhythm.prefix_state import build_prefix_state_from_exec_torch
from tasks.Conan.rhythm.budget_repair import compute_budget_projection_repair_stats


def _safe_mean(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 0:
        return torch.tensor(0.0, device=x.device if isinstance(x, torch.Tensor) else "cpu")
    return x.float().mean()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = values.float()
    if values.dim() <= 1:
        return _safe_mean(values)
    mask = mask.float()
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    reduce_dims = tuple(range(1, values.dim()))
    masked_sum = (values * mask).sum(dim=reduce_dims)
    masked_denom = mask.sum(dim=reduce_dims).clamp_min(1.0)
    return (masked_sum / masked_denom).mean()


def _masked_distribution(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = values.float().clamp_min(0.0)
    mask = mask.float()
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    reduce_dims = tuple(range(1, values.dim()))
    masked_values = values * mask
    total = masked_values.sum(dim=reduce_dims, keepdim=True)
    uniform = mask / mask.sum(dim=reduce_dims, keepdim=True).clamp_min(1.0)
    return torch.where(total > 1e-6, masked_values / total.clamp_min(1e-6), uniform)


def _masked_l1(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return ((pred.float() - tgt.float()).abs() * mask).sum() / denom


def _masked_corr(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    pred_mean = (pred.float() * mask).sum(dim=1) / denom
    tgt_mean = (tgt.float() * mask).sum(dim=1) / denom
    pred_center = (pred.float() - pred_mean[:, None]) * mask
    tgt_center = (tgt.float() - tgt_mean[:, None]) * mask
    cov = (pred_center * tgt_center).sum(dim=1)
    pred_var = (pred_center ** 2).sum(dim=1).clamp_min(1e-6)
    tgt_var = (tgt_center ** 2).sum(dim=1).clamp_min(1e-6)
    corr = cov / (pred_var.sqrt() * tgt_var.sqrt())
    valid = mask.sum(dim=1) > 1
    corr = torch.where(valid, corr, torch.zeros_like(corr))
    return corr.mean()


def _masked_event_f1(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor, *, threshold: float = 0.5):
    mask_bool = mask > 0
    pred_evt = (pred.float() > threshold) & mask_bool
    tgt_evt = (tgt.float() > threshold) & mask_bool
    tp = (pred_evt & tgt_evt).float().sum(dim=1)
    fp = (pred_evt & (~tgt_evt)).float().sum(dim=1)
    fn = ((~pred_evt) & tgt_evt).float().sum(dim=1)
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
    return precision.mean(), recall.mean(), f1.mean()


def _masked_kl(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    pred = (pred.float() * mask).clamp_min(1e-6)
    tgt = (tgt.float() * mask).clamp_min(1e-6)
    pred = pred / pred.sum(dim=1, keepdim=True).clamp_min(1e-6)
    tgt = tgt / tgt.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (tgt * (torch.log(tgt) - torch.log(pred))).sum(dim=1).mean()


def _slice_to_visible_prefix(tensor: torch.Tensor, target_units: int) -> torch.Tensor:
    return tensor[:, :target_units]


def _as_frame_plan(output: dict[str, Any], execution) -> Any | None:
    frame_plan = output.get("rhythm_frame_plan")
    if frame_plan is not None:
        return frame_plan
    return getattr(execution, "frame_plan", None)


def _optional_scalar(x: Any, device: torch.device) -> torch.Tensor | None:
    if isinstance(x, torch.Tensor):
        return _safe_mean(x.float())
    if isinstance(x, (int, float)):
        return torch.tensor(float(x), device=device)
    return None


@dataclass(frozen=True)
class RhythmMetricContext:
    execution: Any
    planner: Any
    blank_exec: torch.Tensor
    unit_batch: Any | None
    unit_mask: torch.Tensor
    anchor_total: torch.Tensor
    visible_units: torch.Tensor
    speech_total: torch.Tensor
    pause_total: torch.Tensor
    total_exec: torch.Tensor
    raw_speech_budget: torch.Tensor
    raw_pause_budget: torch.Tensor
    effective_speech_budget: torch.Tensor
    effective_pause_budget: torch.Tensor
    raw_total_budget: torch.Tensor
    effective_total_budget: torch.Tensor
    raw_exec_gap: torch.Tensor
    feasible_total_budget_delta: torch.Tensor
    projection_total_shift_abs: torch.Tensor
    projection_redistribution_mass: torch.Tensor
    projection_repair_mass: torch.Tensor
    commit_ratio: torch.Tensor
    pred_prefix_clock: torch.Tensor
    pred_prefix_backlog: torch.Tensor
    prefix_budget: torch.Tensor
    final_prefix_clock: torch.Tensor
    pause_shape_entropy: torch.Tensor
    pause_shape_entropy_norm: torch.Tensor


def _build_rhythm_metric_context(output: dict[str, Any], execution) -> RhythmMetricContext:
    planner = execution.planner
    blank_exec = getattr(execution, "blank_duration_exec", execution.pause_after_exec)
    unit_batch = output.get("rhythm_unit_batch")
    if unit_batch is not None:
        unit_mask = unit_batch.unit_mask.float()
        anchor_total = (unit_batch.dur_anchor_src.float() * unit_mask).sum(dim=1)
        visible_units = unit_mask.sum(dim=1).clamp_min(1.0)
    else:
        speech = execution.speech_duration_exec.float()
        unit_mask = speech.new_ones(speech.shape)
        anchor_total = speech.sum(dim=1).clamp_min(1.0)
        visible_units = speech.new_full((speech.size(0),), float(speech.size(1)))

    speech_total = execution.speech_duration_exec.float().sum(dim=1)
    pause_total = blank_exec.float().sum(dim=1)
    total_exec = (speech_total + pause_total).clamp_min(1e-6)
    raw_speech_budget = getattr(planner, "raw_speech_budget_win", planner.speech_budget_win).float().squeeze(-1)
    raw_pause_budget = getattr(planner, "raw_pause_budget_win", planner.pause_budget_win).float().squeeze(-1)
    effective_speech_budget = getattr(planner, "effective_speech_budget_win", planner.speech_budget_win).float().squeeze(-1)
    effective_pause_budget = getattr(planner, "effective_pause_budget_win", planner.pause_budget_win).float().squeeze(-1)
    raw_total_budget = (raw_speech_budget + raw_pause_budget).clamp_min(1e-6)
    effective_total_budget = (effective_speech_budget + effective_pause_budget).clamp_min(1e-6)
    repair_stats = compute_budget_projection_repair_stats(planner)
    raw_exec_gap = (effective_speech_budget - raw_speech_budget).abs() + (
        effective_pause_budget - raw_pause_budget
    ).abs()
    projection_total_shift_abs = repair_stats.total_shift_abs
    projection_redistribution_mass = repair_stats.redistribution_mass
    projection_repair_mass = repair_stats.repair_mass
    feasible_total_budget_delta = getattr(planner, "feasible_total_budget_delta", None)
    if feasible_total_budget_delta is None:
        feasible_total_budget_delta = raw_total_budget.new_zeros(raw_total_budget.shape)
    else:
        feasible_total_budget_delta = feasible_total_budget_delta.float().reshape(feasible_total_budget_delta.size(0), -1)
        feasible_total_budget_delta = feasible_total_budget_delta.mean(dim=1).clamp_min(0.0)
    commit_ratio = execution.commit_frontier.float() / visible_units
    pred_prefix_clock, pred_prefix_backlog = build_prefix_state_from_exec_torch(
        speech_exec=execution.speech_duration_exec,
        pause_exec=blank_exec,
        dur_anchor_src=unit_batch.dur_anchor_src if unit_batch is not None else execution.speech_duration_exec,
        unit_mask=unit_mask,
    )
    prefix_budget = torch.cumsum(
        (unit_batch.dur_anchor_src if unit_batch is not None else execution.speech_duration_exec).float() * unit_mask,
        dim=1,
    ).clamp_min(1.0)
    final_indices = (visible_units.long() - 1).clamp_min(0).unsqueeze(1)
    final_prefix_clock = pred_prefix_clock.gather(1, final_indices).squeeze(1)
    pause_shape_distribution = _masked_distribution(planner.pause_shape_unit.float(), unit_mask)
    pause_shape_entropy = -(
        pause_shape_distribution.clamp_min(1e-6) * pause_shape_distribution.clamp_min(1e-6).log()
    ).sum(dim=1)
    pause_shape_entropy_norm = torch.where(
        visible_units > 1.0,
        pause_shape_entropy / visible_units.float().log().clamp_min(1e-6),
        torch.zeros_like(visible_units),
    )
    return RhythmMetricContext(
        execution=execution,
        planner=planner,
        blank_exec=blank_exec,
        unit_batch=unit_batch,
        unit_mask=unit_mask,
        anchor_total=anchor_total,
        visible_units=visible_units,
        speech_total=speech_total,
        pause_total=pause_total,
        total_exec=total_exec,
        raw_speech_budget=raw_speech_budget,
        raw_pause_budget=raw_pause_budget,
        effective_speech_budget=effective_speech_budget,
        effective_pause_budget=effective_pause_budget,
        raw_total_budget=raw_total_budget,
        effective_total_budget=effective_total_budget,
        raw_exec_gap=raw_exec_gap,
        feasible_total_budget_delta=feasible_total_budget_delta,
        projection_total_shift_abs=projection_total_shift_abs,
        projection_redistribution_mass=projection_redistribution_mass,
        projection_repair_mass=projection_repair_mass,
        commit_ratio=commit_ratio,
        pred_prefix_clock=pred_prefix_clock,
        pred_prefix_backlog=pred_prefix_backlog,
        prefix_budget=prefix_budget,
        final_prefix_clock=final_prefix_clock,
        pause_shape_entropy=pause_shape_entropy,
        pause_shape_entropy_norm=pause_shape_entropy_norm,
    )


def _build_core_rhythm_metric_dict(ctx: RhythmMetricContext) -> dict[str, torch.Tensor]:
    return {
        "rhythm_metric_speech_budget_mean": _safe_mean(ctx.effective_speech_budget),
        "rhythm_metric_pause_budget_mean": _safe_mean(ctx.effective_pause_budget),
        "rhythm_metric_raw_speech_budget_mean": _safe_mean(ctx.raw_speech_budget),
        "rhythm_metric_raw_pause_budget_mean": _safe_mean(ctx.raw_pause_budget),
        "rhythm_metric_effective_speech_budget_mean": _safe_mean(ctx.effective_speech_budget),
        "rhythm_metric_effective_pause_budget_mean": _safe_mean(ctx.effective_pause_budget),
        "rhythm_metric_budget_raw_exec_gap_mean": _safe_mean(ctx.raw_exec_gap),
        "rhythm_metric_budget_raw_exec_gap_ratio_mean": _safe_mean(
            ctx.raw_exec_gap / ctx.raw_total_budget.clamp_min(1.0)
        ),
        "rhythm_metric_budget_repair_ratio_mean": _safe_mean(
            ctx.feasible_total_budget_delta / ctx.anchor_total.clamp_min(1.0)
        ),
        "rhythm_metric_budget_repair_active_rate": _safe_mean(
            (ctx.feasible_total_budget_delta > 1e-6).float()
        ),
        "rhythm_metric_budget_projection_total_shift_abs_mean": _safe_mean(ctx.projection_total_shift_abs),
        "rhythm_metric_budget_projection_redistribution_mean": _safe_mean(ctx.projection_redistribution_mass),
        "rhythm_metric_budget_projection_redistribution_ratio_mean": _safe_mean(
            ctx.projection_redistribution_mass / ctx.anchor_total.clamp_min(1.0)
        ),
        "rhythm_metric_budget_projection_repair_mean": _safe_mean(ctx.projection_repair_mass),
        "rhythm_metric_budget_projection_repair_ratio_mean": _safe_mean(
            ctx.projection_repair_mass / ctx.anchor_total.clamp_min(1.0)
        ),
        "rhythm_metric_budget_projection_repair_active_rate": _safe_mean(
            (ctx.projection_repair_mass > 1e-6).float()
        ),
        "rhythm_metric_dur_shape_abs_mean": _masked_mean(ctx.planner.dur_shape_unit.abs(), ctx.unit_mask),
        "rhythm_metric_pause_shape_entropy": _safe_mean(ctx.pause_shape_entropy),
        "rhythm_metric_pause_shape_entropy_norm": _safe_mean(ctx.pause_shape_entropy_norm),
        "rhythm_metric_boundary_score_mean": _masked_mean(ctx.planner.boundary_score_unit.float(), ctx.unit_mask),
        "rhythm_metric_speech_total_mean": _safe_mean(ctx.speech_total),
        "rhythm_metric_pause_total_mean": _safe_mean(ctx.pause_total),
        "rhythm_metric_pause_share_mean": _safe_mean(ctx.pause_total / ctx.total_exec),
        "rhythm_metric_pause_event_ratio_mean": _safe_mean(
            (ctx.execution.pause_after_exec.float() > 0.5).float().sum(dim=1) / ctx.visible_units
        ),
        "rhythm_metric_blank_slot_ratio_mean": _safe_mean(
            (ctx.blank_exec.float() > 0.5).float().sum(dim=1) / ctx.visible_units
        ),
        "rhythm_metric_expand_ratio_mean": _safe_mean(ctx.speech_total / ctx.anchor_total.clamp_min(1.0)),
        "rhythm_metric_commit_ratio_mean": _safe_mean(ctx.commit_ratio),
        "rhythm_metric_budget_violation_mean": _safe_mean(
            ((ctx.speech_total + ctx.pause_total) - ctx.effective_total_budget).abs()
        ),
        "rhythm_metric_prefix_clock_abs_mean": _masked_mean(ctx.pred_prefix_clock.abs(), ctx.unit_mask),
        "rhythm_metric_prefix_backlog_mean": _masked_mean(ctx.pred_prefix_backlog, ctx.unit_mask),
        "rhythm_metric_prefix_backlog_max": ctx.pred_prefix_backlog.max(),
        "rhythm_metric_deadline_final_abs_mean": _safe_mean(ctx.final_prefix_clock.abs()),
    }


def _update_render_frame_plan_metrics(
    metrics: dict[str, torch.Tensor],
    output: dict[str, Any],
    ctx: RhythmMetricContext,
) -> None:
    render_blank_mask = output.get("rhythm_blank_mask")
    if render_blank_mask is not None:
        render_total_mask = output.get("rhythm_total_mask")
        denom = (
            render_total_mask.float().sum().clamp_min(1.0)
            if render_total_mask is not None
            else float(render_blank_mask.numel())
        )
        metrics["rhythm_metric_render_blank_ratio"] = render_blank_mask.float().sum() / denom
    frame_plan = _as_frame_plan(output, ctx.execution)
    if frame_plan is None:
        metrics["rhythm_metric_frame_plan_present"] = ctx.execution.speech_duration_exec.new_tensor(0.0)
        return
    frame_total_mask = frame_plan.total_mask.float()
    frame_blank_mask = frame_plan.blank_mask.float()
    frame_src_index = frame_plan.frame_src_index
    frame_total = frame_total_mask.sum().clamp_min(1.0)
    frame_blank = (frame_blank_mask * frame_total_mask).sum()
    frame_nonblank = ((1.0 - frame_blank_mask).clamp(0.0, 1.0) * frame_total_mask).sum()
    blank_denom = frame_blank.clamp_min(1.0)
    speech_denom = frame_nonblank.clamp_min(1.0)
    blank_and_neg_src = ((frame_blank_mask > 0.5) & (frame_src_index < 0)).float() * frame_total_mask
    speech_and_pos_src = ((frame_blank_mask <= 0.5) & (frame_src_index >= 0)).float() * frame_total_mask
    metrics["rhythm_metric_frame_plan_present"] = ctx.execution.speech_duration_exec.new_tensor(1.0)
    metrics["rhythm_metric_frame_plan_total_frames_mean"] = _safe_mean(frame_total_mask.sum(dim=1))
    metrics["rhythm_metric_frame_plan_blank_ratio_mean"] = frame_blank / frame_total
    metrics["rhythm_metric_frame_plan_speech_ratio_mean"] = frame_nonblank / frame_total
    metrics["rhythm_metric_frame_plan_blank_src_consistency"] = blank_and_neg_src.sum() / blank_denom
    metrics["rhythm_metric_frame_plan_speech_src_consistency"] = speech_and_pos_src.sum() / speech_denom
    if render_blank_mask is not None and render_blank_mask.shape == frame_blank_mask.shape:
        metrics["rhythm_metric_render_vs_frame_plan_blank_l1"] = _masked_l1(
            render_blank_mask.float(),
            frame_blank_mask.float(),
            frame_total_mask,
        )


def _update_trace_and_boundary_metrics(
    metrics: dict[str, torch.Tensor],
    output: dict[str, Any],
    ctx: RhythmMetricContext,
) -> None:
    metrics["rhythm_metric_local_rate_transfer_corr"] = _masked_corr(
        ctx.execution.speech_duration_exec.float(),
        ctx.planner.trace_context[:, :, 1].float(),
        ctx.unit_mask,
    )
    metrics["rhythm_metric_pause_trace_corr"] = _masked_corr(
        ctx.blank_exec.float(),
        ctx.planner.trace_context[:, :, 0].float(),
        ctx.unit_mask,
    )
    metrics["rhythm_metric_boundary_trace_corr"] = _masked_corr(
        ctx.blank_exec.float(),
        ctx.planner.trace_context[:, :, 2].float(),
        ctx.unit_mask,
    )
    source_boundary_cue = output.get("source_boundary_cue")
    if source_boundary_cue is not None:
        metrics["rhythm_metric_source_boundary_mean"] = _masked_mean(source_boundary_cue.float(), ctx.unit_mask)
        metrics["rhythm_metric_source_boundary_pause_corr"] = _masked_corr(
            ctx.blank_exec.float(),
            source_boundary_cue.float(),
            ctx.unit_mask,
        )


def _update_state_metrics(
    metrics: dict[str, torch.Tensor],
    output: dict[str, Any],
    ctx: RhythmMetricContext,
) -> None:
    device = ctx.execution.speech_duration_exec.device
    for metric_key, field_name in (
        ("rhythm_metric_disable_acoustic_train_path", "disable_acoustic_train_path"),
        ("rhythm_metric_module_only_objective", "rhythm_module_only_objective"),
        ("rhythm_metric_skip_acoustic_objective", "rhythm_skip_acoustic_objective"),
    ):
        scalar = _optional_scalar(output.get(field_name), device=device)
        if scalar is not None:
            metrics[metric_key] = scalar
    state_next = output.get("rhythm_state_next")
    if state_next is not None:
        metrics.update(
            {
                "rhythm_metric_commit_frontier_mean": _safe_mean(state_next.commit_frontier.float()),
                "rhythm_metric_phase_mean": _safe_mean(state_next.phase_ptr),
                "rhythm_metric_backlog_mean": _safe_mean(state_next.backlog),
                "rhythm_metric_clock_delta_mean": _safe_mean(state_next.clock_delta),
                "rhythm_metric_clock_delta_abs_mean": _safe_mean(state_next.clock_delta.abs()),
            }
        )
        trace_tail_reuse_count = getattr(state_next, "trace_tail_reuse_count", None)
        if trace_tail_reuse_count is not None:
            metrics["rhythm_metric_trace_tail_reuse_count_mean"] = _safe_mean(
                trace_tail_reuse_count.float()
            )
        progress_ratio = getattr(state_next, "phase_progress_ratio", None)
        phase_gap = getattr(state_next, "phase_ptr_gap", None)
        if progress_ratio is not None and phase_gap is not None:
            metrics["rhythm_metric_phase_progress_ratio_mean"] = _safe_mean(progress_ratio)
            metrics["rhythm_metric_phase_ptr_vs_progress_l1"] = _safe_mean(phase_gap.abs())
            metrics["rhythm_metric_phase_ptr_below_progress_ratio"] = _safe_mean(
                (phase_gap < -1e-6).float()
            )
    state_prev = output.get("rhythm_state_prev")
    if state_prev is not None and state_next is not None:
        phase_delta = state_next.phase_ptr.float() - state_prev.phase_ptr.float()
        metrics["rhythm_metric_phase_delta_mean"] = _safe_mean(phase_delta)
        metrics["rhythm_metric_phase_delta_min"] = phase_delta.min()
        metrics["rhythm_metric_phase_nonretro_rate"] = _safe_mean((phase_delta >= -1e-6).float())


def _update_reference_conditioning_metrics(metrics: dict[str, torch.Tensor], output: dict[str, Any]) -> None:
    ref_conditioning = output.get("rhythm_ref_conditioning")
    if ref_conditioning is None:
        return
    selector_scores = ref_conditioning.get("selector_meta_scores")
    slow_summary = ref_conditioning.get("slow_rhythm_summary")
    if selector_scores is not None:
        metrics["rhythm_metric_selector_score_mean"] = _safe_mean(selector_scores.float())
        metrics["rhythm_metric_selector_score_max"] = selector_scores.float().max()
    if slow_summary is not None:
        metrics["rhythm_metric_slow_summary_norm_mean"] = _safe_mean(torch.norm(slow_summary.float(), dim=-1))
    planner = getattr(output.get("rhythm_execution"), "planner", None)
    if planner is None:
        planner = output.get("rhythm_planner")
    if planner is not None:
        for metric_key, field_name in (
            ("rhythm_metric_trace_reliability_mean", "trace_reliability"),
            ("rhythm_metric_trace_local_path_weight_mean", "local_trace_path_weight"),
            ("rhythm_metric_trace_boundary_path_weight_mean", "boundary_trace_path_weight"),
            ("rhythm_metric_trace_tail_alpha_mean", "trace_tail_alpha"),
            ("rhythm_metric_trace_gap_alpha_mean", "trace_gap_alpha"),
            ("rhythm_metric_trace_reuse_alpha_mean", "trace_reuse_alpha"),
            ("rhythm_metric_trace_phase_gap_mean", "trace_phase_gap"),
            ("rhythm_metric_trace_tail_reuse_from_planner_mean", "trace_tail_reuse_count"),
        ):
            value = getattr(planner, field_name, None)
            if value is not None:
                metrics[metric_key] = _safe_mean(value.float())


def _update_teacher_alignment_metrics(
    metrics: dict[str, torch.Tensor],
    output: dict[str, Any],
    ctx: RhythmMetricContext,
) -> None:
    offline_execution = output.get("rhythm_offline_execution")
    if offline_execution is not None:
        target_units = ctx.execution.speech_duration_exec.size(1)
        offline_speech = _slice_to_visible_prefix(offline_execution.speech_duration_exec, target_units)
        offline_blank = _slice_to_visible_prefix(
            getattr(offline_execution, "blank_duration_exec", offline_execution.pause_after_exec),
            target_units,
        )
        offline_prefix_clock, offline_prefix_backlog = build_prefix_state_from_exec_torch(
            speech_exec=offline_speech,
            pause_exec=offline_blank,
            dur_anchor_src=ctx.unit_batch.dur_anchor_src if ctx.unit_batch is not None else offline_execution.speech_duration_exec,
            unit_mask=ctx.unit_mask,
        )
        metrics["rhythm_metric_offline_online_speech_l1"] = _masked_l1(
            ctx.execution.speech_duration_exec,
            offline_speech,
            ctx.unit_mask,
        )
        metrics["rhythm_metric_offline_online_pause_l1"] = _masked_l1(
            ctx.blank_exec,
            offline_blank,
            ctx.unit_mask,
        )
        metrics["rhythm_metric_offline_online_total_corr"] = _masked_corr(
            ctx.execution.speech_duration_exec + ctx.blank_exec,
            offline_speech + offline_blank,
            ctx.unit_mask,
        )
        metrics["rhythm_metric_offline_online_alloc_kl"] = _masked_kl(
            ctx.execution.speech_duration_exec + ctx.blank_exec,
            offline_speech + offline_blank,
            ctx.unit_mask,
        )
        metrics["rhythm_metric_offline_online_prefix_clock_l1"] = _masked_l1(
            ctx.pred_prefix_clock,
            offline_prefix_clock,
            ctx.unit_mask,
        )
        metrics["rhythm_metric_offline_online_prefix_backlog_l1"] = _masked_l1(
            ctx.pred_prefix_backlog,
            offline_prefix_backlog,
            ctx.unit_mask,
        )
    algorithmic_teacher = output.get("rhythm_algorithmic_teacher")
    if algorithmic_teacher is not None:
        metrics["rhythm_metric_algorithmic_teacher_confidence"] = _safe_mean(algorithmic_teacher.confidence.float())
        metrics["rhythm_metric_algorithmic_teacher_pause_l1"] = _masked_l1(
            ctx.blank_exec,
            algorithmic_teacher.pause_exec_tgt,
            ctx.unit_mask,
        )
        metrics["rhythm_metric_algorithmic_teacher_alloc_kl"] = _masked_kl(
            ctx.execution.speech_duration_exec + ctx.blank_exec,
            algorithmic_teacher.allocation_tgt,
            ctx.unit_mask,
        )
        metrics["rhythm_metric_algorithmic_teacher_prefix_clock_l1"] = _masked_l1(
            ctx.pred_prefix_clock,
            algorithmic_teacher.prefix_clock_tgt,
            ctx.unit_mask,
        )
        metrics["rhythm_metric_algorithmic_teacher_prefix_backlog_l1"] = _masked_l1(
            ctx.pred_prefix_backlog,
            algorithmic_teacher.prefix_backlog_tgt,
            ctx.unit_mask,
        )


def _update_acoustic_target_metrics(
    metrics: dict[str, torch.Tensor],
    output: dict[str, Any],
    ctx: RhythmMetricContext,
) -> None:
    metrics["rhythm_metric_acoustic_target_is_retimed"] = ctx.execution.speech_duration_exec.new_tensor(
        1.0 if bool(output.get("acoustic_target_is_retimed", False)) else 0.0
    )
    metrics["rhythm_metric_apply_render_mean"] = ctx.execution.speech_duration_exec.new_tensor(
        float(output.get("rhythm_apply_render", 0.0))
    )
    acoustic_target_weight = output.get("acoustic_target_weight")
    if acoustic_target_weight is not None:
        metrics["rhythm_metric_retimed_weight_mean"] = _safe_mean(acoustic_target_weight.float())
    pause_topk_ratio = output.get("rhythm_projector_pause_topk_ratio")
    if pause_topk_ratio is not None:
        metrics["rhythm_metric_pause_topk_ratio_mean"] = _safe_mean(pause_topk_ratio.float())
        metrics["rhythm_metric_pause_topk_sparsity_mean"] = ctx.execution.speech_duration_exec.new_tensor(1.0) - _safe_mean(
            pause_topk_ratio.float()
        )
    source_boundary_scale = output.get("rhythm_source_boundary_scale")
    if source_boundary_scale is not None:
        metrics["rhythm_metric_source_boundary_scale_mean"] = _safe_mean(source_boundary_scale.float())
    teacher_source_boundary_scale = output.get("rhythm_teacher_source_boundary_scale")
    if teacher_source_boundary_scale is not None:
        metrics["rhythm_metric_teacher_source_boundary_scale_mean"] = _safe_mean(
            teacher_source_boundary_scale.float()
        )
    for metric_key, field_name in (
        ("rhythm_metric_pitch_supervision_disabled", "rhythm_pitch_supervision_disabled"),
        ("rhythm_metric_missing_retimed_pitch_target", "rhythm_missing_retimed_pitch_target"),
    ):
        scalar = _optional_scalar(output.get(field_name), ctx.execution.speech_duration_exec.device)
        if scalar is not None:
            metrics[metric_key] = scalar
    for metric_key, field_name in (
        ("rhythm_metric_acoustic_target_length_frames_before_align", "acoustic_target_length_frames_before_align"),
        ("rhythm_metric_acoustic_output_length_frames_before_align", "acoustic_output_length_frames_before_align"),
        ("rhythm_metric_acoustic_target_length_delta_before_align", "acoustic_target_length_delta_before_align"),
        ("rhythm_metric_acoustic_target_length_mismatch_abs_before_align", "acoustic_target_length_mismatch_abs_before_align"),
        ("rhythm_metric_acoustic_target_length_mismatch_present_before_align", "acoustic_target_length_mismatch_present_before_align"),
        ("rhythm_metric_acoustic_target_length_mismatch_ratio_before_align", "acoustic_target_length_mismatch_ratio_before_align"),
        ("rhythm_metric_acoustic_target_resampled_to_output", "acoustic_target_resampled_to_output"),
        ("rhythm_metric_acoustic_target_trimmed_to_output", "acoustic_target_trimmed_to_output"),
        ("rhythm_metric_acoustic_target_length_frames_after_align", "acoustic_target_length_frames_after_align"),
        ("rhythm_metric_acoustic_output_length_frames_after_align", "acoustic_output_length_frames_after_align"),
    ):
        scalar = _optional_scalar(output.get(field_name), ctx.execution.speech_duration_exec.device)
        if scalar is not None:
            metrics[metric_key] = scalar
    acoustic_target_source = output.get("acoustic_target_source")
    acoustic_source_name = str(acoustic_target_source) if acoustic_target_source is not None else "unknown"
    source_to_id = {"source": 0.0, "cached": 1.0, "online": 2.0}
    source_id = source_to_id.get(acoustic_source_name, -1.0)
    metrics["rhythm_metric_acoustic_target_source_id"] = ctx.execution.speech_duration_exec.new_tensor(source_id)
    metrics["rhythm_metric_acoustic_target_source_is_source"] = ctx.execution.speech_duration_exec.new_tensor(
        1.0 if acoustic_source_name == "source" else 0.0
    )
    metrics["rhythm_metric_acoustic_target_source_is_cached"] = ctx.execution.speech_duration_exec.new_tensor(
        1.0 if acoustic_source_name == "cached" else 0.0
    )
    metrics["rhythm_metric_acoustic_target_source_is_online"] = ctx.execution.speech_duration_exec.new_tensor(
        1.0 if acoustic_source_name == "online" else 0.0
    )
    metrics["rhythm_metric_acoustic_target_source_unknown"] = ctx.execution.speech_duration_exec.new_tensor(
        1.0 if source_id < 0.0 else 0.0
    )


def _update_confidence_metrics(
    metrics: dict[str, torch.Tensor],
    output: dict[str, Any],
    ctx: RhythmMetricContext,
) -> None:
    offline_confidence = output.get("rhythm_offline_confidence")
    if offline_confidence is not None:
        metrics["rhythm_metric_offline_confidence_mean"] = _safe_mean(offline_confidence.float())
    offline_component_values = []
    for component in ("exec", "budget", "prefix", "allocation", "shape"):
        component_value = output.get(f"rhythm_offline_confidence_{component}")
        if component_value is not None:
            component_mean = _safe_mean(component_value.float())
            metrics[f"rhythm_metric_offline_confidence_{component}_mean"] = component_mean
            offline_component_values.append(component_mean)
    metrics["rhythm_metric_offline_confidence_component_coverage"] = ctx.execution.speech_duration_exec.new_tensor(
        float(len(offline_component_values)) / 5.0
    )
    if len(offline_component_values) > 0:
        component_stack = torch.stack(offline_component_values, dim=0)
        metrics["rhythm_metric_offline_confidence_component_mean"] = component_stack.mean()
        metrics["rhythm_metric_offline_confidence_component_std"] = component_stack.std(unbiased=False)
        if offline_confidence is not None:
            metrics["rhythm_metric_offline_confidence_overall_component_gap"] = (
                _safe_mean(offline_confidence.float()) - component_stack.mean()
            ).abs()


def _update_alias_metrics(
    metrics: dict[str, torch.Tensor],
    output: dict[str, Any],
    ctx: RhythmMetricContext,
) -> None:
    alias_keys = {
        "L_exec_speech": "rhythm_metric_alias_L_exec_speech",
        "L_exec_pause": "rhythm_metric_alias_L_exec_pause",
        "L_budget": "rhythm_metric_alias_L_budget",
        "L_cumplan": "rhythm_metric_alias_L_cumplan",
        "L_prefix_state": "rhythm_metric_alias_L_prefix_state",
        "L_plan": "rhythm_metric_alias_L_plan",
        "L_plan_local": "rhythm_metric_alias_L_plan_local",
        "L_plan_cum": "rhythm_metric_alias_L_plan_cum",
        "L_guidance": "rhythm_metric_alias_L_guidance",
        "L_kd": "rhythm_metric_alias_L_kd",
        "L_kd_student": "rhythm_metric_alias_L_kd_student",
        "L_kd_same_source": "rhythm_metric_alias_L_kd_same_source",
        "L_kd_same_source_exec": "rhythm_metric_alias_L_kd_same_source_exec",
        "L_kd_same_source_budget": "rhythm_metric_alias_L_kd_same_source_budget",
        "L_kd_same_source_prefix": "rhythm_metric_alias_L_kd_same_source_prefix",
        "L_teacher_aux": "rhythm_metric_alias_L_teacher_aux",
        "L_rhythm_exec": "rhythm_metric_alias_L_rhythm_exec",
        "L_stream_state": "rhythm_metric_alias_L_stream_state",
        "L_base": "rhythm_metric_alias_L_base",
        "L_pitch": "rhythm_metric_alias_L_pitch",
        "rhythm_exec": "rhythm_metric_compact_rhythm_exec",
        "rhythm_prefix_state": "rhythm_metric_compact_prefix_state",
        "rhythm_stream_state": "rhythm_metric_compact_stream_state",
        "base": "rhythm_metric_compact_base",
        "pitch": "rhythm_metric_compact_pitch",
    }
    device = ctx.execution.speech_duration_exec.device
    for src_key, metric_key in alias_keys.items():
        value = _optional_scalar(output.get(src_key), device=device)
        if value is not None:
            metrics[metric_key] = value


def _resolve_sample_pause_keys(sample: dict[str, Any]) -> tuple[str, str, str]:
    pause_target_key = "rhythm_pause_exec_tgt" if "rhythm_pause_exec_tgt" in sample else "rhythm_blank_exec_tgt"
    pause_budget_key = "rhythm_pause_budget_tgt" if "rhythm_pause_budget_tgt" in sample else "rhythm_blank_budget_tgt"
    teacher_pause_key = (
        "rhythm_teacher_pause_exec_tgt"
        if "rhythm_teacher_pause_exec_tgt" in sample
        else "rhythm_teacher_blank_exec_tgt"
    )
    return pause_target_key, pause_budget_key, teacher_pause_key


def _update_sample_supervision_metrics(
    metrics: dict[str, torch.Tensor],
    sample: dict[str, Any],
    ctx: RhythmMetricContext,
) -> None:
    pause_target_key, pause_budget_key, teacher_pause_key = _resolve_sample_pause_keys(sample)
    if "rhythm_speech_budget_tgt" in sample:
        metrics["rhythm_metric_budget_speech_l1"] = (
            ctx.effective_speech_budget.unsqueeze(-1) - sample["rhythm_speech_budget_tgt"].float()
        ).abs().mean()
        metrics["rhythm_metric_raw_budget_speech_l1"] = (
            ctx.raw_speech_budget.unsqueeze(-1) - sample["rhythm_speech_budget_tgt"].float()
        ).abs().mean()
        metrics["rhythm_metric_effective_budget_speech_l1"] = metrics["rhythm_metric_budget_speech_l1"]
    if pause_budget_key in sample:
        metrics["rhythm_metric_budget_pause_l1"] = (
            ctx.effective_pause_budget.unsqueeze(-1) - sample[pause_budget_key].float()
        ).abs().mean()
        metrics["rhythm_metric_raw_budget_pause_l1"] = (
            ctx.raw_pause_budget.unsqueeze(-1) - sample[pause_budget_key].float()
        ).abs().mean()
        metrics["rhythm_metric_effective_budget_pause_l1"] = metrics["rhythm_metric_budget_pause_l1"]
    if "rhythm_speech_exec_tgt" in sample:
        metrics["rhythm_metric_exec_speech_l1"] = _masked_l1(
            ctx.execution.speech_duration_exec,
            sample["rhythm_speech_exec_tgt"],
            ctx.unit_mask,
        )
    if pause_target_key in sample:
        metrics["rhythm_metric_exec_pause_l1"] = _masked_l1(
            ctx.blank_exec,
            sample[pause_target_key],
            ctx.unit_mask,
        )
        pause_p, pause_r, pause_f1 = _masked_event_f1(
            ctx.blank_exec,
            sample[pause_target_key],
            ctx.unit_mask,
        )
        metrics["rhythm_metric_pause_event_precision"] = pause_p
        metrics["rhythm_metric_pause_event_recall"] = pause_r
        metrics["rhythm_metric_pause_event_f1"] = pause_f1
    if "rhythm_speech_exec_tgt" in sample and pause_target_key in sample:
        metrics["rhythm_metric_exec_total_corr"] = _masked_corr(
            ctx.execution.speech_duration_exec + ctx.blank_exec,
            sample["rhythm_speech_exec_tgt"] + sample[pause_target_key],
            ctx.unit_mask,
        )
        pred_prefix = torch.cumsum((ctx.execution.speech_duration_exec + ctx.blank_exec) * ctx.unit_mask, dim=1)
        tgt_prefix = torch.cumsum(
            (sample["rhythm_speech_exec_tgt"] + sample[pause_target_key]).float() * ctx.unit_mask,
            dim=1,
        )
        metrics["rhythm_metric_prefix_drift_l1"] = _masked_l1(pred_prefix, tgt_prefix, ctx.unit_mask)
        metrics["rhythm_metric_cumplan_l1"] = metrics["rhythm_metric_prefix_drift_l1"]
    if "rhythm_teacher_speech_exec_tgt" in sample:
        metrics["rhythm_metric_distill_speech_l1"] = _masked_l1(
            ctx.execution.speech_duration_exec,
            sample["rhythm_teacher_speech_exec_tgt"],
            ctx.unit_mask,
        )
    if teacher_pause_key in sample:
        metrics["rhythm_metric_distill_pause_l1"] = _masked_l1(
            ctx.blank_exec,
            sample[teacher_pause_key],
            ctx.unit_mask,
        )
        pause_p, pause_r, pause_f1 = _masked_event_f1(
            ctx.blank_exec,
            sample[teacher_pause_key],
            ctx.unit_mask,
        )
        metrics["rhythm_metric_distill_pause_event_precision"] = pause_p
        metrics["rhythm_metric_distill_pause_event_recall"] = pause_r
        metrics["rhythm_metric_distill_pause_event_f1"] = pause_f1
    if "rhythm_teacher_speech_exec_tgt" in sample and teacher_pause_key in sample:
        metrics["rhythm_metric_distill_total_corr"] = _masked_corr(
            ctx.execution.speech_duration_exec + ctx.blank_exec,
            sample["rhythm_teacher_speech_exec_tgt"] + sample[teacher_pause_key],
            ctx.unit_mask,
        )
    if "rhythm_teacher_allocation_tgt" in sample:
        metrics["rhythm_metric_distill_alloc_kl"] = _masked_kl(
            ctx.execution.speech_duration_exec + ctx.blank_exec,
            sample["rhythm_teacher_allocation_tgt"],
            ctx.unit_mask,
        )
    if "rhythm_teacher_prefix_clock_tgt" in sample:
        metrics["rhythm_metric_distill_prefix_clock_l1"] = _masked_l1(
            ctx.pred_prefix_clock,
            sample["rhythm_teacher_prefix_clock_tgt"],
            ctx.unit_mask,
        )
        metrics["rhythm_metric_distill_prefix_clock_norm_l1"] = _masked_l1(
            ctx.pred_prefix_clock / ctx.prefix_budget,
            sample["rhythm_teacher_prefix_clock_tgt"].float() / ctx.prefix_budget,
            ctx.unit_mask,
        )
    if "rhythm_teacher_prefix_backlog_tgt" in sample:
        metrics["rhythm_metric_distill_prefix_backlog_l1"] = _masked_l1(
            ctx.pred_prefix_backlog,
            sample["rhythm_teacher_prefix_backlog_tgt"],
            ctx.unit_mask,
        )
        metrics["rhythm_metric_distill_prefix_backlog_norm_l1"] = _masked_l1(
            ctx.pred_prefix_backlog / ctx.prefix_budget,
            sample["rhythm_teacher_prefix_backlog_tgt"].float() / ctx.prefix_budget,
            ctx.unit_mask,
        )
    if "rhythm_target_confidence" in sample:
        metrics["rhythm_metric_target_confidence_mean"] = _safe_mean(sample["rhythm_target_confidence"].float())
    if "rhythm_guidance_confidence" in sample:
        metrics["rhythm_metric_guidance_confidence_mean"] = _safe_mean(sample["rhythm_guidance_confidence"].float())
    if "rhythm_teacher_confidence" in sample:
        metrics["rhythm_metric_teacher_confidence_mean"] = _safe_mean(sample["rhythm_teacher_confidence"].float())
    if "rhythm_retimed_target_confidence" in sample:
        metrics["rhythm_metric_retimed_confidence_mean"] = _safe_mean(sample["rhythm_retimed_target_confidence"].float())
    if "rhythm_retimed_target_source_id" in sample:
        metrics["rhythm_metric_retimed_source_id_mean"] = _safe_mean(sample["rhythm_retimed_target_source_id"].float())
    if "rhythm_cache_version" in sample:
        metrics["rhythm_metric_cache_version_mean"] = _safe_mean(sample["rhythm_cache_version"].float())
    if "rhythm_trace_bins" in sample:
        metrics["rhythm_metric_trace_bins_mean"] = _safe_mean(sample["rhythm_trace_bins"].float())
    if "rhythm_trace_horizon" in sample:
        metrics["rhythm_metric_trace_horizon_mean"] = _safe_mean(sample["rhythm_trace_horizon"].float())
    if "rhythm_stream_prefix_ratio" in sample:
        metrics["rhythm_metric_stream_prefix_ratio"] = _safe_mean(sample["rhythm_stream_prefix_ratio"].float())
    if "rhythm_stream_visible_units" in sample:
        metrics["rhythm_metric_stream_visible_units"] = _safe_mean(sample["rhythm_stream_visible_units"].float())
    if "rhythm_stream_full_units" in sample:
        metrics["rhythm_metric_stream_full_units"] = _safe_mean(sample["rhythm_stream_full_units"].float())
    if "rhythm_reference_is_self" in sample:
        ref_self_rate = _safe_mean(sample["rhythm_reference_is_self"].float())
        metrics["rhythm_metric_reference_self_rate"] = ref_self_rate
        metrics["rhythm_metric_reference_external_rate"] = (ref_self_rate * 0.0) + (1.0 - ref_self_rate)
    if "rhythm_pair_is_identity" in sample:
        metrics["rhythm_metric_pair_identity_rate"] = _safe_mean(sample["rhythm_pair_is_identity"].float())
    if "rhythm_pair_group_id" in sample:
        group_ids = sample["rhythm_pair_group_id"].long().reshape(sample["rhythm_pair_group_id"].size(0), -1)[:, 0]
        metrics["rhythm_metric_pair_group_count"] = torch.tensor(
            float(torch.unique(group_ids).numel()),
            device=group_ids.device,
        )


def _collect_plan_surface_metrics(
    output: dict[str, Any],
    ctx: RhythmMetricContext,
) -> dict[str, torch.Tensor]:
    metrics = _build_core_rhythm_metric_dict(ctx)
    _update_render_frame_plan_metrics(metrics, output, ctx)
    _update_trace_and_boundary_metrics(metrics, output, ctx)
    return metrics


def _collect_runtime_state_metrics(
    output: dict[str, Any],
    ctx: RhythmMetricContext,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    _update_state_metrics(metrics, output, ctx)
    _update_reference_conditioning_metrics(metrics, output)
    return metrics


def _collect_teacher_target_metrics(
    output: dict[str, Any],
    ctx: RhythmMetricContext,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    _update_teacher_alignment_metrics(metrics, output, ctx)
    _update_acoustic_target_metrics(metrics, output, ctx)
    return metrics


def _collect_observability_metrics(
    output: dict[str, Any],
    ctx: RhythmMetricContext,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    _update_confidence_metrics(metrics, output, ctx)
    _update_alias_metrics(metrics, output, ctx)
    return metrics


def _collect_sample_supervision_metric_dict(
    sample: dict[str, Any],
    ctx: RhythmMetricContext,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    _update_sample_supervision_metrics(metrics, sample, ctx)
    return metrics


def build_rhythm_metric_sections(
    output: dict[str, Any],
    sample: dict[str, Any] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    execution = output.get("rhythm_execution")
    if execution is None:
        return {}
    ctx = _build_rhythm_metric_context(output, execution)
    sections = {
        "plan_surfaces": _collect_plan_surface_metrics(output, ctx),
        "runtime_state": _collect_runtime_state_metrics(output, ctx),
        "teacher_targets": _collect_teacher_target_metrics(output, ctx),
        "observability": _collect_observability_metrics(output, ctx),
    }
    if sample is not None:
        sections["sample_supervision"] = _collect_sample_supervision_metric_dict(sample, ctx)
    return sections


def build_rhythm_metric_dict(output: dict[str, Any], sample: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    for section_metrics in build_rhythm_metric_sections(output, sample=sample).values():
        metrics.update(section_metrics)
    return metrics


def build_streaming_chunk_metrics(stream_result) -> dict[str, float]:
    mel_lengths = list(stream_result.mel_lengths)
    commit_history = list(stream_result.commit_history)
    backlog_history = list(getattr(stream_result, "backlog_history", []))
    clock_history = list(getattr(stream_result, "clock_history", []))
    blank_ratio_history = list(getattr(stream_result, "blank_ratio_history", []))
    if len(mel_lengths) <= 0:
        return {
            "stream_num_chunks": 0.0,
            "stream_final_mel_len": 0.0,
            "stream_mean_chunk_mel_delta": 0.0,
            "stream_mean_commit_delta": 0.0,
            "stream_final_committed_mel_len": 0.0,
            "stream_mean_committed_mel_delta": 0.0,
            "stream_mean_prefix_exec_delta": 0.0,
            "stream_max_prefix_exec_delta": 0.0,
            "stream_no_rollback_violations": 0.0,
            "stream_commit_monotonic_violations": 0.0,
            "stream_committed_mel_rollback_violations": 0.0,
            "stream_mean_backlog": 0.0,
            "stream_max_backlog": 0.0,
            "stream_mean_backlog_delta_abs": 0.0,
            "stream_max_backlog_delta_abs": 0.0,
            "stream_mean_clock_abs": 0.0,
            "stream_max_clock_abs": 0.0,
            "stream_mean_clock_step_delta_abs": 0.0,
            "stream_max_clock_step_delta_abs": 0.0,
            "stream_mean_blank_ratio": 0.0,
        }
    mel_deltas = [mel_lengths[0]]
    mel_deltas.extend(max(0, mel_lengths[idx] - mel_lengths[idx - 1]) for idx in range(1, len(mel_lengths)))
    commit_values = [float(hist[0]) if len(hist) > 0 else 0.0 for hist in commit_history]
    commit_deltas = [commit_values[0]]
    commit_deltas.extend(max(0.0, commit_values[idx] - commit_values[idx - 1]) for idx in range(1, len(commit_values)))
    commit_monotonic_violations = sum(
        1 for idx in range(1, len(commit_values)) if commit_values[idx] < commit_values[idx - 1]
    )
    committed_lengths = list(getattr(stream_result, "committed_mel_lengths", []))
    committed_mel_rollback_violations = sum(
        1 for idx in range(1, len(committed_lengths)) if committed_lengths[idx] < committed_lengths[idx - 1]
    ) if len(committed_lengths) > 1 else 0
    committed_deltas = []
    if len(committed_lengths) > 0:
        committed_deltas.append(float(committed_lengths[0]))
        committed_deltas.extend(
            max(0.0, float(committed_lengths[idx] - committed_lengths[idx - 1]))
            for idx in range(1, len(committed_lengths))
        )
    prefix_exec_deltas = list(getattr(stream_result, "prefix_exec_deltas", []))
    rollback_threshold = 1e-4
    prefix_rollback_violations = sum(1 for value in prefix_exec_deltas if float(value) > rollback_threshold)
    backlog_step_deltas = [
        abs(float(backlog_history[idx] - backlog_history[idx - 1]))
        for idx in range(1, len(backlog_history))
    ]
    clock_step_deltas = [
        abs(float(clock_history[idx] - clock_history[idx - 1]))
        for idx in range(1, len(clock_history))
    ]
    return {
        "stream_num_chunks": float(len(mel_lengths)),
        "stream_final_mel_len": float(mel_lengths[-1]),
        "stream_mean_chunk_mel_delta": float(sum(mel_deltas) / max(len(mel_deltas), 1)),
        "stream_mean_commit_delta": float(sum(commit_deltas) / max(len(commit_deltas), 1)),
        "stream_final_committed_mel_len": float(committed_lengths[-1]) if len(committed_lengths) > 0 else 0.0,
        "stream_mean_committed_mel_delta": float(sum(committed_deltas) / max(len(committed_deltas), 1)) if len(committed_deltas) > 0 else 0.0,
        "stream_mean_prefix_exec_delta": float(sum(prefix_exec_deltas) / max(len(prefix_exec_deltas), 1)) if len(prefix_exec_deltas) > 0 else 0.0,
        "stream_max_prefix_exec_delta": float(max(prefix_exec_deltas)) if len(prefix_exec_deltas) > 0 else 0.0,
        "stream_no_rollback_violations": float(prefix_rollback_violations),
        "stream_commit_monotonic_violations": float(commit_monotonic_violations),
        "stream_committed_mel_rollback_violations": float(committed_mel_rollback_violations),
        "stream_mean_backlog": float(sum(backlog_history) / max(len(backlog_history), 1)) if len(backlog_history) > 0 else 0.0,
        "stream_max_backlog": float(max(backlog_history)) if len(backlog_history) > 0 else 0.0,
        "stream_mean_backlog_delta_abs": float(sum(backlog_step_deltas) / max(len(backlog_step_deltas), 1)) if len(backlog_step_deltas) > 0 else 0.0,
        "stream_max_backlog_delta_abs": float(max(backlog_step_deltas)) if len(backlog_step_deltas) > 0 else 0.0,
        "stream_mean_clock_abs": float(sum(abs(x) for x in clock_history) / max(len(clock_history), 1)) if len(clock_history) > 0 else 0.0,
        "stream_max_clock_abs": float(max(abs(x) for x in clock_history)) if len(clock_history) > 0 else 0.0,
        "stream_mean_clock_step_delta_abs": float(sum(clock_step_deltas) / max(len(clock_step_deltas), 1)) if len(clock_step_deltas) > 0 else 0.0,
        "stream_max_clock_step_delta_abs": float(max(clock_step_deltas)) if len(clock_step_deltas) > 0 else 0.0,
        "stream_mean_blank_ratio": float(sum(blank_ratio_history) / max(len(blank_ratio_history), 1)) if len(blank_ratio_history) > 0 else 0.0,
    }

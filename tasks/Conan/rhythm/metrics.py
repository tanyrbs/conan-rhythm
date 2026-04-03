from __future__ import annotations

from typing import Any

import torch


def _safe_mean(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 0:
        return torch.tensor(0.0, device=x.device if isinstance(x, torch.Tensor) else "cpu")
    return x.float().mean()


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


def _build_prefix_carry(
    *,
    speech_exec: torch.Tensor,
    blank_exec: torch.Tensor,
    dur_anchor_src: torch.Tensor,
    unit_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    unit_mask = unit_mask.float()
    prefix_clock = torch.cumsum(
        ((speech_exec.float() + blank_exec.float()) - dur_anchor_src.float()) * unit_mask,
        dim=1,
    ) * unit_mask
    prefix_backlog = prefix_clock.clamp_min(0.0) * unit_mask
    return prefix_clock, prefix_backlog


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


def build_rhythm_metric_dict(output: dict[str, Any], sample: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
    execution = output.get("rhythm_execution")
    if execution is None:
        return {}
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
    commit_ratio = execution.commit_frontier.float() / visible_units
    pred_prefix_clock, pred_prefix_backlog = _build_prefix_carry(
        speech_exec=execution.speech_duration_exec,
        blank_exec=blank_exec,
        dur_anchor_src=unit_batch.dur_anchor_src if unit_batch is not None else execution.speech_duration_exec,
        unit_mask=unit_mask,
    )
    prefix_budget = torch.cumsum(
        (unit_batch.dur_anchor_src if unit_batch is not None else execution.speech_duration_exec).float() * unit_mask,
        dim=1,
    ).clamp_min(1.0)
    final_indices = (visible_units.long() - 1).clamp_min(0).unsqueeze(1)
    final_prefix_clock = pred_prefix_clock.gather(1, final_indices).squeeze(1)

    metrics = {
        "rhythm_metric_speech_budget_mean": _safe_mean(planner.speech_budget_win.squeeze(-1)),
        "rhythm_metric_pause_budget_mean": _safe_mean(planner.pause_budget_win.squeeze(-1)),
        "rhythm_metric_dur_shape_abs_mean": _safe_mean(planner.dur_shape_unit.abs()),
        "rhythm_metric_pause_shape_entropy": _safe_mean(
            -(
                planner.pause_shape_unit.float().clamp_min(1e-6)
                * planner.pause_shape_unit.float().clamp_min(1e-6).log()
            ).sum(dim=1)
        ),
        "rhythm_metric_boundary_score_mean": _safe_mean(planner.boundary_score_unit.float()),
        "rhythm_metric_speech_total_mean": _safe_mean(speech_total),
        "rhythm_metric_pause_total_mean": _safe_mean(pause_total),
        "rhythm_metric_pause_share_mean": _safe_mean(pause_total / total_exec),
        "rhythm_metric_pause_event_ratio_mean": _safe_mean((execution.pause_after_exec.float() > 0.5).float().sum(dim=1) / visible_units),
        "rhythm_metric_blank_slot_ratio_mean": _safe_mean((blank_exec.float() > 0.5).float().sum(dim=1) / visible_units),
        "rhythm_metric_expand_ratio_mean": _safe_mean(speech_total / anchor_total.clamp_min(1.0)),
        "rhythm_metric_commit_ratio_mean": _safe_mean(commit_ratio),
        "rhythm_metric_budget_violation_mean": _safe_mean(
            ((speech_total + pause_total) - (planner.speech_budget_win.squeeze(-1) + planner.pause_budget_win.squeeze(-1))).abs()
        ),
        "rhythm_metric_prefix_clock_abs_mean": _safe_mean(pred_prefix_clock.abs()),
        "rhythm_metric_prefix_backlog_mean": _safe_mean(pred_prefix_backlog),
        "rhythm_metric_prefix_backlog_max": pred_prefix_backlog.max(),
        "rhythm_metric_deadline_final_abs_mean": _safe_mean(final_prefix_clock.abs()),
    }
    render_blank_mask = output.get("rhythm_blank_mask")
    if render_blank_mask is not None:
        render_total_mask = output.get("rhythm_total_mask")
        denom = render_total_mask.float().sum().clamp_min(1.0) if render_total_mask is not None else float(render_blank_mask.numel())
        metrics["rhythm_metric_render_blank_ratio"] = render_blank_mask.float().sum() / denom
    frame_plan = _as_frame_plan(output, execution)
    if frame_plan is not None:
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
        metrics["rhythm_metric_frame_plan_present"] = execution.speech_duration_exec.new_tensor(1.0)
        metrics["rhythm_metric_frame_plan_total_frames_mean"] = _safe_mean(frame_total_mask.sum(dim=1))
        metrics["rhythm_metric_frame_plan_blank_ratio_mean"] = frame_blank / frame_total
        metrics["rhythm_metric_frame_plan_speech_ratio_mean"] = frame_nonblank / frame_total
        metrics["rhythm_metric_frame_plan_blank_src_consistency"] = blank_and_neg_src.sum() / blank_denom
        metrics["rhythm_metric_frame_plan_speech_src_consistency"] = speech_and_pos_src.sum() / speech_denom
        if render_blank_mask is not None:
            mask_for_l1 = frame_total_mask
            if render_blank_mask.shape == frame_blank_mask.shape:
                metrics["rhythm_metric_render_vs_frame_plan_blank_l1"] = _masked_l1(
                    render_blank_mask.float(),
                    frame_blank_mask.float(),
                    mask_for_l1,
                )
    else:
        metrics["rhythm_metric_frame_plan_present"] = execution.speech_duration_exec.new_tensor(0.0)
    metrics["rhythm_metric_local_rate_transfer_corr"] = _masked_corr(
        execution.speech_duration_exec.float(),
        planner.trace_context[:, :, 1].float(),
        unit_mask,
    )
    metrics["rhythm_metric_pause_trace_corr"] = _masked_corr(
        blank_exec.float(),
        planner.trace_context[:, :, 0].float(),
        unit_mask,
    )
    metrics["rhythm_metric_boundary_trace_corr"] = _masked_corr(
        blank_exec.float(),
        planner.trace_context[:, :, 2].float(),
        unit_mask,
    )
    source_boundary_cue = output.get("source_boundary_cue")
    if source_boundary_cue is not None:
        metrics["rhythm_metric_source_boundary_mean"] = _safe_mean(source_boundary_cue.float())
        metrics["rhythm_metric_source_boundary_pause_corr"] = _masked_corr(
            blank_exec.float(),
            source_boundary_cue.float(),
            unit_mask,
        )

    state_next = output.get("rhythm_state_next")
    if state_next is not None:
        metrics.update(
            {
                "rhythm_metric_phase_mean": _safe_mean(state_next.phase_ptr),
                "rhythm_metric_backlog_mean": _safe_mean(state_next.backlog),
                "rhythm_metric_clock_delta_mean": _safe_mean(state_next.clock_delta),
                "rhythm_metric_clock_delta_abs_mean": _safe_mean(state_next.clock_delta.abs()),
            }
        )
        phase_progress = getattr(state_next, "phase_anchor_progress", None)
        phase_total = getattr(state_next, "phase_anchor_total", None)
        if phase_progress is not None and phase_total is not None:
            progress_ratio = phase_progress.float() / phase_total.float().clamp_min(1.0)
            phase_ptr = state_next.phase_ptr.float()
            metrics["rhythm_metric_phase_progress_ratio_mean"] = _safe_mean(progress_ratio)
            metrics["rhythm_metric_phase_ptr_vs_progress_l1"] = _safe_mean((phase_ptr - progress_ratio).abs())
            metrics["rhythm_metric_phase_ptr_below_progress_ratio"] = _safe_mean(
                (phase_ptr + 1e-6 < progress_ratio).float()
            )
    state_prev = output.get("rhythm_state_prev")
    if state_prev is not None and state_next is not None:
        phase_delta = state_next.phase_ptr.float() - state_prev.phase_ptr.float()
        metrics["rhythm_metric_phase_delta_mean"] = _safe_mean(phase_delta)
        metrics["rhythm_metric_phase_delta_min"] = phase_delta.min()
        metrics["rhythm_metric_phase_nonretro_rate"] = _safe_mean((phase_delta >= -1e-6).float())
    ref_conditioning = output.get("rhythm_ref_conditioning")
    if ref_conditioning is not None:
        selector_scores = ref_conditioning.get("selector_meta_scores")
        slow_summary = ref_conditioning.get("slow_rhythm_summary")
        if selector_scores is not None:
            metrics["rhythm_metric_selector_score_mean"] = _safe_mean(selector_scores.float())
            metrics["rhythm_metric_selector_score_max"] = selector_scores.float().max()
        if slow_summary is not None:
            metrics["rhythm_metric_slow_summary_norm_mean"] = _safe_mean(torch.norm(slow_summary.float(), dim=-1))
    offline_execution = output.get("rhythm_offline_execution")
    if offline_execution is not None:
        target_units = execution.speech_duration_exec.size(1)
        offline_speech = _slice_to_visible_prefix(offline_execution.speech_duration_exec, target_units)
        offline_blank = _slice_to_visible_prefix(
            getattr(offline_execution, "blank_duration_exec", offline_execution.pause_after_exec),
            target_units,
        )
        offline_prefix_clock, offline_prefix_backlog = _build_prefix_carry(
            speech_exec=offline_speech,
            blank_exec=offline_blank,
            dur_anchor_src=unit_batch.dur_anchor_src if unit_batch is not None else offline_execution.speech_duration_exec,
            unit_mask=unit_mask,
        )
        metrics["rhythm_metric_offline_online_speech_l1"] = _masked_l1(
            execution.speech_duration_exec,
            offline_speech,
            unit_mask,
        )
        metrics["rhythm_metric_offline_online_pause_l1"] = _masked_l1(
            blank_exec,
            offline_blank,
            unit_mask,
        )
        metrics["rhythm_metric_offline_online_total_corr"] = _masked_corr(
            execution.speech_duration_exec + blank_exec,
            offline_speech + offline_blank,
            unit_mask,
        )
        metrics["rhythm_metric_offline_online_alloc_kl"] = _masked_kl(
            execution.speech_duration_exec + blank_exec,
            offline_speech + offline_blank,
            unit_mask,
        )
        metrics["rhythm_metric_offline_online_prefix_clock_l1"] = _masked_l1(
            pred_prefix_clock,
            offline_prefix_clock,
            unit_mask,
        )
        metrics["rhythm_metric_offline_online_prefix_backlog_l1"] = _masked_l1(
            pred_prefix_backlog,
            offline_prefix_backlog,
            unit_mask,
        )
    algorithmic_teacher = output.get("rhythm_algorithmic_teacher")
    if algorithmic_teacher is not None:
        metrics["rhythm_metric_algorithmic_teacher_confidence"] = _safe_mean(algorithmic_teacher.confidence.float())
        metrics["rhythm_metric_algorithmic_teacher_pause_l1"] = _masked_l1(
            blank_exec,
            algorithmic_teacher.pause_exec_tgt,
            unit_mask,
        )
        metrics["rhythm_metric_algorithmic_teacher_alloc_kl"] = _masked_kl(
            execution.speech_duration_exec + blank_exec,
            algorithmic_teacher.allocation_tgt,
            unit_mask,
        )
        metrics["rhythm_metric_algorithmic_teacher_prefix_clock_l1"] = _masked_l1(
            pred_prefix_clock,
            algorithmic_teacher.prefix_clock_tgt,
            unit_mask,
        )
        metrics["rhythm_metric_algorithmic_teacher_prefix_backlog_l1"] = _masked_l1(
            pred_prefix_backlog,
            algorithmic_teacher.prefix_backlog_tgt,
            unit_mask,
        )
    metrics["rhythm_metric_acoustic_target_is_retimed"] = execution.speech_duration_exec.new_tensor(
        1.0 if bool(output.get("acoustic_target_is_retimed", False)) else 0.0
    )
    metrics["rhythm_metric_apply_render_mean"] = execution.speech_duration_exec.new_tensor(
        float(output.get("rhythm_apply_render", 0.0))
    )
    acoustic_target_weight = output.get("acoustic_target_weight")
    if acoustic_target_weight is not None:
        metrics["rhythm_metric_retimed_weight_mean"] = _safe_mean(acoustic_target_weight.float())
    pause_topk_ratio = output.get("rhythm_projector_pause_topk_ratio")
    if pause_topk_ratio is not None:
        metrics["rhythm_metric_pause_topk_ratio_mean"] = _safe_mean(pause_topk_ratio.float())
        metrics["rhythm_metric_pause_topk_sparsity_mean"] = execution.speech_duration_exec.new_tensor(1.0) - _safe_mean(
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
    acoustic_target_source = output.get("acoustic_target_source")
    acoustic_source_name = str(acoustic_target_source) if acoustic_target_source is not None else "unknown"
    source_to_id = {"source": 0.0, "cached": 1.0, "online": 2.0}
    source_id = source_to_id.get(acoustic_source_name, -1.0)
    metrics["rhythm_metric_acoustic_target_source_id"] = execution.speech_duration_exec.new_tensor(source_id)
    metrics["rhythm_metric_acoustic_target_source_is_source"] = execution.speech_duration_exec.new_tensor(
        1.0 if acoustic_source_name == "source" else 0.0
    )
    metrics["rhythm_metric_acoustic_target_source_is_cached"] = execution.speech_duration_exec.new_tensor(
        1.0 if acoustic_source_name == "cached" else 0.0
    )
    metrics["rhythm_metric_acoustic_target_source_is_online"] = execution.speech_duration_exec.new_tensor(
        1.0 if acoustic_source_name == "online" else 0.0
    )
    metrics["rhythm_metric_acoustic_target_source_unknown"] = execution.speech_duration_exec.new_tensor(
        1.0 if source_id < 0.0 else 0.0
    )
    offline_confidence = output.get("rhythm_offline_confidence")
    if offline_confidence is not None:
        metrics["rhythm_metric_offline_confidence_mean"] = _safe_mean(offline_confidence.float())
    offline_component_values = []
    for component in ("exec", "budget", "prefix", "allocation"):
        component_value = output.get(f"rhythm_offline_confidence_{component}")
        if component_value is not None:
            component_mean = _safe_mean(component_value.float())
            metrics[f"rhythm_metric_offline_confidence_{component}_mean"] = component_mean
            offline_component_values.append(component_mean)
    metrics["rhythm_metric_offline_confidence_component_coverage"] = execution.speech_duration_exec.new_tensor(
        float(len(offline_component_values)) / 4.0
    )
    if len(offline_component_values) > 0:
        component_stack = torch.stack(offline_component_values, dim=0)
        metrics["rhythm_metric_offline_confidence_component_mean"] = component_stack.mean()
        metrics["rhythm_metric_offline_confidence_component_std"] = component_stack.std(unbiased=False)
        if offline_confidence is not None:
            metrics["rhythm_metric_offline_confidence_overall_component_gap"] = (
                _safe_mean(offline_confidence.float()) - component_stack.mean()
            ).abs()

    # Keep compact optimization heads and public aliases observable in metrics.
    alias_keys = {
        "L_exec_speech": "rhythm_metric_alias_L_exec_speech",
        "L_exec_pause": "rhythm_metric_alias_L_exec_pause",
        "L_budget": "rhythm_metric_alias_L_budget",
        "L_cumplan": "rhythm_metric_alias_L_cumplan",
        "L_prefix_state": "rhythm_metric_alias_L_prefix_state",
        "L_rhythm_exec": "rhythm_metric_alias_L_rhythm_exec",
        "L_stream_state": "rhythm_metric_alias_L_stream_state",
        "L_base": "rhythm_metric_alias_L_base",
        "L_pitch": "rhythm_metric_alias_L_pitch",
        "rhythm_exec": "rhythm_metric_compact_rhythm_exec",
        "rhythm_stream_state": "rhythm_metric_compact_stream_state",
        "base": "rhythm_metric_compact_base",
        "pitch": "rhythm_metric_compact_pitch",
    }
    device = execution.speech_duration_exec.device
    for src_key, metric_key in alias_keys.items():
        value = _optional_scalar(output.get(src_key), device=device)
        if value is not None:
            metrics[metric_key] = value

    if sample is not None:
        pause_target_key = "rhythm_pause_exec_tgt" if "rhythm_pause_exec_tgt" in sample else "rhythm_blank_exec_tgt"
        pause_budget_key = "rhythm_pause_budget_tgt" if "rhythm_pause_budget_tgt" in sample else "rhythm_blank_budget_tgt"
        teacher_pause_key = (
            "rhythm_teacher_pause_exec_tgt"
            if "rhythm_teacher_pause_exec_tgt" in sample
            else "rhythm_teacher_blank_exec_tgt"
        )
        if "rhythm_speech_budget_tgt" in sample:
            metrics["rhythm_metric_budget_speech_l1"] = (
                planner.speech_budget_win.float() - sample["rhythm_speech_budget_tgt"].float()
            ).abs().mean()
        if pause_budget_key in sample:
            metrics["rhythm_metric_budget_pause_l1"] = (
                planner.pause_budget_win.float() - sample[pause_budget_key].float()
            ).abs().mean()
        if "rhythm_speech_exec_tgt" in sample:
            metrics["rhythm_metric_exec_speech_l1"] = _masked_l1(
                execution.speech_duration_exec,
                sample["rhythm_speech_exec_tgt"],
                unit_mask,
            )
        if pause_target_key in sample:
            metrics["rhythm_metric_exec_pause_l1"] = _masked_l1(
                blank_exec,
                sample[pause_target_key],
                unit_mask,
            )
            pause_p, pause_r, pause_f1 = _masked_event_f1(
                blank_exec,
                sample[pause_target_key],
                unit_mask,
            )
            metrics["rhythm_metric_pause_event_precision"] = pause_p
            metrics["rhythm_metric_pause_event_recall"] = pause_r
            metrics["rhythm_metric_pause_event_f1"] = pause_f1
        if "rhythm_speech_exec_tgt" in sample and pause_target_key in sample:
            metrics["rhythm_metric_exec_total_corr"] = _masked_corr(
                execution.speech_duration_exec + blank_exec,
                sample["rhythm_speech_exec_tgt"] + sample[pause_target_key],
                unit_mask,
            )
            pred_prefix = torch.cumsum((execution.speech_duration_exec + blank_exec) * unit_mask, dim=1)
            tgt_prefix = torch.cumsum((sample["rhythm_speech_exec_tgt"] + sample[pause_target_key]).float() * unit_mask, dim=1)
            metrics["rhythm_metric_prefix_drift_l1"] = _masked_l1(pred_prefix, tgt_prefix, unit_mask)
            metrics["rhythm_metric_cumplan_l1"] = metrics["rhythm_metric_prefix_drift_l1"]
        if "rhythm_teacher_speech_exec_tgt" in sample:
            metrics["rhythm_metric_distill_speech_l1"] = _masked_l1(
                execution.speech_duration_exec,
                sample["rhythm_teacher_speech_exec_tgt"],
                unit_mask,
            )
        if teacher_pause_key in sample:
            metrics["rhythm_metric_distill_pause_l1"] = _masked_l1(
                blank_exec,
                sample[teacher_pause_key],
                unit_mask,
            )
            pause_p, pause_r, pause_f1 = _masked_event_f1(
                blank_exec,
                sample[teacher_pause_key],
                unit_mask,
            )
            metrics["rhythm_metric_distill_pause_event_precision"] = pause_p
            metrics["rhythm_metric_distill_pause_event_recall"] = pause_r
            metrics["rhythm_metric_distill_pause_event_f1"] = pause_f1
        if "rhythm_teacher_speech_exec_tgt" in sample and teacher_pause_key in sample:
            metrics["rhythm_metric_distill_total_corr"] = _masked_corr(
                execution.speech_duration_exec + blank_exec,
                sample["rhythm_teacher_speech_exec_tgt"] + sample[teacher_pause_key],
                unit_mask,
            )
        if "rhythm_teacher_allocation_tgt" in sample:
            metrics["rhythm_metric_distill_alloc_kl"] = _masked_kl(
                execution.speech_duration_exec + blank_exec,
                sample["rhythm_teacher_allocation_tgt"],
                unit_mask,
            )
        if "rhythm_teacher_prefix_clock_tgt" in sample:
            metrics["rhythm_metric_distill_prefix_clock_l1"] = _masked_l1(
                pred_prefix_clock,
                sample["rhythm_teacher_prefix_clock_tgt"],
                unit_mask,
            )
            metrics["rhythm_metric_distill_prefix_clock_norm_l1"] = _masked_l1(
                pred_prefix_clock / prefix_budget,
                sample["rhythm_teacher_prefix_clock_tgt"].float() / prefix_budget,
                unit_mask,
            )
        if "rhythm_teacher_prefix_backlog_tgt" in sample:
            metrics["rhythm_metric_distill_prefix_backlog_l1"] = _masked_l1(
                pred_prefix_backlog,
                sample["rhythm_teacher_prefix_backlog_tgt"],
                unit_mask,
            )
            metrics["rhythm_metric_distill_prefix_backlog_norm_l1"] = _masked_l1(
                pred_prefix_backlog / prefix_budget,
                sample["rhythm_teacher_prefix_backlog_tgt"].float() / prefix_budget,
                unit_mask,
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

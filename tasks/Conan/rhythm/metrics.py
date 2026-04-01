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
        offline_blank = getattr(offline_execution, "blank_duration_exec", offline_execution.pause_after_exec)
        offline_prefix_clock, offline_prefix_backlog = _build_prefix_carry(
            speech_exec=offline_execution.speech_duration_exec,
            blank_exec=offline_blank,
            dur_anchor_src=unit_batch.dur_anchor_src if unit_batch is not None else offline_execution.speech_duration_exec,
            unit_mask=unit_mask,
        )
        metrics["rhythm_metric_offline_online_speech_l1"] = _masked_l1(
            execution.speech_duration_exec,
            offline_execution.speech_duration_exec,
            unit_mask,
        )
        metrics["rhythm_metric_offline_online_pause_l1"] = _masked_l1(
            blank_exec,
            offline_blank,
            unit_mask,
        )
        metrics["rhythm_metric_offline_online_total_corr"] = _masked_corr(
            execution.speech_duration_exec + blank_exec,
            offline_execution.speech_duration_exec + offline_blank,
            unit_mask,
        )
        metrics["rhythm_metric_offline_online_alloc_kl"] = _masked_kl(
            execution.speech_duration_exec + blank_exec,
            offline_execution.speech_duration_exec + offline_blank,
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

    if sample is not None:
        pause_target_key = "rhythm_blank_exec_tgt" if "rhythm_blank_exec_tgt" in sample else "rhythm_pause_exec_tgt"
        pause_budget_key = "rhythm_blank_budget_tgt" if "rhythm_blank_budget_tgt" in sample else "rhythm_pause_budget_tgt"
        teacher_pause_key = (
            "rhythm_teacher_blank_exec_tgt"
            if "rhythm_teacher_blank_exec_tgt" in sample
            else "rhythm_teacher_pause_exec_tgt"
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
            "stream_mean_backlog": 0.0,
            "stream_max_backlog": 0.0,
            "stream_mean_clock_abs": 0.0,
            "stream_max_clock_abs": 0.0,
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
    committed_deltas = []
    if len(committed_lengths) > 0:
        committed_deltas.append(float(committed_lengths[0]))
        committed_deltas.extend(
            max(0.0, float(committed_lengths[idx] - committed_lengths[idx - 1]))
            for idx in range(1, len(committed_lengths))
        )
    prefix_exec_deltas = list(getattr(stream_result, "prefix_exec_deltas", []))
    return {
        "stream_num_chunks": float(len(mel_lengths)),
        "stream_final_mel_len": float(mel_lengths[-1]),
        "stream_mean_chunk_mel_delta": float(sum(mel_deltas) / max(len(mel_deltas), 1)),
        "stream_mean_commit_delta": float(sum(commit_deltas) / max(len(commit_deltas), 1)),
        "stream_final_committed_mel_len": float(committed_lengths[-1]) if len(committed_lengths) > 0 else 0.0,
        "stream_mean_committed_mel_delta": float(sum(committed_deltas) / max(len(committed_deltas), 1)) if len(committed_deltas) > 0 else 0.0,
        "stream_mean_prefix_exec_delta": float(sum(prefix_exec_deltas) / max(len(prefix_exec_deltas), 1)) if len(prefix_exec_deltas) > 0 else 0.0,
        "stream_max_prefix_exec_delta": float(max(prefix_exec_deltas)) if len(prefix_exec_deltas) > 0 else 0.0,
        "stream_commit_monotonic_violations": float(commit_monotonic_violations),
        "stream_mean_backlog": float(sum(backlog_history) / max(len(backlog_history), 1)) if len(backlog_history) > 0 else 0.0,
        "stream_max_backlog": float(max(backlog_history)) if len(backlog_history) > 0 else 0.0,
        "stream_mean_clock_abs": float(sum(abs(x) for x in clock_history) / max(len(clock_history), 1)) if len(clock_history) > 0 else 0.0,
        "stream_max_clock_abs": float(max(abs(x) for x in clock_history)) if len(clock_history) > 0 else 0.0,
        "stream_mean_blank_ratio": float(sum(blank_ratio_history) / max(len(blank_ratio_history), 1)) if len(blank_ratio_history) > 0 else 0.0,
    }

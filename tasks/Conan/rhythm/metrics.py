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


def build_rhythm_metric_dict(output: dict[str, Any], sample: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
    execution = output.get("rhythm_execution")
    if execution is None:
        return {}
    planner = execution.planner
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
    pause_total = execution.pause_after_exec.float().sum(dim=1)
    total_exec = (speech_total + pause_total).clamp_min(1e-6)
    commit_ratio = execution.commit_frontier.float() / visible_units

    metrics = {
        "rhythm_metric_speech_budget_mean": _safe_mean(planner.speech_budget_win.squeeze(-1)),
        "rhythm_metric_pause_budget_mean": _safe_mean(planner.pause_budget_win.squeeze(-1)),
        "rhythm_metric_speech_total_mean": _safe_mean(speech_total),
        "rhythm_metric_pause_total_mean": _safe_mean(pause_total),
        "rhythm_metric_pause_share_mean": _safe_mean(pause_total / total_exec),
        "rhythm_metric_expand_ratio_mean": _safe_mean(speech_total / anchor_total.clamp_min(1.0)),
        "rhythm_metric_commit_ratio_mean": _safe_mean(commit_ratio),
    }

    state_next = output.get("rhythm_state_next")
    if state_next is not None:
        metrics.update(
            {
                "rhythm_metric_phase_mean": _safe_mean(state_next.phase_ptr),
                "rhythm_metric_backlog_mean": _safe_mean(state_next.backlog),
                "rhythm_metric_clock_delta_mean": _safe_mean(state_next.clock_delta),
            }
        )

    if sample is not None:
        if "rhythm_speech_budget_tgt" in sample:
            metrics["rhythm_metric_budget_speech_l1"] = (
                planner.speech_budget_win.float() - sample["rhythm_speech_budget_tgt"].float()
            ).abs().mean()
        if "rhythm_pause_budget_tgt" in sample:
            metrics["rhythm_metric_budget_pause_l1"] = (
                planner.pause_budget_win.float() - sample["rhythm_pause_budget_tgt"].float()
            ).abs().mean()
        if "rhythm_speech_exec_tgt" in sample:
            metrics["rhythm_metric_exec_speech_l1"] = _masked_l1(
                execution.speech_duration_exec,
                sample["rhythm_speech_exec_tgt"],
                unit_mask,
            )
        if "rhythm_pause_exec_tgt" in sample:
            metrics["rhythm_metric_exec_pause_l1"] = _masked_l1(
                execution.pause_after_exec,
                sample["rhythm_pause_exec_tgt"],
                unit_mask,
            )
    return metrics


def build_streaming_chunk_metrics(stream_result) -> dict[str, float]:
    mel_lengths = list(stream_result.mel_lengths)
    commit_history = list(stream_result.commit_history)
    if len(mel_lengths) <= 0:
        return {
            "stream_num_chunks": 0.0,
            "stream_final_mel_len": 0.0,
            "stream_mean_chunk_mel_delta": 0.0,
            "stream_mean_commit_delta": 0.0,
        }
    mel_deltas = [mel_lengths[0]]
    mel_deltas.extend(max(0, mel_lengths[idx] - mel_lengths[idx - 1]) for idx in range(1, len(mel_lengths)))
    commit_values = [float(hist[0]) if len(hist) > 0 else 0.0 for hist in commit_history]
    commit_deltas = [commit_values[0]]
    commit_deltas.extend(max(0.0, commit_values[idx] - commit_values[idx - 1]) for idx in range(1, len(commit_values)))
    return {
        "stream_num_chunks": float(len(mel_lengths)),
        "stream_final_mel_len": float(mel_lengths[-1]),
        "stream_mean_chunk_mel_delta": float(sum(mel_deltas) / max(len(mel_deltas), 1)),
        "stream_mean_commit_delta": float(sum(commit_deltas) / max(len(commit_deltas), 1)),
    }

from __future__ import annotations

import torch


def _rounded_effective_durations(output: dict) -> torch.Tensor | None:
    execution = output.get("rhythm_execution")
    if execution is None:
        return None
    speech = torch.round(execution.speech_duration_exec.float()).long().clamp_min(0)
    pause = torch.round(execution.pause_after_exec.float()).long().clamp_min(0)
    return speech + pause


def compute_committed_mel_length(output: dict, batch_index: int = 0) -> int:
    mel_out = output.get("mel_out")
    if mel_out is None:
        return 0
    effective = _rounded_effective_durations(output)
    commit_frontier = output.get("commit_frontier")
    if effective is None or commit_frontier is None:
        return int(mel_out[batch_index].shape[0])
    frontier = int(commit_frontier[batch_index].item())
    frontier = max(0, min(frontier, int(effective[batch_index].shape[0])))
    return int(effective[batch_index, :frontier].sum().item())


def extract_incremental_committed_mel(
    output: dict,
    *,
    prev_committed_len: int,
    batch_index: int = 0,
) -> tuple[torch.Tensor, int]:
    mel_out = output["mel_out"][batch_index]
    committed_len = compute_committed_mel_length(output, batch_index=batch_index)
    committed_len = max(prev_committed_len, min(committed_len, int(mel_out.shape[0])))
    mel_new = mel_out[prev_committed_len:committed_len]
    return mel_new, committed_len

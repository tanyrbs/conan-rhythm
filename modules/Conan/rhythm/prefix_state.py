from __future__ import annotations

import numpy as np
import torch


def build_prefix_state_from_exec_torch(
    speech_exec: torch.Tensor,
    pause_exec: torch.Tensor,
    dur_anchor_src: torch.Tensor,
    unit_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Authoritative prefix clock/backlog construction for torch code paths."""
    unit_mask = unit_mask.float()
    prefix_clock = torch.cumsum(
        ((speech_exec.float() + pause_exec.float()) - dur_anchor_src.float()) * unit_mask,
        dim=1,
    ) * unit_mask
    prefix_backlog = prefix_clock.clamp_min(0.0) * unit_mask
    return prefix_clock, prefix_backlog


def normalize_prefix_state_torch(
    prefix_state: torch.Tensor,
    *,
    dur_anchor_src: torch.Tensor,
    unit_mask: torch.Tensor,
) -> torch.Tensor:
    prefix_budget = torch.cumsum(dur_anchor_src.float() * unit_mask.float(), dim=1).clamp_min(1.0)
    return prefix_state.float() / prefix_budget


def build_prefix_state_from_exec_numpy(
    speech_exec,
    pause_exec,
    dur_anchor_src,
    unit_mask=None,
) -> tuple[np.ndarray, np.ndarray]:
    speech_exec = np.asarray(speech_exec, dtype=np.float32).reshape(-1)
    pause_exec = np.asarray(pause_exec, dtype=np.float32).reshape(-1)
    dur_anchor_src = np.asarray(dur_anchor_src, dtype=np.float32).reshape(-1)
    visible = min(len(speech_exec), len(pause_exec), len(dur_anchor_src))
    speech_exec = speech_exec[:visible]
    pause_exec = pause_exec[:visible]
    dur_anchor_src = dur_anchor_src[:visible]
    if unit_mask is None:
        unit_mask = (dur_anchor_src > 0).astype(np.float32)
    else:
        unit_mask = np.asarray(unit_mask, dtype=np.float32).reshape(-1)[:visible]
    prefix_clock = (
        np.cumsum(((speech_exec + pause_exec) - dur_anchor_src) * unit_mask, axis=0) * unit_mask
    ).astype(np.float32)
    prefix_backlog = np.maximum(prefix_clock, 0.0).astype(np.float32) * unit_mask
    return prefix_clock.astype(np.float32), prefix_backlog.astype(np.float32)

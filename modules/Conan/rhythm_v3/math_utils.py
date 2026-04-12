from __future__ import annotations

import torch


def apply_analytic_gap_clip(
    analytic_gap: torch.Tensor,
    clip_value: float | None,
) -> torch.Tensor:
    if clip_value is None:
        return analytic_gap
    clip = float(clip_value)
    if clip <= 0.0:
        return analytic_gap
    return analytic_gap.clamp(min=-clip, max=clip)


def build_causal_local_rate_seq(
    *,
    observed_log: torch.Tensor,
    speech_mask: torch.Tensor,
    init_rate: torch.Tensor | None,
    default_init_rate: torch.Tensor | float | None = None,
    decay: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    observed_log: [B, U], usually normalized log-duration observations.
    speech_mask:  [B, U], 1 only for speech units that may update the EMA.
    init_rate:    [B, 1] or None, carried runtime state.
    Returns:
        local_rate_seq:  [B, U], rate state BEFORE consuming current unit
        local_rate_last: [B, 1], state AFTER consuming the whole chunk
    """
    batch_size, num_units = observed_log.shape
    if init_rate is None:
        if isinstance(default_init_rate, torch.Tensor):
            prev = default_init_rate.to(device=observed_log.device, dtype=observed_log.dtype).reshape(-1)
            if prev.numel() == 1:
                prev = prev.view(1, 1).expand(batch_size, 1)
            elif prev.numel() == batch_size:
                prev = prev.reshape(batch_size, 1)
            else:
                raise ValueError(
                    f"default_init_rate must be scalar or batch-sized tensor, got shape={tuple(default_init_rate.shape)}"
                )
        elif default_init_rate is None:
            prev = observed_log.new_zeros((batch_size, 1))
        else:
            prev = observed_log.new_full((batch_size, 1), float(default_init_rate))
    else:
        prev = init_rate.float().reshape(batch_size, 1)
    decay = float(max(0.0, min(0.999, decay)))

    seq: list[torch.Tensor] = []
    for unit_idx in range(num_units):
        seq.append(prev)
        use_t = speech_mask[:, unit_idx : unit_idx + 1] > 0.5
        cur_t = observed_log[:, unit_idx : unit_idx + 1]
        prev = torch.where(use_t, decay * prev + (1.0 - decay) * cur_t, prev)

    if seq:
        local_rate_seq = torch.cat(seq, dim=1)
    else:
        local_rate_seq = observed_log.new_zeros((batch_size, 0))
    return local_rate_seq, prev


__all__ = ["apply_analytic_gap_clip", "build_causal_local_rate_seq"]

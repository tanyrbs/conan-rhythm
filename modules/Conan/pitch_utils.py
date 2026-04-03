from __future__ import annotations

import torch


F0_MIN = 6.0
F0_MAX = 10.0


def f0_minmax_norm(
    x: torch.Tensor,
    uv: torch.Tensor | None = None,
    *,
    strict_upper: bool = False,
) -> torch.Tensor:
    if strict_upper and torch.any(x > F0_MAX):
        raise ValueError("check minmax_norm!!")
    normed_x = (x - F0_MIN) / (F0_MAX - F0_MIN) * 2 - 1
    if uv is not None:
        normed_x = normed_x.masked_fill(uv > 0, 0)
    return normed_x


def f0_minmax_denorm(x: torch.Tensor, uv: torch.Tensor | None = None) -> torch.Tensor:
    denormed_x = (x + 1) / 2 * (F0_MAX - F0_MIN) + F0_MIN
    if uv is not None:
        denormed_x = denormed_x.masked_fill(uv > 0, 0)
    return denormed_x


def apply_silent_content_to_uv(
    uv: torch.Tensor,
    *,
    content: torch.Tensor | None,
    silent_token: int | None,
) -> torch.Tensor:
    if content is None or silent_token is None or content.shape != uv.shape:
        return uv
    uv = uv.clone()
    fill_value = True if uv.dtype == torch.bool else 1
    uv[content == silent_token] = fill_value
    return uv


def infer_uv_from_logits(
    uv_logits: torch.Tensor,
    *,
    content: torch.Tensor | None = None,
    silent_token: int | None = None,
) -> torch.Tensor:
    uv = uv_logits[:, :, 0] > 0
    return apply_silent_content_to_uv(uv, content=content, silent_token=silent_token)


def midi_to_log2_f0(midi_notes: torch.Tensor) -> torch.Tensor:
    return (2 ** ((midi_notes - 69) / 12) * 440).log2()


def build_midi_norm_f0_bounds(
    midi_notes: torch.Tensor,
    *,
    semitone_margin: float = 3.0,
    strict_upper: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower_bound = midi_notes - semitone_margin
    upper_bound = midi_notes + semitone_margin
    upper_norm_f0 = f0_minmax_norm(
        midi_to_log2_f0(upper_bound),
        strict_upper=strict_upper,
    ).clamp(-1.0, 1.0)
    lower_norm_f0 = f0_minmax_norm(
        midi_to_log2_f0(lower_bound),
        strict_upper=strict_upper,
    ).clamp(-1.0, 1.0)
    return lower_norm_f0, upper_norm_f0


def pack_flow_f0_target(norm_f0: torch.Tensor) -> torch.Tensor:
    if norm_f0.ndim != 2:
        raise ValueError(f"Unexpected norm_f0 shape during training: {norm_f0.shape}")
    return norm_f0.unsqueeze(1).unsqueeze(1)

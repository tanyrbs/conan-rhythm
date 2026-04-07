from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .frame_plan import build_frame_plan_from_execution, build_frame_weight_from_plan, sample_tensor_by_frame_plan


def _as_mel_tensor(mel) -> torch.Tensor:
    if torch.is_tensor(mel):
        mel_tensor = mel.detach().float().cpu()
    else:
        mel_tensor = torch.tensor(np.asarray(mel), dtype=torch.float32)
    if mel_tensor.dim() == 2:
        mel_tensor = mel_tensor.unsqueeze(0)
    return mel_tensor


def _infer_silence_frame(mel: torch.Tensor) -> torch.Tensor:
    if mel.size(0) <= 0:
        return mel.new_zeros((mel.size(-1),))
    energy = mel.mean(dim=-1)
    # A single argmin frame is brittle: it may land on an accidental low-energy
    # voiced frame or on an outlier. Averaging a small bottom-k pool yields a
    # more stable pause prototype for both cached and online retimed targets.
    low_k = max(1, min(int(mel.size(0)), max(3, int(round(float(mel.size(0)) * 0.10)))))
    _, low_idx = torch.topk(energy, k=low_k, largest=False)
    return mel.index_select(0, low_idx).mean(dim=0)


def _as_runtime_batched_tensor(value, *, dtype=torch.float32) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.float() if dtype == torch.float32 else value.to(dtype=dtype)
    else:
        tensor = torch.tensor(np.asarray(value), dtype=dtype)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def build_online_retimed_bundle(
    *,
    mel,
    frame_plan,
    f0=None,
    uv=None,
    pause_frame_weight: float = 0.20,
    stretch_weight_min: float = 0.35,
) -> dict[str, torch.Tensor]:
    mel_tensor = _as_runtime_batched_tensor(mel, dtype=torch.float32)
    if mel_tensor.dim() != 3:
        raise ValueError(f"Expected mel with shape [B,T,C], got {tuple(mel_tensor.shape)}")
    silence_fill = torch.stack([
        _infer_silence_frame(mel_tensor[batch_idx])
        for batch_idx in range(mel_tensor.size(0))
    ], dim=0)
    mel_tgt = sample_tensor_by_frame_plan(
        mel_tensor,
        frame_plan,
        blank_fill=silence_fill,
    )
    frame_weight = build_frame_weight_from_plan(
        frame_plan,
        pause_frame_weight=float(pause_frame_weight),
        stretch_weight_min=float(stretch_weight_min),
    )
    bundle: dict[str, torch.Tensor] = {
        "mel_tgt": mel_tgt,
        "frame_weight": frame_weight,
        "mel_len": frame_plan.total_mask.sum(dim=1).long(),
    }
    if f0 is not None:
        f0_tensor = _as_runtime_batched_tensor(f0, dtype=torch.float32)
        bundle["f0_tgt"] = sample_tensor_by_frame_plan(
            f0_tensor,
            frame_plan,
            blank_fill=0.0,
        )
    if uv is not None:
        uv_tensor = _as_runtime_batched_tensor(uv, dtype=torch.float32)
        bundle["uv_tgt"] = sample_tensor_by_frame_plan(
            uv_tensor,
            frame_plan,
            blank_fill=1.0,
        )
    return bundle


def build_retimed_mel_target(
    *,
    mel,
    dur_anchor_src,
    speech_exec_tgt,
    pause_exec_tgt,
    unit_mask=None,
    pause_frame_weight: float = 0.20,
    stretch_weight_min: float = 0.35,
) -> dict[str, np.ndarray]:
    mel = _as_mel_tensor(mel)
    dur_anchor_src = torch.tensor(np.asarray(dur_anchor_src), dtype=torch.float32).unsqueeze(0)
    speech_exec_tgt = torch.tensor(np.asarray(speech_exec_tgt), dtype=torch.float32).unsqueeze(0)
    pause_exec_tgt = torch.tensor(np.asarray(pause_exec_tgt), dtype=torch.float32).unsqueeze(0)
    if unit_mask is None:
        unit_mask = dur_anchor_src.gt(0).float()
    else:
        unit_mask = torch.tensor(np.asarray(unit_mask), dtype=torch.float32).unsqueeze(0)

    visible = int(unit_mask.sum().item())
    if visible <= 0:
        empty = mel.new_zeros((0, mel.size(-1)))
        empty_weight = mel.new_zeros((0,))
        return {
            "rhythm_retimed_mel_tgt": empty.cpu().numpy().astype(np.float32),
            "rhythm_retimed_mel_len": np.asarray([0], dtype=np.int64),
            "rhythm_retimed_frame_weight": empty_weight.cpu().numpy().astype(np.float32),
        }
    frame_plan = build_frame_plan_from_execution(
        dur_anchor_src=dur_anchor_src,
        speech_exec=speech_exec_tgt,
        pause_exec=pause_exec_tgt,
        unit_mask=unit_mask,
    )
    bundle = build_online_retimed_bundle(
        mel=mel,
        frame_plan=frame_plan,
        pause_frame_weight=float(pause_frame_weight),
        stretch_weight_min=float(stretch_weight_min),
    )
    retimed_len = int(bundle["mel_len"][0].item())
    retimed = bundle["mel_tgt"][0, :retimed_len]
    frame_weight = bundle["frame_weight"][0, :retimed_len]
    return {
        "rhythm_retimed_mel_tgt": retimed.cpu().numpy().astype(np.float32),
        "rhythm_retimed_mel_len": np.asarray([int(retimed.size(0))], dtype=np.int64),
        "rhythm_retimed_frame_weight": frame_weight.cpu().numpy().astype(np.float32),
    }


__all__ = [
    "build_online_retimed_bundle",
    "build_retimed_mel_target",
]

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.Conan.rhythm.unit_frontend import RhythmUnitFrontend

from .contracts import SourceUnitBatch


class NominalDurationBaseline(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int = 128,
        kernel_size: int = 5,
        min_frames: float = 1.0,
        max_frames: float = 12.0,
    ) -> None:
        super().__init__()
        self.min_frames = float(max(0.25, min_frames))
        self.max_frames = float(max(self.min_frames + 1.0e-3, max_frames))
        self.unit_embedding = nn.Embedding(vocab_size, hidden_size)
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(
        self,
        content_units: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        if content_units.size(1) <= 0:
            return unit_mask.new_zeros(unit_mask.shape, dtype=torch.float32)
        unit_mask = unit_mask.float()
        embed = self.unit_embedding(content_units.long())
        hidden = F.silu(self.in_proj(embed))
        conv_input = F.pad(hidden.transpose(1, 2), (self.conv.kernel_size[0] - 1, 0))
        conv_hidden = self.conv(conv_input).transpose(1, 2)
        hidden = F.silu(hidden + conv_hidden)
        scale = torch.sigmoid(self.out_proj(hidden).squeeze(-1))
        anchor = self.min_frames + (self.max_frames - self.min_frames) * scale
        return anchor * unit_mask


class DurationUnitFrontend(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        silent_token: int | None = None,
        separator_aware: bool = True,
        tail_open_units: int = 1,
        anchor_hidden_size: int = 128,
        anchor_min_frames: float = 1.0,
        anchor_max_frames: float = 12.0,
    ) -> None:
        super().__init__()
        self.base_frontend = RhythmUnitFrontend(
            silent_token=silent_token,
            separator_aware=separator_aware,
            tail_open_units=tail_open_units,
        )
        self.baseline = NominalDurationBaseline(
            vocab_size=vocab_size,
            hidden_size=anchor_hidden_size,
            min_frames=anchor_min_frames,
            max_frames=anchor_max_frames,
        )
        # Compatibility alias; v3 mainline should use `.baseline`.
        self.anchor_net = self.baseline

    @staticmethod
    def _validate_precomputed_shapes(**named_tensors) -> None:
        expected_shape = None
        for name, value in named_tensors.items():
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor when provided, got {type(value)!r}")
            if value.dim() != 2:
                raise ValueError(f"{name} must be rank-2 [B, U], got shape={tuple(value.shape)}")
            if expected_shape is None:
                expected_shape = tuple(value.shape)
                continue
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"Precomputed source cache shape mismatch for {name}: expected {expected_shape}, got {tuple(value.shape)}"
                )

    def _resolve_anchor_base(
        self,
        *,
        base_batch,
        unit_anchor_base: torch.Tensor | None,
    ) -> torch.Tensor:
        if unit_anchor_base is not None:
            return unit_anchor_base.float() * base_batch.unit_mask.float()
        return self.baseline(
            content_units=base_batch.content_units,
            unit_mask=base_batch.unit_mask,
        )

    def _convert_batch(
        self,
        base_batch,
        *,
        unit_anchor_base: torch.Tensor | None = None,
    ) -> SourceUnitBatch:
        unit_mask = base_batch.unit_mask.float()
        sep_mask = base_batch.sep_hint.float() * unit_mask
        resolved_anchor_base = self._resolve_anchor_base(base_batch=base_batch, unit_anchor_base=unit_anchor_base)
        return SourceUnitBatch(
            content_units=base_batch.content_units,
            source_duration_obs=base_batch.dur_anchor_src.float() * unit_mask,
            unit_anchor_base=resolved_anchor_base,
            unit_mask=unit_mask,
            sealed_mask=base_batch.sealed_mask.float() * unit_mask,
            sep_mask=sep_mask,
        )

    def from_content_tensor(
        self,
        content: torch.Tensor,
        *,
        content_lengths: torch.Tensor | None = None,
        mark_last_open: bool = True,
    ) -> SourceUnitBatch:
        base_batch = self.base_frontend.from_content_tensor(
            content,
            content_lengths=content_lengths,
            mark_last_open=mark_last_open,
        )
        return self._convert_batch(base_batch)

    def from_precomputed(
        self,
        *,
        content_units: torch.Tensor,
        source_duration_obs: torch.Tensor,
        unit_mask: torch.Tensor | None = None,
        sealed_mask: torch.Tensor | None = None,
        sep_mask: torch.Tensor | None = None,
        unit_anchor_base: torch.Tensor | None = None,
    ) -> SourceUnitBatch:
        self._validate_precomputed_shapes(
            content_units=content_units,
            source_duration_obs=source_duration_obs,
            unit_mask=unit_mask,
            sealed_mask=sealed_mask,
            sep_mask=sep_mask,
            unit_anchor_base=unit_anchor_base,
        )
        base_batch = self.base_frontend.from_precomputed(
            content_units=content_units,
            dur_anchor_src=source_duration_obs,
            unit_mask=unit_mask,
            sealed_mask=sealed_mask,
            sep_hint=sep_mask,
        )
        return self._convert_batch(base_batch, unit_anchor_base=unit_anchor_base)

    def init_stream_state(self, batch_size: int, *, device: torch.device | None = None):
        return self.base_frontend.init_stream_state(batch_size=batch_size, device=device)

    def step_content_tensor(
        self,
        content: torch.Tensor,
        state,
        *,
        content_lengths: torch.Tensor | None = None,
        mark_last_open: bool = True,
    ):
        base_batch, next_state = self.base_frontend.step_content_tensor(
            content,
            state,
            content_lengths=content_lengths,
            mark_last_open=mark_last_open,
        )
        return self._convert_batch(base_batch), next_state


AnchorNet = NominalDurationBaseline

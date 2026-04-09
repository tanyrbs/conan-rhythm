from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.Conan.rhythm.source_boundary import build_source_boundary_cue
from modules.Conan.rhythm.unit_frontend import RhythmUnitFrontend

from .contracts import SourceUnitBatch


class AnchorNet(nn.Module):
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
        self.in_proj = nn.Linear(hidden_size + 2, hidden_size)
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(
        self,
        content_units: torch.Tensor,
        unit_mask: torch.Tensor,
        edge_cue: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if content_units.size(1) <= 0:
            return unit_mask.new_zeros(unit_mask.shape, dtype=torch.float32)
        content_units = content_units.long()
        unit_mask = unit_mask.float()
        if edge_cue is None:
            edge_cue = unit_mask.new_zeros(unit_mask.shape)
        edge_cue = edge_cue.float() * unit_mask
        embed = self.unit_embedding(content_units)
        boundary_feat = edge_cue.unsqueeze(-1)
        mask_feat = unit_mask.unsqueeze(-1)
        hidden = self.in_proj(torch.cat([embed, boundary_feat, mask_feat], dim=-1))
        hidden = F.silu(hidden)
        conv_inp = hidden.transpose(1, 2)
        conv_inp = F.pad(conv_inp, (self.conv.kernel_size[0] - 1, 0))
        conv = self.conv(conv_inp).transpose(1, 2)
        hidden = F.silu(hidden + conv)
        raw = self.out_proj(hidden).squeeze(-1)
        scale = torch.sigmoid(raw)
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
        self.anchor_net = AnchorNet(
            vocab_size=vocab_size,
            hidden_size=anchor_hidden_size,
            min_frames=anchor_min_frames,
            max_frames=anchor_max_frames,
        )

    def _convert_batch(
        self,
        base_batch,
        *,
        unit_anchor_base: torch.Tensor | None = None,
        edge_cue: torch.Tensor | None = None,
    ) -> SourceUnitBatch:
        if edge_cue is None:
            edge_cue = build_source_boundary_cue(
                dur_anchor_src=base_batch.dur_anchor_src.float(),
                unit_mask=base_batch.unit_mask.float(),
                sep_hint=base_batch.sep_hint,
                open_run_mask=base_batch.open_run_mask,
                sealed_mask=base_batch.sealed_mask,
                boundary_confidence=base_batch.boundary_confidence,
            )
        if unit_anchor_base is None:
            unit_anchor_base = self.anchor_net(
                content_units=base_batch.content_units,
                unit_mask=base_batch.unit_mask,
                edge_cue=edge_cue,
            )
        return SourceUnitBatch(
            content_units=base_batch.content_units,
            source_runlen_src=base_batch.dur_anchor_src.float() * base_batch.unit_mask.float(),
            unit_anchor_base=unit_anchor_base.float() * base_batch.unit_mask.float(),
            unit_mask=base_batch.unit_mask.float(),
            edge_cue=edge_cue.float() * base_batch.unit_mask.float(),
            open_run_mask=base_batch.open_run_mask.long(),
            sealed_mask=base_batch.sealed_mask.float(),
            sep_hint=base_batch.sep_hint.long(),
            boundary_confidence=base_batch.boundary_confidence.float(),
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
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor | None = None,
        open_run_mask: torch.Tensor | None = None,
        sealed_mask: torch.Tensor | None = None,
        sep_hint: torch.Tensor | None = None,
        boundary_confidence: torch.Tensor | None = None,
        unit_anchor_base: torch.Tensor | None = None,
        edge_cue: torch.Tensor | None = None,
    ) -> SourceUnitBatch:
        base_batch = self.base_frontend.from_precomputed(
            content_units=content_units,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            open_run_mask=open_run_mask,
            sealed_mask=sealed_mask,
            sep_hint=sep_hint,
            boundary_confidence=boundary_confidence,
        )
        return self._convert_batch(base_batch, unit_anchor_base=unit_anchor_base, edge_cue=edge_cue)

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

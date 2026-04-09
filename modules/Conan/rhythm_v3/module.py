from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import (
    DurationExecution,
    DurationRuntimeState,
    ReferenceDurationMemory,
    SourceUnitBatch,
    ensure_reference_duration_memory_batch,
)
from .projector import StreamingDurationProjector
from .reference_memory import PromptDurationMemoryBuilder


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


def grad_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    return GradientReverse.apply(x, float(scale))


class LocalRoleEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        role_dim: int,
        window_left: int = 4,
        window_right: int = 0,
    ) -> None:
        super().__init__()
        self.window_left = max(0, int(window_left))
        self.window_right = max(0, int(window_right))
        kernel_size = self.window_left + self.window_right + 1
        self.unit_embedding = nn.Embedding(vocab_size, hidden_size)
        self.in_proj = nn.Linear(hidden_size + 2, hidden_size)
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, role_dim)

    def forward(
        self,
        *,
        content_units: torch.Tensor,
        unit_anchor_base: torch.Tensor,
        edge_cue: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        if content_units.size(1) <= 0:
            return self.out_proj.weight.new_zeros(
                (content_units.size(0), 0, self.out_proj.out_features)
            )
        content_embed = self.unit_embedding(content_units.long())
        features = torch.cat(
            [
                content_embed,
                torch.log1p(unit_anchor_base.float().clamp_min(0.0)).unsqueeze(-1),
                edge_cue.float().unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden = F.silu(self.in_proj(features))
        conv_inp = hidden.transpose(1, 2)
        conv_inp = F.pad(conv_inp, (self.window_left, self.window_right))
        conv = self.conv(conv_inp).transpose(1, 2)
        hidden = F.silu(hidden + conv)
        return self.out_proj(hidden) * unit_mask.unsqueeze(-1)


class DurationHead(nn.Module):
    def __init__(
        self,
        *,
        role_dim: int,
        hidden_size: int,
        max_logstretch: float = 1.0,
        anti_pos_bins: int = 8,
        anti_pos_grl_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_logstretch = float(max(0.05, max_logstretch))
        self.anti_pos_grl_scale = float(max(0.0, anti_pos_grl_scale))
        self.query_proj = nn.Linear(role_dim, role_dim)
        self.fusion = nn.Sequential(
            nn.Linear(role_dim + 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.anti_pos_probe = nn.Linear(role_dim, anti_pos_bins)

    def forward(
        self,
        *,
        role_query: torch.Tensor,
        ref_memory: ReferenceDurationMemory,
        drift_correction: torch.Tensor,
        unit_anchor_base: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if role_query.size(1) <= 0:
            batch_size = role_query.size(0)
            empty_exec = unit_anchor_base.float()[:, :0]
            empty_attn = ref_memory.role_value.new_zeros((batch_size, 0, ref_memory.role_value.size(1)))
            empty_logits = role_query.new_zeros(
                (batch_size, 0, self.anti_pos_probe.out_features)
            )
            return empty_exec, empty_exec, empty_attn, empty_logits
        q = F.normalize(self.query_proj(role_query), dim=-1)
        role_keys = F.normalize(ref_memory.role_keys, dim=-1)
        scores = torch.einsum("bud,md->bum", q, role_keys)
        scores = scores + torch.log(ref_memory.role_coverage.clamp_min(1.0e-6)).unsqueeze(1)
        role_attention = F.softmax(scores, dim=-1)
        retrieved = (role_attention * ref_memory.role_value.unsqueeze(1)).sum(dim=-1, keepdim=True)
        global_rate = ref_memory.global_rate.unsqueeze(1).expand(-1, role_query.size(1), -1)
        fusion = torch.cat([role_query, retrieved, global_rate], dim=-1)
        base_logstretch = self.fusion(fusion).squeeze(-1)
        if drift_correction.dim() == 2:
            drift = drift_correction.expand(-1, role_query.size(1))
        else:
            drift = drift_correction
        unit_logstretch = (base_logstretch + drift).clamp(-self.max_logstretch, self.max_logstretch)
        unit_logstretch = unit_logstretch * unit_mask.float()
        unit_duration_exec = unit_anchor_base.float() * torch.exp(unit_logstretch) * unit_mask.float()
        anti_input = grad_reverse(role_query, self.anti_pos_grl_scale)
        anti_pos_logits = self.anti_pos_probe(anti_input) * unit_mask.unsqueeze(-1)
        return unit_logstretch, unit_duration_exec, role_attention, anti_pos_logits


class StreamingDurationModule(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int = 256,
        role_dim: int = 64,
        codebook_size: int = 12,
        role_window_left: int = 4,
        role_window_right: int = 0,
        trace_bins: int = 24,
        coverage_floor: float = 0.05,
        prefix_drift_gain: float = 0.25,
        prefix_drift_clip: float = 0.05,
        max_logstretch: float = 1.0,
        anti_pos_bins: int = 8,
        anti_pos_grl_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.reference_memory_builder = PromptDurationMemoryBuilder(
            trace_bins=trace_bins,
            role_dim=role_dim,
            codebook_size=codebook_size,
            coverage_floor=coverage_floor,
        )
        self.local_role_encoder = LocalRoleEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            role_dim=role_dim,
            window_left=role_window_left,
            window_right=role_window_right,
        )
        self.duration_head = DurationHead(
            role_dim=role_dim,
            hidden_size=hidden_size,
            max_logstretch=max_logstretch,
            anti_pos_bins=anti_pos_bins,
            anti_pos_grl_scale=anti_pos_grl_scale,
        )
        self.projector = StreamingDurationProjector(
            drift_gain=prefix_drift_gain,
            drift_clip=prefix_drift_clip,
        )

    def init_state(self, batch_size: int, device: torch.device) -> DurationRuntimeState:
        return self.projector.init_state(batch_size=batch_size, device=device)

    def encode_reference(
        self,
        ref_mel: torch.Tensor,
        *,
        ref_lengths: torch.Tensor | None = None,
    ) -> ReferenceDurationMemory:
        return self.reference_memory_builder(ref_mel=ref_mel, ref_lengths=ref_lengths)

    def build_reference_conditioning(
        self,
        *,
        ref_conditioning=None,
        ref_rhythm_stats: torch.Tensor | None = None,
        ref_rhythm_trace: torch.Tensor | None = None,
        ref_mel: torch.Tensor | None = None,
        ref_lengths: torch.Tensor | None = None,
    ) -> ReferenceDurationMemory:
        if ref_conditioning is not None:
            return self.reference_memory_builder(ref_conditioning=ref_conditioning)
        if ref_rhythm_stats is not None and ref_rhythm_trace is not None:
            return self.reference_memory_builder(
                ref_conditioning={
                    "ref_rhythm_stats": ref_rhythm_stats,
                    "ref_rhythm_trace": ref_rhythm_trace,
                }
            )
        if ref_mel is None:
            raise ValueError("Either reference conditioning or reference mel must be provided.")
        return self.reference_memory_builder(ref_mel=ref_mel, ref_lengths=ref_lengths)

    def _freeze_committed_prefix(
        self,
        *,
        unit_duration_exec: torch.Tensor,
        unit_logstretch: torch.Tensor,
        unit_anchor_base: torch.Tensor,
        state: DurationRuntimeState | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None or state.cached_duration_exec is None:
            return unit_duration_exec, unit_logstretch
        frozen = state.cached_duration_exec.to(device=unit_duration_exec.device, dtype=unit_duration_exec.dtype)
        max_units = min(frozen.size(1), unit_duration_exec.size(1))
        if max_units <= 0:
            return unit_duration_exec, unit_logstretch
        for batch_idx in range(unit_duration_exec.size(0)):
            frontier = int(min(int(state.committed_units[batch_idx].item()), max_units))
            if frontier <= 0:
                continue
            unit_duration_exec[batch_idx, :frontier] = frozen[batch_idx, :frontier]
            denom = unit_anchor_base[batch_idx, :frontier].float().clamp_min(1.0e-6)
            unit_logstretch[batch_idx, :frontier] = torch.log(
                unit_duration_exec[batch_idx, :frontier].float().clamp_min(1.0e-6) / denom
            )
        return unit_duration_exec, unit_logstretch

    def forward(
        self,
        *,
        source_batch: SourceUnitBatch,
        ref_memory: ReferenceDurationMemory,
        state: DurationRuntimeState | None = None,
    ) -> DurationExecution:
        batch_size = int(source_batch.content_units.size(0))
        ref_memory = ensure_reference_duration_memory_batch(ref_memory, batch_size=batch_size)
        if state is None:
            state = self.init_state(
                batch_size=batch_size,
                device=source_batch.content_units.device,
            )
        role_query = self.local_role_encoder(
            content_units=source_batch.content_units,
            unit_anchor_base=source_batch.unit_anchor_base,
            edge_cue=source_batch.edge_cue,
            unit_mask=source_batch.unit_mask,
        )
        drift = self.projector.compute_drift_correction(state).to(
            device=role_query.device,
            dtype=role_query.dtype,
        )
        unit_logstretch, unit_duration_exec, role_attention, anti_pos_logits = self.duration_head(
            role_query=role_query,
            ref_memory=ref_memory,
            drift_correction=drift,
            unit_anchor_base=source_batch.unit_anchor_base,
            unit_mask=source_batch.unit_mask,
        )
        unit_duration_exec, unit_logstretch = self._freeze_committed_prefix(
            unit_duration_exec=unit_duration_exec,
            unit_logstretch=unit_logstretch,
            unit_anchor_base=source_batch.unit_anchor_base,
            state=state,
        )
        return self.projector.finalize_execution(
            unit_logstretch=unit_logstretch,
            unit_duration_exec=unit_duration_exec,
            role_attention=role_attention,
            anti_pos_logits=anti_pos_logits,
            prompt_reconstruction=ref_memory.prompt_reconstruction,
            prompt_rel_stretch=ref_memory.prompt_rel_stretch,
            prompt_mask=ref_memory.prompt_mask,
            source_runlen_src=source_batch.source_runlen_src,
            unit_mask=source_batch.unit_mask,
            sealed_mask=source_batch.sealed_mask,
            state=state,
        )

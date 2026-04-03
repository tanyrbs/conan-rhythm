from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .controller import ResidualTemporalBlock, masked_mean, masked_softmax
from .contracts import RhythmPlannerOutputs
from .source_boundary import _masked_standardize, build_deterministic_boundary_score


@dataclass
class OfflineTeacherConfig:
    """Extra capacity for the learned offline planner teacher.

    The teacher stays on the same public contract as the student
    (planner surface -> shared projector), but it uses a richer
    non-causal temporal stack and stronger global conditioning.
    """

    num_blocks: int = 6
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8, 2, 1)
    phrase_kernel_sizes: tuple[int, ...] = (3, 7)
    global_gate_scale: float = 0.12
    pause_trace_weight: float = 0.30
    boundary_trace_weight: float = 0.30
    confidence_agreement_weight: float = 0.25
    confidence_floor: float = 0.05
    confidence_ceiling: float = 1.0
    max_total_logratio: float = 0.8
    max_unit_logratio: float = 0.6
    pause_share_max: float = 0.45
    boundary_feature_scale: float = 0.35
    boundary_source_cue_weight: float = 0.35
    pause_boundary_latent_weight: float = 0.35
    pause_source_boundary_weight: float = 0.20
    min_speech_frames: float = 1.0

    def resolve_dilations(self) -> tuple[int, ...]:
        if len(self.dilations) >= self.num_blocks:
            return tuple(int(x) for x in self.dilations[: self.num_blocks])
        if len(self.dilations) == 0:
            return tuple(1 for _ in range(self.num_blocks))
        values = list(int(x) for x in self.dilations)
        while len(values) < self.num_blocks:
            values.extend(values)
        return tuple(values[: self.num_blocks])


class _GlobalFiLM(nn.Module):
    def __init__(self, hidden_size: int, gate_scale: float) -> None:
        super().__init__()
        self.gate_scale = float(gate_scale)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
        )

    def forward(self, x: torch.Tensor, global_hidden: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(global_hidden).chunk(2, dim=-1)
        gamma = self.gate_scale * torch.tanh(gamma).unsqueeze(1)
        beta = self.gate_scale * beta.unsqueeze(1)
        return x * (1.0 + gamma) + beta


class _TeacherTemporalBlock(nn.Module):
    def __init__(self, hidden_size: int, *, kernel_size: int, dilation: int, gate_scale: float) -> None:
        super().__init__()
        self.block = ResidualTemporalBlock(
            hidden_size,
            kernel_size=kernel_size,
            dilation=dilation,
            causal=False,
        )
        self.film = _GlobalFiLM(hidden_size, gate_scale=gate_scale)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, global_hidden: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        y = self.film(y, global_hidden)
        return self.norm(y)


def _masked_avg_pool1d(x: torch.Tensor, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1 or x.size(1) <= 1:
        return x
    kernel_size = min(kernel_size, int(x.size(1)))
    pad = kernel_size // 2
    x_t = x.transpose(1, 2)
    mask_t = mask.float().unsqueeze(1)
    pooled_x = F.avg_pool1d(x_t * mask_t, kernel_size=kernel_size, stride=1, padding=pad)
    pooled_m = F.avg_pool1d(mask_t, kernel_size=kernel_size, stride=1, padding=pad)
    if pooled_x.size(-1) > x.size(1):
        pooled_x = pooled_x[..., : x.size(1)]
    if pooled_m.size(-1) > x.size(1):
        pooled_m = pooled_m[..., : x.size(1)]
    pooled = pooled_x / pooled_m.clamp_min(1e-6)
    return pooled.transpose(1, 2)

def _masked_cosine_similarity(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    x_masked = x * mask
    y_masked = y * mask
    dot = (x_masked * y_masked).sum(dim=1)
    x_norm = x_masked.square().sum(dim=1).clamp_min(1e-6).sqrt()
    y_norm = y_masked.square().sum(dim=1).clamp_min(1e-6).sqrt()
    return dot / (x_norm * y_norm).clamp_min(1e-6)


class OfflineRhythmTeacherPlanner(nn.Module):
    """Stronger non-causal offline planner used as the learned teacher.

    Design goals:
    - keep the same public teacher surface as before
    - be substantially stronger than the streaming scheduler
    - distill only an executable surface through the shared projector
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        stats_dim: int,
        trace_dim: int,
        config: OfflineTeacherConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or OfflineTeacherConfig()
        self.max_total_logratio = float(self.config.max_total_logratio)
        self.max_unit_logratio = float(self.config.max_unit_logratio)
        self.pause_share_max = float(max(0.0, self.config.pause_share_max))
        self.boundary_feature_scale = float(self.config.boundary_feature_scale)
        self.boundary_source_cue_weight = float(self.config.boundary_source_cue_weight)
        self.pause_boundary_latent_weight = float(self.config.pause_boundary_latent_weight)
        self.pause_source_boundary_weight = float(self.config.pause_source_boundary_weight)
        self.pause_trace_weight = float(self.config.pause_trace_weight)
        self.boundary_trace_weight = float(self.config.boundary_trace_weight)
        self.min_speech_frames = float(self.config.min_speech_frames)
        self.confidence_agreement_weight = float(self.config.confidence_agreement_weight)
        self.confidence_floor = float(self.config.confidence_floor)
        self.confidence_ceiling = float(max(self.confidence_floor, self.config.confidence_ceiling))

        self.unit_in = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.anchor_proj = nn.Linear(1, hidden_size)
        self.trace_proj = nn.Linear(trace_dim, hidden_size)
        self.trace_delta_proj = nn.Linear(trace_dim, hidden_size)
        self.boundary_proj = nn.Linear(1, hidden_size)
        self.progress_proj = nn.Linear(3, hidden_size)
        self.slow_proj = nn.Linear(trace_dim, hidden_size)
        self.stats_proj = nn.Linear(stats_dim, hidden_size)
        self.local_fuse = nn.Sequential(
            nn.Linear(hidden_size * (1 + len(self.config.phrase_kernel_sizes)), hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        self.blocks = nn.ModuleList(
            [
                _TeacherTemporalBlock(
                    hidden_size,
                    kernel_size=int(self.config.kernel_size),
                    dilation=int(dilation),
                    gate_scale=float(self.config.global_gate_scale),
                )
                for dilation in self.config.resolve_dilations()
            ]
        )
        global_input_dim = hidden_size + trace_dim + trace_dim + stats_dim + 7
        self.global_summary = nn.Sequential(
            nn.Linear(global_input_dim, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
        )
        self.global_refine = nn.Sequential(
            nn.Linear(hidden_size + trace_dim + trace_dim + stats_dim + 7, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.total_budget_head = nn.Linear(hidden_size, 1)
        self.pause_share_head = nn.Linear(hidden_size, 1)
        self.anchor_gate_head = nn.Linear(hidden_size, 1)
        self.logratio_head = nn.Linear(hidden_size, 1)
        self.pause_head = nn.Linear(hidden_size, 1)
        # Compatibility-only parameter surface. Kept for older checkpoints, unused in forward.
        self.boundary_head = nn.Linear(hidden_size, 1)
        for param in self.boundary_head.parameters():
            param.requires_grad = False
        self.confidence_trunk = nn.Sequential(
            nn.Linear(hidden_size + stats_dim + trace_dim + 5, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.confidence_heads = nn.ModuleDict(
            {
                "overall": nn.Linear(hidden_size, 1),
                "exec": nn.Linear(hidden_size, 1),
                "budget": nn.Linear(hidden_size, 1),
                "prefix": nn.Linear(hidden_size, 1),
                "allocation": nn.Linear(hidden_size, 1),
            }
        )

    def _build_global_hidden(
        self,
        *,
        hidden: torch.Tensor,
        unit_mask: torch.Tensor,
        trace_context: torch.Tensor,
        slow_rhythm_summary: torch.Tensor,
        ref_rhythm_stats: torch.Tensor,
        dur_anchor_src: torch.Tensor,
        source_boundary_cue: torch.Tensor,
        agreement: torch.Tensor,
    ) -> torch.Tensor:
        pooled_hidden = masked_mean(hidden, unit_mask, dim=1)
        pooled_trace = masked_mean(trace_context, unit_mask, dim=1)
        visible = unit_mask.sum(dim=1).clamp_min(1.0)
        src_total = (dur_anchor_src.float() * unit_mask).sum(dim=1)
        pooled_anchor = src_total / visible
        pooled_boundary = (source_boundary_cue.float() * unit_mask).sum(dim=1) / visible
        global_input = torch.cat(
            [
                pooled_hidden,
                pooled_trace,
                slow_rhythm_summary,
                ref_rhythm_stats,
                pooled_anchor.unsqueeze(-1),
                src_total.log1p().unsqueeze(-1),
                visible.log1p().unsqueeze(-1),
                pooled_boundary.unsqueeze(-1),
                agreement.unsqueeze(-1),
                ref_rhythm_stats[:, 0:1],
                ref_rhythm_stats[:, 4:5],
            ],
            dim=-1,
        )
        return self.global_summary(global_input)

    @staticmethod
    def _build_trace_delta(trace_context: torch.Tensor) -> torch.Tensor:
        trace_delta = trace_context.new_zeros(trace_context.shape)
        if trace_context.size(1) > 1:
            trace_delta[:, 1:] = trace_context[:, 1:] - trace_context[:, :-1]
        return trace_delta

    def forward(
        self,
        *,
        unit_states: torch.Tensor,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor,
        ref_conditioning: dict[str, torch.Tensor],
        trace_context: torch.Tensor,
        source_boundary_cue: torch.Tensor | None = None,
    ) -> tuple[RhythmPlannerOutputs, dict[str, torch.Tensor]]:
        unit_mask = unit_mask.float()
        if source_boundary_cue is None:
            source_boundary_cue = unit_mask.new_zeros(unit_mask.shape)
        slow_rhythm_summary = ref_conditioning.get("slow_rhythm_summary")
        if slow_rhythm_summary is None:
            slow_rhythm_summary = masked_mean(trace_context, unit_mask, dim=1)
        src_log = torch.log1p(dur_anchor_src.float().clamp_min(0.0)).unsqueeze(-1)
        trace_delta = self._build_trace_delta(trace_context)
        boundary_feat = (source_boundary_cue.float() * self.boundary_feature_scale).unsqueeze(-1)

        anchor_mass = dur_anchor_src.float() * unit_mask
        total_anchor = anchor_mass.sum(dim=1, keepdim=True).clamp_min(1.0)
        progress = torch.cumsum(anchor_mass, dim=1) / total_anchor
        remaining = (1.0 - progress).clamp(0.0, 1.0)
        centered = (progress - 0.5) * 2.0
        position_feat = torch.stack([progress, remaining, centered], dim=-1)

        x = (
            self.unit_in(unit_states)
            + self.anchor_proj(src_log)
            + self.trace_proj(trace_context)
            + self.trace_delta_proj(trace_delta)
            + self.boundary_proj(boundary_feat)
            + self.progress_proj(position_feat)
            + self.slow_proj(slow_rhythm_summary).unsqueeze(1)
            + self.stats_proj(ref_conditioning["ref_rhythm_stats"]).unsqueeze(1)
        )
        local_features = [x]
        for kernel_size in self.config.phrase_kernel_sizes:
            local_features.append(_masked_avg_pool1d(x, unit_mask, kernel_size=int(kernel_size)))
        x = self.local_fuse(torch.cat(local_features, dim=-1))
        x = self.in_proj(x)

        agreement = _masked_cosine_similarity(
            _masked_standardize(source_boundary_cue.float(), unit_mask),
            _masked_standardize(trace_context[:, :, 2].float(), unit_mask),
            unit_mask,
        ).clamp(-1.0, 1.0)
        global_hidden = self._build_global_hidden(
            hidden=x,
            unit_mask=unit_mask,
            trace_context=trace_context,
            slow_rhythm_summary=slow_rhythm_summary,
            ref_rhythm_stats=ref_conditioning["ref_rhythm_stats"],
            dur_anchor_src=dur_anchor_src,
            source_boundary_cue=source_boundary_cue,
            agreement=agreement,
        )
        for block in self.blocks:
            x = block(x, global_hidden)

        pooled_hidden = masked_mean(x, unit_mask, dim=1)
        pooled_trace = masked_mean(trace_context, unit_mask, dim=1)
        trace_var = (
            ((trace_context[:, :, 1] - pooled_trace[:, None, 1]) ** 2) * unit_mask
        ).sum(dim=1) / unit_mask.sum(dim=1).clamp_min(1.0)
        pooled_boundary = masked_mean(source_boundary_cue.unsqueeze(-1), unit_mask, dim=1).squeeze(-1)
        refined_input = torch.cat(
            [
                pooled_hidden,
                pooled_trace,
                slow_rhythm_summary,
                ref_conditioning["ref_rhythm_stats"],
                pooled_boundary.unsqueeze(-1),
                trace_var.unsqueeze(-1),
                agreement.unsqueeze(-1),
                ref_conditioning["ref_rhythm_stats"][:, 0:1],
                ref_conditioning["ref_rhythm_stats"][:, 4:5],
                total_anchor.log1p(),
                unit_mask.sum(dim=1, keepdim=True).log1p(),
            ],
            dim=-1,
        )
        global_hidden = self.global_refine(refined_input)

        raw_total_logratio = torch.tanh(self.total_budget_head(global_hidden)) * self.max_total_logratio
        raw_pause_share = torch.sigmoid(self.pause_share_head(global_hidden))
        anchor_gate = torch.sigmoid(self.anchor_gate_head(global_hidden))

        src_total = total_anchor
        total_budget = src_total * torch.exp(raw_total_logratio * anchor_gate)
        pause_share = self.pause_share_max * raw_pause_share
        pause_budget = total_budget * pause_share
        min_speech_budget = unit_mask.sum(dim=1, keepdim=True).clamp_min(1.0) * self.min_speech_frames
        speech_budget = (total_budget - pause_budget).clamp_min(min_speech_budget)
        pause_budget = (total_budget - speech_budget).clamp_min(0.0)

        raw_logratio = torch.tanh(self.logratio_head(x).squeeze(-1)) * self.max_unit_logratio
        mean_logratio = masked_mean(raw_logratio.unsqueeze(-1), unit_mask, dim=1, keepdim=True).squeeze(-1)
        dur_logratio = (raw_logratio - mean_logratio) * unit_mask

        boundary_latent = build_deterministic_boundary_score(
            source_boundary_cue=source_boundary_cue,
            boundary_trace=trace_context[:, :, 2] if trace_context.size(-1) > 2 else None,
            unit_mask=unit_mask,
            source_weight=self.boundary_source_cue_weight,
            trace_weight=self.boundary_trace_weight,
        )
        pause_logits = self.pause_head(x).squeeze(-1)
        pause_logits = pause_logits + self.pause_boundary_latent_weight * boundary_latent
        pause_logits = pause_logits + self.pause_source_boundary_weight * source_boundary_cue.float()
        pause_logits = pause_logits + self.pause_trace_weight * trace_context[:, :, 0].float()
        pause_weight = masked_softmax(pause_logits, unit_mask, dim=1) * unit_mask

        planner = RhythmPlannerOutputs(
            speech_budget_win=speech_budget,
            pause_budget_win=pause_budget,
            dur_logratio_unit=dur_logratio,
            pause_weight_unit=pause_weight,
            total_budget_win=total_budget,
            pause_share_win=pause_share,
            anchor_gate=anchor_gate,
            boundary_latent=boundary_latent,
            trace_context=trace_context,
            source_boundary_cue=source_boundary_cue,
        )

        confidence_input = torch.cat(
            [
                global_hidden,
                ref_conditioning["ref_rhythm_stats"],
                pooled_trace,
                agreement.unsqueeze(-1),
                trace_var.unsqueeze(-1),
                pooled_boundary.unsqueeze(-1),
                pause_share,
                raw_total_logratio,
            ],
            dim=-1,
        )
        confidence_hidden = self.confidence_trunk(confidence_input)
        agreement_gain = 1.0 + self.confidence_agreement_weight * agreement.unsqueeze(-1)
        confidence = {}
        for name, head in self.confidence_heads.items():
            value = torch.sigmoid(head(confidence_hidden)) * agreement_gain
            value = value.clamp(self.confidence_floor, self.confidence_ceiling)
            confidence[name] = value
        return planner, confidence

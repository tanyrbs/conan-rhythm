from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import (
    PromptConditioningEvidence,
    ReferenceDurationMemory,
    StructuredDurationOperatorMemory,
    StructuredRoleDurationMemory,
    validate_reference_duration_memory,
)


def _masked_median(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch_size = values.size(0)
    median = values.new_zeros((batch_size, 1))
    for batch_idx in range(batch_size):
        valid = mask[batch_idx] > 0.5
        if bool(valid.any().item()):
            median[batch_idx, 0] = values[batch_idx][valid].median()
    return median


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (values * mask_f.unsqueeze(-1)).sum(dim=1) / denom


def _masked_std(values: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    diff2 = ((values - mean.unsqueeze(1)) ** 2) * mask_f.unsqueeze(-1)
    return torch.sqrt(diff2.sum(dim=1) / denom.clamp_min(1.0e-6) + 1.0e-6)


class SharedRoleCodebook(nn.Module):
    """Compatibility stub kept so old imports/configs do not break.

    The latest mainline no longer relies on slot codebooks for duration writing,
    but some construction sites still instantiate this object. The tensor remains
    trainable in case downstream experiments want to reuse it.
    """

    def __init__(self, *, num_slots: int, dim: int) -> None:
        super().__init__()
        self.num_slots = int(max(1, num_slots))
        self.dim = int(max(4, dim))
        self.role_key = nn.Parameter(torch.randn(self.num_slots, self.dim) * 0.02)


class CausalStretchQueryEncoder(nn.Module):
    """Small causal content encoder for summary-conditioned residual stretch.

    This intentionally avoids boundary/event features. The only dynamic source-side
    scalar is the centered log-anchor, which keeps the model focused on "source as
    anchor, reference as residual style".
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.dim = int(max(8, dim))
        self.unit_emb = nn.Embedding(int(vocab_size), self.dim)
        self.proj_aux = nn.Linear(1, self.dim)
        self.conv1 = nn.Conv1d(self.dim, self.dim, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(self.dim, self.dim, kernel_size=kernel_size)
        self.norm = nn.LayerNorm(self.dim)

    @staticmethod
    def _causal_conv(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        y = x.transpose(1, 2)
        pad = int(conv.kernel_size[0]) - 1
        y = F.pad(y, (pad, 0))
        y = conv(y)
        return y.transpose(1, 2)

    def forward(
        self,
        *,
        unit_ids: torch.Tensor,
        scalar_feat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.unit_emb(unit_ids.long()) + self.proj_aux(scalar_feat.float().unsqueeze(-1))
        hidden = F.gelu(self._causal_conv(hidden, self.conv1))
        hidden = F.gelu(self._causal_conv(hidden, self.conv2))
        return self.norm(hidden) * mask.unsqueeze(-1).float()


class CausalRoleQueryEncoder(CausalStretchQueryEncoder):
    """Backward-compatible alias for legacy imports."""


class PromptDurationMemoryEncoder(nn.Module):
    """Reference-side summary encoder.

    The implementation keeps the old class name for compatibility, but the actual
    mechanism is now the simplified source-anchored summary model discussed in the
    latest design iteration:

      1) compute speech-only global prompt rate
      2) center prompt log-duration by that rate
      3) build a compact summary vector with masked statistics pooling
      4) keep only a scalar residual probe surface for diagnostics / optional loss

    `operator_coeff` is repurposed as the compact prompt summary vector. This keeps
    the public contract stable without forcing slot memories or local bases back in.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        num_slots: int,
        operator_rank: int,
        coverage_floor: float = 0.05,
        codebook: SharedRoleCodebook | None = None,
    ) -> None:
        super().__init__()
        del num_slots, operator_rank, codebook
        self.coverage_floor = float(max(1.0e-4, coverage_floor))
        self.summary_dim = int(max(8, dim))
        self.prompt_encoder = CausalStretchQueryEncoder(vocab_size=vocab_size, dim=self.summary_dim)
        self.summary_proj = nn.Sequential(
            nn.Linear(self.summary_dim * 2, self.summary_dim),
            nn.GELU(),
            nn.Linear(self.summary_dim, self.summary_dim),
        )

    def forward(
        self,
        *,
        prompt_content_units: torch.Tensor,
        prompt_duration_obs: torch.Tensor,
        prompt_mask: torch.Tensor,
        prompt_edge_cue: torch.Tensor | None = None,
    ) -> ReferenceDurationMemory:
        del prompt_edge_cue
        mask = prompt_mask.float().clamp(0.0, 1.0)
        logdur = torch.log(prompt_duration_obs.float().clamp_min(1.0e-4)) * mask
        global_rate = _masked_median(logdur, mask)
        ref_residual = (logdur - global_rate) * mask

        hidden = self.prompt_encoder(
            unit_ids=prompt_content_units.long(),
            scalar_feat=ref_residual,
            mask=mask,
        )
        mean = _masked_mean(hidden, mask)
        std = _masked_std(hidden, mask, mean)
        summary_state = torch.tanh(self.summary_proj(torch.cat([mean, std], dim=-1)))

        support_raw = mask.sum(dim=1, keepdim=True)
        support = support_raw.clamp_min(1.0)
        summary_state = torch.where(support_raw > 0.0, summary_state, torch.zeros_like(summary_state))
        coverage = (support_raw / float(max(1, mask.size(1)))).clamp_min(self.coverage_floor)
        residual_mean = (ref_residual * mask).sum(dim=1, keepdim=True) / support
        residual_var = (((ref_residual - residual_mean) ** 2) * mask).sum(dim=1, keepdim=True) / support
        residual_var = residual_var.clamp_min(1.0e-4)

        attn = (mask / support).unsqueeze(-1)
        prompt_role_fit = residual_mean.expand_as(ref_residual) * mask

        return validate_reference_duration_memory(
            ReferenceDurationMemory(
                global_rate=global_rate,
                operator=StructuredDurationOperatorMemory(operator_coeff=summary_state),
                role=StructuredRoleDurationMemory(
                    role_value=residual_mean,
                    role_var=residual_var,
                    role_coverage=coverage,
                ),
                prompt=PromptConditioningEvidence(
                    prompt_mask=mask,
                    prompt_log_duration=logdur,
                    prompt_log_residual=ref_residual,
                    prompt_role_attn=attn,
                    prompt_role_fit=prompt_role_fit,
                    prompt_operator_coeff_norm=summary_state.norm(dim=-1, keepdim=True),
                ),
            )
        )


class StreamingDurationHead(nn.Module):
    """Single-head source-anchored residual stretch writer.

    Final prediction form:
        logstretch = (g_ref - g_src_prefix) + residual(source_window, ref_summary)

    The head keeps the old role-memory method signature so the rest of the codebase
    does not need a broad rewrite, but internally it is a compact summary model with
    no slots, no pointer, and no boundary features.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        num_slots: int,
        max_logstretch: float = 1.2,
        codebook: SharedRoleCodebook | None = None,
    ) -> None:
        super().__init__()
        del num_slots, codebook
        self.max_logstretch = float(max(0.1, max_logstretch))
        self.dim = int(max(8, dim))
        self.query_encoder = CausalStretchQueryEncoder(vocab_size=vocab_size, dim=self.dim)
        self.residual_head = nn.Sequential(
            nn.Linear((2 * self.dim) + 3, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, 1),
        )
        self.residual_scale = 0.35

    def forward(
        self,
        *,
        content_units: torch.Tensor,
        log_anchor: torch.Tensor,
        unit_mask: torch.Tensor,
        sealed_mask: torch.Tensor,
        sep_hint: torch.Tensor,
        edge_cue: torch.Tensor,
        global_rate: torch.Tensor,
        role_value: torch.Tensor,
        role_var: torch.Tensor,
        role_coverage: torch.Tensor,
        local_rate_ema: torch.Tensor,
        summary_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del edge_cue
        mask = unit_mask.float().clamp(0.0, 1.0)
        speech_mask = mask * (1.0 - sep_hint.float().clamp(0.0, 1.0))
        if local_rate_ema.dim() == 2 and local_rate_ema.size(1) == 1:
            local_rate = local_rate_ema.float().expand_as(log_anchor)
        else:
            local_rate = local_rate_ema.float()
            if tuple(local_rate.shape) != tuple(log_anchor.shape):
                raise ValueError(
                    f"StreamingDurationHead.local_rate_ema must be [B,1] or [B,U], got {tuple(local_rate.shape)}"
                )
        centered_anchor = (log_anchor.float() - local_rate) * speech_mask
        query = self.query_encoder(
            unit_ids=content_units,
            scalar_feat=centered_anchor,
            mask=mask,
        )

        summary = summary_state.float().unsqueeze(1).expand(-1, log_anchor.size(1), -1)
        global_shift = global_rate.float().expand_as(log_anchor) - local_rate
        summary_mean = role_value.float().expand_as(log_anchor)
        summary_conf = (role_coverage.float() / (1.0 + role_var.float().sqrt())).expand_as(log_anchor)

        residual_input = torch.cat(
            [
                query,
                summary,
                global_shift.unsqueeze(-1),
                summary_mean.unsqueeze(-1),
                summary_conf.unsqueeze(-1),
            ],
            dim=-1,
        )
        residual = self.residual_scale * torch.tanh(self.residual_head(residual_input).squeeze(-1))
        pred = (global_shift + residual).clamp(min=-self.max_logstretch, max=self.max_logstretch)
        pred = pred * sealed_mask.float() * speech_mask

        attn = mask.unsqueeze(-1)
        role_var_unit = role_var.float().expand_as(log_anchor).clamp_min(1.0e-4)
        return {
            "unit_logstretch": pred,
            "role_attn_unit": attn,
            "role_value_unit": summary_mean * mask,
            "role_var_unit": role_var_unit * mask,
            "role_conf_unit": summary_conf * mask,
            "role_query_unit": query,
            "local_response": residual * speech_mask,
        }

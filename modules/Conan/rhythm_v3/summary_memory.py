from __future__ import annotations

import math

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


def _build_causal_local_rate_seq(
    *,
    observed_log: torch.Tensor,
    speech_mask: torch.Tensor,
    init_rate: torch.Tensor | None,
    decay: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    observed_log: [B, U], usually log(source_observed_duration)
    speech_mask:  [B, U], 1 only for sealed speech units
    init_rate:    [B, 1] or None
    Returns:
        local_rate_seq:  [B, U], rate state BEFORE consuming current unit
        local_rate_last: [B, 1], state AFTER consuming the whole chunk
    """
    batch_size, num_units = observed_log.shape
    if init_rate is None:
        prev = observed_log.new_zeros((batch_size, 1))
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


class SharedSummaryCodebook(nn.Module):
    """Prompt-summary sidecar kept for diagnostics and legacy compatibility."""

    def __init__(self, *, num_slots: int, dim: int) -> None:
        super().__init__()
        self.num_slots = int(max(1, num_slots))
        self.dim = int(max(4, dim))
        self.summary_key = nn.Parameter(torch.randn(self.num_slots, self.dim) * 0.02)

    @property
    def role_key(self) -> torch.Tensor:
        return self.summary_key


class CausalRoleQueryEncoder(nn.Module):
    """Shared causal encoder for prompt-side residual summaries and source queries."""

    def __init__(
        self,
        *,
        vocab_size: int = 2048,
        dim: int = 64,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.dim = int(max(8, dim))
        self.unit_emb = nn.Embedding(int(vocab_size), self.dim)
        self.proj_aux = nn.Linear(4, self.dim)
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
        log_anchor: torch.Tensor,
        edge_cue: torch.Tensor,
        sep_hint: torch.Tensor,
        local_rate: torch.Tensor,
    ) -> torch.Tensor:
        aux = torch.stack(
            [log_anchor.float(), edge_cue.float(), sep_hint.float(), local_rate.float()],
            dim=-1,
        )
        hidden = self.unit_emb(unit_ids.long()) + self.proj_aux(aux)
        hidden = F.gelu(self._causal_conv(hidden, self.conv1))
        hidden = F.gelu(self._causal_conv(hidden, self.conv2))
        return self.norm(hidden)

    def encode_prompt(
        self,
        unit_ids: torch.Tensor,
        logdur: torch.Tensor,
        mask: torch.Tensor,
        edge_cue: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_cue is None:
            edge_cue = torch.zeros_like(logdur)
        sep = torch.zeros_like(logdur)
        local_rate = torch.zeros_like(logdur)
        hidden = self.forward(
            unit_ids=unit_ids,
            log_anchor=logdur,
            edge_cue=edge_cue,
            sep_hint=sep,
            local_rate=local_rate,
        )
        return hidden * mask.unsqueeze(-1).float()


class CausalStretchQueryEncoder(CausalRoleQueryEncoder):
    """Backward-compatible alias for legacy imports."""


class PromptDurationMemoryEncoder(nn.Module):
    """Reference-side summary encoder with slotwise diagnostics."""

    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        num_slots: int,
        operator_rank: int,
        coverage_floor: float = 0.05,
        codebook: SharedSummaryCodebook | None = None,
    ) -> None:
        super().__init__()
        del operator_rank
        self.coverage_floor = float(max(1.0e-4, coverage_floor))
        self.summary_dim = int(max(8, dim))
        self.prompt_encoder = CausalRoleQueryEncoder(vocab_size=vocab_size, dim=self.summary_dim)
        self.codebook = codebook if codebook is not None else SharedSummaryCodebook(num_slots=num_slots, dim=self.summary_dim)
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

        hidden = self.prompt_encoder.encode_prompt(
            prompt_content_units.long(),
            ref_residual,
            mask,
            edge_cue=None,
        )
        mean = _masked_mean(hidden, mask)
        std = _masked_std(hidden, mask, mean)
        summary_state = torch.tanh(self.summary_proj(torch.cat([mean, std], dim=-1)))

        support_raw = mask.sum(dim=1, keepdim=True)
        support = support_raw.clamp_min(1.0)
        summary_state = torch.where(support_raw > 0.0, summary_state, torch.zeros_like(summary_state))

        score = torch.einsum("btd,md->btm", hidden, self.codebook.role_key)
        score = score / math.sqrt(float(max(1, self.codebook.dim)))
        score = score.masked_fill(mask.unsqueeze(-1) <= 0.0, -1.0e4)
        attn = F.softmax(score, dim=-1) * mask.unsqueeze(-1)

        role_mass = attn.sum(dim=1)
        role_denom = role_mass.clamp_min(1.0e-6)
        role_value = torch.einsum("btm,bt->bm", attn, ref_residual) / role_denom
        diff2 = (ref_residual.unsqueeze(-1) - role_value.unsqueeze(1)) ** 2
        role_var = (attn * diff2).sum(dim=1) / role_denom
        role_var = torch.where(role_mass > 0.0, role_var, torch.full_like(role_var, 1.0e-4)).clamp_min(1.0e-4)
        role_coverage = (role_mass / support).clamp_min(self.coverage_floor)
        prompt_role_fit = torch.einsum("btm,bm->bt", attn, role_value) * mask

        return validate_reference_duration_memory(
            ReferenceDurationMemory(
                global_rate=global_rate,
                operator=StructuredDurationOperatorMemory(operator_coeff=summary_state),
                role=StructuredRoleDurationMemory(
                    role_value=role_value,
                    role_var=role_var,
                    role_coverage=role_coverage,
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
    """Source-anchored residual stretch writer with explicit global-rate shift."""

    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        num_slots: int,
        max_logstretch: float = 1.2,
        codebook: SharedSummaryCodebook | None = None,
    ) -> None:
        super().__init__()
        self.max_logstretch = float(max(0.1, max_logstretch))
        self.query_encoder = CausalRoleQueryEncoder(vocab_size=vocab_size, dim=dim)
        self.codebook = codebook if codebook is not None else SharedSummaryCodebook(num_slots=num_slots, dim=dim)
        self.residual_head = nn.Sequential(
            nn.Linear(int(max(8, dim)) + 4, int(max(8, dim))),
            nn.GELU(),
            nn.Linear(int(max(8, dim)), 1),
        )

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
    ) -> dict[str, torch.Tensor]:
        del edge_cue
        mask = unit_mask.float().clamp(0.0, 1.0)
        sealed = sealed_mask.float().clamp(0.0, 1.0)
        sep = sep_hint.float().clamp(0.0, 1.0)
        speech_mask = mask * sealed * (1.0 - sep)

        init_local_rate = None
        if isinstance(local_rate_ema, torch.Tensor):
            init_local_rate = local_rate_ema.float()
            if init_local_rate.dim() != 2 or init_local_rate.size(1) != 1:
                raise ValueError(
                    f"StreamingDurationHead.local_rate_ema must have shape [B, 1], got {tuple(init_local_rate.shape)}"
                )

        local_rate_seq, local_rate_final = _build_causal_local_rate_seq(
            observed_log=log_anchor.float(),
            speech_mask=speech_mask,
            init_rate=init_local_rate,
            decay=0.95,
        )

        query = self.query_encoder(
            unit_ids=content_units,
            log_anchor=log_anchor.float(),
            edge_cue=torch.zeros_like(log_anchor),
            sep_hint=torch.zeros_like(log_anchor),
            local_rate=local_rate_seq,
        ) * mask.unsqueeze(-1)

        score = torch.einsum("bud,md->bum", query, self.codebook.role_key)
        score = score / math.sqrt(float(max(1, self.codebook.dim)))
        score = score + torch.log(role_coverage.float().clamp_min(1.0e-4)).unsqueeze(1)
        score = score.masked_fill(mask.unsqueeze(-1) <= 0.0, -1.0e4)
        attn = F.softmax(score, dim=-1) * mask.unsqueeze(-1)

        role_value_unit = torch.einsum("bum,bm->bu", attn, role_value.float())
        role_var_unit = torch.einsum("bum,bm->bu", attn, role_var.float()).clamp_min(1.0e-4)
        role_cov_unit = torch.einsum("bum,bm->bu", attn, role_coverage.float()).clamp_min(1.0e-4)
        role_conf_unit = role_cov_unit / (role_cov_unit + role_var_unit)

        global_shift = (global_rate.float() - local_rate_seq.float()) * speech_mask
        residual_input = torch.cat(
            [
                query,
                global_shift.unsqueeze(-1),
                role_value_unit.unsqueeze(-1),
                role_conf_unit.unsqueeze(-1),
                role_var_unit.log().unsqueeze(-1),
            ],
            dim=-1,
        )
        residual = 0.35 * torch.tanh(self.residual_head(residual_input).squeeze(-1))
        residual = residual * speech_mask
        pred = (global_shift + residual).clamp(
            min=-self.max_logstretch,
            max=self.max_logstretch,
        ) * speech_mask

        return {
            "unit_logstretch": pred,
            "unit_global_shift": global_shift * mask,
            "unit_residual_logstretch": residual * mask,
            "role_attn_unit": attn,
            "role_value_unit": role_value_unit * mask,
            "role_var_unit": role_var_unit * mask,
            "role_conf_unit": role_conf_unit * mask,
            "role_query_unit": query,
            "local_response": residual * mask,
            "local_rate_seq": local_rate_seq * mask,
            "local_rate_final": local_rate_final,
        }


PromptSummaryEncoder = PromptDurationMemoryEncoder
PromptSummaryDurationHead = StreamingDurationHead
CausalSummaryQueryEncoder = CausalStretchQueryEncoder
SharedRoleCodebook = SharedSummaryCodebook

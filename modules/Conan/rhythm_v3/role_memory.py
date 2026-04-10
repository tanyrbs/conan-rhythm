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


class SharedRoleCodebook(nn.Module):
    def __init__(self, *, num_slots: int, dim: int) -> None:
        super().__init__()
        self.num_slots = int(max(1, num_slots))
        self.dim = int(max(4, dim))
        self.role_key = nn.Parameter(torch.randn(self.num_slots, self.dim) * 0.02)


class CausalRoleQueryEncoder(nn.Module):
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
            [
                log_anchor.float(),
                edge_cue.float(),
                sep_hint.float(),
                local_rate.float(),
            ],
            dim=-1,
        )
        hidden = self.unit_emb(unit_ids.long()) + self.proj_aux(aux)
        hidden = F.gelu(self._causal_conv(hidden, self.conv1))
        hidden = F.gelu(self._causal_conv(hidden, self.conv2))
        return self.norm(hidden)

    def encode_prompt(
        self,
        *,
        unit_ids: torch.Tensor,
        logdur: torch.Tensor,
        mask: torch.Tensor,
        edge_cue: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_cue is None:
            edge_cue = torch.zeros_like(logdur)
        sep_hint = torch.zeros_like(logdur)
        local_rate = torch.zeros_like(logdur)
        hidden = self.forward(
            unit_ids=unit_ids,
            log_anchor=logdur,
            edge_cue=edge_cue,
            sep_hint=sep_hint,
            local_rate=local_rate,
        )
        return hidden * mask.unsqueeze(-1).float()


class PromptDurationMemoryEncoder(nn.Module):
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
        self.coverage_floor = float(max(1.0e-4, coverage_floor))
        self.role_encoder = CausalRoleQueryEncoder(vocab_size=vocab_size, dim=dim)
        self.codebook = codebook if codebook is not None else SharedRoleCodebook(num_slots=num_slots, dim=dim)
        self.operator_rank = int(max(1, operator_rank))

    def forward(
        self,
        *,
        prompt_content_units: torch.Tensor,
        prompt_duration_obs: torch.Tensor,
        prompt_mask: torch.Tensor,
        prompt_edge_cue: torch.Tensor | None = None,
    ) -> ReferenceDurationMemory:
        mask = prompt_mask.float().clamp(0.0, 1.0)
        logdur = torch.log(prompt_duration_obs.float().clamp_min(1.0e-4)) * mask
        hidden = self.role_encoder.encode_prompt(
            unit_ids=prompt_content_units.long(),
            logdur=logdur,
            mask=mask,
            edge_cue=prompt_edge_cue,
        )
        score = torch.einsum("btd,md->btm", hidden, self.codebook.role_key)
        score = score / math.sqrt(float(max(1, self.codebook.dim)))
        score = score.masked_fill(mask.unsqueeze(-1) <= 0.0, -1.0e4)
        attn = F.softmax(score, dim=-1) * mask.unsqueeze(-1)

        support = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        global_rate = _masked_median(logdur, mask)
        ref_residual = (logdur - global_rate) * mask

        coverage = attn.sum(dim=1).clamp_min(1.0e-6)
        role_value = torch.einsum("btm,bt->bm", attn, ref_residual) / coverage
        diff2 = (ref_residual.unsqueeze(-1) - role_value.unsqueeze(1)) ** 2
        role_var = (attn * diff2).sum(dim=1) / coverage
        role_cov = (coverage / support).clamp_min(self.coverage_floor)
        prompt_role_fit = torch.einsum("btm,bm->bt", attn, role_value) * mask

        batch_size = int(prompt_content_units.size(0))
        zero_operator = prompt_duration_obs.new_zeros((batch_size, self.operator_rank))
        return validate_reference_duration_memory(
            ReferenceDurationMemory(
                global_rate=global_rate,
                operator=StructuredDurationOperatorMemory(operator_coeff=zero_operator),
                role=StructuredRoleDurationMemory(
                    role_value=role_value,
                    role_var=role_var.clamp_min(1.0e-4),
                    role_coverage=role_cov,
                ),
                prompt=PromptConditioningEvidence(
                    prompt_mask=mask,
                    prompt_log_duration=logdur,
                    prompt_log_residual=ref_residual,
                    prompt_role_attn=attn,
                    prompt_role_fit=prompt_role_fit,
                ),
            )
        )


class StreamingDurationHead(nn.Module):
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
        self.max_logstretch = float(max(0.1, max_logstretch))
        self.query_encoder = CausalRoleQueryEncoder(vocab_size=vocab_size, dim=dim)
        self.codebook = codebook if codebook is not None else SharedRoleCodebook(num_slots=num_slots, dim=dim)
        self.prior_head = nn.Sequential(
            nn.Linear(dim + 3, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(dim + 4, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
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
        mask = unit_mask.float().clamp(0.0, 1.0)
        local_rate = local_rate_ema.float().expand_as(log_anchor)
        query = self.query_encoder(
            unit_ids=content_units,
            log_anchor=log_anchor,
            edge_cue=edge_cue,
            sep_hint=sep_hint,
            local_rate=local_rate,
        ) * mask.unsqueeze(-1)
        score = torch.einsum("bud,md->bum", query, self.codebook.role_key)
        score = score / math.sqrt(float(max(1, self.codebook.dim)))
        score = score + torch.log(role_coverage.float().clamp_min(1.0e-4)).unsqueeze(1)
        score = score.masked_fill(mask.unsqueeze(-1) <= 0.0, -1.0e4)
        attn = F.softmax(score, dim=-1) * mask.unsqueeze(-1)

        role_value_unit = torch.einsum("bum,bm->bu", attn, role_value.float())
        role_var_unit = torch.einsum("bum,bm->bu", attn, role_var.float()).clamp_min(1.0e-4)
        role_cov_unit = torch.einsum("bum,bm->bu", attn, role_coverage.float()).clamp_min(1.0e-4)
        role_conf = role_cov_unit / (role_cov_unit + role_var_unit)

        global_shift = (global_rate.float() - local_rate_ema.float()).expand_as(role_value_unit)
        prior_input = torch.cat(
            [
                query,
                global_shift.unsqueeze(-1),
                (role_conf * role_value_unit).unsqueeze(-1),
                role_var_unit.log().unsqueeze(-1),
            ],
            dim=-1,
        )
        prior = self.prior_head(prior_input).squeeze(-1)
        residual_input = torch.cat(
            [
                query,
                global_shift.unsqueeze(-1),
                role_value_unit.unsqueeze(-1),
                role_conf.unsqueeze(-1),
                role_var_unit.log().unsqueeze(-1),
            ],
            dim=-1,
        )
        residual = 0.25 * torch.tanh(self.residual_head(residual_input).squeeze(-1))
        pred = (prior + residual).clamp(min=-self.max_logstretch, max=self.max_logstretch)
        pred = pred * sealed_mask.float() * (1.0 - sep_hint.float().clamp(0.0, 1.0))
        return {
            "unit_logstretch": pred,
            "role_attn_unit": attn,
            "role_value_unit": role_value_unit * mask,
            "role_var_unit": role_var_unit * mask,
            "role_conf_unit": role_conf * mask,
            "role_query_unit": query,
            "local_response": residual * mask,
        }

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


def _masked_prefix_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float().clamp(0.0, 1.0)
    prefix_num = torch.cumsum(values * mask_f.unsqueeze(-1), dim=1)
    prefix_den = torch.cumsum(mask_f, dim=1).clamp_min(1.0).unsqueeze(-1)
    prefix_mean = prefix_num / prefix_den
    has_support = (torch.cumsum(mask_f, dim=1) > 0.0).unsqueeze(-1)
    return torch.where(has_support, prefix_mean, torch.zeros_like(prefix_mean))


def _masked_mean_1d(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float().clamp(0.0, 1.0)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * mask_f).sum(dim=1, keepdim=True) / denom
    has_support = mask_f.sum(dim=1, keepdim=True) > 0.0
    return torch.where(has_support, mean, torch.zeros_like(mean))


def _build_causal_local_rate_seq(
    *,
    observed_log: torch.Tensor,
    speech_mask: torch.Tensor,
    init_rate: torch.Tensor | None,
    default_init_rate: torch.Tensor | float | None = None,
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


class CausalUnitRunEncoder(nn.Module):
    """Minimal source-side run encoder for the maintained unit-run stretch head."""

    def __init__(
        self,
        *,
        vocab_size: int = 2048,
        dim: int = 64,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.dim = int(max(8, dim))
        self.kernel_size = int(max(2, kernel_size))
        self.dilations = tuple(max(1, int(value)) for value in dilations)
        self.unit_emb = nn.Embedding(int(vocab_size), self.dim)
        self.in_proj = nn.Linear(self.dim + 3, self.dim)
        self.dw = nn.ModuleList(
            [
                nn.Conv1d(
                    self.dim,
                    self.dim,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    groups=self.dim,
                )
                for dilation in self.dilations
            ]
        )
        self.pw = nn.ModuleList([nn.Conv1d(self.dim, self.dim, kernel_size=1) for _ in self.dilations])
        self.norm = nn.LayerNorm(self.dim)

    def forward(
        self,
        *,
        unit_ids: torch.Tensor,
        log_anchor: torch.Tensor,
        source_rate: torch.Tensor,
        silence_mask: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = unit_mask.float().clamp(0.0, 1.0)
        centered = (log_anchor.float() - source_rate.float()) * mask
        silence = silence_mask.float().clamp(0.0, 1.0) * mask
        hidden = torch.cat(
            [
                self.unit_emb(unit_ids.long()),
                log_anchor.float().unsqueeze(-1),
                centered.unsqueeze(-1),
                silence.unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden = self.in_proj(hidden) * mask.unsqueeze(-1)
        hidden_t = hidden.transpose(1, 2)
        for depthwise, pointwise, dilation in zip(self.dw, self.pw, self.dilations):
            padded = F.pad(hidden_t, (dilation * (self.kernel_size - 1), 0))
            update = pointwise(F.gelu(depthwise(padded)))
            hidden_t = hidden_t + update
        hidden = self.norm(hidden_t.transpose(1, 2))
        return hidden * mask.unsqueeze(-1)


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
        summary_pool_speech_only: bool = True,
        codebook: SharedSummaryCodebook | None = None,
    ) -> None:
        super().__init__()
        del operator_rank
        self.coverage_floor = float(max(1.0e-4, coverage_floor))
        self.summary_pool_speech_only = bool(summary_pool_speech_only)
        self.summary_dim = int(max(8, dim))
        self.prompt_unit_emb = nn.Embedding(vocab_size, self.summary_dim)
        self.prompt_in_proj = nn.Linear(self.summary_dim + 2, self.summary_dim)
        self.prompt_conv1 = nn.Conv1d(self.summary_dim, self.summary_dim, kernel_size=3, padding=1)
        self.prompt_conv2 = nn.Conv1d(self.summary_dim, self.summary_dim, kernel_size=3, padding=1)
        self.prompt_norm = nn.LayerNorm(self.summary_dim)
        self.codebook = codebook if codebook is not None else SharedSummaryCodebook(num_slots=num_slots, dim=self.summary_dim)
        self.summary_proj = nn.Sequential(
            nn.Linear(self.summary_dim * 2, self.summary_dim),
            nn.GELU(),
            nn.Linear(self.summary_dim, self.summary_dim),
        )

    def _encode_prompt_summary(
        self,
        *,
        prompt_content_units: torch.Tensor,
        centered_logdur: torch.Tensor,
        valid_mask: torch.Tensor,
        speech_mask: torch.Tensor,
    ) -> torch.Tensor:
        sil_flag = (valid_mask.float() - speech_mask.float()).clamp_min(0.0)
        hidden = torch.cat(
            [
                self.prompt_unit_emb(prompt_content_units.long()),
                centered_logdur.float().unsqueeze(-1),
                sil_flag.unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden = self.prompt_in_proj(hidden)
        hidden = F.gelu(self.prompt_conv1(hidden.transpose(1, 2)).transpose(1, 2))
        hidden = F.gelu(self.prompt_conv2(hidden.transpose(1, 2)).transpose(1, 2))
        hidden = self.prompt_norm(hidden)
        return hidden * valid_mask.unsqueeze(-1).float()

    def forward(
        self,
        *,
        prompt_content_units: torch.Tensor,
        prompt_duration_obs: torch.Tensor,
        prompt_mask: torch.Tensor | None = None,
        prompt_valid_mask: torch.Tensor | None = None,
        prompt_speech_mask: torch.Tensor | None = None,
        prompt_spk_embed: torch.Tensor | None = None,
        prompt_edge_cue: torch.Tensor | None = None,
    ) -> ReferenceDurationMemory:
        del prompt_edge_cue
        valid_mask = prompt_valid_mask
        if not isinstance(valid_mask, torch.Tensor):
            valid_mask = prompt_mask
        if not isinstance(valid_mask, torch.Tensor):
            valid_mask = prompt_duration_obs.float().gt(0.0).float()
        valid_mask = valid_mask.float().clamp(0.0, 1.0)
        speech_mask = prompt_speech_mask.float().clamp(0.0, 1.0) if isinstance(prompt_speech_mask, torch.Tensor) else valid_mask
        speech_mask = speech_mask * valid_mask
        logdur = torch.log(prompt_duration_obs.float().clamp_min(1.0e-4)) * valid_mask
        global_rate = _masked_median(logdur, speech_mask)
        ref_residual = (logdur - global_rate) * valid_mask

        hidden = self._encode_prompt_summary(
            prompt_content_units=prompt_content_units.long(),
            centered_logdur=ref_residual,
            valid_mask=valid_mask,
            speech_mask=speech_mask,
        )
        summary_mask = speech_mask if self.summary_pool_speech_only else valid_mask
        mean = _masked_mean(hidden, summary_mask)
        std = _masked_std(hidden, summary_mask, mean)
        summary_state = torch.tanh(self.summary_proj(torch.cat([mean, std], dim=-1)))

        support_raw = valid_mask.sum(dim=1, keepdim=True)
        support = support_raw.clamp_min(1.0)
        summary_state = torch.where(support_raw > 0.0, summary_state, torch.zeros_like(summary_state))

        score = torch.einsum("btd,md->btm", hidden, self.codebook.role_key)
        score = score / math.sqrt(float(max(1, self.codebook.dim)))
        score = score.masked_fill(valid_mask.unsqueeze(-1) <= 0.0, -1.0e4)
        attn = F.softmax(score, dim=-1) * valid_mask.unsqueeze(-1)

        role_mass = attn.sum(dim=1)
        role_denom = role_mass.clamp_min(1.0e-6)
        role_value = torch.einsum("btm,bt->bm", attn, ref_residual) / role_denom
        diff2 = (ref_residual.unsqueeze(-1) - role_value.unsqueeze(1)) ** 2
        role_var = (attn * diff2).sum(dim=1) / role_denom
        role_var = torch.where(role_mass > 0.0, role_var, torch.full_like(role_var, 1.0e-4)).clamp_min(1.0e-4)
        role_coverage = (role_mass / support).clamp_min(self.coverage_floor)
        prompt_role_fit = torch.einsum("btm,bm->bt", attn, role_value) * valid_mask

        return validate_reference_duration_memory(
            ReferenceDurationMemory(
                global_rate=global_rate,
                operator=StructuredDurationOperatorMemory(operator_coeff=summary_state),
                summary_state=summary_state,
                spk_embed=(
                    prompt_spk_embed.float()
                    if isinstance(prompt_spk_embed, torch.Tensor)
                    else None
                ),
                prompt_valid_mask=valid_mask,
                prompt_speech_mask=speech_mask,
                role=StructuredRoleDurationMemory(
                    role_value=role_value,
                    role_var=role_var,
                    role_coverage=role_coverage,
                ),
                prompt=PromptConditioningEvidence(
                    prompt_mask=valid_mask,
                    prompt_log_duration=logdur,
                    prompt_log_residual=ref_residual,
                    prompt_role_attn=attn,
                    prompt_role_fit=prompt_role_fit,
                    prompt_operator_coeff_norm=summary_state.norm(dim=-1, keepdim=True),
                ),
            )
        )


class StreamingDurationHead(nn.Module):
    """Source-anchored unit-run writer with prefix coarse control and local residual."""

    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        num_slots: int,
        spk_dim: int | None = None,
        max_logstretch: float = 1.2,
        max_silence_logstretch: float = 0.35,
        codebook: SharedSummaryCodebook | None = None,
    ) -> None:
        super().__init__()
        self.max_logstretch = float(max(0.1, max_logstretch))
        self.max_silence_logstretch = float(max(0.01, min(self.max_logstretch, max_silence_logstretch)))
        self.query_dim = int(max(8, dim))
        self.query_encoder = CausalUnitRunEncoder(vocab_size=vocab_size, dim=self.query_dim)
        self.codebook = codebook if codebook is not None else SharedSummaryCodebook(num_slots=num_slots, dim=self.query_dim)
        self.spk_dim = int(max(8, spk_dim if spk_dim is not None else dim))
        self.spk_proj = nn.Linear(self.spk_dim, self.query_dim)
        self.coarse_head = nn.Sequential(
            nn.Linear(self.query_dim * 3, self.query_dim),
            nn.GELU(),
            nn.Linear(self.query_dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear((self.query_dim * 3) + 1, self.query_dim),
            nn.GELU(),
            nn.Linear(self.query_dim, 1),
        )
        self.coarse_delta_scale = 0.20
        self.local_residual_scale = 0.35
        self.src_rate_init = nn.Parameter(torch.full((1,), math.log(2.0)))

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
        summary_state: torch.Tensor | None = None,
        spk_embed: torch.Tensor | None = None,
        role_value: torch.Tensor | None = None,
        role_var: torch.Tensor | None = None,
        role_coverage: torch.Tensor | None = None,
        local_rate_ema: torch.Tensor,
        silence_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del edge_cue
        mask = unit_mask.float().clamp(0.0, 1.0)
        sealed = sealed_mask.float().clamp(0.0, 1.0)
        silence = silence_mask.float().clamp(0.0, 1.0) if isinstance(silence_mask, torch.Tensor) else torch.zeros_like(mask)
        commit_valid_mask = mask * sealed
        silence_commit_mask = commit_valid_mask * silence
        speech_mask = commit_valid_mask * (1.0 - silence)

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
            default_init_rate=self.src_rate_init,
            decay=0.95,
        )

        query = self.query_encoder(
            unit_ids=content_units.long(),
            log_anchor=log_anchor.float(),
            source_rate=local_rate_seq.float(),
            silence_mask=silence if silence_mask is not None else sep_hint.float(),
            unit_mask=mask,
        ) * mask.unsqueeze(-1)

        attn = None
        role_value_unit = None
        role_var_unit = None
        role_conf_unit = None
        if (
            isinstance(role_value, torch.Tensor)
            and isinstance(role_var, torch.Tensor)
            and isinstance(role_coverage, torch.Tensor)
        ):
            score = torch.einsum("bud,md->bum", query, self.codebook.role_key)
            score = score / math.sqrt(float(max(1, self.codebook.dim)))
            score = score + torch.log(role_coverage.float().clamp_min(1.0e-4)).unsqueeze(1)
            score = score.masked_fill(mask.unsqueeze(-1) <= 0.0, -1.0e4)
            attn = F.softmax(score, dim=-1) * mask.unsqueeze(-1)
            role_value_unit = torch.einsum("bum,bm->bu", attn, role_value.float())
            role_var_unit = torch.einsum("bum,bm->bu", attn, role_var.float()).clamp_min(1.0e-4)
            role_cov_unit = torch.einsum("bum,bm->bu", attn, role_coverage.float()).clamp_min(1.0e-4)
            role_conf_unit = role_cov_unit / (role_cov_unit + role_var_unit)

        summary = (
            summary_state.float()
            if isinstance(summary_state, torch.Tensor)
            else query.new_zeros((query.size(0), self.query_dim))
        )
        if isinstance(spk_embed, torch.Tensor):
            spk = spk_embed.float()
            if spk.dim() == 3 and spk.size(-1) == 1:
                spk = spk.squeeze(-1)
            elif spk.dim() == 3 and spk.size(1) == 1:
                spk = spk.squeeze(1)
            if spk.dim() != 2:
                raise ValueError(f"StreamingDurationHead.spk_embed must have shape [B,H], got {tuple(spk.shape)}")
            if spk.size(-1) != self.spk_dim:
                if spk.size(-1) > self.spk_dim:
                    spk = spk[:, : self.spk_dim]
                else:
                    pad = spk.new_zeros((spk.size(0), self.spk_dim - spk.size(-1)))
                    spk = torch.cat([spk, pad], dim=-1)
            spk_ctx = torch.tanh(self.spk_proj(spk))
        else:
            spk_ctx = summary.new_zeros(summary.shape)

        global_shift_analytic = (global_rate.float() - local_rate_seq.float()) * commit_valid_mask
        summary_expand = summary.unsqueeze(1).expand(-1, query.size(1), -1)
        spk_expand = spk_ctx.unsqueeze(1).expand(-1, query.size(1), -1)
        prefix_query = _masked_prefix_mean(query, speech_mask)
        coarse_input = torch.cat(
            [
                prefix_query,
                summary_expand,
                spk_expand,
            ],
            dim=-1,
        )
        coarse_correction = self.coarse_delta_scale * torch.tanh(self.coarse_head(coarse_input).squeeze(-1))
        coarse_correction = coarse_correction * commit_valid_mask
        global_term = global_shift_analytic + coarse_correction
        residual_input = torch.cat(
            [
                query,
                summary_expand,
                spk_expand,
                global_term.unsqueeze(-1),
            ],
            dim=-1,
        )
        residual = self.local_residual_scale * torch.tanh(self.residual_head(residual_input).squeeze(-1))
        residual = residual * speech_mask
        pred_speech = (global_term + residual).clamp(
            min=-self.max_logstretch,
            max=self.max_logstretch,
        ) * speech_mask
        pred_silence = global_term.clamp(
            min=-self.max_silence_logstretch,
            max=self.max_silence_logstretch,
        ) * silence_commit_mask
        pred = pred_speech + pred_silence
        global_bias_scalar = _masked_mean_1d(coarse_correction, speech_mask)

        return {
            "unit_logstretch": pred,
            "unit_global_shift": global_term * mask,
            "unit_global_shift_analytic": global_shift_analytic * mask,
            "global_bias_scalar": global_bias_scalar,
            "unit_coarse_logstretch": global_term * mask,
            "unit_coarse_correction": coarse_correction * mask,
            "unit_residual_logstretch": residual * mask,
            "role_attn_unit": (attn if attn is not None else mask.unsqueeze(-1)),
            "role_value_unit": (
                role_value_unit * mask
                if isinstance(role_value_unit, torch.Tensor)
                else mask.new_zeros(mask.shape)
            ),
            "role_var_unit": (
                role_var_unit * mask
                if isinstance(role_var_unit, torch.Tensor)
                else mask.new_zeros(mask.shape)
            ),
            "role_conf_unit": (
                role_conf_unit * mask
                if isinstance(role_conf_unit, torch.Tensor)
                else mask.new_zeros(mask.shape)
            ),
            "role_query_unit": query,
            "local_response": residual * mask,
            "local_rate_seq": local_rate_seq * mask,
            "local_rate_final": local_rate_final,
            "source_rate_seq": local_rate_seq * mask,
            "source_prefix_summary": prefix_query * mask.unsqueeze(-1),
        }


PromptSummaryEncoder = PromptDurationMemoryEncoder
PromptSummaryDurationHead = StreamingDurationHead
CausalSummaryQueryEncoder = CausalStretchQueryEncoder
SharedRoleCodebook = SharedSummaryCodebook

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .math_utils import build_causal_local_rate_seq
from .g_stats import (
    build_global_rate_support_mask,
    compute_global_rate,
    normalize_falsification_eval_mode,
    normalize_global_rate_variant,
)
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


def _masked_mean_1d(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float().clamp(0.0, 1.0)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * mask_f).sum(dim=1, keepdim=True) / denom
    has_support = mask_f.sum(dim=1, keepdim=True) > 0.0
    return torch.where(has_support, mean, torch.zeros_like(mean))


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
        self.in_proj = nn.Linear(self.dim + 6, self.dim)
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
        log_base: torch.Tensor | None,
        use_log_base_rate: bool = True,
        source_rate: torch.Tensor,
        silence_mask: torch.Tensor,
        sep_hint: torch.Tensor,
        edge_cue: torch.Tensor,
        phrase_final_mask: torch.Tensor | None,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = unit_mask.float().clamp(0.0, 1.0)
        base = (
            log_base.float()
            if (bool(use_log_base_rate) and isinstance(log_base, torch.Tensor))
            else torch.zeros_like(log_anchor.float())
        )
        normalized_anchor = (log_anchor.float() - base) * mask
        centered = (normalized_anchor - source_rate.float()) * mask
        silence = silence_mask.float().clamp(0.0, 1.0) * mask
        sep = sep_hint.float().clamp(0.0, 1.0) * mask
        edge = edge_cue.float().clamp(0.0, 1.0) * mask
        phrase_final = (
            phrase_final_mask.float().clamp(0.0, 1.0) * mask
            if isinstance(phrase_final_mask, torch.Tensor)
            else torch.zeros_like(mask)
        )
        hidden = torch.cat(
            [
                self.unit_emb(unit_ids.long()),
                log_anchor.float().unsqueeze(-1),
                centered.unsqueeze(-1),
                silence.unsqueeze(-1),
                sep.unsqueeze(-1),
                edge.unsqueeze(-1),
                phrase_final.unsqueeze(-1),
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
        summary_use_unit_embedding: bool = False,
        simple_global_stats: bool = False,
        use_log_base_rate: bool = False,
        emit_prompt_diagnostics: bool = True,
        g_variant: str = "raw_median",
        g_trim_ratio: float = 0.2,
        drop_edge_runs_for_g: int = 0,
        codebook: SharedSummaryCodebook | None = None,
    ) -> None:
        super().__init__()
        self.operator_rank = int(max(1, operator_rank))
        self.simple_global_stats = bool(simple_global_stats)
        self.rate_mode = "simple_global" if self.simple_global_stats else "log_base"
        self.coverage_floor = float(max(1.0e-4, coverage_floor))
        self.summary_pool_speech_only = bool(summary_pool_speech_only)
        self.summary_use_unit_embedding = bool(summary_use_unit_embedding)
        self.use_log_base_rate = bool(use_log_base_rate) and not self.simple_global_stats
        self.emit_prompt_diagnostics = bool(emit_prompt_diagnostics)
        self.g_variant = normalize_global_rate_variant(g_variant)
        self.g_trim_ratio = float(max(0.0, min(0.49, g_trim_ratio)))
        self.g_drop_edge_runs = max(0, int(drop_edge_runs_for_g))
        self.summary_dim = int(max(8, dim))
        self.prompt_unit_emb = (
            nn.Embedding(vocab_size, self.summary_dim)
            if self.summary_use_unit_embedding
            else None
        )
        self.prompt_in_proj = nn.Linear(self.summary_dim + 4, self.summary_dim)
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
        edge_cue: torch.Tensor | None = None,
        phrase_final_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sil_flag = (valid_mask.float() - speech_mask.float()).clamp_min(0.0)
        edge = (
            edge_cue.float().clamp(0.0, 1.0)
            if isinstance(edge_cue, torch.Tensor)
            else torch.zeros_like(centered_logdur)
        )
        phrase_final = (
            phrase_final_mask.float().clamp(0.0, 1.0)
            if isinstance(phrase_final_mask, torch.Tensor)
            else torch.zeros_like(centered_logdur)
        )
        content_embed = (
            self.prompt_unit_emb(prompt_content_units.long())
            if self.prompt_unit_emb is not None
            else centered_logdur.new_zeros(
                prompt_content_units.size(0),
                prompt_content_units.size(1),
                self.summary_dim,
            )
        )
        hidden = torch.cat(
            [
                content_embed,
                centered_logdur.float().unsqueeze(-1),
                sil_flag.unsqueeze(-1),
                edge.unsqueeze(-1),
                phrase_final.unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden = self.prompt_in_proj(hidden) * valid_mask.unsqueeze(-1).float()
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
        prompt_unit_anchor_base: torch.Tensor | None = None,
        prompt_log_base: torch.Tensor | None = None,
        prompt_spk_embed: torch.Tensor | None = None,
        prompt_edge_cue: torch.Tensor | None = None,
        prompt_phrase_final_mask: torch.Tensor | None = None,
        prompt_global_weight: torch.Tensor | None = None,
        prompt_unit_log_prior: torch.Tensor | None = None,
    ) -> ReferenceDurationMemory:
        valid_mask = prompt_valid_mask
        if not isinstance(valid_mask, torch.Tensor):
            valid_mask = prompt_mask
        if not isinstance(valid_mask, torch.Tensor):
            valid_mask = prompt_duration_obs.float().gt(0.0).float()
        valid_mask = valid_mask.float().clamp(0.0, 1.0)
        speech_mask = prompt_speech_mask.float().clamp(0.0, 1.0) if isinstance(prompt_speech_mask, torch.Tensor) else valid_mask
        speech_mask = speech_mask * valid_mask
        logdur = torch.log(prompt_duration_obs.float().clamp_min(1.0e-4)) * valid_mask
        if self.use_log_base_rate:
            if isinstance(prompt_log_base, torch.Tensor):
                log_base = prompt_log_base.float().detach() * valid_mask
            elif isinstance(prompt_unit_anchor_base, torch.Tensor):
                log_base = torch.log(prompt_unit_anchor_base.float().detach().clamp_min(1.0e-6)) * valid_mask
            else:
                log_base = torch.zeros_like(logdur)
        else:
            log_base = torch.zeros_like(logdur)
        rate_logdur = (
            (logdur - log_base) * valid_mask
            if self.use_log_base_rate
            else logdur
        )
        support_mask = build_global_rate_support_mask(
            speech_mask=speech_mask,
            valid_mask=valid_mask,
            drop_edge_runs=self.g_drop_edge_runs,
        )
        support_has_mass = support_mask.any(dim=1, keepdim=True)
        rate_speech_mask = speech_mask
        if bool((~support_has_mass).any().item()):
            # Generic prompt-summary pooling remains lenient for all-silence prompts:
            # use valid support only to avoid a hard failure, then zero the
            # speech-only pooled summary when requested.
            rate_speech_mask = torch.where(support_has_mass, speech_mask, valid_mask)
        global_rate = compute_global_rate(
            log_dur=rate_logdur,
            speech_mask=rate_speech_mask,
            valid_mask=valid_mask,
            variant=self.g_variant,
            weight=prompt_global_weight,
            trim_ratio=self.g_trim_ratio,
            drop_edge_runs=self.g_drop_edge_runs,
            unit_ids=prompt_content_units,
            unit_prior=prompt_unit_log_prior,
        )
        if self.summary_pool_speech_only:
            global_rate = torch.where(support_has_mass, global_rate, torch.zeros_like(global_rate))
        residual_mask = (
            support_mask
            if self.summary_pool_speech_only
            else build_global_rate_support_mask(
                speech_mask=rate_speech_mask,
                valid_mask=valid_mask,
                drop_edge_runs=self.g_drop_edge_runs,
            )
        )
        ref_residual = (rate_logdur - global_rate) * residual_mask
        if self.simple_global_stats and not self.emit_prompt_diagnostics:
            operator_coeff = global_rate.new_zeros((global_rate.size(0), self.operator_rank))
            return validate_reference_duration_memory(
                ReferenceDurationMemory(
                    global_rate=global_rate,
                    operator=StructuredDurationOperatorMemory(operator_coeff=operator_coeff),
                    summary_state=None,
                    spk_embed=(
                        prompt_spk_embed.float()
                        if isinstance(prompt_spk_embed, torch.Tensor)
                        else None
                    ),
                    prompt_valid_mask=valid_mask,
                    prompt_speech_mask=speech_mask,
                    prompt=PromptConditioningEvidence(
                        prompt_mask=valid_mask,
                        prompt_log_base=log_base,
                        prompt_log_duration=logdur,
                        prompt_log_residual=ref_residual,
                    ),
                )
            )

        summary_mask = support_mask if self.summary_pool_speech_only else valid_mask
        hidden = self._encode_prompt_summary(
            prompt_content_units=prompt_content_units.long(),
            centered_logdur=ref_residual,
            valid_mask=summary_mask,
            speech_mask=speech_mask,
            edge_cue=prompt_edge_cue,
            phrase_final_mask=prompt_phrase_final_mask,
        )
        mean = _masked_mean(hidden, summary_mask)
        std = _masked_std(hidden, summary_mask, mean)
        summary_state = torch.tanh(self.summary_proj(torch.cat([mean, std], dim=-1)))

        support_raw = summary_mask.sum(dim=1, keepdim=True)
        support = support_raw.clamp_min(1.0)
        summary_state = torch.where(support_raw > 0.0, summary_state, torch.zeros_like(summary_state))

        score = torch.einsum("btd,md->btm", hidden, self.codebook.role_key)
        score = score / math.sqrt(float(max(1, self.codebook.dim)))
        score = score.masked_fill(summary_mask.unsqueeze(-1) <= 0.0, -1.0e4)
        attn = F.softmax(score, dim=-1) * summary_mask.unsqueeze(-1)

        role_mass = attn.sum(dim=1)
        role_denom = role_mass.clamp_min(1.0e-6)
        role_value = torch.einsum("btm,bt->bm", attn, ref_residual) / role_denom
        diff2 = (ref_residual.unsqueeze(-1) - role_value.unsqueeze(1)) ** 2
        role_var = (attn * diff2).sum(dim=1) / role_denom
        role_var = torch.where(role_mass > 0.0, role_var, torch.full_like(role_var, 1.0e-4)).clamp_min(1.0e-4)
        role_coverage = (role_mass / support).clamp_min(self.coverage_floor)
        prompt_role_fit = torch.einsum("btm,bm->bt", attn, role_value) * summary_mask

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
                    prompt_log_base=log_base,
                    prompt_log_duration=logdur,
                    prompt_log_residual=ref_residual,
                    prompt_role_attn=attn,
                    prompt_role_fit=prompt_role_fit,
                    prompt_operator_coeff_norm=summary_state.norm(dim=-1, keepdim=True),
                ),
            )
        )


class PromptGlobalConditionEncoderV1G(nn.Module):
    """Minimal prompt encoder that only exposes speech-only global tempo and speaker state."""

    def __init__(
        self,
        *,
        operator_rank: int,
        min_speech_ratio: float = 0.6,
        use_log_base_rate: bool = False,
        g_variant: str = "raw_median",
        g_trim_ratio: float = 0.2,
        drop_edge_runs_for_g: int = 0,
    ) -> None:
        super().__init__()
        self.operator_rank = int(max(1, operator_rank))
        self.min_speech_ratio = float(max(0.0, min(1.0, min_speech_ratio)))
        self.use_log_base_rate = bool(use_log_base_rate)
        self.rate_mode = "log_base" if self.use_log_base_rate else "simple_global"
        self.simple_global_stats = not self.use_log_base_rate
        self.g_variant = normalize_global_rate_variant(g_variant)
        self.g_trim_ratio = float(max(0.0, min(0.49, g_trim_ratio)))
        self.g_drop_edge_runs = max(0, int(drop_edge_runs_for_g))

    def forward(
        self,
        *,
        prompt_content_units: torch.Tensor,
        prompt_duration_obs: torch.Tensor,
        prompt_mask: torch.Tensor | None = None,
        prompt_valid_mask: torch.Tensor | None = None,
        prompt_speech_mask: torch.Tensor | None = None,
        prompt_unit_anchor_base: torch.Tensor | None = None,
        prompt_log_base: torch.Tensor | None = None,
        prompt_spk_embed: torch.Tensor | None = None,
        prompt_edge_cue: torch.Tensor | None = None,
        prompt_phrase_final_mask: torch.Tensor | None = None,
        prompt_global_weight: torch.Tensor | None = None,
        prompt_unit_log_prior: torch.Tensor | None = None,
    ) -> ReferenceDurationMemory:
        del prompt_edge_cue, prompt_phrase_final_mask
        valid_mask = prompt_valid_mask
        if not isinstance(valid_mask, torch.Tensor):
            valid_mask = prompt_mask
        if not isinstance(valid_mask, torch.Tensor):
            valid_mask = prompt_duration_obs.float().gt(0.0).float()
        valid_mask = valid_mask.float().clamp(0.0, 1.0)
        speech_mask = (
            prompt_speech_mask.float().clamp(0.0, 1.0)
            if isinstance(prompt_speech_mask, torch.Tensor)
            else valid_mask
        )
        speech_mask = speech_mask * valid_mask
        speech_mass = speech_mask.sum(dim=1, keepdim=True)
        if bool((speech_mass <= 0.0).any().item()):
            raise ValueError("V1-G prompt conditioning requires at least one speech run.")
        valid_mass = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        speech_ratio = speech_mass / valid_mass
        if self.min_speech_ratio > 0.0 and bool((speech_ratio < self.min_speech_ratio).any().item()):
            raise ValueError(
                f"V1-G prompt conditioning requires speech-dominant prompts "
                f"(speech_ratio >= {self.min_speech_ratio:.2f})."
            )

        logdur = torch.log(prompt_duration_obs.float().clamp_min(1.0e-4)) * valid_mask
        if self.use_log_base_rate:
            if isinstance(prompt_log_base, torch.Tensor):
                log_base = prompt_log_base.float().detach() * valid_mask
            elif isinstance(prompt_unit_anchor_base, torch.Tensor):
                log_base = torch.log(prompt_unit_anchor_base.float().detach().clamp_min(1.0e-6)) * valid_mask
            else:
                log_base = torch.zeros_like(logdur)
        else:
            log_base = torch.zeros_like(logdur)
        rate_logdur = ((logdur - log_base) * valid_mask) if self.use_log_base_rate else logdur
        global_rate = compute_global_rate(
            log_dur=rate_logdur,
            speech_mask=speech_mask,
            valid_mask=valid_mask,
            variant=self.g_variant,
            weight=prompt_global_weight,
            trim_ratio=self.g_trim_ratio,
            drop_edge_runs=self.g_drop_edge_runs,
            unit_ids=prompt_content_units,
            unit_prior=prompt_unit_log_prior,
        )
        prompt_residual = (rate_logdur - global_rate) * speech_mask
        operator_coeff = global_rate.new_zeros((global_rate.size(0), self.operator_rank))

        return validate_reference_duration_memory(
            ReferenceDurationMemory(
                global_rate=global_rate,
                operator=StructuredDurationOperatorMemory(operator_coeff=operator_coeff),
                summary_state=None,
                spk_embed=(
                    prompt_spk_embed.float()
                    if isinstance(prompt_spk_embed, torch.Tensor)
                    else None
                ),
                prompt_valid_mask=valid_mask,
                prompt_speech_mask=speech_mask,
                prompt=PromptConditioningEvidence(
                    prompt_mask=valid_mask,
                    prompt_log_base=log_base,
                    prompt_log_duration=logdur,
                    prompt_log_residual=prompt_residual,
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
        simple_global_stats: bool = False,
        use_log_base_rate: bool = False,
        use_learned_residual_gate: bool = False,
        max_logstretch: float = 1.2,
        max_silence_logstretch: float = 0.35,
        local_cold_start_runs: int = 2,
        local_short_run_min_duration: float = 2.0,
        local_rate_decay: float = 0.95,
        short_gap_silence_scale: float = 0.35,
        leading_silence_scale: float = 0.0,
        eval_mode: str = "learned",
        disable_local_residual: bool = False,
        disable_coarse_bias: bool = False,
        codebook: SharedSummaryCodebook | None = None,
    ) -> None:
        super().__init__()
        self.simple_global_stats = bool(simple_global_stats)
        self.rate_mode = "simple_global" if self.simple_global_stats else "log_base"
        self.use_log_base_rate = bool(use_log_base_rate) and not self.simple_global_stats
        self.use_learned_residual_gate = bool(use_learned_residual_gate)
        self.max_logstretch = float(max(0.1, max_logstretch))
        self.max_silence_logstretch = float(max(0.01, min(self.max_logstretch, max_silence_logstretch)))
        self.local_cold_start_runs = int(max(0, local_cold_start_runs))
        self.local_short_run_min_duration = float(max(1.0, local_short_run_min_duration))
        self.local_rate_decay = float(max(0.0, min(0.999, local_rate_decay)))
        self.short_gap_silence_scale = float(max(0.0, min(1.0, short_gap_silence_scale)))
        self.leading_silence_scale = float(max(0.0, min(1.0, leading_silence_scale)))
        self.eval_mode = normalize_falsification_eval_mode(eval_mode)
        self.disable_local_residual = bool(disable_local_residual)
        self.disable_coarse_bias = bool(disable_coarse_bias)
        self.query_dim = int(max(8, dim))
        self.query_encoder = CausalUnitRunEncoder(vocab_size=vocab_size, dim=self.query_dim)
        self.codebook = codebook if codebook is not None else SharedSummaryCodebook(num_slots=num_slots, dim=self.query_dim)
        self.spk_dim = int(max(8, spk_dim if spk_dim is not None else dim))
        self.spk_proj = nn.Linear(self.spk_dim, self.query_dim)
        self.coarse_head = nn.Sequential(
            nn.Linear((self.query_dim * 2) + 1, self.query_dim),
            nn.GELU(),
            nn.Linear(self.query_dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear((self.query_dim * 3) + 1, self.query_dim),
            nn.GELU(),
            nn.Linear(self.query_dim, 1),
        )
        self.residual_gate_head = (
            nn.Sequential(
                nn.Linear((self.query_dim * 3) + 1, self.query_dim),
                nn.GELU(),
                nn.Linear(self.query_dim, 1),
            )
            if self.use_learned_residual_gate
            else None
        )
        self.coarse_delta_scale = 0.20
        self.local_residual_scale = 0.35
        self.src_rate_init = nn.Parameter(torch.zeros((1,)))

    def forward(
        self,
        *,
        content_units: torch.Tensor,
        log_anchor: torch.Tensor,
        log_base: torch.Tensor | None = None,
        unit_mask: torch.Tensor,
        sealed_mask: torch.Tensor,
        sep_hint: torch.Tensor,
        edge_cue: torch.Tensor,
        phrase_final_mask: torch.Tensor | None = None,
        global_rate: torch.Tensor,
        summary_state: torch.Tensor | None = None,
        spk_embed: torch.Tensor | None = None,
        role_value: torch.Tensor | None = None,
        role_var: torch.Tensor | None = None,
        role_coverage: torch.Tensor | None = None,
        local_rate_ema: torch.Tensor,
        silence_mask: torch.Tensor | None = None,
        run_stability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mask = unit_mask.float().clamp(0.0, 1.0)
        sealed = sealed_mask.float().clamp(0.0, 1.0)
        silence = silence_mask.float().clamp(0.0, 1.0) if isinstance(silence_mask, torch.Tensor) else torch.zeros_like(mask)
        runtime_stability = (
            run_stability.float().clamp(0.0, 1.0) * mask
            if isinstance(run_stability, torch.Tensor)
            else torch.ones_like(mask)
        )
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

        observed_log_anchor = (
            (log_anchor.float() - log_base.float())
            if (self.use_log_base_rate and isinstance(log_base, torch.Tensor))
            else log_anchor.float()
        )
        local_rate_seq, local_rate_final = build_causal_local_rate_seq(
            observed_log=observed_log_anchor,
            speech_mask=speech_mask,
            init_rate=init_local_rate,
            default_init_rate=self.src_rate_init,
            decay=self.local_rate_decay,
        )

        query = self.query_encoder(
            unit_ids=content_units.long(),
            log_anchor=log_anchor.float(),
            log_base=log_base.float() if isinstance(log_base, torch.Tensor) else None,
            use_log_base_rate=self.use_log_base_rate,
            source_rate=local_rate_seq.float(),
            silence_mask=silence,
            sep_hint=sep_hint.float(),
            edge_cue=edge_cue.float(),
            phrase_final_mask=phrase_final_mask,
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
        coarse_context = torch.cat(
            [
                summary,
                spk_ctx,
                global_rate.float(),
            ],
            dim=-1,
        )
        coarse_scalar = self.coarse_delta_scale * torch.tanh(self.coarse_head(coarse_context).squeeze(-1))
        predicted_coarse = coarse_scalar.unsqueeze(1).expand(-1, query.size(1)) * commit_valid_mask
        coarse_correction = predicted_coarse
        if self.eval_mode == "analytic" or self.disable_coarse_bias:
            coarse_correction = torch.zeros_like(predicted_coarse)
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
        prefix_speech_prev = (torch.cumsum(speech_mask, dim=1) - speech_mask).clamp_min(0.0)
        if self.local_cold_start_runs > 0:
            cold_gate = (prefix_speech_prev / float(self.local_cold_start_runs)).clamp(0.0, 1.0)
        else:
            cold_gate = torch.ones_like(prefix_speech_prev)
        min_duration = max(1.0, float(self.local_short_run_min_duration))
        short_gate = (
            (torch.exp(log_anchor.float()).clamp_min(1.0) - 1.0)
            / max(1.0, min_duration - 1.0)
        ).clamp(0.0, 1.0)
        deterministic_gate = cold_gate * short_gate * runtime_stability * speech_mask
        if self.use_learned_residual_gate:
            if self.residual_gate_head is None:
                raise RuntimeError("residual_gate_head is missing while learned residual gate is enabled.")
            gate_raw = torch.sigmoid(self.residual_gate_head(residual_input).squeeze(-1))
            residual_gate = gate_raw * deterministic_gate
        else:
            residual_gate = deterministic_gate
        residual = residual * residual_gate
        predicted_residual = residual
        if self.eval_mode in {"analytic", "coarse_only"} or self.disable_local_residual:
            residual = torch.zeros_like(predicted_residual)
        pred_speech = (global_term + residual).clamp(
            min=-self.max_logstretch,
            max=self.max_logstretch,
        ) * speech_mask
        pause_shape = torch.sigmoid(log_anchor.float() - math.log(3.0))
        boundary_shape = torch.maximum(sep_hint.float().clamp(0.0, 1.0), edge_cue.float().clamp(0.0, 1.0))
        silence_shape = torch.maximum(pause_shape, boundary_shape)
        silence_tau = self.max_silence_logstretch * (
            self.short_gap_silence_scale
            + ((1.0 - self.short_gap_silence_scale) * silence_shape)
        )
        leading_gate = torch.where(
            prefix_speech_prev > 0.0,
            torch.ones_like(prefix_speech_prev),
            torch.full_like(prefix_speech_prev, self.leading_silence_scale),
        )
        pred_silence = torch.clamp(global_term, min=-silence_tau, max=silence_tau) * silence_commit_mask * leading_gate
        pred = pred_speech + pred_silence
        global_bias_scalar = coarse_scalar.reshape(-1, 1)

        return {
            "unit_logstretch": pred,
            "unit_global_shift": global_term * mask,
            "unit_analytic_gap": global_shift_analytic * mask,
            "unit_global_shift_analytic": global_shift_analytic * mask,
            "global_bias_scalar": global_bias_scalar,
            "unit_coarse_logstretch": global_term * mask,
            "unit_coarse_correction_used": coarse_correction * mask,
            "unit_coarse_correction": coarse_correction * mask,
            "unit_coarse_correction_predicted": predicted_coarse * mask,
            "unit_coarse_correction_pred": predicted_coarse * mask,
            "unit_local_residual_used": residual * mask,
            "unit_residual_logstretch": residual * mask,
            "unit_residual_logstretch_pred": predicted_residual * mask,
            "unit_residual_gate": residual_gate * mask,
            "unit_runtime_stability": runtime_stability * mask,
            "unit_silence_tau": silence_tau * silence_commit_mask,
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
            "local_response_pred": predicted_residual * mask,
            "local_rate_seq": local_rate_seq * mask,
            "local_rate_final": local_rate_final,
            "source_rate_seq": local_rate_seq * mask,
            "g_ref_scalar": global_rate.float().reshape(global_rate.size(0), -1)[:, 0],
            "g_ref": global_rate.float().reshape(global_rate.size(0), -1)[:, 0],
            "g_src_prefix_seq": local_rate_seq * mask,
            "g_src_prefix": local_rate_seq * mask,
            "eval_mode": self.eval_mode,
            "falsification_eval_mode": mask.new_full((mask.size(0), 1), {"analytic": 0.0, "coarse_only": 1.0, "learned": 2.0}[self.eval_mode]),
        }


PromptSummaryEncoder = PromptDurationMemoryEncoder
PromptSummaryDurationHead = StreamingDurationHead
CausalSummaryQueryEncoder = CausalStretchQueryEncoder
SharedRoleCodebook = SharedSummaryCodebook

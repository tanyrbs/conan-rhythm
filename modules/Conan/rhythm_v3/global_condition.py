from __future__ import annotations

import torch
import torch.nn as nn

from .contracts import (
    PromptConditioningEvidence,
    ReferenceDurationMemory,
    StructuredDurationOperatorMemory,
    validate_reference_duration_memory,
)
from .g_stats import (
    build_softclean_weights,
    compute_duration_weighted_speech_ratio,
    compute_global_rate,
    is_softclean_global_rate_variant,
    normalize_global_rate_variant,
    summarize_global_rate_support,
)


class PromptGlobalConditionEncoderV1G(nn.Module):
    """Minimal prompt encoder that only exposes speech-only global tempo and speaker state."""

    def __init__(
        self,
        *,
        operator_rank: int,
        prompt_domain_mode: str = "minimal_strict",
        prompt_require_clean_support: bool = True,
        min_speech_ratio: float = 0.6,
        min_ref_len_sec: float = 3.0,
        max_ref_len_sec: float = 8.0,
        use_log_base_rate: bool = False,
        g_variant: str = "raw_median",
        g_trim_ratio: float = 0.2,
        drop_edge_runs_for_g: int = 0,
        min_boundary_confidence: float | None = None,
        strict_eval_invalid_g: bool = False,
    ) -> None:
        super().__init__()
        self.operator_rank = int(max(1, operator_rank))
        self.prompt_domain_mode = str(prompt_domain_mode or "minimal_strict").strip().lower()
        self.prompt_require_clean_support = bool(prompt_require_clean_support)
        self.min_speech_ratio = float(max(0.0, min(1.0, min_speech_ratio)))
        self.min_ref_len_sec = float(max(0.0, min_ref_len_sec))
        self.max_ref_len_sec = float(max(self.min_ref_len_sec, max_ref_len_sec))
        self.use_log_base_rate = bool(use_log_base_rate)
        self.rate_mode = "log_base" if self.use_log_base_rate else "simple_global"
        self.simple_global_stats = not self.use_log_base_rate
        self.g_variant = normalize_global_rate_variant(g_variant)
        self.g_trim_ratio = float(max(0.0, min(0.49, g_trim_ratio)))
        self.g_drop_edge_runs = max(0, int(drop_edge_runs_for_g))
        self.min_boundary_confidence = (
            None if min_boundary_confidence is None else float(max(0.0, min(1.0, min_boundary_confidence)))
        )
        self.strict_eval_invalid_g = bool(strict_eval_invalid_g)

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
        prompt_closed_mask: torch.Tensor | None = None,
        prompt_boundary_confidence: torch.Tensor | None = None,
        prompt_global_weight: torch.Tensor | None = None,
        prompt_unit_log_prior: torch.Tensor | None = None,
        prompt_ref_len_sec: torch.Tensor | None = None,
        prompt_speech_ratio_scalar: torch.Tensor | None = None,
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
        speech_count = speech_mask.sum(dim=1, keepdim=True)
        speech_ratio = compute_duration_weighted_speech_ratio(
            duration_obs=prompt_duration_obs.float(),
            speech_mask=speech_mask,
            valid_mask=valid_mask,
        )
        if isinstance(prompt_speech_ratio_scalar, torch.Tensor):
            provided_ratio = prompt_speech_ratio_scalar.float().reshape(-1, 1)
            if int(provided_ratio.size(0)) != int(speech_ratio.size(0)):
                raise ValueError(
                    "V1-G prompt conditioning prompt_speech_ratio_scalar batch mismatch: "
                    f"got {tuple(provided_ratio.shape)} for batch_size={int(speech_ratio.size(0))}."
                )
            if bool((torch.abs(provided_ratio - speech_ratio) > 5.0e-3).any().item()):
                raise ValueError("V1-G prompt conditioning prompt_speech_ratio_scalar mismatch.")
        no_speech_rows = speech_count <= 0.0
        low_speech_ratio_rows = (
            speech_ratio < self.min_speech_ratio
            if self.min_speech_ratio > 0.0
            else torch.zeros_like(speech_ratio, dtype=torch.bool)
        )
        ref_len = prompt_ref_len_sec.float().reshape(-1, 1) if isinstance(prompt_ref_len_sec, torch.Tensor) else None
        if isinstance(ref_len, torch.Tensor) and int(ref_len.size(0)) != int(speech_ratio.size(0)):
            raise ValueError(
                "V1-G prompt conditioning prompt_ref_len_sec batch mismatch: "
                f"got {tuple(ref_len.shape)} for batch_size={int(speech_ratio.size(0))}."
            )
        missing_ref_len_rows = (
            torch.ones_like(speech_ratio, dtype=torch.bool)
            if not isinstance(ref_len, torch.Tensor)
            else torch.zeros_like(speech_ratio, dtype=torch.bool)
        )
        if self.prompt_domain_mode == "minimal_strict":
            invalid_ref_len_rows = (
                ((~torch.isfinite(ref_len)) | (ref_len < self.min_ref_len_sec) | (ref_len > self.max_ref_len_sec))
                if isinstance(ref_len, torch.Tensor)
                else torch.zeros_like(speech_ratio, dtype=torch.bool)
            )
            invalid_ref_len_rows = invalid_ref_len_rows | missing_ref_len_rows
        else:
            invalid_ref_len_rows = torch.zeros_like(speech_ratio, dtype=torch.bool)
        if self.training:
            if bool(no_speech_rows.any().item()):
                raise ValueError("V1-G prompt conditioning requires at least one speech run.")
            if bool(low_speech_ratio_rows.any().item()):
                raise ValueError(
                    f"V1-G prompt conditioning requires speech-dominant prompts "
                    f"(speech_ratio >= {self.min_speech_ratio:.2f})."
                )
            if bool(invalid_ref_len_rows.any().item()):
                raise ValueError(
                    f"V1-G prompt conditioning requires {self.min_ref_len_sec:.0f}-{self.max_ref_len_sec:.0f}s reference duration."
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
        closed_mask = (
            prompt_closed_mask.float().clamp(0.0, 1.0) * valid_mask
            if isinstance(prompt_closed_mask, torch.Tensor)
            else None
        )
        boundary_confidence = (
            prompt_boundary_confidence.float().clamp(0.0, 1.0) * valid_mask
            if isinstance(prompt_boundary_confidence, torch.Tensor)
            else None
        )
        support_stats = summarize_global_rate_support(
            speech_mask=speech_mask,
            valid_mask=valid_mask,
            duration_obs=prompt_duration_obs.float(),
            drop_edge_runs=self.g_drop_edge_runs,
            closed_mask=closed_mask,
            boundary_confidence=boundary_confidence,
            min_boundary_confidence=self.min_boundary_confidence,
        )
        support_mask = support_stats.support_mask
        if self.prompt_require_clean_support:
            missing_closed_sidecar_rows = (
                torch.ones_like(support_stats.support_count, dtype=torch.bool)
                if not isinstance(prompt_closed_mask, torch.Tensor)
                else torch.zeros_like(support_stats.support_count, dtype=torch.bool)
            )
            missing_boundary_sidecar_rows = (
                torch.ones_like(support_stats.support_count, dtype=torch.bool)
                if self.min_boundary_confidence is not None and not isinstance(prompt_boundary_confidence, torch.Tensor)
                else torch.zeros_like(support_stats.support_count, dtype=torch.bool)
            )
            invalid_clean_rows = support_stats.clean_count <= 0.0
        else:
            missing_closed_sidecar_rows = torch.zeros_like(support_stats.support_count, dtype=torch.bool)
            missing_boundary_sidecar_rows = torch.zeros_like(support_stats.support_count, dtype=torch.bool)
            invalid_clean_rows = torch.zeros_like(support_stats.support_count, dtype=torch.bool)
        invalid_clean_support_rows = invalid_clean_rows | missing_closed_sidecar_rows | missing_boundary_sidecar_rows
        invalid_support_rows = support_stats.support_count <= 0.0
        resolved_weight = (
            prompt_global_weight.float()
            if isinstance(prompt_global_weight, torch.Tensor)
            else None
        )
        estimator_support_mask = support_mask
        if is_softclean_global_rate_variant(self.g_variant):
            estimator_support_mask = (speech_mask > 0.5) & (valid_mask > 0.5)
            if resolved_weight is None:
                resolved_weight = build_softclean_weights(
                    speech_mask=speech_mask,
                    valid_mask=valid_mask,
                    closed_mask=closed_mask,
                    boundary_confidence=boundary_confidence,
                )
        support_weight = support_stats.support_count
        if isinstance(prompt_global_weight, torch.Tensor):
            support_mass = torch.where(
                support_mask,
                prompt_global_weight.float().clamp_min(0.0),
                torch.zeros_like(prompt_global_weight.float()),
            ).sum(dim=1, keepdim=True)
            support_weight = support_mass
            invalid_support_rows = invalid_support_rows | (support_mass <= 0.0)
        domain_invalid_rows = no_speech_rows | low_speech_ratio_rows | invalid_ref_len_rows
        invalid_rows = invalid_support_rows | invalid_clean_support_rows | domain_invalid_rows
        if bool(invalid_rows.any().item()):
            if self.training or self.strict_eval_invalid_g:
                support_error = (
                    "V1-G prompt conditioning requires non-empty closed/boundary-clean support for g."
                    if self.prompt_require_clean_support
                    else "V1-G prompt conditioning requires non-empty support for g."
                )
                raise ValueError(
                    support_error
                )
        global_rate = rate_logdur.new_zeros((rate_logdur.size(0), 1))
        valid_rows = ~invalid_rows.squeeze(1)
        if bool(valid_rows.any().item()):
            row_rate = compute_global_rate(
                log_dur=rate_logdur[valid_rows],
                speech_mask=speech_mask[valid_rows],
                valid_mask=valid_mask[valid_rows],
                variant=self.g_variant,
                weight=None if resolved_weight is None else resolved_weight[valid_rows],
                trim_ratio=self.g_trim_ratio,
                drop_edge_runs=self.g_drop_edge_runs,
                unit_ids=prompt_content_units[valid_rows],
                unit_prior=None if prompt_unit_log_prior is None else prompt_unit_log_prior[valid_rows],
                support_mask=estimator_support_mask[valid_rows],
                invalid_weight_behavior="raise",
            )
            global_rate[valid_rows] = row_rate
        effective_support_mask = torch.where(
            invalid_rows.expand_as(support_mask),
            torch.zeros_like(support_mask),
            support_mask,
        )
        residual_mask = torch.where(
            invalid_rows.expand_as(effective_support_mask),
            torch.zeros_like(effective_support_mask, dtype=rate_logdur.dtype),
            effective_support_mask.float(),
        )
        prompt_residual = (rate_logdur - global_rate) * residual_mask
        operator_coeff = global_rate.new_zeros((global_rate.size(0), self.operator_rank))
        clean_mask = support_stats.clean_mask
        if not isinstance(clean_mask, torch.Tensor):
            clean_mask = support_mask
        effective_clean_mask = torch.where(
            invalid_rows.expand_as(clean_mask),
            torch.zeros_like(clean_mask),
            clean_mask,
        )
        effective_clean_count = effective_clean_mask.sum(dim=1, keepdim=True).float()
        prompt_domain_valid = (~invalid_rows).float()
        effective_support_count = effective_support_mask.sum(dim=1, keepdim=True).float()
        effective_support_weight = torch.where(
            invalid_rows,
            torch.zeros_like(support_weight),
            support_weight,
        )
        valid_count = valid_mask.sum(dim=1, keepdim=True).float().clamp_min(1.0)
        speech_count = speech_mask.sum(dim=1, keepdim=True).float().clamp_min(1.0)

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
                    prompt_g_support_mask=effective_support_mask.float(),
                    prompt_g_clean_mask=effective_clean_mask.float(),
                    prompt_g_support_count=effective_support_count.detach(),
                    prompt_g_clean_count=effective_clean_count.detach(),
                    prompt_g_support_weight=effective_support_weight.detach(),
                    prompt_g_domain_valid=prompt_domain_valid.detach(),
                    prompt_g_support_ratio_vs_speech=(
                        effective_support_count / speech_count
                    ).detach(),
                    prompt_g_support_ratio_vs_valid=(
                        effective_support_count / valid_count
                    ).detach(),
                    prompt_g_clean_ratio_vs_speech=(
                        effective_clean_count / speech_count
                    ).detach(),
                    prompt_g_clean_ratio_vs_valid=(
                        effective_clean_count / valid_count
                    ).detach(),
                    prompt_g_speech_ratio_weighted=speech_ratio.detach(),
                    prompt_g_speech_ratio_count=support_stats.speech_ratio_count.detach(),
                    prompt_g_invalid_no_speech=no_speech_rows.float().detach(),
                    prompt_g_invalid_low_speech_ratio=low_speech_ratio_rows.float().detach(),
                    prompt_g_invalid_ref_len=invalid_ref_len_rows.float().detach(),
                    prompt_g_invalid_support=invalid_support_rows.float().detach(),
                    prompt_g_invalid_clean=invalid_clean_rows.float().detach(),
                    prompt_g_invalid_missing_closed=missing_closed_sidecar_rows.float().detach(),
                    prompt_g_invalid_missing_boundary=missing_boundary_sidecar_rows.float().detach(),
                ),
            )
        )

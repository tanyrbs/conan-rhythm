from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .g_stats import (
    build_global_rate_support_mask,
    compute_global_rate,
    normalize_global_rate_variant,
)
from .contracts import (
    PromptConditioningEvidence,
    ReferenceDurationMemory,
    StructuredProgressDurationMemory,
    StructuredDetectorDurationMemory,
    StructuredDurationOperatorMemory,
    StructuredRoleDurationMemory,
    validate_reference_duration_memory,
)

_V3_MEMORY_REQUIRED_FIELDS = ("global_rate",)
_V3_MEMORY_OPTIONAL_FIELDS = (
    "operator_coeff",
    "progress_profile",
    "detector_coeff",
    "role_value",
    "role_var",
    "role_coverage",
    "prompt_valid_mask",
    "prompt_speech_mask",
    "prompt_basis_activation",
    "prompt_operator_fit",
    "prompt_operator_cv_fit",
    "prompt_random_target",
    "prompt_mask",
    "prompt_fit_mask",
    "prompt_eval_mask",
    "prompt_log_base",
    "prompt_log_duration",
    "prompt_log_residual",
    "prompt_progress_fit",
    "prompt_operator_support",
    "prompt_operator_condition_number",
    "prompt_short_fallback",
    "prompt_operator_coeff_norm",
    "prompt_detector_fit",
    "prompt_detector_support",
    "prompt_detector_condition_number",
    "prompt_detector_coeff_norm",
    "prompt_role_attn",
    "prompt_role_fit",
)
_V3_PROMPT_UNIT_REQUIRED_FIELDS = (
    "prompt_content_units",
    "prompt_duration_obs",
)
_V3_PROMPT_UNIT_OPTIONAL_FIELDS = (
    "prompt_unit_anchor_base",
    "prompt_log_base",
    "prompt_unit_mask",
    "prompt_valid_mask",
    "prompt_speech_mask",
    "prompt_global_weight",
    "prompt_unit_log_prior",
    "prompt_spk_embed",
    "prompt_source_boundary_cue",
    "prompt_phrase_group_pos",
    "prompt_phrase_final_mask",
)
_V3_NESTED_PROMPT_FIELDS = (
    "prompt_basis_activation",
    "prompt_operator_fit",
    "prompt_operator_cv_fit",
    "prompt_random_target",
    "prompt_mask",
    "prompt_fit_mask",
    "prompt_eval_mask",
    "prompt_log_base",
    "prompt_log_duration",
    "prompt_log_residual",
    "prompt_progress_fit",
    "prompt_operator_support",
    "prompt_operator_condition_number",
    "prompt_short_fallback",
    "prompt_operator_coeff_norm",
    "prompt_detector_fit",
    "prompt_detector_support",
    "prompt_detector_condition_number",
    "prompt_detector_coeff_norm",
    "prompt_role_attn",
    "prompt_role_fit",
)

_DETECTOR_BANK_DIM = 4

_V3_REMOVED_MEMORY_ALIASES = {
    "coarse_profile": "progress_profile",
    "prompt_coarse_fit": "prompt_progress_fit",
    "coarse": "progress",
}


@dataclass(frozen=True)
class PromptOperatorSummary:
    prompt_mask: torch.Tensor
    prompt_valid_mask: torch.Tensor
    prompt_speech_mask: torch.Tensor
    prompt_log_base: torch.Tensor
    prompt_log_duration: torch.Tensor | None
    prompt_log_residual: torch.Tensor
    prompt_random_target: torch.Tensor
    global_rate: torch.Tensor


@dataclass(frozen=True)
class PromptOperatorDiagnostics:
    fit_mask: torch.Tensor
    eval_mask: torch.Tensor
    operator_cv_fit: torch.Tensor
    operator_support: torch.Tensor
    operator_condition_number: torch.Tensor
    short_fallback: torch.Tensor
    operator_coeff_norm: torch.Tensor


@dataclass(frozen=True)
class PromptDetectorDiagnostics:
    detector_fit: torch.Tensor
    detector_support: torch.Tensor
    detector_condition_number: torch.Tensor
    detector_coeff_norm: torch.Tensor


def _reject_removed_v3_memory_aliases(source: Mapping[str, Any]) -> None:
    for removed_key, replacement in _V3_REMOVED_MEMORY_ALIASES.items():
        if removed_key in source:
            raise ValueError(
                f"{removed_key} has been removed from rhythm_v3 conditioning. Use {replacement} instead."
            )


def _collect_flat_v3_memory(source: Mapping[str, Any]) -> dict[str, Any] | None:
    _reject_removed_v3_memory_aliases(source)
    if not all(key in source for key in _V3_MEMORY_REQUIRED_FIELDS):
        return None
    normalized = {key: source[key] for key in _V3_MEMORY_REQUIRED_FIELDS}
    for key in _V3_MEMORY_OPTIONAL_FIELDS:
        value = source.get(key)
        if value is not None:
            normalized[key] = value
    return normalized


def _lookup_mapping_or_attr(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _collect_nested_v3_memory(source: Mapping[str, Any]) -> dict[str, Any] | None:
    _reject_removed_v3_memory_aliases(source)
    global_rate = source.get("global_rate")
    operator = source.get("operator")
    operator_coeff = _lookup_mapping_or_attr(operator, "operator_coeff")
    detector = source.get("detector")
    detector_coeff = _lookup_mapping_or_attr(detector, "detector_coeff")
    if global_rate is None:
        return None
    normalized = {
        "global_rate": global_rate,
    }
    if operator_coeff is not None:
        normalized["operator_coeff"] = operator_coeff
    if detector_coeff is not None:
        normalized["detector_coeff"] = detector_coeff
    for key in ("prompt_valid_mask", "prompt_speech_mask"):
        value = source.get(key)
        if value is not None:
            normalized[key] = value
    progress = source.get("progress")
    progress_profile = _lookup_mapping_or_attr(progress, "progress_profile")
    if progress_profile is not None:
        normalized["progress_profile"] = progress_profile
    prompt = source.get("prompt")
    if prompt is None:
        return normalized
    if isinstance(prompt, Mapping):
        _reject_removed_v3_memory_aliases(prompt)
    for key in _V3_NESTED_PROMPT_FIELDS:
        value = _lookup_mapping_or_attr(prompt, key)
        if value is not None:
            normalized[key] = value
    return normalized


def _collect_prompt_unit_conditioning(source: Mapping[str, Any]) -> dict[str, Any] | None:
    if any(key not in source for key in _V3_PROMPT_UNIT_REQUIRED_FIELDS):
        return None
    normalized = {key: source[key] for key in _V3_PROMPT_UNIT_REQUIRED_FIELDS}
    for key in _V3_PROMPT_UNIT_OPTIONAL_FIELDS:
        value = source.get(key)
        if value is not None:
            normalized[key] = value
    return normalized


def normalize_duration_v3_conditioning(
    source: Any,
) -> ReferenceDurationMemory | dict[str, Any] | None:
    if source is None or isinstance(source, ReferenceDurationMemory):
        return source
    if not isinstance(source, Mapping):
        raise TypeError(f"Unsupported reference conditioning type: {type(source)!r}")

    prompt_units = _collect_prompt_unit_conditioning(source)
    normalized_v3 = _collect_flat_v3_memory(source)
    normalized_nested_v3 = _collect_nested_v3_memory(source)
    if prompt_units is not None:
        if normalized_v3 is not None or normalized_nested_v3 is not None:
            raise ValueError(
                "Ambiguous rhythm_v3 conditioning: prompt-unit evidence cannot be mixed with "
                "prebuilt operator memory."
            )
        return prompt_units

    if normalized_v3 is not None:
        return normalized_v3

    if normalized_nested_v3 is not None:
        return normalized_nested_v3

    nested = source.get("rhythm_ref_conditioning")
    if nested is not None and nested is not source:
        normalized_nested = normalize_duration_v3_conditioning(nested)
        if normalized_nested is not None:
            return normalized_nested
    return None


class PromptConditionedOperatorEstimator(nn.Module):
    def __init__(
        self,
        *,
        progress_bins: int = 4,
        ridge_lambda: float = 1.0,
        global_shrink_tau: float = 8.0,
        progress_support_tau: float = 8.0,
        ridge_support_tau: float = 8.0,
        holdout_ratio: float = 0.30,
        min_operator_support_factor: float = 1.0,
        simple_global_stats: bool = False,
        use_log_base_rate: bool = False,
        g_variant: str = "raw_median",
        g_trim_ratio: float = 0.2,
        drop_edge_runs_for_g: int = 0,
    ) -> None:
        super().__init__()
        self.progress_bins = int(max(1, progress_bins))
        self.ridge_lambda = float(max(1.0e-4, ridge_lambda))
        self.global_shrink_tau = float(max(0.0, global_shrink_tau))
        self.progress_support_tau = float(max(0.0, progress_support_tau))
        self.ridge_support_tau = float(max(0.0, ridge_support_tau))
        self.holdout_ratio = float(max(0.0, min(0.95, holdout_ratio)))
        self.min_operator_support_factor = float(max(0.0, min_operator_support_factor))
        self.simple_global_stats = bool(simple_global_stats)
        self.rate_mode = "simple_global" if self.simple_global_stats else "log_base"
        self.use_log_base_rate = bool(use_log_base_rate) and not self.simple_global_stats
        self.g_variant = normalize_global_rate_variant(g_variant)
        self.g_trim_ratio = float(max(0.0, min(0.49, g_trim_ratio)))
        self.g_drop_edge_runs = max(0, int(drop_edge_runs_for_g))

    @staticmethod
    def _masked_median(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = values.size(0)
        median = values.new_zeros((batch_size, 1))
        for batch_idx in range(batch_size):
            valid = mask[batch_idx] > 0.5
            if bool(valid.any().item()):
                median[batch_idx, 0] = values[batch_idx][valid].median()
        return median

    @staticmethod
    def _masked_count(mask: torch.Tensor) -> torch.Tensor:
        return mask.float().sum(dim=1, keepdim=True)

    @staticmethod
    def _support_shrink(value: torch.Tensor, support: torch.Tensor, tau: float) -> torch.Tensor:
        if tau <= 0.0:
            return value
        support = support.float().clamp_min(0.0)
        conf = support / (support + float(tau))
        return value * conf

    @staticmethod
    def _build_progress_from_log_base(
        *,
        prompt_log_base: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = prompt_mask.float().clamp(0.0, 1.0)
        if prompt_log_base.numel() <= 0:
            return prompt_log_base.new_zeros(prompt_log_base.shape)
        mass = torch.exp(prompt_log_base.float()).clamp_min(1.0e-6) * mask
        total_mass = mass.sum(dim=1, keepdim=True)
        fallback_mass = mask
        use_fallback = total_mass <= 1.0e-6
        mass = torch.where(use_fallback, fallback_mass, mass)
        total_mass = mass.sum(dim=1, keepdim=True).clamp_min(1.0)
        centered_cum = torch.cumsum(mass, dim=1) - (0.5 * mass)
        progress = (centered_cum / total_mass).clamp(0.0, 1.0)
        return progress * mask

    @staticmethod
    def _masked_center_scale(values: torch.Tensor, mask: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
        mask = mask.float()
        total = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (values * mask).sum(dim=1, keepdim=True) / total
        centered = (values - mean) * mask
        var = ((centered ** 2) * mask).sum(dim=1, keepdim=True) / total
        scale = var.clamp_min(eps).sqrt()
        return centered / scale

    def _build_detector_design(
        self,
        *,
        prompt_log_base: torch.Tensor,
        prompt_mask: torch.Tensor,
        boundary_cue: torch.Tensor | None = None,
        phrase_group_pos: torch.Tensor | None = None,
        phrase_final_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = prompt_mask.float().clamp(0.0, 1.0)
        progress = self._build_progress_from_log_base(
            prompt_log_base=prompt_log_base,
            prompt_mask=prompt_mask,
        )
        if isinstance(boundary_cue, torch.Tensor):
            boundary = boundary_cue.float() * mask
        else:
            boundary = torch.zeros_like(progress)
        if isinstance(phrase_group_pos, torch.Tensor):
            phrase_pos = phrase_group_pos.float() * mask
        else:
            phrase_pos = progress
        if isinstance(phrase_final_mask, torch.Tensor):
            phrase_final = phrase_final_mask.float() * mask
        else:
            phrase_final = torch.zeros_like(progress)
        features = torch.stack(
            [
                self._masked_center_scale(2.0 * progress - 1.0, mask),
                self._masked_center_scale(boundary, mask),
                self._masked_center_scale(2.0 * phrase_pos - 1.0, mask),
                self._masked_center_scale(phrase_final, mask),
            ],
            dim=-1,
        )
        return features * mask.unsqueeze(-1)

    def _build_prompt_progress_components(
        self,
        *,
        prompt_random_target: torch.Tensor,
        prompt_log_base: torch.Tensor,
        prompt_mask: torch.Tensor,
        need_progress: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        prompt_mask = prompt_mask.float().clamp(0.0, 1.0)
        zero_fit = torch.zeros_like(prompt_random_target.float()) * prompt_mask
        if not need_progress or prompt_random_target.size(1) <= 0:
            return None, zero_fit
        batch_size, num_units = prompt_random_target.shape
        progress_profile = prompt_random_target.new_zeros((batch_size, self.progress_bins))
        progress_fit = prompt_random_target.new_zeros((batch_size, num_units))
        progress = self._build_progress_from_log_base(
            prompt_log_base=prompt_log_base,
            prompt_mask=prompt_mask,
        )
        for batch_idx in range(batch_size):
            visible = prompt_mask[batch_idx] > 0.5
            if not bool(visible.any().item()):
                continue
            progress_b = progress[batch_idx]
            target_b = prompt_random_target[batch_idx].float()
            for bin_idx in range(self.progress_bins):
                lo = float(bin_idx) / float(self.progress_bins)
                hi = float(bin_idx + 1) / float(self.progress_bins)
                in_bin = visible & (progress_b >= lo)
                if bin_idx + 1 < self.progress_bins:
                    in_bin = in_bin & (progress_b < hi)
                if not bool(in_bin.any().item()):
                    continue
                stat = target_b[in_bin].median()
                support = target_b.new_tensor([[float(in_bin.float().sum().item())]])
                shrunk = self._support_shrink(
                    stat.reshape(1, 1),
                    support,
                    self.progress_support_tau,
                )[0, 0]
                progress_profile[batch_idx, bin_idx] = shrunk
                progress_fit[batch_idx, in_bin] = shrunk
        return progress_profile, progress_fit * prompt_mask

    @staticmethod
    def _zero_operator_coeff(
        *,
        batch_size: int,
        response_encoder,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        operator_rank = int(getattr(getattr(response_encoder, "out_proj", None), "out_features", 0))
        return torch.zeros((batch_size, operator_rank), device=device, dtype=dtype)

    @staticmethod
    def _zero_detector_coeff(
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros((batch_size, _DETECTOR_BANK_DIM), device=device, dtype=dtype)

    @staticmethod
    def _resolve_prompt_mask(
        *,
        prompt_duration_obs: torch.Tensor,
        prompt_unit_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(prompt_unit_mask, torch.Tensor):
            return prompt_unit_mask.float().clamp(0.0, 1.0)
        return prompt_duration_obs.float().gt(0.0).float()

    @staticmethod
    def _resolve_prompt_log_base(
        *,
        prompt_unit_anchor_base: torch.Tensor | None,
        prompt_log_base: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(prompt_log_base, torch.Tensor):
            return prompt_log_base.float().detach()
        if not isinstance(prompt_unit_anchor_base, torch.Tensor):
            raise ValueError("Prompt unit conditioning requires prompt_unit_anchor_base or prompt_log_base.")
        return torch.log(prompt_unit_anchor_base.float().detach().clamp_min(1.0e-6))

    def _build_prompt_summary_from_units(
        self,
        *,
        prompt_content_units: torch.Tensor,
        prompt_duration_obs: torch.Tensor,
        prompt_unit_anchor_base: torch.Tensor | None,
        prompt_log_base: torch.Tensor | None,
        prompt_unit_mask: torch.Tensor | None,
        prompt_valid_mask: torch.Tensor | None,
        prompt_speech_mask: torch.Tensor | None = None,
        prompt_global_weight: torch.Tensor | None = None,
        prompt_unit_log_prior: torch.Tensor | None = None,
    ) -> PromptOperatorSummary:
        prompt_mask = (
            prompt_valid_mask.float().clamp(0.0, 1.0)
            if isinstance(prompt_valid_mask, torch.Tensor)
            else self._resolve_prompt_mask(
                prompt_duration_obs=prompt_duration_obs,
                prompt_unit_mask=prompt_unit_mask,
            )
        )
        if self.use_log_base_rate:
            resolved_log_base = self._resolve_prompt_log_base(
                prompt_unit_anchor_base=prompt_unit_anchor_base,
                prompt_log_base=prompt_log_base,
            )
        else:
            resolved_log_base = torch.zeros_like(prompt_duration_obs.float())
        prompt_log_duration = torch.log(prompt_duration_obs.float().clamp_min(1.0e-6)).detach() * prompt_mask
        prompt_log_base = resolved_log_base.float().detach() * prompt_mask
        if self.use_log_base_rate:
            prompt_log_residual = (prompt_log_duration - prompt_log_base).detach() * prompt_mask
        else:
            prompt_log_residual = prompt_log_duration
        speech_mask = (
            prompt_speech_mask.float().clamp(0.0, 1.0) * prompt_mask
            if isinstance(prompt_speech_mask, torch.Tensor)
            else prompt_mask
        )
        support_mask = build_global_rate_support_mask(
            speech_mask=speech_mask,
            valid_mask=prompt_mask,
            drop_edge_runs=self.g_drop_edge_runs,
        )
        global_rate = compute_global_rate(
            log_dur=prompt_log_residual,
            speech_mask=speech_mask,
            valid_mask=prompt_mask,
            variant=self.g_variant,
            weight=prompt_global_weight,
            trim_ratio=self.g_trim_ratio,
            drop_edge_runs=self.g_drop_edge_runs,
            unit_ids=prompt_content_units,
            unit_prior=prompt_unit_log_prior,
            support_mask=support_mask,
        )
        global_rate = self._support_shrink(
            global_rate,
            self._masked_count(support_mask),
            self.global_shrink_tau,
        ).detach()
        prompt_random_target = (prompt_log_residual - global_rate) * prompt_mask
        return PromptOperatorSummary(
            prompt_mask=prompt_mask,
            prompt_valid_mask=prompt_mask,
            prompt_speech_mask=speech_mask,
            prompt_log_base=prompt_log_base,
            prompt_log_duration=prompt_log_duration,
            prompt_log_residual=prompt_log_residual,
            prompt_random_target=prompt_random_target,
            global_rate=global_rate,
        )

    def _solve_detector_coeff(
        self,
        *,
        detector_design: torch.Tensor,
        prompt_random_target: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, detector_dim = detector_design.shape
        if detector_dim <= 0:
            return detector_design.new_zeros((batch_size, 0))
        mask = prompt_mask.float().unsqueeze(-1)
        design = detector_design.float() * mask
        lhs = torch.matmul(design.transpose(1, 2), design)
        eye = torch.eye(detector_dim, device=design.device, dtype=design.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        support = prompt_mask.float().sum(dim=1).clamp_min(1.0)
        ridge = self.ridge_lambda * (1.0 + (self.ridge_support_tau / support))
        lhs = lhs + ridge[:, None, None] * eye
        rhs = torch.matmul(design.transpose(1, 2), prompt_random_target.float().unsqueeze(-1))
        return torch.linalg.solve(lhs, rhs).squeeze(-1)

    def _resolve_detector_enable_mask(
        self,
        *,
        detector_design: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, detector_dim = detector_design.shape
        if detector_dim <= 0:
            support = torch.zeros((batch_size, 1), device=prompt_mask.device, dtype=prompt_mask.dtype)
            enabled = torch.zeros_like(support, dtype=torch.bool)
            return support, enabled
        support = prompt_mask.float().sum(dim=1, keepdim=True)
        min_support = max(1, int(round(self.min_operator_support_factor * float(detector_dim))))
        enabled = support >= float(min_support)
        return support, enabled

    def _estimate_detector_condition_number(
        self,
        *,
        detector_design: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, detector_dim = detector_design.shape
        cond = detector_design.new_zeros((batch_size, 1), dtype=torch.float32)
        if detector_dim <= 0:
            return cond
        mask = prompt_mask.float().unsqueeze(-1)
        design = detector_design.float() * mask
        gram = torch.matmul(design.transpose(1, 2), design)
        eye = torch.eye(detector_dim, device=design.device, dtype=design.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        support = prompt_mask.float().sum(dim=1).clamp_min(1.0)
        ridge = self.ridge_lambda * (1.0 + (self.ridge_support_tau / support))
        gram = gram + ridge[:, None, None] * eye
        for batch_idx in range(batch_size):
            try:
                cond_value = torch.linalg.cond(gram[batch_idx])
            except RuntimeError:
                cond_value = gram.new_tensor(1.0e6)
            if not torch.isfinite(cond_value):
                cond_value = gram.new_tensor(1.0e6)
            cond[batch_idx, 0] = cond_value.float()
        return cond

    def _solve_operator_coeff(
        self,
        *,
        prompt_basis_activation: torch.Tensor,
        prompt_random_target: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, basis_rank = prompt_basis_activation.shape
        if basis_rank <= 0:
            return prompt_basis_activation.new_zeros((batch_size, 0))
        mask = prompt_mask.float().unsqueeze(-1)
        basis = prompt_basis_activation.float() * mask
        lhs = torch.matmul(basis.transpose(1, 2), basis)
        eye = torch.eye(basis_rank, device=basis.device, dtype=basis.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        support = prompt_mask.float().sum(dim=1).clamp_min(1.0)
        ridge = self.ridge_lambda * (1.0 + (self.ridge_support_tau / support))
        lhs = lhs + ridge[:, None, None] * eye
        rhs = torch.matmul(basis.transpose(1, 2), prompt_random_target.float().unsqueeze(-1))
        coeff = torch.linalg.solve(lhs, rhs).squeeze(-1)
        return coeff

    def _resolve_operator_enable_mask(
        self,
        *,
        prompt_basis_activation: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, basis_rank = prompt_basis_activation.shape
        if basis_rank <= 0:
            support = torch.zeros((batch_size, 1), device=prompt_mask.device, dtype=prompt_mask.dtype)
            enabled = torch.zeros_like(support, dtype=torch.bool)
            return support, enabled
        support = prompt_mask.float().sum(dim=1, keepdim=True)
        min_support = max(1, int(round(self.min_operator_support_factor * float(basis_rank))))
        enabled = support >= float(min_support)
        return support, enabled

    def _estimate_operator_condition_number(
        self,
        *,
        prompt_basis_activation: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, basis_rank = prompt_basis_activation.shape
        cond = prompt_basis_activation.new_zeros((batch_size, 1), dtype=torch.float32)
        if basis_rank <= 0:
            return cond
        mask = prompt_mask.float().unsqueeze(-1)
        basis = prompt_basis_activation.float() * mask
        gram = torch.matmul(basis.transpose(1, 2), basis)
        eye = torch.eye(basis_rank, device=basis.device, dtype=basis.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        support = prompt_mask.float().sum(dim=1).clamp_min(1.0)
        ridge = self.ridge_lambda * (1.0 + (self.ridge_support_tau / support))
        gram = gram + ridge[:, None, None] * eye
        for batch_idx in range(batch_size):
            try:
                cond_value = torch.linalg.cond(gram[batch_idx])
            except RuntimeError:
                cond_value = gram.new_tensor(1.0e6)
            if not torch.isfinite(cond_value):
                cond_value = gram.new_tensor(1.0e6)
            cond[batch_idx, 0] = cond_value.float()
        return cond

    def _build_blocked_holdout_masks(self, prompt_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fit_mask = prompt_mask.float().clone()
        eval_mask = torch.zeros_like(fit_mask)
        if self.holdout_ratio <= 0.0:
            return fit_mask, eval_mask
        for batch_idx in range(prompt_mask.size(0)):
            visible = torch.where(prompt_mask[batch_idx] > 0.5)[0]
            visible_count = int(visible.numel())
            if visible_count < 4:
                continue
            holdout = max(1, int(round(visible_count * self.holdout_ratio)))
            holdout = min(max(1, visible_count - 1), holdout)
            start = max(0, (visible_count - holdout) // 2)
            chosen = visible[start : start + holdout]
            fit_mask[batch_idx, chosen] = 0.0
            eval_mask[batch_idx, chosen] = 1.0
        return fit_mask, eval_mask

    def _build_prompt_operator_diagnostics(
        self,
        *,
        prompt_basis_activation: torch.Tensor,
        prompt_random_target: torch.Tensor,
        prompt_mask: torch.Tensor,
        operator_coeff: torch.Tensor,
        operator_enabled: torch.Tensor,
    ) -> PromptOperatorDiagnostics:
        fit_mask, eval_mask = self._build_blocked_holdout_masks(prompt_mask)
        if bool(eval_mask.any().item()):
            coeff = self._solve_operator_coeff(
                prompt_basis_activation=prompt_basis_activation,
                prompt_random_target=prompt_random_target,
                prompt_mask=fit_mask,
            )
            coeff = torch.where(operator_enabled, coeff, torch.zeros_like(coeff))
            operator_cv_fit = self._build_prompt_operator_fit(
                prompt_basis_activation=prompt_basis_activation,
                operator_coeff=coeff,
                prompt_mask=prompt_mask,
            )
        else:
            operator_cv_fit = self._build_prompt_operator_fit(
                prompt_basis_activation=prompt_basis_activation,
                operator_coeff=self._solve_operator_coeff(
                    prompt_basis_activation=prompt_basis_activation,
                    prompt_random_target=prompt_random_target,
                    prompt_mask=prompt_mask,
                ),
                prompt_mask=prompt_mask,
            )
        operator_support, _ = self._resolve_operator_enable_mask(
            prompt_basis_activation=prompt_basis_activation,
            prompt_mask=prompt_mask,
        )
        operator_condition_number = self._estimate_operator_condition_number(
            prompt_basis_activation=prompt_basis_activation,
            prompt_mask=prompt_mask,
        )
        return PromptOperatorDiagnostics(
            fit_mask=fit_mask,
            eval_mask=eval_mask,
            operator_cv_fit=operator_cv_fit,
            operator_support=operator_support.float(),
            operator_condition_number=operator_condition_number.float(),
            short_fallback=(~operator_enabled).float(),
            operator_coeff_norm=operator_coeff.float().norm(dim=1, keepdim=True),
        )

    @staticmethod
    def _build_prompt_operator_fit(
        *,
        prompt_basis_activation: torch.Tensor,
        operator_coeff: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        operator_fit = torch.einsum("bnk,bk->bn", prompt_basis_activation.float(), operator_coeff.float())
        return operator_fit * prompt_mask.float()

    @staticmethod
    def _build_prompt_detector_fit(
        *,
        detector_design: torch.Tensor,
        detector_coeff: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        detector_fit = torch.einsum("bnd,bd->bn", detector_design.float(), detector_coeff.float())
        return detector_fit * prompt_mask.float()

    def _build_prompt_detector_diagnostics(
        self,
        *,
        detector_design: torch.Tensor,
        prompt_mask: torch.Tensor,
        detector_coeff: torch.Tensor,
    ) -> PromptDetectorDiagnostics:
        detector_support, _ = self._resolve_detector_enable_mask(
            detector_design=detector_design,
            prompt_mask=prompt_mask,
        )
        detector_condition_number = self._estimate_detector_condition_number(
            detector_design=detector_design,
            prompt_mask=prompt_mask,
        )
        detector_fit = self._build_prompt_detector_fit(
            detector_design=detector_design,
            detector_coeff=detector_coeff,
            prompt_mask=prompt_mask,
        )
        return PromptDetectorDiagnostics(
            detector_fit=detector_fit,
            detector_support=detector_support.float(),
            detector_condition_number=detector_condition_number.float(),
            detector_coeff_norm=detector_coeff.float().norm(dim=1, keepdim=True),
        )

    def build_from_prompt_units(
        self,
        *,
        prompt_content_units: torch.Tensor,
        prompt_duration_obs: torch.Tensor,
        prompt_unit_anchor_base: torch.Tensor | None,
        prompt_log_base: torch.Tensor | None,
        prompt_unit_mask: torch.Tensor | None,
        prompt_valid_mask: torch.Tensor | None,
        prompt_speech_mask: torch.Tensor | None,
        prompt_global_weight: torch.Tensor | None,
        prompt_unit_log_prior: torch.Tensor | None,
        prompt_source_boundary_cue: torch.Tensor | None,
        prompt_phrase_group_pos: torch.Tensor | None,
        prompt_phrase_final_mask: torch.Tensor | None,
        response_encoder,
        need_progress: bool = True,
        need_detector: bool = False,
        need_operator: bool = True,
    ) -> ReferenceDurationMemory:
        summary = self._build_prompt_summary_from_units(
            prompt_content_units=prompt_content_units,
            prompt_duration_obs=prompt_duration_obs,
            prompt_unit_anchor_base=prompt_unit_anchor_base,
            prompt_log_base=prompt_log_base,
            prompt_unit_mask=prompt_unit_mask,
            prompt_valid_mask=prompt_valid_mask,
            prompt_speech_mask=prompt_speech_mask,
            prompt_global_weight=prompt_global_weight,
            prompt_unit_log_prior=prompt_unit_log_prior,
        )
        progress_profile, prompt_progress_fit = self._build_prompt_progress_components(
            prompt_random_target=summary.prompt_random_target,
            prompt_log_base=summary.prompt_log_base,
            prompt_mask=summary.prompt_mask,
            need_progress=need_progress,
        )
        prompt_local_target = (summary.prompt_random_target - prompt_progress_fit) * summary.prompt_mask
        prompt_basis_activation = None
        operator_coeff = self._zero_operator_coeff(
            batch_size=summary.prompt_mask.size(0),
            response_encoder=response_encoder,
            device=summary.prompt_mask.device,
            dtype=summary.prompt_mask.dtype,
        )
        detector_coeff = self._zero_detector_coeff(
            batch_size=summary.prompt_mask.size(0),
            device=summary.prompt_mask.device,
            dtype=summary.prompt_mask.dtype,
        )
        diagnostics = None
        detector_diagnostics = None
        prompt_operator_fit = None
        if need_detector:
            detector_design = self._build_detector_design(
                prompt_log_base=summary.prompt_log_base,
                prompt_mask=summary.prompt_mask,
                boundary_cue=prompt_source_boundary_cue,
                phrase_group_pos=prompt_phrase_group_pos,
                phrase_final_mask=prompt_phrase_final_mask,
            )
            detector_support, detector_enabled = self._resolve_detector_enable_mask(
                detector_design=detector_design,
                prompt_mask=summary.prompt_mask,
            )
            detector_coeff = self._solve_detector_coeff(
                detector_design=detector_design,
                prompt_random_target=summary.prompt_random_target,
                prompt_mask=summary.prompt_mask,
            )
            detector_coeff = torch.where(detector_enabled, detector_coeff, torch.zeros_like(detector_coeff))
            detector_diagnostics = self._build_prompt_detector_diagnostics(
                detector_design=detector_design,
                prompt_mask=summary.prompt_mask,
                detector_coeff=detector_coeff,
            )
            prompt_local_target = (summary.prompt_random_target - detector_diagnostics.detector_fit) * summary.prompt_mask
        if need_operator:
            prompt_basis_activation = response_encoder.encode_prompt_units(
                content_units=prompt_content_units.long(),
                log_anchor_base=summary.prompt_log_base.detach(),
                prompt_mask=summary.prompt_mask,
            )
            operator_support, operator_enabled = self._resolve_operator_enable_mask(
                prompt_basis_activation=prompt_basis_activation,
                prompt_mask=summary.prompt_mask,
            )
            operator_coeff = self._solve_operator_coeff(
                prompt_basis_activation=prompt_basis_activation,
                prompt_random_target=prompt_local_target,
                prompt_mask=summary.prompt_mask,
            )
            operator_coeff = torch.where(operator_enabled, operator_coeff, torch.zeros_like(operator_coeff))
            diagnostics = self._build_prompt_operator_diagnostics(
                prompt_basis_activation=prompt_basis_activation,
                prompt_random_target=prompt_local_target,
                prompt_mask=summary.prompt_mask,
                operator_coeff=operator_coeff,
                operator_enabled=operator_enabled,
            )
            prompt_operator_fit = self._build_prompt_operator_fit(
                prompt_basis_activation=prompt_basis_activation,
                operator_coeff=operator_coeff,
                prompt_mask=summary.prompt_mask,
            )
        return validate_reference_duration_memory(
            ReferenceDurationMemory(
                global_rate=summary.global_rate,
                operator=StructuredDurationOperatorMemory(operator_coeff=operator_coeff),
                progress=(
                    None
                    if progress_profile is None
                    else StructuredProgressDurationMemory(progress_profile=progress_profile)
                ),
                detector=(
                    None
                    if not need_detector
                    else StructuredDetectorDurationMemory(detector_coeff=detector_coeff)
                ),
                prompt=PromptConditioningEvidence(
                    prompt_basis_activation=prompt_basis_activation,
                    prompt_random_target=prompt_local_target,
                    prompt_mask=summary.prompt_mask,
                    prompt_fit_mask=None if diagnostics is None else diagnostics.fit_mask,
                    prompt_eval_mask=None if diagnostics is None else diagnostics.eval_mask,
                    prompt_operator_fit=prompt_operator_fit,
                    prompt_operator_cv_fit=None if diagnostics is None else diagnostics.operator_cv_fit,
                    prompt_log_base=summary.prompt_log_base,
                    prompt_log_duration=summary.prompt_log_duration,
                    prompt_log_residual=summary.prompt_log_residual,
                    prompt_progress_fit=prompt_progress_fit,
                    prompt_operator_support=None if diagnostics is None else diagnostics.operator_support,
                    prompt_operator_condition_number=(
                        None if diagnostics is None else diagnostics.operator_condition_number
                    ),
                    prompt_short_fallback=None if diagnostics is None else diagnostics.short_fallback,
                    prompt_operator_coeff_norm=None if diagnostics is None else diagnostics.operator_coeff_norm,
                    prompt_detector_fit=None if detector_diagnostics is None else detector_diagnostics.detector_fit,
                    prompt_detector_support=(
                        None if detector_diagnostics is None else detector_diagnostics.detector_support
                    ),
                    prompt_detector_condition_number=(
                        None if detector_diagnostics is None else detector_diagnostics.detector_condition_number
                    ),
                    prompt_detector_coeff_norm=(
                        None if detector_diagnostics is None else detector_diagnostics.detector_coeff_norm
                    ),
                ),
                prompt_valid_mask=summary.prompt_valid_mask,
                prompt_speech_mask=summary.prompt_speech_mask,
            )
        )

    def _build_memory_from_flat_conditioning(
        self,
        *,
        ref_conditioning: Mapping[str, Any],
        response_encoder,
    ) -> ReferenceDurationMemory:
        def _detach_float(value):
            if not isinstance(value, torch.Tensor):
                return value
            return value.float().detach()

        prompt_kwargs = {
            "prompt_basis_activation": _detach_float(ref_conditioning.get("prompt_basis_activation")),
            "prompt_random_target": _detach_float(ref_conditioning.get("prompt_random_target")),
            "prompt_mask": _detach_float(ref_conditioning.get("prompt_mask")),
            "prompt_fit_mask": _detach_float(ref_conditioning.get("prompt_fit_mask")),
            "prompt_eval_mask": _detach_float(ref_conditioning.get("prompt_eval_mask")),
            "prompt_operator_fit": _detach_float(ref_conditioning.get("prompt_operator_fit")),
            "prompt_operator_cv_fit": _detach_float(ref_conditioning.get("prompt_operator_cv_fit")),
            "prompt_log_base": _detach_float(ref_conditioning.get("prompt_log_base")),
            "prompt_log_duration": _detach_float(ref_conditioning.get("prompt_log_duration")),
            "prompt_log_residual": _detach_float(ref_conditioning.get("prompt_log_residual")),
            "prompt_progress_fit": _detach_float(ref_conditioning.get("prompt_progress_fit")),
            "prompt_operator_support": _detach_float(ref_conditioning.get("prompt_operator_support")),
            "prompt_operator_condition_number": _detach_float(ref_conditioning.get("prompt_operator_condition_number")),
            "prompt_short_fallback": _detach_float(ref_conditioning.get("prompt_short_fallback")),
            "prompt_operator_coeff_norm": _detach_float(ref_conditioning.get("prompt_operator_coeff_norm")),
            "prompt_detector_fit": _detach_float(ref_conditioning.get("prompt_detector_fit")),
            "prompt_detector_support": _detach_float(ref_conditioning.get("prompt_detector_support")),
            "prompt_detector_condition_number": _detach_float(ref_conditioning.get("prompt_detector_condition_number")),
            "prompt_detector_coeff_norm": _detach_float(ref_conditioning.get("prompt_detector_coeff_norm")),
            "prompt_role_attn": _detach_float(ref_conditioning.get("prompt_role_attn")),
            "prompt_role_fit": _detach_float(ref_conditioning.get("prompt_role_fit")),
        }
        prompt = (
            None
            if all(value is None for value in prompt_kwargs.values())
            else PromptConditioningEvidence(**prompt_kwargs)
        )
        return validate_reference_duration_memory(
            ReferenceDurationMemory(
                global_rate=ref_conditioning["global_rate"].float().detach(),
                operator=StructuredDurationOperatorMemory(
                    operator_coeff=(
                        ref_conditioning["operator_coeff"].float().detach()
                        if isinstance(ref_conditioning.get("operator_coeff"), torch.Tensor)
                        else self._zero_operator_coeff(
                            batch_size=int(ref_conditioning["global_rate"].size(0)),
                            response_encoder=response_encoder,
                            device=ref_conditioning["global_rate"].device,
                            dtype=ref_conditioning["global_rate"].dtype,
                        )
                    ),
                ),
                progress=(
                    None
                    if ref_conditioning.get("progress_profile") is None
                    else StructuredProgressDurationMemory(
                        progress_profile=ref_conditioning["progress_profile"].float().detach(),
                    )
                ),
                detector=(
                    None
                    if ref_conditioning.get("detector_coeff") is None
                    else StructuredDetectorDurationMemory(
                        detector_coeff=ref_conditioning["detector_coeff"].float().detach(),
                    )
                ),
                role=(
                    None
                    if any(ref_conditioning.get(key) is None for key in ("role_value", "role_var", "role_coverage"))
                    else StructuredRoleDurationMemory(
                        role_value=ref_conditioning["role_value"].float().detach(),
                        role_var=ref_conditioning["role_var"].float().detach(),
                        role_coverage=ref_conditioning["role_coverage"].float().detach(),
                    )
                ),
                prompt=prompt,
                prompt_valid_mask=_detach_float(ref_conditioning.get("prompt_valid_mask")),
                prompt_speech_mask=_detach_float(ref_conditioning.get("prompt_speech_mask")),
            )
        )

    def from_conditioning(
        self,
        ref_conditioning,
        *,
        response_encoder,
        need_progress: bool = True,
        need_detector: bool = False,
        need_operator: bool = True,
    ) -> ReferenceDurationMemory:
        normalized = normalize_duration_v3_conditioning(ref_conditioning)
        if isinstance(normalized, ReferenceDurationMemory):
            return normalized
        if normalized is None:
            raise ValueError("Reference conditioning is required for rhythm_v3.")
        if all(key in normalized for key in _V3_MEMORY_REQUIRED_FIELDS):
            return self._build_memory_from_flat_conditioning(
                ref_conditioning=normalized,
                response_encoder=response_encoder,
            )
        if all(key in normalized for key in _V3_PROMPT_UNIT_REQUIRED_FIELDS):
            return self.build_from_prompt_units(
                prompt_content_units=normalized["prompt_content_units"],
                prompt_duration_obs=normalized["prompt_duration_obs"],
                prompt_unit_anchor_base=normalized.get("prompt_unit_anchor_base"),
                prompt_log_base=normalized.get("prompt_log_base"),
                prompt_unit_mask=normalized.get("prompt_unit_mask"),
                prompt_valid_mask=normalized.get("prompt_valid_mask"),
                prompt_speech_mask=normalized.get("prompt_speech_mask"),
                prompt_global_weight=normalized.get("prompt_global_weight"),
                prompt_unit_log_prior=normalized.get("prompt_unit_log_prior"),
                prompt_source_boundary_cue=normalized.get("prompt_source_boundary_cue"),
                prompt_phrase_group_pos=normalized.get("prompt_phrase_group_pos"),
                prompt_phrase_final_mask=normalized.get("prompt_phrase_final_mask"),
                response_encoder=response_encoder,
                need_progress=need_progress,
                need_detector=need_detector,
                need_operator=need_operator,
            )
        raise ValueError(
            "Reference conditioning must provide prebuilt duration memory or explicit prompt units."
        )

    def forward(
        self,
        *,
        response_encoder,
        ref_conditioning=None,
        need_progress: bool = True,
        need_detector: bool = False,
        need_operator: bool = True,
    ) -> ReferenceDurationMemory:
        if ref_conditioning is None:
            raise ValueError("rhythm_v3 now requires explicit prompt-unit conditioning or prebuilt reference memory.")
        return self.from_conditioning(
            ref_conditioning,
            response_encoder=response_encoder,
            need_progress=need_progress,
            need_detector=need_detector,
            need_operator=need_operator,
        )


PromptDurationOperatorBuilder = PromptConditionedOperatorEstimator

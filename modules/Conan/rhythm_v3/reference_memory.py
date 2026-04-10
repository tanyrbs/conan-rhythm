from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from modules.Conan.rhythm.reference_encoder import ReferenceRhythmEncoder

from .contracts import (
    PromptConditioningEvidence,
    ReferenceDurationMemory,
    StructuredDurationOperatorMemory,
    validate_reference_duration_memory,
)

_V3_MEMORY_REQUIRED_FIELDS = ("global_rate", "operator_coeff")
_V3_MEMORY_OPTIONAL_FIELDS = (
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
)
_V3_PROMPT_UNIT_REQUIRED_FIELDS = (
    "prompt_content_units",
    "prompt_duration_obs",
)
_V3_PROMPT_UNIT_OPTIONAL_FIELDS = (
    "prompt_unit_anchor_base",
    "prompt_log_base",
    "prompt_unit_mask",
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
)


@dataclass(frozen=True)
class PromptOperatorSummary:
    prompt_mask: torch.Tensor
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


def _collect_flat_v3_memory(source: Mapping[str, Any]) -> dict[str, Any] | None:
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
    global_rate = source.get("global_rate")
    operator = source.get("operator")
    operator_coeff = _lookup_mapping_or_attr(operator, "operator_coeff")
    if global_rate is None or operator_coeff is None:
        return None
    normalized = {
        "global_rate": global_rate,
        "operator_coeff": operator_coeff,
    }
    prompt = source.get("prompt")
    if prompt is None:
        return normalized
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


def normalize_duration_v3_conditioning(source: Any) -> ReferenceDurationMemory | dict[str, Any] | None:
    if source is None or isinstance(source, ReferenceDurationMemory):
        return source
    if not isinstance(source, Mapping):
        raise TypeError(f"Unsupported reference conditioning type: {type(source)!r}")

    nested = source.get("rhythm_ref_conditioning")
    if nested is not None and nested is not source:
        normalized_nested = normalize_duration_v3_conditioning(nested)
        if normalized_nested is not None:
            return normalized_nested

    normalized_v3 = _collect_flat_v3_memory(source)
    if normalized_v3 is not None:
        return normalized_v3

    normalized_nested_v3 = _collect_nested_v3_memory(source)
    if normalized_nested_v3 is not None:
        return normalized_nested_v3

    prompt_units = _collect_prompt_unit_conditioning(source)
    if prompt_units is not None:
        return prompt_units

    ref_stats = source.get("ref_rhythm_stats")
    ref_trace = source.get("ref_rhythm_trace")
    if ref_stats is None or ref_trace is None:
        return None
    normalized = {
        "ref_rhythm_stats": ref_stats,
        "ref_rhythm_trace": ref_trace,
    }
    for key in _V3_MEMORY_OPTIONAL_FIELDS:
        value = source.get(key)
        if value is not None:
            normalized[key] = value
    return normalized


class PromptConditionedOperatorEstimator(nn.Module):
    def __init__(
        self,
        *,
        trace_bins: int = 24,
        ridge_lambda: float = 1.0,
        speech_threshold: float = 0.10,
        global_shrink_tau: float = 8.0,
        ridge_support_tau: float = 8.0,
        holdout_ratio: float = 0.30,
    ) -> None:
        super().__init__()
        self.reference_encoder = ReferenceRhythmEncoder(trace_bins=trace_bins)
        self.ridge_lambda = float(max(1.0e-4, ridge_lambda))
        self.speech_threshold = float(max(0.0, min(1.0, speech_threshold)))
        self.global_shrink_tau = float(max(0.0, global_shrink_tau))
        self.ridge_support_tau = float(max(0.0, ridge_support_tau))
        self.holdout_ratio = float(max(0.0, min(0.95, holdout_ratio)))

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

    def _build_proxy_prompt_mask(self, ref_rhythm_trace: torch.Tensor) -> torch.Tensor:
        if ref_rhythm_trace.size(1) <= 0:
            return ref_rhythm_trace.new_zeros(ref_rhythm_trace.shape[:2])
        pause_indicator = ref_rhythm_trace[..., 0].float().clamp(0.0, 1.0)
        return (1.0 - pause_indicator).clamp(0.0, 1.0)

    @staticmethod
    def derive_prompt_proxy_residual(trace: torch.Tensor) -> torch.Tensor:
        pause = trace[..., 0]
        local_rate = trace[..., 1]
        segment_bias = trace[..., 3]
        voiced = trace[..., 4]
        return 0.60 * segment_bias - 0.30 * local_rate + 0.10 * voiced - 0.10 * pause

    def _build_prompt_summary_from_proxy(
        self,
        *,
        ref_rhythm_trace: torch.Tensor,
    ) -> PromptOperatorSummary:
        trace = ref_rhythm_trace.float()
        prompt_mask = self._build_proxy_prompt_mask(trace)
        prompt_log_base = torch.zeros_like(prompt_mask)
        prompt_log_duration = None
        prompt_log_residual = self.derive_prompt_proxy_residual(trace).detach() * prompt_mask
        speech_support = (prompt_mask > self.speech_threshold).float()
        global_rate = self._support_shrink(
            self._masked_median(prompt_log_residual, speech_support),
            self._masked_count(speech_support),
            self.global_shrink_tau,
        ).detach()
        prompt_random_target = (prompt_log_residual - global_rate) * prompt_mask
        return PromptOperatorSummary(
            prompt_mask=prompt_mask,
            prompt_log_base=prompt_log_base,
            prompt_log_duration=prompt_log_duration,
            prompt_log_residual=prompt_log_residual,
            prompt_random_target=prompt_random_target,
            global_rate=global_rate,
        )

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
        prompt_duration_obs: torch.Tensor,
        prompt_unit_anchor_base: torch.Tensor | None,
        prompt_log_base: torch.Tensor | None,
        prompt_unit_mask: torch.Tensor | None,
    ) -> PromptOperatorSummary:
        prompt_mask = self._resolve_prompt_mask(
            prompt_duration_obs=prompt_duration_obs,
            prompt_unit_mask=prompt_unit_mask,
        )
        resolved_log_base = self._resolve_prompt_log_base(
            prompt_unit_anchor_base=prompt_unit_anchor_base,
            prompt_log_base=prompt_log_base,
        )
        prompt_log_duration = torch.log(prompt_duration_obs.float().clamp_min(1.0e-6)).detach() * prompt_mask
        prompt_log_base = resolved_log_base.float().detach() * prompt_mask
        prompt_log_residual = (prompt_log_duration - prompt_log_base).detach() * prompt_mask
        global_rate = self._support_shrink(
            self._masked_median(prompt_log_residual, prompt_mask),
            self._masked_count(prompt_mask),
            self.global_shrink_tau,
        ).detach()
        prompt_random_target = (prompt_log_residual - global_rate) * prompt_mask
        return PromptOperatorSummary(
            prompt_mask=prompt_mask,
            prompt_log_base=prompt_log_base,
            prompt_log_duration=prompt_log_duration,
            prompt_log_residual=prompt_log_residual,
            prompt_random_target=prompt_random_target,
            global_rate=global_rate,
        )

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
    ) -> PromptOperatorDiagnostics:
        fit_mask, eval_mask = self._build_blocked_holdout_masks(prompt_mask)
        if bool(eval_mask.any().item()):
            coeff = self._solve_operator_coeff(
                prompt_basis_activation=prompt_basis_activation,
                prompt_random_target=prompt_random_target,
                prompt_mask=fit_mask,
            )
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
        return PromptOperatorDiagnostics(
            fit_mask=fit_mask,
            eval_mask=eval_mask,
            operator_cv_fit=operator_cv_fit,
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

    def encode_reference_features(
        self,
        ref_mel: torch.Tensor,
        *,
        ref_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.reference_encoder(ref_mel, ref_lengths=ref_lengths)
        return encoded["ref_rhythm_stats"], encoded["ref_rhythm_trace"]

    def build_from_prompt_units(
        self,
        *,
        prompt_content_units: torch.Tensor,
        prompt_duration_obs: torch.Tensor,
        prompt_unit_anchor_base: torch.Tensor | None,
        prompt_log_base: torch.Tensor | None,
        prompt_unit_mask: torch.Tensor | None,
        response_encoder,
    ) -> ReferenceDurationMemory:
        summary = self._build_prompt_summary_from_units(
            prompt_duration_obs=prompt_duration_obs,
            prompt_unit_anchor_base=prompt_unit_anchor_base,
            prompt_log_base=prompt_log_base,
            prompt_unit_mask=prompt_unit_mask,
        )
        prompt_basis_activation = response_encoder.encode_prompt_units(
            content_units=prompt_content_units.long(),
            log_anchor_base=summary.prompt_log_base.detach(),
            prompt_mask=summary.prompt_mask,
        )
        operator_coeff = self._solve_operator_coeff(
            prompt_basis_activation=prompt_basis_activation,
            prompt_random_target=summary.prompt_random_target,
            prompt_mask=summary.prompt_mask,
        )
        diagnostics = self._build_prompt_operator_diagnostics(
            prompt_basis_activation=prompt_basis_activation,
            prompt_random_target=summary.prompt_random_target,
            prompt_mask=summary.prompt_mask,
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
                prompt=PromptConditioningEvidence(
                    prompt_basis_activation=prompt_basis_activation,
                    prompt_random_target=summary.prompt_random_target,
                    prompt_mask=summary.prompt_mask,
                    prompt_fit_mask=diagnostics.fit_mask,
                    prompt_eval_mask=diagnostics.eval_mask,
                    prompt_operator_fit=prompt_operator_fit,
                    prompt_operator_cv_fit=diagnostics.operator_cv_fit,
                    prompt_log_base=summary.prompt_log_base,
                    prompt_log_duration=summary.prompt_log_duration,
                    prompt_log_residual=summary.prompt_log_residual,
                ),
            )
        )

    def build_from_trace_proxy(
        self,
        *,
        ref_rhythm_stats: torch.Tensor,
        ref_rhythm_trace: torch.Tensor,
        response_encoder,
    ) -> ReferenceDurationMemory:
        summary = self._build_prompt_summary_from_proxy(ref_rhythm_trace=ref_rhythm_trace)
        prompt_basis_activation = response_encoder.encode_prompt_proxy(
            ref_rhythm_trace=ref_rhythm_trace.float(),
            ref_rhythm_stats=ref_rhythm_stats.float(),
            prompt_mask=summary.prompt_mask.float(),
        )
        operator_coeff = self._solve_operator_coeff(
            prompt_basis_activation=prompt_basis_activation,
            prompt_random_target=summary.prompt_random_target,
            prompt_mask=summary.prompt_mask,
        )
        diagnostics = self._build_prompt_operator_diagnostics(
            prompt_basis_activation=prompt_basis_activation,
            prompt_random_target=summary.prompt_random_target,
            prompt_mask=summary.prompt_mask,
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
                prompt=PromptConditioningEvidence(
                    prompt_basis_activation=prompt_basis_activation,
                    prompt_random_target=summary.prompt_random_target,
                    prompt_mask=summary.prompt_mask,
                    prompt_fit_mask=diagnostics.fit_mask,
                    prompt_eval_mask=diagnostics.eval_mask,
                    prompt_operator_fit=prompt_operator_fit,
                    prompt_operator_cv_fit=diagnostics.operator_cv_fit,
                    prompt_log_base=summary.prompt_log_base,
                    prompt_log_duration=summary.prompt_log_duration,
                    prompt_log_residual=summary.prompt_log_residual,
                ),
            )
        )

    def _build_memory_from_flat_conditioning(
        self,
        *,
        ref_conditioning: Mapping[str, Any],
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
                    operator_coeff=ref_conditioning["operator_coeff"].float().detach(),
                ),
                prompt=prompt,
            )
        )

    def from_conditioning(
        self,
        ref_conditioning,
        *,
        response_encoder,
    ) -> ReferenceDurationMemory:
        normalized = normalize_duration_v3_conditioning(ref_conditioning)
        if isinstance(normalized, ReferenceDurationMemory):
            return normalized
        if normalized is None:
            raise ValueError("Reference conditioning is required when no reference mel is provided.")
        if all(key in normalized for key in _V3_MEMORY_REQUIRED_FIELDS):
            return self._build_memory_from_flat_conditioning(ref_conditioning=normalized)
        if all(key in normalized for key in _V3_PROMPT_UNIT_REQUIRED_FIELDS):
            return self.build_from_prompt_units(
                prompt_content_units=normalized["prompt_content_units"],
                prompt_duration_obs=normalized["prompt_duration_obs"],
                prompt_unit_anchor_base=normalized.get("prompt_unit_anchor_base"),
                prompt_log_base=normalized.get("prompt_log_base"),
                prompt_unit_mask=normalized.get("prompt_unit_mask"),
                response_encoder=response_encoder,
            )
        ref_rhythm_stats = normalized.get("ref_rhythm_stats")
        ref_rhythm_trace = normalized.get("ref_rhythm_trace")
        if ref_rhythm_stats is None or ref_rhythm_trace is None:
            raise ValueError(
                "Reference conditioning must provide operator memory, explicit prompt units, "
                "or ref_rhythm_stats/ref_rhythm_trace proxy features."
            )
        return self.build_from_trace_proxy(
            ref_rhythm_stats=ref_rhythm_stats,
            ref_rhythm_trace=ref_rhythm_trace,
            response_encoder=response_encoder,
        )
    def forward(
        self,
        *,
        response_encoder,
        ref_conditioning=None,
        ref_mel: torch.Tensor | None = None,
        ref_lengths: torch.Tensor | None = None,
    ) -> ReferenceDurationMemory:
        if ref_conditioning is not None:
            return self.from_conditioning(
                ref_conditioning,
                response_encoder=response_encoder,
            )
        if ref_mel is None:
            raise ValueError("Either ref_conditioning or ref_mel must be provided.")
        ref_rhythm_stats, ref_rhythm_trace = self.encode_reference_features(ref_mel, ref_lengths=ref_lengths)
        return self.build_from_trace_proxy(
            ref_rhythm_stats=ref_rhythm_stats,
            ref_rhythm_trace=ref_rhythm_trace,
            response_encoder=response_encoder,
        )


PromptDurationOperatorBuilder = PromptConditionedOperatorEstimator

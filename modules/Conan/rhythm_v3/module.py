from __future__ import annotations

from dataclasses import replace
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import (
    DurationExecution,
    DurationRuntimeState,
    ReferenceDurationMemory,
    SourceUnitBatch,
    ensure_duration_runtime_state_batch,
    ensure_reference_duration_memory_batch,
    validate_reference_duration_memory,
)
from .projector import StreamingDurationProjector
from .reference_memory import PromptConditionedOperatorEstimator
from .g_stats import (
    build_global_rate_support_mask,
    compute_duration_weighted_speech_ratio,
    compute_global_rate_batch,
    normalize_falsification_eval_mode,
    normalize_global_rate_variant,
)
from .math_utils import (
    build_causal_source_prefix_rate_seq,
    first_valid_speech_init,
    normalize_src_prefix_stat_mode,
)
from .global_condition import PromptGlobalConditionEncoderV1G
from .minimal_writer import MinimalStreamingDurationHeadV1G, MinimalStreamingDurationWriterV1G
from .summary_memory import (
    PromptDurationMemoryEncoder,
    SharedSummaryCodebook,
    StreamingDurationHead,
)

_MINIMAL_V1_PUBLIC_BACKBONE = "minimal_v1_global"


_PROMPT_SUMMARY_BACKBONE_ALIASES = {
    "prompt_summary",
    "role_memory",
    "unit_run",
    "minimal_v1_global",
    "v1g_minimal",
}


def _normalize_prompt_summary_backbone(backbone_mode: str | None) -> str:
    normalized = str(backbone_mode or "global_only").strip().lower()
    if normalized in _PROMPT_SUMMARY_BACKBONE_ALIASES - {"prompt_summary"}:
        return "prompt_summary"
    return normalized


def _prompt_summary_public_label(*, minimal_v1_profile: bool) -> str:
    return _MINIMAL_V1_PUBLIC_BACKBONE if minimal_v1_profile else "prompt_summary"


def _prompt_summary_public_with_aliases(*, minimal_v1_profile: bool) -> str:
    if minimal_v1_profile:
        return f"{_MINIMAL_V1_PUBLIC_BACKBONE} (compat: prompt_summary; legacy: role_memory, unit_run)"
    return "prompt_summary (public minimal alias: minimal_v1_global; legacy aliases: role_memory, unit_run)"


def compute_duration_weighted_prompt_speech_ratio(
    *,
    prompt_log_duration: torch.Tensor | None,
    prompt_mask: torch.Tensor | None,
    prompt_speech_mask: torch.Tensor | None,
    eps: float = 1.0e-6,
) -> torch.Tensor | None:
    if not isinstance(prompt_log_duration, torch.Tensor):
        return None
    if not isinstance(prompt_mask, torch.Tensor) or not isinstance(prompt_speech_mask, torch.Tensor):
        return None
    mask = prompt_mask.float().clamp(0.0, 1.0)
    speech_mask = prompt_speech_mask.float().clamp(0.0, 1.0)
    duration = torch.exp(prompt_log_duration.float()) * mask
    numerator = (duration * speech_mask).sum(dim=1, keepdim=True)
    denominator = duration.sum(dim=1, keepdim=True).clamp_min(float(eps))
    return numerator / denominator


def _init_accepts_kwarg(module_cls: type[nn.Module], kwarg_name: str) -> bool:
    return kwarg_name in inspect.signature(module_cls.__init__).parameters


_MISSING = object()


def _resolve_reference_summary_usage(
    *,
    minimal_v1_profile: bool,
    requested_use_reference_summary,
) -> bool:
    use_reference_summary = False if requested_use_reference_summary is _MISSING else bool(requested_use_reference_summary)
    if minimal_v1_profile and use_reference_summary:
        raise ValueError(
            "rhythm_v3_minimal_v1_profile selects the minimal_v1_global path directly; "
            "use_reference_summary only enables optional non-minimal reference-summary tensors and must be false."
        )
    return use_reference_summary


class SharedCausalBasisEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        basis_rank: int,
        window_left: int = 4,
        window_right: int = 0,
    ) -> None:
        super().__init__()
        self.window_left = max(0, int(window_left))
        self.window_right = max(0, int(window_right))
        kernel_size = self.window_left + self.window_right + 1
        self.unit_embedding = nn.Embedding(vocab_size, hidden_size)
        self.source_adapter = nn.Linear(hidden_size + 1, hidden_size)
        self.shared_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, bias=True)
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, basis_rank)

    def _run_shared(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if hidden.size(1) <= 0:
            return self.out_proj.weight.new_zeros((hidden.size(0), 0, self.out_proj.out_features))
        conv_input = hidden.transpose(1, 2)
        conv_input = F.pad(conv_input, (self.window_left, self.window_right))
        conv_hidden = self.shared_conv(conv_input).transpose(1, 2)
        hidden = self.hidden_norm(F.silu(hidden + conv_hidden))
        basis = self.out_proj(hidden)
        basis = F.normalize(basis, p=2.0, dim=-1, eps=1.0e-6)
        return basis * mask.unsqueeze(-1)

    def encode_source(
        self,
        *,
        content_units: torch.Tensor,
        log_anchor_base: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        if content_units.size(1) <= 0:
            return self.out_proj.weight.new_zeros((content_units.size(0), 0, self.out_proj.out_features))
        content_embed = self.unit_embedding(content_units.long())
        features = torch.cat([content_embed, log_anchor_base.unsqueeze(-1)], dim=-1)
        hidden = F.silu(self.source_adapter(features))
        return self._run_shared(hidden, unit_mask.float())

    def encode_prompt_units(
        self,
        *,
        content_units: torch.Tensor,
        log_anchor_base: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_source(
            content_units=content_units,
            log_anchor_base=log_anchor_base,
            unit_mask=prompt_mask,
        )

def _resolve_duration_runtime_surface(
    *,
    backbone_mode: str | None,
    warp_mode: str | None,
    allow_hybrid: bool | None,
    source_residual_gain: float,
) -> tuple[str, str, bool, str]:
    resolved_backbone = _normalize_prompt_summary_backbone(backbone_mode)
    resolved_warp = str(warp_mode or "none").strip().lower()
    resolved_allow_hybrid = bool(allow_hybrid) if allow_hybrid is not None else False
    if resolved_backbone not in {"global_only", "operator", "prompt_summary"}:
        raise ValueError(
            "Unsupported rhythm_v3 backbone mode: "
            f"{backbone_mode!r}. Supported values are global_only, operator, prompt_summary "
            "(public minimal alias: minimal_v1_global; legacy aliases: role_memory, unit_run)."
        )
    if resolved_warp not in {"none", "progress", "detector"}:
        raise ValueError(f"Unsupported rhythm_v3 warp mode: {warp_mode!r}")
    if resolved_backbone == "prompt_summary":
        if resolved_warp != "none":
            raise ValueError("rhythm_v3_backbone='prompt_summary' (public minimal alias: 'minimal_v1_global'; legacy aliases: 'role_memory', 'unit_run') only supports rhythm_v3_warp_mode='none'.")
        if resolved_allow_hybrid:
            raise ValueError("rhythm_v3_allow_hybrid is not used by rhythm_v3_backbone='prompt_summary' (public minimal alias: 'minimal_v1_global'; legacy aliases: 'role_memory', 'unit_run').")
        if float(source_residual_gain) > 0.0:
            raise ValueError("rhythm_v3_source_residual_gain is not supported by rhythm_v3_backbone='prompt_summary' (public minimal alias: 'minimal_v1_global'; legacy aliases: 'role_memory', 'unit_run').")
        return resolved_backbone, resolved_warp, False, "prompt_summary"
    if resolved_backbone == "global_only":
        if resolved_allow_hybrid:
            raise ValueError("rhythm_v3_allow_hybrid is only valid when rhythm_v3_backbone='operator'.")
        canonical = (
            "progress_only"
            if resolved_warp == "progress"
            else "detector_only"
            if resolved_warp == "detector"
            else "global_only"
        )
        return resolved_backbone, resolved_warp, False, canonical
    if resolved_warp == "detector":
        raise ValueError(
            "Detector bank is a global-only candidate layer. "
            "Use rhythm_v3_backbone='global_only' with rhythm_v3_warp_mode='detector'."
        )
    if resolved_warp == "progress":
        if not resolved_allow_hybrid:
            raise ValueError(
                "Operator + progress warp must be explicit: set rhythm_v3_allow_hybrid=true "
                "when rhythm_v3_backbone='operator' and rhythm_v3_warp_mode='progress'."
            )
        return resolved_backbone, resolved_warp, True, "operator_progress"
    runtime_mode = "operator_srcres" if float(source_residual_gain) > 0.0 else "operator"
    return resolved_backbone, resolved_warp, False, runtime_mode


def _enforce_minimal_v1_runtime_contract(
    *,
    minimal_v1_profile: bool,
    strict_minimal_claim_profile: bool,
    use_src_gap_in_coarse_head: bool,
    g_variant: str,
    rate_mode: str,
    simple_global_stats: bool,
    use_log_base_rate: bool,
    use_reference_summary: bool,
    use_learned_residual_gate: bool,
    streaming_mode: str,
    backbone_mode: str,
    warp_mode: str,
    allow_hybrid: bool,
    source_residual_gain: float,
    short_gap_silence_scale: float,
    leading_silence_scale: float,
    detach_global_term_in_local_head: bool,
    freeze_src_rate_init: bool,
) -> None:
    if not minimal_v1_profile:
        return
    errors: list[str] = []
    if strict_minimal_claim_profile and str(g_variant).strip().lower() == "unit_norm":
        errors.append("g_variant must not be unit_norm when strict minimal claim profile is enabled")
    if str(rate_mode).strip().lower() != "simple_global":
        errors.append("rate_mode must be simple_global")
    if not bool(simple_global_stats):
        errors.append("simple_global_stats must be true")
    if bool(use_log_base_rate):
        errors.append("use_log_base_rate must be false")
    if bool(use_reference_summary):
        errors.append("use_reference_summary must be false")
    if bool(use_learned_residual_gate):
        errors.append("use_learned_residual_gate must be false")
    if str(streaming_mode).strip().lower() != "strict":
        errors.append("streaming_mode must be strict")
    if str(backbone_mode).strip().lower() != "prompt_summary":
        errors.append(f"backbone_mode must resolve to {_MINIMAL_V1_PUBLIC_BACKBONE}")
    if str(warp_mode).strip().lower() != "none":
        errors.append("warp_mode must resolve to none")
    if bool(allow_hybrid):
        errors.append("allow_hybrid must be false")
    if float(source_residual_gain) > 0.0:
        errors.append("source_residual_gain must be 0")
    if abs(float(short_gap_silence_scale) - 0.35) > 1.0e-6:
        errors.append("short_gap_silence_scale must stay at the constant-clip default (0.35)")
    if abs(float(leading_silence_scale) - 0.0) > 1.0e-6:
        errors.append("leading_silence_scale must stay at the constant-clip default (0.0)")
    if not bool(detach_global_term_in_local_head):
        errors.append("detach_global_term_in_local_head must be true")
    if not bool(freeze_src_rate_init):
        errors.append("freeze_src_rate_init must be true")
    if bool(use_src_gap_in_coarse_head):
        errors.append("use_src_gap_in_coarse_head must be false")
    if errors:
        raise ValueError(
            "rhythm_v3_minimal_v1_profile runtime contract violation: "
            + "; ".join(errors)
        )


def _assert_minimal_reference_memory(
    *,
    minimal_v1_profile: bool,
    ref_memory: ReferenceDurationMemory,
    context: str,
) -> None:
    if not minimal_v1_profile:
        return
    violations: list[str] = []
    if ref_memory.summary_state is not None:
        violations.append("summary_state")
    if ref_memory.progress is not None:
        violations.append("progress")
    if ref_memory.detector is not None:
        violations.append("detector")
    if ref_memory.role is not None:
        violations.append("role")
    if violations:
        raise RuntimeError(
            "rhythm_v3_minimal_v1_profile forbids non-minimal reference memory fields "
            f"at {context}: {', '.join(violations)}."
        )


class DurationBackbone(nn.Module):
    backbone_mode = "global_only"
    warp_mode = "none"
    allow_hybrid = False
    use_source_residual = False
    need_operator = False
    need_progress = False
    need_detector = False

    @staticmethod
    def _global_response(*, ref_memory: ReferenceDurationMemory, unit_mask: torch.Tensor, speech_commit_mask: torch.Tensor) -> torch.Tensor:
        return ref_memory.global_rate.float().expand(-1, unit_mask.size(1)) * speech_commit_mask.float()

    def forward(
        self,
        *,
        module,
        source_batch: SourceUnitBatch,
        ref_memory: ReferenceDurationMemory,
        unit_mask: torch.Tensor,
        speech_commit_mask: torch.Tensor,
        detached_log_anchor: torch.Tensor,
        basis_activation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class GlobalOnlyBackbone(DurationBackbone):
    backbone_mode = "global_only"
    warp_mode = "none"
    allow_hybrid = False
    use_source_residual = False
    need_operator = False
    need_progress = False

    def forward(
        self,
        *,
        module,
        source_batch: SourceUnitBatch,
        ref_memory: ReferenceDurationMemory,
        unit_mask: torch.Tensor,
        speech_commit_mask: torch.Tensor,
        detached_log_anchor: torch.Tensor,
        basis_activation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del module, source_batch, detached_log_anchor, basis_activation
        zeros = speech_commit_mask.new_zeros(speech_commit_mask.shape)
        return self._global_response(
            ref_memory=ref_memory,
            unit_mask=unit_mask,
            speech_commit_mask=speech_commit_mask,
        ), zeros, zeros, zeros


class ProgressWarpBackbone(DurationBackbone):
    backbone_mode = "global_only"
    warp_mode = "progress"
    allow_hybrid = False
    use_source_residual = False
    need_operator = False
    need_progress = True

    def forward(
        self,
        *,
        module,
        source_batch: SourceUnitBatch,
        ref_memory: ReferenceDurationMemory,
        unit_mask: torch.Tensor,
        speech_commit_mask: torch.Tensor,
        detached_log_anchor: torch.Tensor,
        basis_activation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del detached_log_anchor, basis_activation
        zeros = speech_commit_mask.new_zeros(speech_commit_mask.shape)
        progress_response = module._sample_progress_response(
            ref_memory=ref_memory,
            source_batch=source_batch,
            speech_commit_mask=speech_commit_mask,
        )
        return self._global_response(
            ref_memory=ref_memory,
            unit_mask=unit_mask,
            speech_commit_mask=speech_commit_mask,
        ), progress_response, zeros, zeros


class DetectorBankBackbone(DurationBackbone):
    backbone_mode = "global_only"
    warp_mode = "detector"
    allow_hybrid = False
    use_source_residual = False
    need_operator = False
    need_progress = False
    need_detector = True

    def forward(
        self,
        *,
        module,
        source_batch: SourceUnitBatch,
        ref_memory: ReferenceDurationMemory,
        unit_mask: torch.Tensor,
        speech_commit_mask: torch.Tensor,
        detached_log_anchor: torch.Tensor,
        basis_activation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del detached_log_anchor, basis_activation
        zeros = speech_commit_mask.new_zeros(speech_commit_mask.shape)
        detector_response = module._predict_detector_response(
            source_batch=source_batch,
            ref_memory=ref_memory,
            speech_commit_mask=speech_commit_mask,
        )
        return self._global_response(
            ref_memory=ref_memory,
            unit_mask=unit_mask,
            speech_commit_mask=speech_commit_mask,
        ), detector_response, zeros, zeros


class OperatorBackbone(DurationBackbone):
    backbone_mode = "operator"
    need_operator = True

    def __init__(self, *, allow_hybrid: bool = False, use_source_residual: bool = False) -> None:
        super().__init__()
        self.allow_hybrid = bool(allow_hybrid)
        self.use_source_residual = bool(use_source_residual)
        self.warp_mode = "progress" if self.allow_hybrid else "none"
        self.need_progress = self.allow_hybrid

    def forward(
        self,
        *,
        module,
        source_batch: SourceUnitBatch,
        ref_memory: ReferenceDurationMemory,
        unit_mask: torch.Tensor,
        speech_commit_mask: torch.Tensor,
        detached_log_anchor: torch.Tensor,
        basis_activation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        global_response = self._global_response(
            ref_memory=ref_memory,
            unit_mask=unit_mask,
            speech_commit_mask=speech_commit_mask,
        )
        local_response = module._predict_local_response(
            basis_activation=basis_activation,
            ref_memory=ref_memory,
        ) * speech_commit_mask.float()
        progress_response = speech_commit_mask.new_zeros(speech_commit_mask.shape)
        if self.allow_hybrid:
            progress_response = module._sample_progress_response(
                ref_memory=ref_memory,
                source_batch=source_batch,
                speech_commit_mask=speech_commit_mask,
            )
        source_residual_response = speech_commit_mask.new_zeros(speech_commit_mask.shape)
        if self.use_source_residual and module.source_residual_gain > 0.0:
            centered_source_residual = module._build_centered_source_residual(
                source_batch=source_batch,
                detached_log_anchor=detached_log_anchor,
                speech_commit_mask=speech_commit_mask,
            )
            source_residual_response = module.source_residual_gain * centered_source_residual
        return global_response, progress_response, local_response, source_residual_response


class MixedEffectsDurationModule(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int = 256,
        basis_rank: int = 12,
        response_window_left: int = 4,
        response_window_right: int = 0,
        streaming_mode: str = "strict",
        micro_lookahead_units: int | None = None,
        ridge_lambda: float = 1.0,
        backbone_mode: str | None = None,
        warp_mode: str | None = None,
        allow_hybrid: bool | None = None,
        source_residual_gain: float = 0.0,
        **unused_kwargs,
    ) -> None:
        super().__init__()
        summary_dim = int(unused_kwargs.pop("summary_dim", unused_kwargs.pop("role_dim", hidden_size)))
        minimal_profile_requested = bool(
            unused_kwargs.get("minimal_v1_profile", unused_kwargs.get("rhythm_v3_minimal_v1_profile", False))
        )
        summary_slots_default = 1 if minimal_profile_requested else max(4, basis_rank)
        summary_slots = int(
            unused_kwargs.pop(
                "num_summary_slots",
                unused_kwargs.pop("num_role_slots", summary_slots_default),
            )
        )
        summary_cov_floor = float(unused_kwargs.pop("summary_cov_floor", unused_kwargs.pop("role_cov_floor", 0.05)))
        summary_pool_speech_only = bool(unused_kwargs.pop("summary_pool_speech_only", True))
        summary_use_unit_embedding = bool(unused_kwargs.pop("summary_use_unit_embedding", False))
        max_logstretch = float(unused_kwargs.pop("max_logstretch", 1.2))
        max_silence_logstretch = float(unused_kwargs.pop("max_silence_logstretch", 0.35))
        local_cold_start_runs = int(unused_kwargs.pop("local_cold_start_runs", 2))
        local_short_run_min_duration = float(unused_kwargs.pop("local_short_run_min_duration", 2.0))
        local_rate_decay = float(unused_kwargs.pop("local_rate_decay", 0.95))
        analytic_gap_clip = float(
            unused_kwargs.pop(
                "analytic_gap_clip",
                unused_kwargs.pop("rhythm_v3_analytic_gap_clip", 0.35),
            )
            or 0.0
        )
        short_gap_silence_scale = float(unused_kwargs.pop("short_gap_silence_scale", 0.35))
        leading_silence_scale = float(unused_kwargs.pop("leading_silence_scale", 0.0))
        rate_mode = str(
            unused_kwargs.pop(
                "rate_mode",
                unused_kwargs.pop("rhythm_v3_rate_mode", ""),
            )
            or ""
        ).strip().lower()
        self.minimal_v1_profile = bool(
            unused_kwargs.pop(
                "minimal_v1_profile",
                unused_kwargs.pop("rhythm_v3_minimal_v1_profile", False),
            )
        )
        requested_simple_global_stats = bool(
            unused_kwargs.pop(
                "simple_global_stats",
                unused_kwargs.pop("rhythm_v3_simple_global_stats", self.minimal_v1_profile),
            )
        )
        if rate_mode in {"", "none", "auto"}:
            rate_mode = "simple_global" if requested_simple_global_stats else "log_base"
        self.rate_mode = rate_mode
        self.simple_global_stats = bool(requested_simple_global_stats) or self.rate_mode == "simple_global"
        use_log_base_rate = unused_kwargs.pop(
            "use_log_base_rate",
            unused_kwargs.pop("rhythm_v3_use_log_base_rate", _MISSING),
        )
        if use_log_base_rate is _MISSING:
            use_log_base_rate = True
        self.use_log_base_rate = bool(use_log_base_rate) and not self.simple_global_stats
        disable_learned_gate = unused_kwargs.pop(
            "disable_learned_gate",
            unused_kwargs.pop("rhythm_v3_disable_learned_gate", _MISSING),
        )
        use_learned_residual_gate_default = False if self.minimal_v1_profile else True
        self.use_learned_residual_gate = bool(
            unused_kwargs.pop(
                "use_learned_residual_gate",
                unused_kwargs.pop("rhythm_v3_use_learned_residual_gate", use_learned_residual_gate_default),
            )
        )
        if disable_learned_gate is not _MISSING and bool(disable_learned_gate):
            self.use_learned_residual_gate = False
        use_reference_summary = unused_kwargs.pop(
            "use_reference_summary",
            unused_kwargs.pop("rhythm_v3_use_reference_summary", _MISSING),
        )
        self.use_reference_summary = _resolve_reference_summary_usage(
            minimal_v1_profile=self.minimal_v1_profile,
            requested_use_reference_summary=use_reference_summary,
        )
        global_shrink_tau = float(unused_kwargs.pop("global_shrink_tau", 8.0))
        progress_support_tau = float(unused_kwargs.pop("progress_support_tau", 8.0))
        progress_bins = int(unused_kwargs.pop("progress_bins", 4))
        ridge_support_tau = float(unused_kwargs.pop("ridge_support_tau", 8.0))
        operator_holdout_ratio = float(unused_kwargs.pop("operator_holdout_ratio", 0.30))
        prefix_budget_pos = int(unused_kwargs.pop("prefix_budget_pos", unused_kwargs.pop("unit_budget_pos", 24)))
        prefix_budget_neg = int(unused_kwargs.pop("prefix_budget_neg", unused_kwargs.pop("unit_budget_neg", 24)))
        dynamic_budget_ratio = float(unused_kwargs.pop("dynamic_budget_ratio", 0.0))
        min_prefix_budget = int(unused_kwargs.pop("min_prefix_budget", 0))
        max_prefix_budget = int(unused_kwargs.pop("max_prefix_budget", 0))
        budget_mode = str(unused_kwargs.pop("budget_mode", unused_kwargs.pop("rhythm_v3_budget_mode", "total")))
        boundary_carry_decay = float(unused_kwargs.pop("boundary_carry_decay", 0.25))
        boundary_offset_decay = unused_kwargs.pop(
            "boundary_offset_decay",
            unused_kwargs.pop("rhythm_v3_boundary_offset_decay", None),
        )
        boundary_reset_thresh = float(unused_kwargs.pop("boundary_reset_thresh", 0.5))
        emit_prompt_diagnostics = bool(
            unused_kwargs.pop("emit_prompt_diagnostics", unused_kwargs.pop("rhythm_v3_emit_prompt_diagnostics", True))
        )
        self.g_variant = normalize_global_rate_variant(
            unused_kwargs.pop("g_variant", unused_kwargs.pop("rhythm_v3_g_variant", "raw_median"))
        )
        self.g_trim_ratio = float(
            max(
                0.0,
                min(
                    0.49,
                    float(unused_kwargs.pop("g_trim_ratio", unused_kwargs.pop("rhythm_v3_g_trim_ratio", 0.2))),
                ),
            )
        )
        self.g_drop_edge_runs = max(
            0,
            int(
                unused_kwargs.pop(
                    "drop_edge_runs_for_g",
                    unused_kwargs.pop("rhythm_v3_drop_edge_runs_for_g", 0),
                )
                or 0
            ),
        )
        self.eval_mode = normalize_falsification_eval_mode(
            unused_kwargs.pop("eval_mode", unused_kwargs.pop("rhythm_v3_eval_mode", "learned"))
        )
        self.disable_local_residual = bool(
            unused_kwargs.pop(
                "disable_local_residual",
                unused_kwargs.pop("rhythm_v3_disable_local_residual", False),
            )
        )
        self.disable_coarse_bias = bool(
            unused_kwargs.pop(
                "disable_coarse_bias",
                unused_kwargs.pop("rhythm_v3_disable_coarse_bias", False),
            )
        )
        self.debug_export = bool(
            unused_kwargs.pop("debug_export", unused_kwargs.pop("rhythm_v3_debug_export", False))
        )
        self.export_projector_telemetry = bool(
            unused_kwargs.pop(
                "export_projector_telemetry",
                unused_kwargs.pop("rhythm_v3_export_projector_telemetry", self.debug_export),
            )
        )
        self.min_prompt_speech_ratio = float(
            unused_kwargs.pop(
                "min_prompt_speech_ratio",
                unused_kwargs.pop("rhythm_v3_min_prompt_speech_ratio", 0.6),
            )
        )
        self.min_prompt_ref_len_sec = float(
            unused_kwargs.pop(
                "min_prompt_ref_len_sec",
                unused_kwargs.pop("rhythm_v3_min_prompt_ref_len_sec", 3.0),
            )
        )
        self.max_prompt_ref_len_sec = float(
            unused_kwargs.pop(
                "max_prompt_ref_len_sec",
                unused_kwargs.pop("rhythm_v3_max_prompt_ref_len_sec", 8.0),
            )
        )
        min_boundary_confidence_for_g = unused_kwargs.pop(
            "min_boundary_confidence_for_g",
            unused_kwargs.pop("rhythm_v3_min_boundary_confidence_for_g", None),
        )
        self.min_boundary_confidence_for_g = (
            None if min_boundary_confidence_for_g is None else float(min_boundary_confidence_for_g)
        )
        self.coarse_delta_scale = float(
            unused_kwargs.pop(
                "coarse_delta_scale",
                unused_kwargs.pop("rhythm_v3_coarse_delta_scale", 0.20),
            )
        )
        self.local_residual_scale = float(
            unused_kwargs.pop(
                "local_residual_scale",
                unused_kwargs.pop("rhythm_v3_local_residual_scale", 0.35),
            )
        )
        source_rate_init_mode_default = "first_speech" if self.minimal_v1_profile else "learned"
        self.src_rate_init_mode = str(
            unused_kwargs.pop(
                "src_rate_init_mode",
                unused_kwargs.pop("rhythm_v3_src_rate_init_mode", source_rate_init_mode_default),
            )
            or source_rate_init_mode_default
        ).strip().lower()
        self.src_rate_init_value = float(
            unused_kwargs.pop(
                "src_rate_init_value",
                unused_kwargs.pop("rhythm_v3_src_rate_init_value", 0.0),
            )
        )
        self.freeze_src_rate_init = bool(
            unused_kwargs.pop(
                "freeze_src_rate_init",
                unused_kwargs.pop("rhythm_v3_freeze_src_rate_init", bool(self.minimal_v1_profile)),
            )
        )
        self.src_prefix_min_support = int(
            max(
                1,
                unused_kwargs.pop(
                    "src_prefix_min_support",
                    unused_kwargs.pop("rhythm_v3_src_prefix_min_support", 3),
                ),
            )
        )
        self.src_prefix_stat_mode = normalize_src_prefix_stat_mode(
            unused_kwargs.pop(
                "src_prefix_stat_mode",
                unused_kwargs.pop("rhythm_v3_src_prefix_stat_mode", "ema"),
            )
        )
        self.strict_eval_invalid_g = bool(
            unused_kwargs.pop(
                "strict_eval_invalid_g",
                unused_kwargs.pop("rhythm_v3_strict_eval_invalid_g", bool(self.minimal_v1_profile)),
            )
        )
        detach_global_term_default = bool(self.minimal_v1_profile)
        self.detach_global_term_in_local_head = bool(
            unused_kwargs.pop(
                "detach_global_term_in_local_head",
                unused_kwargs.pop("rhythm_v3_detach_global_term_in_local_head", detach_global_term_default),
            )
        )
        self.use_src_gap_in_coarse_head = bool(
            unused_kwargs.pop(
                "use_src_gap_in_coarse_head",
                unused_kwargs.pop("rhythm_v3_use_src_gap_in_coarse_head", False),
            )
        )
        self.strict_minimal_claim_profile = bool(
            unused_kwargs.pop(
                "strict_minimal_claim_profile",
                unused_kwargs.pop("rhythm_v3_strict_minimal_claim_profile", True),
            )
        )
        del unused_kwargs
        self.analytic_gap_clip = float(max(0.0, analytic_gap_clip))
        self.streaming_mode = str(streaming_mode or "strict").strip().lower()
        if self.streaming_mode not in {"strict", "micro_lookahead"}:
            raise ValueError(f"Unsupported streaming_mode={streaming_mode!r}")
        self.source_residual_gain = float(max(0.0, source_residual_gain))
        (
            self.backbone_mode,
            self.warp_mode,
            self.allow_hybrid,
            self.runtime_mode,
        ) = _resolve_duration_runtime_surface(
            backbone_mode=backbone_mode,
            warp_mode=warp_mode,
            allow_hybrid=allow_hybrid,
            source_residual_gain=self.source_residual_gain,
        )
        self.is_minimal_v1 = bool(self.minimal_v1_profile and self.backbone_mode == "prompt_summary")
        self.public_backbone_mode = (
            _prompt_summary_public_label(minimal_v1_profile=True)
            if self.is_minimal_v1
            else self.backbone_mode
        )
        self.public_runtime_mode = (
            _prompt_summary_public_label(minimal_v1_profile=True)
            if self.is_minimal_v1
            else self.runtime_mode
        )
        _enforce_minimal_v1_runtime_contract(
            minimal_v1_profile=self.minimal_v1_profile,
            strict_minimal_claim_profile=self.strict_minimal_claim_profile,
            use_src_gap_in_coarse_head=self.use_src_gap_in_coarse_head,
            g_variant=self.g_variant,
            rate_mode=self.rate_mode,
            simple_global_stats=self.simple_global_stats,
            use_log_base_rate=self.use_log_base_rate,
            use_reference_summary=self.use_reference_summary,
            use_learned_residual_gate=self.use_learned_residual_gate,
            streaming_mode=self.streaming_mode,
            backbone_mode=self.backbone_mode,
            warp_mode=self.warp_mode,
            allow_hybrid=self.allow_hybrid,
            source_residual_gain=self.source_residual_gain,
            short_gap_silence_scale=short_gap_silence_scale,
            leading_silence_scale=leading_silence_scale,
            detach_global_term_in_local_head=self.detach_global_term_in_local_head,
            freeze_src_rate_init=self.freeze_src_rate_init,
        )
        effective_window_right = int(response_window_right)
        if self.streaming_mode == "strict":
            effective_window_right = 0
        elif micro_lookahead_units is not None:
            effective_window_right = max(0, int(micro_lookahead_units))

        self.response_encoder = None
        self.reference_memory_builder = None
        if self.backbone_mode != "prompt_summary":
            self.response_encoder = SharedCausalBasisEncoder(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                basis_rank=basis_rank,
                window_left=response_window_left,
                window_right=effective_window_right,
            )
            self.reference_memory_builder = PromptConditionedOperatorEstimator(
                progress_bins=progress_bins,
                ridge_lambda=ridge_lambda,
                global_shrink_tau=global_shrink_tau,
                progress_support_tau=progress_support_tau,
                ridge_support_tau=ridge_support_tau,
                holdout_ratio=operator_holdout_ratio,
                simple_global_stats=self.simple_global_stats,
                use_log_base_rate=self.use_log_base_rate,
                g_variant=self.g_variant,
                g_trim_ratio=self.g_trim_ratio,
                drop_edge_runs_for_g=self.g_drop_edge_runs,
            )
            self.reference_memory_builder.rate_mode = self.rate_mode
        self.projector = StreamingDurationProjector(
            prefix_budget_pos=prefix_budget_pos,
            prefix_budget_neg=prefix_budget_neg,
            dynamic_budget_ratio=dynamic_budget_ratio,
            min_prefix_budget=min_prefix_budget,
            max_prefix_budget=max_prefix_budget,
            budget_mode=budget_mode,
            boundary_carry_decay=boundary_carry_decay,
            boundary_offset_decay=boundary_offset_decay,
            boundary_reset_thresh=boundary_reset_thresh,
            export_projector_telemetry=self.export_projector_telemetry,
        )
        if self.use_reference_summary:
            emit_prompt_diagnostics = True
        self.summary_codebook = None
        self.role_codebook = None
        self.prompt_memory_encoder = None
        self.duration_head = None
        if self.backbone_mode == "prompt_summary":
            if self.is_minimal_v1:
                self.prompt_memory_encoder = PromptGlobalConditionEncoderV1G(
                    operator_rank=basis_rank,
                    min_speech_ratio=self.min_prompt_speech_ratio,
                    min_ref_len_sec=self.min_prompt_ref_len_sec,
                    max_ref_len_sec=self.max_prompt_ref_len_sec,
                    use_log_base_rate=self.use_log_base_rate,
                    g_variant=self.g_variant,
                    g_trim_ratio=self.g_trim_ratio,
                    drop_edge_runs_for_g=self.g_drop_edge_runs,
                    min_boundary_confidence=self.min_boundary_confidence_for_g,
                    strict_eval_invalid_g=self.strict_eval_invalid_g,
                )
                self.duration_head = MinimalStreamingDurationWriterV1G(
                    vocab_size=vocab_size,
                    dim=summary_dim,
                    num_slots=1,
                    spk_dim=hidden_size,
                    simple_global_stats=self.simple_global_stats,
                    use_log_base_rate=self.use_log_base_rate,
                    use_learned_residual_gate=self.use_learned_residual_gate,
                    max_logstretch=max_logstretch,
                    max_silence_logstretch=max_silence_logstretch,
                    local_cold_start_runs=local_cold_start_runs,
                    local_short_run_min_duration=local_short_run_min_duration,
                    local_rate_decay=local_rate_decay,
                    analytic_gap_clip=self.analytic_gap_clip,
                    eval_mode=self.eval_mode,
                    disable_local_residual=self.disable_local_residual,
                    disable_coarse_bias=self.disable_coarse_bias,
                    detach_global_term_in_local_head=self.detach_global_term_in_local_head,
                    coarse_delta_scale=self.coarse_delta_scale,
                    use_src_gap_in_coarse_head=self.use_src_gap_in_coarse_head,
                    local_residual_scale=self.local_residual_scale,
                    src_rate_init_mode=self.src_rate_init_mode,
                    src_rate_init_value=self.src_rate_init_value,
                    freeze_src_rate_init=self.freeze_src_rate_init,
                    g_variant=self.g_variant,
                    g_trim_ratio=self.g_trim_ratio,
                    src_prefix_stat_mode=self.src_prefix_stat_mode,
                    src_prefix_min_support=self.src_prefix_min_support,
                    g_drop_edge_runs=self.g_drop_edge_runs,
                    min_boundary_confidence_for_g=self.min_boundary_confidence_for_g,
                )
            else:
                summary_codebook = SharedSummaryCodebook(num_slots=summary_slots, dim=summary_dim)
                self.summary_codebook = summary_codebook
                self.role_codebook = summary_codebook
                prompt_memory_kwargs = dict(
                    vocab_size=vocab_size,
                    dim=summary_dim,
                    num_slots=summary_slots,
                    operator_rank=basis_rank,
                    coverage_floor=summary_cov_floor,
                    summary_pool_speech_only=summary_pool_speech_only,
                    summary_use_unit_embedding=summary_use_unit_embedding,
                    emit_prompt_diagnostics=emit_prompt_diagnostics,
                    g_variant=self.g_variant,
                    g_trim_ratio=self.g_trim_ratio,
                    drop_edge_runs_for_g=self.g_drop_edge_runs,
                    min_boundary_confidence=self.min_boundary_confidence_for_g,
                    codebook=self.summary_codebook,
                )
                if _init_accepts_kwarg(PromptDurationMemoryEncoder, "simple_global_stats"):
                    prompt_memory_kwargs["simple_global_stats"] = self.simple_global_stats
                if _init_accepts_kwarg(PromptDurationMemoryEncoder, "use_log_base_rate"):
                    prompt_memory_kwargs["use_log_base_rate"] = self.use_log_base_rate
                self.prompt_memory_encoder = PromptDurationMemoryEncoder(**prompt_memory_kwargs)
            self.prompt_memory_encoder.rate_mode = self.rate_mode
            self.prompt_memory_encoder.simple_global_stats = self.simple_global_stats
            self.prompt_memory_encoder.use_log_base_rate = self.use_log_base_rate
            self.prompt_memory_encoder.g_variant = self.g_variant
            self.prompt_memory_encoder.g_trim_ratio = self.g_trim_ratio
            self.prompt_memory_encoder.g_drop_edge_runs = self.g_drop_edge_runs
            if hasattr(self.prompt_memory_encoder, "min_speech_ratio"):
                self.prompt_memory_encoder.min_speech_ratio = self.min_prompt_speech_ratio
            if hasattr(self.prompt_memory_encoder, "min_ref_len_sec"):
                self.prompt_memory_encoder.min_ref_len_sec = self.min_prompt_ref_len_sec
            if hasattr(self.prompt_memory_encoder, "max_ref_len_sec"):
                self.prompt_memory_encoder.max_ref_len_sec = self.max_prompt_ref_len_sec
            if hasattr(self.prompt_memory_encoder, "min_boundary_confidence"):
                self.prompt_memory_encoder.min_boundary_confidence = self.min_boundary_confidence_for_g

            if not self.is_minimal_v1:
                duration_head_kwargs = dict(
                    vocab_size=vocab_size,
                    dim=summary_dim,
                    num_slots=summary_slots,
                    spk_dim=hidden_size,
                    max_logstretch=max_logstretch,
                    max_silence_logstretch=max_silence_logstretch,
                    local_cold_start_runs=local_cold_start_runs,
                    local_short_run_min_duration=local_short_run_min_duration,
                    local_rate_decay=local_rate_decay,
                    analytic_gap_clip=self.analytic_gap_clip,
                    short_gap_silence_scale=short_gap_silence_scale,
                    leading_silence_scale=leading_silence_scale,
                    eval_mode=self.eval_mode,
                    disable_local_residual=self.disable_local_residual,
                    disable_coarse_bias=self.disable_coarse_bias,
                    codebook=self.summary_codebook,
                )
                if _init_accepts_kwarg(StreamingDurationHead, "simple_global_stats"):
                    duration_head_kwargs["simple_global_stats"] = self.simple_global_stats
                if _init_accepts_kwarg(StreamingDurationHead, "use_log_base_rate"):
                    duration_head_kwargs["use_log_base_rate"] = self.use_log_base_rate
                if _init_accepts_kwarg(StreamingDurationHead, "use_learned_residual_gate"):
                    duration_head_kwargs["use_learned_residual_gate"] = self.use_learned_residual_gate
                if _init_accepts_kwarg(StreamingDurationHead, "coarse_delta_scale"):
                    duration_head_kwargs["coarse_delta_scale"] = self.coarse_delta_scale
                if _init_accepts_kwarg(StreamingDurationHead, "local_residual_scale"):
                    duration_head_kwargs["local_residual_scale"] = self.local_residual_scale
                if _init_accepts_kwarg(StreamingDurationHead, "src_rate_init_mode"):
                    duration_head_kwargs["src_rate_init_mode"] = self.src_rate_init_mode
                if _init_accepts_kwarg(StreamingDurationHead, "src_rate_init_value"):
                    duration_head_kwargs["src_rate_init_value"] = self.src_rate_init_value
                if _init_accepts_kwarg(StreamingDurationHead, "freeze_src_rate_init"):
                    duration_head_kwargs["freeze_src_rate_init"] = self.freeze_src_rate_init
                if _init_accepts_kwarg(StreamingDurationHead, "g_variant"):
                    duration_head_kwargs["g_variant"] = self.g_variant
                if _init_accepts_kwarg(StreamingDurationHead, "g_trim_ratio"):
                    duration_head_kwargs["g_trim_ratio"] = self.g_trim_ratio
                if _init_accepts_kwarg(StreamingDurationHead, "src_prefix_stat_mode"):
                    duration_head_kwargs["src_prefix_stat_mode"] = self.src_prefix_stat_mode
                if _init_accepts_kwarg(StreamingDurationHead, "src_prefix_min_support"):
                    duration_head_kwargs["src_prefix_min_support"] = self.src_prefix_min_support
                if _init_accepts_kwarg(StreamingDurationHead, "g_drop_edge_runs"):
                    duration_head_kwargs["g_drop_edge_runs"] = self.g_drop_edge_runs
                if _init_accepts_kwarg(StreamingDurationHead, "min_boundary_confidence_for_g"):
                    duration_head_kwargs["min_boundary_confidence_for_g"] = self.min_boundary_confidence_for_g
                self.duration_head = StreamingDurationHead(**duration_head_kwargs)
            self.duration_head.rate_mode = self.rate_mode
            self.duration_head.simple_global_stats = self.simple_global_stats
            self.duration_head.use_log_base_rate = self.use_log_base_rate
            self.duration_head.use_learned_residual_gate = self.use_learned_residual_gate
            self.duration_head.analytic_gap_clip = self.analytic_gap_clip
            self.duration_head.eval_mode = self.eval_mode
            self.duration_head.disable_local_residual = self.disable_local_residual
            self.duration_head.disable_coarse_bias = self.disable_coarse_bias
            if hasattr(self.duration_head, "g_variant"):
                self.duration_head.g_variant = self.g_variant
            if hasattr(self.duration_head, "g_trim_ratio"):
                self.duration_head.g_trim_ratio = self.g_trim_ratio
            if hasattr(self.duration_head, "src_prefix_stat_mode"):
                self.duration_head.src_prefix_stat_mode = self.src_prefix_stat_mode
            if hasattr(self.duration_head, "src_prefix_min_support"):
                self.duration_head.src_prefix_min_support = self.src_prefix_min_support
            if hasattr(self.duration_head, "g_drop_edge_runs"):
                self.duration_head.g_drop_edge_runs = self.g_drop_edge_runs
            if hasattr(self.duration_head, "min_boundary_confidence_for_g"):
                self.duration_head.min_boundary_confidence_for_g = self.min_boundary_confidence_for_g
        if self.backbone_mode == "global_only" and self.warp_mode == "progress":
            self.backbone = ProgressWarpBackbone()
        elif self.backbone_mode == "global_only" and self.warp_mode == "detector":
            self.backbone = DetectorBankBackbone()
        elif self.backbone_mode == "operator":
            self.backbone = OperatorBackbone(
                allow_hybrid=self.allow_hybrid,
                use_source_residual=(self.source_residual_gain > 0.0),
            )
        else:
            self.backbone = GlobalOnlyBackbone()

    def _use_progress_response(self) -> bool:
        return bool(getattr(self.backbone, "need_progress", False))

    def _use_local_operator(self) -> bool:
        return bool(getattr(self.backbone, "need_operator", False))

    def _use_detector_bank(self) -> bool:
        return bool(getattr(self.backbone, "need_detector", False))

    def init_state(self, batch_size: int, device: torch.device) -> DurationRuntimeState:
        return self.projector.init_state(batch_size=batch_size, device=device)

    def build_reference_conditioning(
        self,
        *,
        ref_conditioning=None,
    ) -> ReferenceDurationMemory:
        if self.backbone_mode == "prompt_summary":
            if isinstance(ref_conditioning, ReferenceDurationMemory):
                _assert_minimal_reference_memory(
                    minimal_v1_profile=self.minimal_v1_profile,
                    ref_memory=ref_conditioning,
                    context="build_reference_conditioning",
                )
                return ref_conditioning
            if not isinstance(ref_conditioning, dict):
                raise ValueError(
                    f"rhythm_v3 {_prompt_summary_public_with_aliases(minimal_v1_profile=self.is_minimal_v1)} requires explicit prompt-unit conditioning."
                )
            prompt_content_units = ref_conditioning.get("prompt_content_units")
            prompt_duration_obs = ref_conditioning.get("prompt_duration_obs")
            prompt_mask = ref_conditioning.get("prompt_unit_mask")
            if not (
                isinstance(prompt_content_units, torch.Tensor)
                and isinstance(prompt_duration_obs, torch.Tensor)
                and isinstance(prompt_mask, torch.Tensor)
            ):
                raise ValueError(
                    f"rhythm_v3 {_prompt_summary_public_with_aliases(minimal_v1_profile=self.is_minimal_v1)} requires prompt_content_units / prompt_duration_obs / prompt_unit_mask."
                )
            prompt_speech_mask = ref_conditioning.get("prompt_speech_mask")
            if not isinstance(prompt_speech_mask, torch.Tensor):
                raise ValueError(
                    f"rhythm_v3 {_prompt_summary_public_with_aliases(minimal_v1_profile=self.is_minimal_v1)} requires prompt_speech_mask "
                    "to preserve speech-only global rate semantics."
                )
            prompt_unit_anchor_base = (
                ref_conditioning.get("prompt_unit_anchor_base")
                if self.use_log_base_rate
                else None
            )
            prompt_log_base = (
                ref_conditioning.get("prompt_log_base")
                if self.use_log_base_rate
                else None
            )
            prompt_memory_kwargs = dict(
                prompt_content_units=prompt_content_units,
                prompt_duration_obs=prompt_duration_obs,
                prompt_mask=prompt_mask,
                prompt_valid_mask=ref_conditioning.get("prompt_valid_mask"),
                prompt_speech_mask=prompt_speech_mask,
                prompt_unit_anchor_base=prompt_unit_anchor_base,
                prompt_log_base=prompt_log_base,
                prompt_spk_embed=ref_conditioning.get("prompt_spk_embed"),
                prompt_edge_cue=ref_conditioning.get("prompt_source_boundary_cue"),
                prompt_phrase_final_mask=ref_conditioning.get("prompt_phrase_final_mask"),
                prompt_closed_mask=ref_conditioning.get("prompt_closed_mask"),
                prompt_boundary_confidence=ref_conditioning.get("prompt_boundary_confidence"),
                prompt_global_weight=ref_conditioning.get("prompt_global_weight"),
                prompt_unit_log_prior=ref_conditioning.get("prompt_unit_log_prior"),
            )
            if self.minimal_v1_profile:
                prompt_valid_mask = prompt_memory_kwargs["prompt_valid_mask"]
                if not isinstance(prompt_valid_mask, torch.Tensor):
                    prompt_valid_mask = prompt_mask.float()
                prompt_valid_mask = prompt_valid_mask.float().clamp(0.0, 1.0)
                prompt_ref_len_sec = ref_conditioning.get("prompt_ref_len_sec")
                if isinstance(prompt_ref_len_sec, torch.Tensor):
                    prompt_ref_len_sec = prompt_ref_len_sec.float().reshape(int(prompt_duration_obs.size(0)), -1)[:, :1]
                prompt_speech_ratio_scalar = ref_conditioning.get("prompt_speech_ratio_scalar")
                if not isinstance(prompt_speech_ratio_scalar, torch.Tensor):
                    prompt_speech_ratio_scalar = compute_duration_weighted_speech_ratio(
                        duration_obs=prompt_duration_obs.float(),
                        speech_mask=prompt_speech_mask.float(),
                        valid_mask=prompt_valid_mask,
                    )
                else:
                    prompt_speech_ratio_scalar = prompt_speech_ratio_scalar.float().reshape(
                        int(prompt_duration_obs.size(0)),
                        -1,
                    )[:, :1]
                if isinstance(prompt_ref_len_sec, torch.Tensor):
                    prompt_memory_kwargs["prompt_ref_len_sec"] = prompt_ref_len_sec
                prompt_memory_kwargs["prompt_speech_ratio_scalar"] = prompt_speech_ratio_scalar
            return self.prompt_memory_encoder(**prompt_memory_kwargs)
        need_progress = self._use_progress_response()
        need_detector = self._use_detector_bank()
        need_operator = self._use_local_operator()
        if ref_conditioning is not None:
            return self.reference_memory_builder(
                response_encoder=self.response_encoder,
                ref_conditioning=ref_conditioning,
                need_progress=need_progress,
                need_detector=need_detector,
                need_operator=need_operator,
            )
        raise ValueError("rhythm_v3 now requires explicit prompt-unit conditioning or prebuilt reference memory.")

    @staticmethod
    def _freeze_committed_prefix(
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

    @staticmethod
    def _compute_prompt_speech_ratio_from_memory(
        ref_memory: ReferenceDurationMemory,
    ) -> torch.Tensor | None:
        prompt_speech_mask = getattr(ref_memory, "prompt_speech_mask", None)
        if not isinstance(prompt_speech_mask, torch.Tensor):
            return None
        prompt_valid_mask = getattr(ref_memory, "prompt_valid_mask", None)
        prompt_log_duration = getattr(ref_memory, "prompt_log_duration", None)
        prompt_duration_obs = (
            torch.exp(prompt_log_duration.float()).clamp_min(0.0)
            if isinstance(prompt_log_duration, torch.Tensor)
            else None
        )
        return compute_duration_weighted_speech_ratio(
            duration_obs=prompt_duration_obs,
            speech_mask=prompt_speech_mask.float(),
            valid_mask=prompt_valid_mask.float() if isinstance(prompt_valid_mask, torch.Tensor) else None,
        )

    @staticmethod
    def _zero_invalid_prompt_rows(
        value: torch.Tensor | None,
        invalid_rows: torch.Tensor,
    ) -> torch.Tensor | None:
        if not isinstance(value, torch.Tensor) or value.dim() <= 0:
            return value
        row_mask = invalid_rows.bool().reshape(int(invalid_rows.size(0)), 1)
        if int(value.size(0)) != int(row_mask.size(0)):
            return value
        expand_mask = row_mask
        while expand_mask.dim() < value.dim():
            expand_mask = expand_mask.unsqueeze(-1)
        return torch.where(expand_mask, torch.zeros_like(value), value)

    @classmethod
    def _neutralize_minimal_prompt_invalid_rows(
        cls,
        *,
        role_plan: dict[str, torch.Tensor | str],
        invalid_rows: torch.Tensor,
    ) -> dict[str, torch.Tensor | str]:
        zero_keys = (
            "unit_logstretch",
            "unit_global_shift",
            "unit_global_shift_analytic",
            "unit_analytic_gap",
            "unit_analytic_logstretch",
            "global_bias_scalar",
            "unit_coarse_logstretch",
            "unit_coarse_path_logstretch",
            "unit_coarse_correction",
            "unit_coarse_correction_used",
            "unit_coarse_delta",
            "unit_coarse_correction_pred",
            "unit_coarse_correction_predicted",
            "coarse_scalar_raw",
            "unit_global_term_before_local",
            "unit_residual_logstretch",
            "unit_local_residual_used",
            "unit_residual_logstretch_pred",
            "unit_residual_gate",
            "residual_gate_mean",
            "unit_residual_cold_gate",
            "unit_residual_short_gate",
            "unit_residual_gate_stability",
            "residual_gate_cold",
            "residual_gate_short",
            "residual_gate_stability",
            "unit_speech_pred",
            "unit_silence_pred",
            "local_response",
            "local_response_pred",
            "g_ref",
        )
        updated = dict(role_plan)
        for key in zero_keys:
            if key in updated:
                updated[key] = cls._zero_invalid_prompt_rows(updated.get(key), invalid_rows)
        return updated

    def _encode_source_basis(
        self,
        *,
        source_batch: SourceUnitBatch,
        unit_mask: torch.Tensor,
        log_anchor: torch.Tensor,
    ) -> torch.Tensor:
        if not self._use_local_operator() or self.response_encoder is None:
            batch_size, num_units = source_batch.content_units.shape
            basis_rank = int(self.duration_head.query_dim if self.duration_head is not None else 1)
            return log_anchor.new_zeros((batch_size, num_units, basis_rank))
        return self.response_encoder.encode_source(
            content_units=source_batch.content_units,
            log_anchor_base=log_anchor,
            unit_mask=unit_mask,
        )

    @staticmethod
    def _predict_local_response(
        *,
        basis_activation: torch.Tensor,
        ref_memory: ReferenceDurationMemory,
    ) -> torch.Tensor:
        if basis_activation.size(1) <= 0:
            return basis_activation.new_zeros((basis_activation.size(0), 0))
        return torch.einsum("buk,bk->bu", basis_activation.float(), ref_memory.operator_coeff.float())

    @staticmethod
    def _build_prefix_progress(
        *,
        unit_anchor_base: torch.Tensor,
        speech_commit_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = speech_commit_mask.float().clamp(0.0, 1.0)
        if unit_anchor_base.numel() <= 0:
            return unit_anchor_base.new_zeros(unit_anchor_base.shape)
        mass = unit_anchor_base.float().detach().clamp_min(1.0e-6) * mask
        total_mass = mass.sum(dim=1, keepdim=True)
        fallback_mass = mask
        use_fallback = total_mass <= 1.0e-6
        mass = torch.where(use_fallback, fallback_mass, mass)
        total_mass = mass.sum(dim=1, keepdim=True).clamp_min(1.0)
        centered_cum = torch.cumsum(mass, dim=1) - (0.5 * mass)
        return (centered_cum / total_mass).clamp(0.0, 1.0) * mask

    @staticmethod
    def _sample_progress_response(
        *,
        ref_memory: ReferenceDurationMemory,
        source_batch: SourceUnitBatch,
        speech_commit_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(getattr(ref_memory, "progress_profile", None), torch.Tensor):
            return speech_commit_mask.new_zeros(speech_commit_mask.shape)
        progress_profile = ref_memory.progress_profile.float()
        if progress_profile.numel() <= 0 or progress_profile.size(1) <= 0 or source_batch.content_units.size(1) <= 0:
            return speech_commit_mask.new_zeros(speech_commit_mask.shape)
        progress = MixedEffectsDurationModule._build_prefix_progress(
            unit_anchor_base=source_batch.unit_anchor_base,
            speech_commit_mask=speech_commit_mask,
        )
        num_bins = int(progress_profile.size(1))
        indices = torch.clamp((progress * float(num_bins)).long(), min=0, max=max(0, num_bins - 1))
        sampled = progress_profile.gather(1, indices)
        return sampled * speech_commit_mask.float()

    @staticmethod
    def _build_detector_features(
        *,
        source_batch: SourceUnitBatch,
        speech_commit_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = speech_commit_mask.float().clamp(0.0, 1.0)
        progress = MixedEffectsDurationModule._build_prefix_progress(
            unit_anchor_base=source_batch.unit_anchor_base,
            speech_commit_mask=speech_commit_mask,
        )
        boundary = (
            source_batch.source_boundary_cue.float() * mask
            if isinstance(getattr(source_batch, "source_boundary_cue", None), torch.Tensor)
            else torch.zeros_like(progress)
        )
        phrase_pos = (
            source_batch.phrase_group_pos.float() * mask
            if isinstance(getattr(source_batch, "phrase_group_pos", None), torch.Tensor)
            else progress
        )
        phrase_final = (
            source_batch.phrase_final_mask.float() * mask
            if isinstance(getattr(source_batch, "phrase_final_mask", None), torch.Tensor)
            else torch.zeros_like(progress)
        )
        return torch.stack(
            [
                2.0 * progress - 1.0,
                boundary,
                2.0 * phrase_pos - 1.0,
                phrase_final,
            ],
            dim=-1,
        ) * mask.unsqueeze(-1)

    def _predict_detector_response(
        self,
        *,
        source_batch: SourceUnitBatch,
        ref_memory: ReferenceDurationMemory,
        speech_commit_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(getattr(ref_memory, "detector_coeff", None), torch.Tensor):
            return speech_commit_mask.new_zeros(speech_commit_mask.shape)
        detector_features = self._build_detector_features(
            source_batch=source_batch,
            speech_commit_mask=speech_commit_mask,
        )
        if detector_features.size(1) <= 0:
            return speech_commit_mask.new_zeros(speech_commit_mask.shape)
        response = torch.einsum("bud,bd->bu", detector_features.float(), ref_memory.detector_coeff.float())
        return response * speech_commit_mask.float()

    @staticmethod
    def _resolve_speech_commit_mask(
        *,
        source_batch: SourceUnitBatch,
        commit_mask: torch.Tensor,
    ) -> torch.Tensor:
        speech_commit_mask = commit_mask.float()
        if isinstance(getattr(source_batch, "source_silence_mask", None), torch.Tensor):
            speech_commit_mask = speech_commit_mask * (1.0 - source_batch.source_silence_mask.float().clamp(0.0, 1.0))
        return speech_commit_mask

    @staticmethod
    def _build_centered_source_residual(
        *,
        source_batch: SourceUnitBatch,
        detached_log_anchor: torch.Tensor,
        speech_commit_mask: torch.Tensor,
    ) -> torch.Tensor:
        raw_source_residual = (
            torch.log(source_batch.source_duration_obs.float().clamp_min(1.0e-6))
            - detached_log_anchor.float()
        ) * speech_commit_mask.float()
        prefix_sum = torch.cumsum(raw_source_residual, dim=1)
        prefix_den = torch.cumsum(speech_commit_mask.float(), dim=1).clamp_min(1.0)
        prefix_mean = (prefix_sum / prefix_den).detach()
        return (raw_source_residual - prefix_mean) * speech_commit_mask.float()

    @staticmethod
    def _resolve_source_boundary_confidence(
        source_batch: SourceUnitBatch,
    ) -> torch.Tensor | None:
        boundary_confidence = getattr(source_batch, "boundary_confidence", None)
        if isinstance(boundary_confidence, torch.Tensor):
            return boundary_confidence.float()
        source_boundary_cue = getattr(source_batch, "source_boundary_cue", None)
        if isinstance(source_boundary_cue, torch.Tensor):
            return source_boundary_cue.float()
        return None

    def _compute_source_global_rate(
        self,
        *,
        source_batch: SourceUnitBatch,
    ) -> torch.Tensor | None:
        valid_mask = source_batch.unit_mask.float()
        if valid_mask.numel() <= 0:
            return None
        speech_mask = valid_mask
        if isinstance(getattr(source_batch, "source_silence_mask", None), torch.Tensor):
            speech_mask = speech_mask * (
                1.0 - source_batch.source_silence_mask.float().clamp(0.0, 1.0)
            )
        log_dur = torch.log(source_batch.source_duration_obs.float().clamp_min(1.0e-4)) * valid_mask
        if self.use_log_base_rate:
            if isinstance(getattr(source_batch, "unit_rate_log_base", None), torch.Tensor):
                log_base = source_batch.unit_rate_log_base.float().detach() * valid_mask
            else:
                log_base = torch.zeros_like(log_dur)
            rate_logdur = (log_dur - log_base) * valid_mask
        else:
            rate_logdur = log_dur
        unit_prior = getattr(source_batch, "unit_log_prior", None)
        if self.g_variant == "unit_norm" and not isinstance(unit_prior, torch.Tensor):
            return None
        boundary_confidence = self._resolve_source_boundary_confidence(source_batch)
        support_mask = build_global_rate_support_mask(
            speech_mask=speech_mask,
            valid_mask=valid_mask,
            drop_edge_runs=self.g_drop_edge_runs,
            closed_mask=(
                source_batch.sealed_mask.float()
                if isinstance(getattr(source_batch, "sealed_mask", None), torch.Tensor)
                else None
            ),
            boundary_confidence=boundary_confidence,
            min_boundary_confidence=self.min_boundary_confidence_for_g,
        )
        try:
            return compute_global_rate_batch(
                log_dur=rate_logdur,
                speech_mask=speech_mask,
                valid_mask=valid_mask,
                variant=self.g_variant,
                trim_ratio=self.g_trim_ratio,
                drop_edge_runs=self.g_drop_edge_runs,
                support_mask=support_mask,
                unit_ids=source_batch.content_units,
                unit_prior=unit_prior if isinstance(unit_prior, torch.Tensor) else None,
            )
        except ValueError:
            return None

    @staticmethod
    def _compute_source_prefix_mean(
        *,
        source_rate_seq: torch.Tensor | None,
        speech_commit_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if not isinstance(source_rate_seq, torch.Tensor):
            return None
        mask = speech_commit_mask.float().clamp(0.0, 1.0)
        mass = mask.sum(dim=1, keepdim=True)
        if not bool((mass > 0.0).any().item()):
            return None
        return (source_rate_seq.float() * mask).sum(dim=1, keepdim=True) / mass.clamp_min(1.0)

    def _compute_source_prefix_rate_seq(
        self,
        *,
        source_batch: SourceUnitBatch,
        speech_commit_mask: torch.Tensor,
        state: DurationRuntimeState | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        valid_mask = source_batch.unit_mask.float()
        if valid_mask.numel() <= 0:
            return None, None
        log_dur = torch.log(source_batch.source_duration_obs.float().clamp_min(1.0e-4)) * valid_mask
        if self.use_log_base_rate:
            if isinstance(getattr(source_batch, "unit_rate_log_base", None), torch.Tensor):
                log_base = source_batch.unit_rate_log_base.float().detach() * valid_mask
            else:
                log_base = torch.zeros_like(log_dur)
            observed_log = (log_dur - log_base) * valid_mask
        else:
            observed_log = log_dur
        init_rate = (
            state.local_rate_ema.float()
            if isinstance(getattr(state, "local_rate_ema", None), torch.Tensor)
            else None
        )
        default_init_rate = getattr(
            self.duration_head,
            "src_rate_init",
            observed_log.new_zeros((valid_mask.size(0), 1)),
        )
        if self.src_rate_init_mode == "zero":
            default_init_rate = observed_log.new_zeros((valid_mask.size(0), 1))
        elif self.src_rate_init_mode == "first_speech":
            default_init_rate = first_valid_speech_init(
                observed_log,
                speech_commit_mask.float() * valid_mask,
            )
        prefix_weight = None
        if (
            self.g_variant in {"weighted_median", "softclean_wmed", "softclean_wtmean"}
            and isinstance(getattr(source_batch, "source_run_stability", None), torch.Tensor)
        ):
            prefix_weight = (
                source_batch.source_run_stability.float().clamp(0.0, 1.0)
                * speech_commit_mask.float()
                * valid_mask
            )
        boundary_confidence = self._resolve_source_boundary_confidence(source_batch)
        source_rate_seq, source_rate_final = build_causal_source_prefix_rate_seq(
            observed_log=observed_log,
            speech_mask=speech_commit_mask.float() * valid_mask,
            init_rate=init_rate,
            default_init_rate=default_init_rate,
            stat_mode=self.src_prefix_stat_mode,
            decay=float(getattr(self.duration_head, "local_rate_decay", 0.95)),
            variant=self.g_variant,
            trim_ratio=self.g_trim_ratio,
            min_support=self.src_prefix_min_support,
            weight=prefix_weight,
            valid_mask=valid_mask,
            closed_mask=(
                source_batch.sealed_mask.float()
                if isinstance(getattr(source_batch, "sealed_mask", None), torch.Tensor)
                else None
            ),
            boundary_confidence=(
                boundary_confidence * valid_mask
                if isinstance(boundary_confidence, torch.Tensor)
                else None
            ),
            min_boundary_confidence=self.min_boundary_confidence_for_g,
            drop_edge_runs=self.g_drop_edge_runs,
            min_speech_ratio=0.0,
            unit_ids=source_batch.content_units.long(),
        )
        return source_rate_seq * valid_mask, source_rate_final

    @staticmethod
    def _resolve_runtime_state(
        *,
        state: DurationRuntimeState | None,
        batch_size: int,
        device: torch.device,
        init_state,
    ) -> DurationRuntimeState:
        if state is None:
            return init_state(batch_size=batch_size, device=device)
        return ensure_duration_runtime_state_batch(state, batch_size=batch_size)

    @staticmethod
    def _resolve_commit_mask(source_batch: SourceUnitBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unit_mask = source_batch.unit_mask.float()
        sealed_mask = source_batch.sealed_mask.float() if source_batch.sealed_mask is not None else unit_mask
        commit_mask = unit_mask * sealed_mask
        return unit_mask, sealed_mask, commit_mask

    @staticmethod
    def _build_causal_source_rate_context(
        *,
        source_batch: SourceUnitBatch,
        speech_commit_mask: torch.Tensor,
        state: DurationRuntimeState,
        decay: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed_log = torch.log(source_batch.source_duration_obs.float().clamp_min(1.0e-4))
        context = observed_log.new_zeros(observed_log.shape)
        final_rate = observed_log.new_zeros((observed_log.size(0), 1))
        if state.local_rate_ema is None:
            prev_rate = final_rate.new_zeros(final_rate.shape)
        else:
            prev_rate = state.local_rate_ema.float()
        has_history = state.committed_units.long() > 0
        decay = float(max(0.0, min(0.999, decay)))
        for batch_idx in range(int(observed_log.size(0))):
            ema = float(prev_rate[batch_idx, 0].item()) if bool(has_history[batch_idx].item()) else None
            for unit_idx in range(int(observed_log.size(1))):
                is_speech = float(speech_commit_mask[batch_idx, unit_idx].item()) > 0.5
                obs = float(observed_log[batch_idx, unit_idx].item())
                if ema is None:
                    context[batch_idx, unit_idx] = obs if is_speech else 0.0
                else:
                    context[batch_idx, unit_idx] = ema
                if is_speech:
                    ema = obs if ema is None else ((decay * ema) + ((1.0 - decay) * obs))
            if ema is None:
                ema = float(prev_rate[batch_idx, 0].item()) if bool(has_history[batch_idx].item()) else 0.0
            final_rate[batch_idx, 0] = ema
        return context, final_rate

    def _resolve_prediction_anchor(
        self,
        *,
        source_batch: SourceUnitBatch,
        speech_commit_mask: torch.Tensor,
        commit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del speech_commit_mask, commit_mask
        if self.backbone_mode == "prompt_summary":
            observed = source_batch.source_duration_obs.float().clamp_min(1.0e-4)
            baseline = source_batch.unit_anchor_base.float().clamp_min(1.0e-4)
            visible_mask = source_batch.unit_mask.float()
            return torch.where(visible_mask > 0.5, observed, baseline * visible_mask)
        return source_batch.unit_anchor_base.float()

    @staticmethod
    def _update_prompt_summary_state(
        *,
        final_rate_ema: torch.Tensor,
        next_state: DurationRuntimeState,
    ) -> DurationRuntimeState:
        return replace(
            next_state,
            local_rate_ema=final_rate_ema.detach(),
            since_last_boundary=(
                next_state.since_last_boundary.detach()
                if isinstance(next_state.since_last_boundary, torch.Tensor)
                else final_rate_ema.new_zeros(final_rate_ema.shape)
            ),
        )

    @staticmethod
    def _predict_unit_duration(
        *,
        prediction_anchor: torch.Tensor,
        unit_logstretch: torch.Tensor,
    ) -> torch.Tensor:
        return prediction_anchor.float() * torch.exp(unit_logstretch)

    def forward(
        self,
        *,
        source_batch: SourceUnitBatch,
        ref_memory: ReferenceDurationMemory,
        state: DurationRuntimeState | None = None,
    ) -> DurationExecution:
        batch_size = int(source_batch.content_units.size(0))
        ref_memory = validate_reference_duration_memory(ref_memory)
        ref_memory = ensure_reference_duration_memory_batch(ref_memory, batch_size=batch_size)
        _assert_minimal_reference_memory(
            minimal_v1_profile=self.minimal_v1_profile,
            ref_memory=ref_memory,
            context="forward",
        )
        state = self._resolve_runtime_state(
            state=state,
            batch_size=batch_size,
            device=source_batch.content_units.device,
            init_state=self.init_state,
        )
        unit_mask, sealed_mask, commit_mask = self._resolve_commit_mask(source_batch)
        detached_log_anchor = torch.log(source_batch.unit_anchor_base.float().detach().clamp_min(1.0e-6))
        basis_activation = self._encode_source_basis(
            source_batch=source_batch,
            unit_mask=unit_mask,
            log_anchor=detached_log_anchor,
        )
        speech_commit_mask = self._resolve_speech_commit_mask(
            source_batch=source_batch,
            commit_mask=commit_mask,
        )
        prediction_anchor = self._resolve_prediction_anchor(
            source_batch=source_batch,
            speech_commit_mask=speech_commit_mask,
            commit_mask=commit_mask,
        )
        if self.backbone_mode == "prompt_summary":
            if self.duration_head is None:
                raise RuntimeError(
                    f"{_prompt_summary_public_with_aliases(minimal_v1_profile=self.is_minimal_v1)} is missing StreamingDurationHead."
                )
            local_rate_ema = (
                state.local_rate_ema.float()
                if isinstance(getattr(state, "local_rate_ema", None), torch.Tensor)
                else None
            )
            log_base = None
            if self.use_log_base_rate:
                if isinstance(getattr(source_batch, "unit_rate_log_base", None), torch.Tensor):
                    log_base = source_batch.unit_rate_log_base.float()
                else:
                    log_base = torch.log(source_batch.unit_anchor_base.float().clamp_min(1.0e-4))
            # Minimal v1 uses the prompt_summary backbone too, but it never consumes the
            # optional reference-summary tensors gated by use_reference_summary.
            summary_state = ref_memory.summary_state if self.use_reference_summary else None
            role_value = ref_memory.role_value if self.use_reference_summary else None
            role_var = ref_memory.role_var if self.use_reference_summary else None
            role_coverage = ref_memory.role_coverage if self.use_reference_summary else None
            role_plan = self.duration_head(
                content_units=source_batch.content_units,
                log_anchor=torch.log(prediction_anchor.clamp_min(1.0e-4)),
                log_base=log_base,
                unit_mask=unit_mask,
                sealed_mask=sealed_mask,
                sep_hint=(source_batch.sep_mask.float() if isinstance(getattr(source_batch, "sep_mask", None), torch.Tensor) else unit_mask.new_zeros(unit_mask.shape)),
                edge_cue=(
                    source_batch.source_boundary_cue.float()
                    if isinstance(getattr(source_batch, "source_boundary_cue", None), torch.Tensor)
                    else unit_mask.new_zeros(unit_mask.shape)
                ),
                phrase_final_mask=getattr(source_batch, "phrase_final_mask", None),
                global_rate=ref_memory.global_rate,
                summary_state=summary_state,
                spk_embed=ref_memory.spk_embed,
                role_value=role_value,
                role_var=role_var,
                role_coverage=role_coverage,
                local_rate_ema=local_rate_ema,
                silence_mask=getattr(source_batch, "source_silence_mask", None),
                run_stability=getattr(source_batch, "source_run_stability", None),
            )
            if self.minimal_v1_profile and isinstance(getattr(source_batch, "source_silence_mask", None), torch.Tensor):
                residual = role_plan.get("unit_residual_logstretch")
                if isinstance(residual, torch.Tensor):
                    silence_local = residual.float().abs() * source_batch.source_silence_mask.float().clamp(0.0, 1.0)
                    if bool((silence_local > 1.0e-6).any().item()):
                        raise RuntimeError("rhythm_v3_minimal_v1_profile forbids silence local residual.")
            prompt_domain_valid = getattr(ref_memory, "prompt_g_domain_valid", None)
            if self.minimal_v1_profile and isinstance(prompt_domain_valid, torch.Tensor):
                invalid_prompt_rows = prompt_domain_valid.float().reshape(
                    int(prompt_domain_valid.size(0)),
                    -1,
                )[:, :1] <= 0.5
                if bool(invalid_prompt_rows.any().item()):
                    role_plan = self._neutralize_minimal_prompt_invalid_rows(
                        role_plan=role_plan,
                        invalid_rows=invalid_prompt_rows,
                    )
            unit_logstretch = role_plan["unit_logstretch"] * commit_mask.float()
            unit_duration_exec = self._predict_unit_duration(
                prediction_anchor=prediction_anchor,
                unit_logstretch=unit_logstretch,
            )
            unit_logstretch_raw = unit_logstretch.clone()
            unit_duration_raw = unit_duration_exec.clone()
            unit_duration_exec, unit_logstretch = self._freeze_committed_prefix(
                unit_duration_exec=unit_duration_exec,
                unit_logstretch=unit_logstretch,
                unit_anchor_base=prediction_anchor.float().clamp_min(1.0e-4),
                state=state,
            )
            execution = self.projector.finalize_execution(
                unit_logstretch=unit_logstretch,
                unit_duration_exec=unit_duration_exec,
                basis_activation=basis_activation * 0.0,
                source_duration_obs=source_batch.source_duration_obs.float(),
                unit_mask=unit_mask,
                sealed_mask=sealed_mask,
                speech_commit_mask=speech_commit_mask,
                state=state,
                progress_response=None,
                detector_response=None,
                local_response=role_plan["unit_residual_logstretch"] * speech_commit_mask.float(),
                role_attn_unit=role_plan["role_attn_unit"] * unit_mask.unsqueeze(-1),
                role_value_unit=role_plan["role_value_unit"] * unit_mask,
                role_var_unit=role_plan["role_var_unit"] * unit_mask,
                role_conf_unit=role_plan["role_conf_unit"] * unit_mask,
                unit_logstretch_raw=unit_logstretch_raw,
                unit_duration_raw=unit_duration_raw,
                coarse_only_commit_mask=(
                    commit_mask.float() * source_batch.source_silence_mask.float().clamp(0.0, 1.0)
                    if isinstance(getattr(source_batch, "source_silence_mask", None), torch.Tensor)
                    else None
                ),
                source_boundary_cue=getattr(source_batch, "source_boundary_cue", None),
                phrase_final_mask=getattr(source_batch, "phrase_final_mask", None),
                global_bias_scalar=role_plan.get("global_bias_scalar"),
                global_shift_analytic=role_plan.get("unit_analytic_logstretch", role_plan.get("unit_global_shift_analytic")),
                coarse_logstretch=role_plan.get("unit_coarse_path_logstretch", role_plan.get("unit_coarse_logstretch")),
                coarse_path_logstretch=role_plan.get("unit_coarse_path_logstretch", role_plan.get("unit_coarse_logstretch")),
                coarse_correction=role_plan.get("unit_coarse_delta", role_plan.get("unit_coarse_correction")),
                coarse_correction_pred=role_plan.get(
                    "unit_coarse_correction_pred",
                    role_plan.get("unit_coarse_correction_predicted"),
                ),
                local_residual=role_plan.get("unit_local_residual_used", role_plan.get("unit_residual_logstretch")),
                local_residual_pred=role_plan.get(
                    "unit_residual_logstretch_pred",
                    role_plan.get("local_response_pred"),
                ),
                speech_pred=role_plan.get("unit_speech_pred"),
                silence_pred=role_plan.get("unit_silence_pred"),
                source_rate_seq=role_plan.get("source_rate_seq"),
                source_prefix_summary=role_plan.get("source_prefix_summary"),
            )
            execution.next_state = self._update_prompt_summary_state(
                final_rate_ema=role_plan["local_rate_final"],
                next_state=execution.next_state,
            )
            execution.g_ref = (
                role_plan.get("g_ref")
                if isinstance(role_plan.get("g_ref"), torch.Tensor)
                else ref_memory.global_rate.detach()
            )
            execution.g_src_prefix = (
                role_plan.get("g_src_prefix")
                if isinstance(role_plan.get("g_src_prefix"), torch.Tensor)
                else execution.source_rate_seq
            )
            execution.g_src_utt = self._compute_source_global_rate(source_batch=source_batch)
            execution.g_src_prefix_mean = self._compute_source_prefix_mean(
                source_rate_seq=execution.g_src_prefix,
                speech_commit_mask=speech_commit_mask,
            )
            execution.coarse_scalar_raw = role_plan.get("coarse_scalar_raw")
            execution.global_term_before_local = role_plan.get("unit_global_term_before_local")
            execution.analytic_gap_clip_value = role_plan.get("analytic_gap_clip_value")
            execution.unit_residual_gate = role_plan.get("unit_residual_gate")
            execution.unit_residual_cold_gate = role_plan.get("unit_residual_cold_gate")
            execution.unit_residual_short_gate = role_plan.get("unit_residual_short_gate")
            execution.unit_residual_gate_stability = role_plan.get("unit_residual_gate_stability")
            execution.unit_runtime_stability = role_plan.get("unit_runtime_stability")
            execution.residual_gate_mean = role_plan.get("residual_gate_mean")
            execution.detach_global_term_in_local_head = role_plan.get("detach_global_term_in_local_head")
            execution.eval_mode = str(role_plan.get("eval_mode", self.eval_mode))
            if isinstance(ref_memory.prompt_valid_mask, torch.Tensor):
                valid_len = ref_memory.prompt_valid_mask.float().sum(dim=1, keepdim=True)
                execution.prompt_valid_len = valid_len
                prompt_speech_ratio = self._compute_prompt_speech_ratio_from_memory(ref_memory)
                if isinstance(prompt_speech_ratio, torch.Tensor):
                    execution.prompt_speech_ratio = prompt_speech_ratio
            return execution
        global_response, structure_response, local_response, source_residual_response = self.backbone(
            module=self,
            ref_memory=ref_memory,
            source_batch=source_batch,
            unit_mask=unit_mask,
            speech_commit_mask=speech_commit_mask,
            detached_log_anchor=detached_log_anchor,
            basis_activation=basis_activation,
        )
        if self.warp_mode == "progress":
            progress_response = structure_response
            detector_response = None
        elif self.warp_mode == "detector":
            progress_response = speech_commit_mask.new_zeros(speech_commit_mask.shape)
            detector_response = structure_response
        else:
            progress_response = structure_response
            detector_response = None
        detector_term = (
            detector_response
            if isinstance(detector_response, torch.Tensor)
            else speech_commit_mask.new_zeros(speech_commit_mask.shape)
        )
        unit_logstretch = global_response + progress_response + detector_term + local_response + source_residual_response
        unit_duration_exec = self._predict_unit_duration(
            prediction_anchor=prediction_anchor,
            unit_logstretch=unit_logstretch,
        )
        unit_logstretch_raw = unit_logstretch.clone()
        unit_duration_raw = unit_duration_exec.clone()
        unit_duration_exec, unit_logstretch = self._freeze_committed_prefix(
            unit_duration_exec=unit_duration_exec,
            unit_logstretch=unit_logstretch,
            unit_anchor_base=prediction_anchor.float().clamp_min(1.0e-4),
            state=state,
        )
        execution = self.projector.finalize_execution(
            unit_logstretch=unit_logstretch,
            unit_duration_exec=unit_duration_exec,
            basis_activation=basis_activation * speech_commit_mask.unsqueeze(-1),
            source_duration_obs=source_batch.source_duration_obs.float(),
            unit_mask=unit_mask,
            sealed_mask=sealed_mask,
            speech_commit_mask=speech_commit_mask,
            state=state,
            progress_response=progress_response * speech_commit_mask,
            detector_response=(
                None
                if detector_response is None
                else detector_response * speech_commit_mask
            ),
            local_response=local_response * speech_commit_mask,
            source_boundary_cue=getattr(source_batch, "source_boundary_cue", None),
            phrase_final_mask=getattr(source_batch, "phrase_final_mask", None),
            unit_logstretch_raw=unit_logstretch_raw,
            unit_duration_raw=unit_duration_raw,
        )
        source_prefix_seq, source_prefix_final = self._compute_source_prefix_rate_seq(
            source_batch=source_batch,
            speech_commit_mask=speech_commit_mask,
            state=state,
        )
        if isinstance(source_prefix_final, torch.Tensor):
            execution.next_state = replace(
                execution.next_state,
                local_rate_ema=source_prefix_final.detach(),
            )
        execution.g_ref = ref_memory.global_rate.detach()
        execution.source_rate_seq = source_prefix_seq
        execution.g_src_prefix = source_prefix_seq
        execution.g_src_utt = self._compute_source_global_rate(source_batch=source_batch)
        execution.g_src_prefix_mean = self._compute_source_prefix_mean(
            source_rate_seq=source_prefix_seq,
            speech_commit_mask=speech_commit_mask,
        )
        execution.eval_mode = self.eval_mode
        if isinstance(ref_memory.prompt_valid_mask, torch.Tensor):
            valid_len = ref_memory.prompt_valid_mask.float().sum(dim=1, keepdim=True)
            execution.prompt_valid_len = valid_len
            prompt_speech_ratio = self._compute_prompt_speech_ratio_from_memory(ref_memory)
            if isinstance(prompt_speech_ratio, torch.Tensor):
                execution.prompt_speech_ratio = prompt_speech_ratio
        return execution


SharedResponseEncoder = SharedCausalBasisEncoder
StreamingDurationModule = MixedEffectsDurationModule

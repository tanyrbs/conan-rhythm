from __future__ import annotations

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


class SharedCausalBasisEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        basis_rank: int,
        prompt_trace_dim: int = 5,
        prompt_stats_dim: int = 6,
        window_left: int = 4,
        window_right: int = 0,
    ) -> None:
        super().__init__()
        self.window_left = max(0, int(window_left))
        self.window_right = max(0, int(window_right))
        kernel_size = self.window_left + self.window_right + 1
        self.unit_embedding = nn.Embedding(vocab_size, hidden_size)
        self.source_adapter = nn.Linear(hidden_size + 1, hidden_size)
        self.prompt_proxy_adapter = nn.Linear(prompt_trace_dim + prompt_stats_dim, hidden_size)
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

    def encode_prompt_proxy(
        self,
        *,
        ref_rhythm_trace: torch.Tensor,
        ref_rhythm_stats: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        if ref_rhythm_trace.size(1) <= 0:
            return self.out_proj.weight.new_zeros((ref_rhythm_trace.size(0), 0, self.out_proj.out_features))
        stats_feat = ref_rhythm_stats.float().unsqueeze(1).expand(-1, ref_rhythm_trace.size(1), -1)
        hidden = F.silu(self.prompt_proxy_adapter(torch.cat([ref_rhythm_trace.float(), stats_feat], dim=-1)))
        return self._run_shared(hidden, prompt_mask.float())


class MixedEffectsDurationModule(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int = 256,
        basis_rank: int = 12,
        response_window_left: int = 4,
        response_window_right: int = 0,
        trace_bins: int = 24,
        streaming_mode: str = "strict",
        micro_lookahead_units: int | None = None,
        ridge_lambda: float = 1.0,
        ablation_mode: str = "coarse_operator",
        source_residual_gain: float = 0.0,
        **unused_kwargs,
    ) -> None:
        super().__init__()
        global_shrink_tau = float(unused_kwargs.pop("global_shrink_tau", 8.0))
        coarse_support_tau = float(unused_kwargs.pop("coarse_support_tau", 8.0))
        coarse_bins = int(unused_kwargs.pop("coarse_bins", 4))
        ridge_support_tau = float(unused_kwargs.pop("ridge_support_tau", 8.0))
        operator_holdout_ratio = float(unused_kwargs.pop("operator_holdout_ratio", 0.30))
        del unused_kwargs
        self.streaming_mode = str(streaming_mode or "strict").strip().lower()
        if self.streaming_mode not in {"strict", "micro_lookahead"}:
            raise ValueError(f"Unsupported streaming_mode={streaming_mode!r}")
        self.ablation_mode = str(ablation_mode or "coarse_operator").strip().lower()
        if self.ablation_mode not in {"global_only", "coarse_only", "operator", "coarse_operator", "operator_srcres"}:
            raise ValueError(f"Unsupported rhythm_v3 ablation mode: {ablation_mode!r}")
        self.source_residual_gain = float(max(0.0, source_residual_gain))
        effective_window_right = int(response_window_right)
        if self.streaming_mode == "strict":
            effective_window_right = 0
        elif micro_lookahead_units is not None:
            effective_window_right = max(0, int(micro_lookahead_units))

        self.response_encoder = SharedCausalBasisEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            basis_rank=basis_rank,
            window_left=response_window_left,
            window_right=effective_window_right,
        )
        self.reference_memory_builder = PromptConditionedOperatorEstimator(
            trace_bins=trace_bins,
            coarse_bins=coarse_bins,
            ridge_lambda=ridge_lambda,
            global_shrink_tau=global_shrink_tau,
            coarse_support_tau=coarse_support_tau,
            ridge_support_tau=ridge_support_tau,
            holdout_ratio=operator_holdout_ratio,
        )
        self.projector = StreamingDurationProjector()

    def _use_coarse_response(self) -> bool:
        return self.ablation_mode in {"coarse_only", "coarse_operator", "operator_srcres"}

    def _use_local_operator(self) -> bool:
        return self.ablation_mode in {"operator", "coarse_operator", "operator_srcres"}

    def init_state(self, batch_size: int, device: torch.device) -> DurationRuntimeState:
        return self.projector.init_state(batch_size=batch_size, device=device)

    def build_reference_conditioning(
        self,
        *,
        ref_conditioning=None,
        ref_rhythm_stats: torch.Tensor | None = None,
        ref_rhythm_trace: torch.Tensor | None = None,
        ref_mel: torch.Tensor | None = None,
        ref_lengths: torch.Tensor | None = None,
    ) -> ReferenceDurationMemory:
        need_coarse = self._use_coarse_response()
        need_operator = self._use_local_operator()
        if ref_conditioning is not None:
            return self.reference_memory_builder(
                response_encoder=self.response_encoder,
                ref_conditioning=ref_conditioning,
                need_coarse=need_coarse,
                need_operator=need_operator,
            )
        if ref_rhythm_stats is not None and ref_rhythm_trace is not None:
            return self.reference_memory_builder(
                response_encoder=self.response_encoder,
                ref_conditioning={
                    "ref_rhythm_stats": ref_rhythm_stats,
                    "ref_rhythm_trace": ref_rhythm_trace,
                },
                need_coarse=need_coarse,
                need_operator=need_operator,
            )
        if ref_mel is None:
            raise ValueError("Either reference conditioning or reference mel must be provided.")
        return self.reference_memory_builder(
            response_encoder=self.response_encoder,
            ref_mel=ref_mel,
            ref_lengths=ref_lengths,
            need_coarse=need_coarse,
            need_operator=need_operator,
        )

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

    def _encode_source_basis(
        self,
        *,
        source_batch: SourceUnitBatch,
        unit_mask: torch.Tensor,
        log_anchor: torch.Tensor,
    ) -> torch.Tensor:
        if not self._use_local_operator():
            batch_size, num_units = source_batch.content_units.shape
            basis_rank = int(self.response_encoder.out_proj.out_features)
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
    def _sample_coarse_response(
        *,
        ref_memory: ReferenceDurationMemory,
        source_batch: SourceUnitBatch,
        speech_commit_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(getattr(ref_memory, "coarse_profile", None), torch.Tensor):
            return speech_commit_mask.new_zeros(speech_commit_mask.shape)
        coarse_profile = ref_memory.coarse_profile.float()
        if coarse_profile.numel() <= 0 or coarse_profile.size(1) <= 0 or source_batch.content_units.size(1) <= 0:
            return speech_commit_mask.new_zeros(speech_commit_mask.shape)
        progress = MixedEffectsDurationModule._build_prefix_progress(
            unit_anchor_base=source_batch.unit_anchor_base,
            speech_commit_mask=speech_commit_mask,
        )
        num_bins = int(coarse_profile.size(1))
        indices = torch.clamp((progress * float(num_bins)).long(), min=0, max=max(0, num_bins - 1))
        sampled = coarse_profile.gather(1, indices)
        return sampled * speech_commit_mask.float()

    @staticmethod
    def _resolve_speech_commit_mask(
        *,
        source_batch: SourceUnitBatch,
        commit_mask: torch.Tensor,
    ) -> torch.Tensor:
        speech_commit_mask = commit_mask.float()
        if isinstance(getattr(source_batch, "sep_mask", None), torch.Tensor):
            speech_commit_mask = speech_commit_mask * (1.0 - source_batch.sep_mask.float().clamp(0.0, 1.0))
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
    def _predict_unit_duration(
        *,
        source_batch: SourceUnitBatch,
        unit_logstretch: torch.Tensor,
    ) -> torch.Tensor:
        return source_batch.unit_anchor_base.float() * torch.exp(unit_logstretch)

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
        local_response = self._predict_local_response(
            basis_activation=basis_activation,
            ref_memory=ref_memory,
        )
        speech_commit_mask = self._resolve_speech_commit_mask(
            source_batch=source_batch,
            commit_mask=commit_mask,
        )
        coarse_response = self._sample_coarse_response(
            ref_memory=ref_memory,
            source_batch=source_batch,
            speech_commit_mask=speech_commit_mask,
        )
        if not self._use_local_operator():
            local_response = torch.zeros_like(local_response)
        if not self._use_coarse_response():
            coarse_response = torch.zeros_like(coarse_response)
        source_residual_response = torch.zeros_like(local_response)
        if self.ablation_mode == "operator_srcres" and self.source_residual_gain > 0.0:
            centered_source_residual = self._build_centered_source_residual(
                source_batch=source_batch,
                detached_log_anchor=detached_log_anchor,
                speech_commit_mask=speech_commit_mask,
            )
            source_residual_response = self.source_residual_gain * centered_source_residual
        global_stretch = ref_memory.global_rate.float().expand(-1, unit_mask.size(1))
        unit_logstretch = (global_stretch + coarse_response + local_response + source_residual_response) * speech_commit_mask
        unit_duration_exec = self._predict_unit_duration(
            source_batch=source_batch,
            unit_logstretch=unit_logstretch,
        )
        unit_duration_exec, unit_logstretch = self._freeze_committed_prefix(
            unit_duration_exec=unit_duration_exec,
            unit_logstretch=unit_logstretch,
            unit_anchor_base=source_batch.unit_anchor_base,
            state=state,
        )
        return self.projector.finalize_execution(
            unit_logstretch=unit_logstretch,
            unit_duration_exec=unit_duration_exec,
            basis_activation=basis_activation * speech_commit_mask.unsqueeze(-1),
            source_duration_obs=source_batch.source_duration_obs,
            unit_mask=unit_mask,
            sealed_mask=sealed_mask,
            state=state,
            coarse_response=coarse_response * speech_commit_mask,
            local_response=local_response * speech_commit_mask,
        )


SharedResponseEncoder = SharedCausalBasisEncoder
StreamingDurationModule = MixedEffectsDurationModule

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
    resolved_backbone = str(backbone_mode or "global_only").strip().lower()
    resolved_warp = str(warp_mode or "none").strip().lower()
    resolved_allow_hybrid = bool(allow_hybrid) if allow_hybrid is not None else False
    if resolved_backbone not in {"global_only", "operator"}:
        raise ValueError(f"Unsupported rhythm_v3 backbone mode: {backbone_mode!r}")
    if resolved_warp not in {"none", "progress", "detector"}:
        raise ValueError(f"Unsupported rhythm_v3 warp mode: {warp_mode!r}")
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
        global_shrink_tau = float(unused_kwargs.pop("global_shrink_tau", 8.0))
        progress_support_tau = float(unused_kwargs.pop("progress_support_tau", 8.0))
        progress_bins = int(unused_kwargs.pop("progress_bins", 4))
        ridge_support_tau = float(unused_kwargs.pop("ridge_support_tau", 8.0))
        operator_holdout_ratio = float(unused_kwargs.pop("operator_holdout_ratio", 0.30))
        del unused_kwargs
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
            progress_bins=progress_bins,
            ridge_lambda=ridge_lambda,
            global_shrink_tau=global_shrink_tau,
            progress_support_tau=progress_support_tau,
            ridge_support_tau=ridge_support_tau,
            holdout_ratio=operator_holdout_ratio,
        )
        self.projector = StreamingDurationProjector()
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
        speech_commit_mask = self._resolve_speech_commit_mask(
            source_batch=source_batch,
            commit_mask=commit_mask,
        )
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
            source_batch=source_batch,
            unit_logstretch=unit_logstretch,
        )
        unit_logstretch_raw = unit_logstretch.clone()
        unit_duration_raw = unit_duration_exec.clone()
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
            speech_commit_mask=speech_commit_mask,
            state=state,
            progress_response=progress_response * speech_commit_mask,
            detector_response=(
                None
                if detector_response is None
                else detector_response * speech_commit_mask
            ),
            local_response=local_response * speech_commit_mask,
            unit_logstretch_raw=unit_logstretch_raw,
            unit_duration_raw=unit_duration_raw,
        )


SharedResponseEncoder = SharedCausalBasisEncoder
StreamingDurationModule = MixedEffectsDurationModule

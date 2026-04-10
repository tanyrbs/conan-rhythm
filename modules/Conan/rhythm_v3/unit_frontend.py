from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.Conan.rhythm.unit_frontend import RhythmUnitFrontend

from .contracts import SourceUnitBatch


class TableDurationPrior(nn.Module):
    """Frozen speaker-free unigram prior for nominal duration anchoring."""

    def __init__(self, *, vocab_size: int, default_log_anchor: float) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.default_log_anchor = float(default_log_anchor)
        self.register_buffer("log_prior_delta", torch.zeros(self.vocab_size))
        self.register_buffer("prior_mask", torch.zeros(self.vocab_size))

    def forward(self, content_units: torch.Tensor, unit_mask: torch.Tensor) -> torch.Tensor:
        if content_units.size(1) <= 0:
            return unit_mask.new_zeros(unit_mask.shape, dtype=torch.float32)
        mask = unit_mask.float()
        indices = content_units.long().clamp(min=0, max=max(0, self.vocab_size - 1))
        delta = self.log_prior_delta[indices]
        active = self.prior_mask[indices]
        return delta * active * mask

    def load_prior_tensor(self, prior: torch.Tensor, *, is_log: bool = False) -> None:
        prior = prior.detach().float().reshape(-1)
        if prior.numel() != self.vocab_size:
            raise ValueError(
                f"Baseline table prior size mismatch: expected {self.vocab_size}, got {int(prior.numel())}."
            )
        finite = torch.isfinite(prior)
        prior = torch.where(finite, prior, torch.zeros_like(prior))
        if not is_log:
            prior = torch.log(prior.clamp_min(1.0e-6))
        delta = prior - float(self.default_log_anchor)
        delta = torch.where(finite, delta, torch.zeros_like(delta))
        with torch.no_grad():
            self.log_prior_delta.copy_(delta.to(device=self.log_prior_delta.device, dtype=self.log_prior_delta.dtype))
            self.prior_mask.copy_(finite.float().to(device=self.prior_mask.device, dtype=self.prior_mask.dtype))

    def load_prior_file(self, path: str | Path) -> None:
        payload = torch.load(str(path), map_location="cpu")
        is_log = False
        prior = payload
        if isinstance(payload, Mapping):
            if isinstance(payload.get("log_anchor_prior"), torch.Tensor):
                prior = payload["log_anchor_prior"]
                is_log = True
            elif isinstance(payload.get("baseline_log_prior"), torch.Tensor):
                prior = payload["baseline_log_prior"]
                is_log = True
            elif isinstance(payload.get("anchor_prior_frames"), torch.Tensor):
                prior = payload["anchor_prior_frames"]
            elif isinstance(payload.get("anchor_prior"), torch.Tensor):
                prior = payload["anchor_prior"]
            elif isinstance(payload.get("table_prior"), torch.Tensor):
                prior = payload["table_prior"]
            elif isinstance(payload.get("unit_anchor_base"), torch.Tensor):
                prior = payload["unit_anchor_base"]
            else:
                raise ValueError(
                    "Baseline table prior file must contain one of: "
                    "log_anchor_prior, baseline_log_prior, anchor_prior_frames, anchor_prior, table_prior, unit_anchor_base."
                )
        if not isinstance(prior, torch.Tensor):
            raise TypeError(f"Baseline table prior must be a tensor, got {type(prior)!r}.")
        self.load_prior_tensor(prior, is_log=is_log)


class StrictLocalDurationTrunk(nn.Module):
    """Strictly causal local content encoder for nominal-duration residuals."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int = 128,
        kernel_size: int = 5,
        delta_scale: float = 0.75,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.delta_scale = float(max(1.0e-3, delta_scale))
        self.unit_embedding = nn.Embedding(vocab_size, hidden_size)
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(
        self,
        *,
        content_units: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = unit_mask.float()
        if content_units.size(1) <= 0:
            hidden = self.in_proj.weight.new_zeros((content_units.size(0), 0, self.hidden_size))
            return hidden, hidden.new_zeros(mask.shape)
        embed = self.unit_embedding(content_units.long())
        hidden = F.silu(self.in_proj(embed))
        conv_input = F.pad(hidden.transpose(1, 2), (self.conv.kernel_size[0] - 1, 0))
        conv_hidden = self.conv(conv_input).transpose(1, 2)
        hidden = F.silu(hidden + conv_hidden)
        delta = self.delta_scale * torch.tanh(self.out_proj(hidden).squeeze(-1))
        return hidden, delta * mask


class ProtocolDurationBaseline(nn.Module):
    """Nominal-duration baseline as a protocol, not a single opaque regressor."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int = 128,
        kernel_size: int = 5,
        min_frames: float = 1.0,
        max_frames: float = 12.0,
    ) -> None:
        super().__init__()
        self.min_frames = float(max(0.25, min_frames))
        self.max_frames = float(max(self.min_frames + 1.0e-3, max_frames))
        default_log_anchor = 0.5 * (
            torch.log(torch.tensor(self.min_frames)) + torch.log(torch.tensor(self.max_frames))
        )
        self.register_buffer("log_min_frames", torch.log(torch.tensor(self.min_frames)))
        self.register_buffer("log_max_frames", torch.log(torch.tensor(self.max_frames)))
        self.register_buffer("default_log_anchor", default_log_anchor.float())
        self.table_prior = TableDurationPrior(
            vocab_size=vocab_size,
            default_log_anchor=float(self.default_log_anchor.item()),
        )
        self.local_trunk = StrictLocalDurationTrunk(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
        )

    def forward(
        self,
        content_units: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = unit_mask.float()
        if content_units.size(1) <= 0:
            return mask.new_zeros(mask.shape, dtype=torch.float32)
        table_delta = self.table_prior(content_units, mask)
        _, local_delta = self.local_trunk(content_units=content_units, unit_mask=mask)
        log_anchor = (
            self.default_log_anchor
            + table_delta
            + local_delta
        )
        log_anchor = log_anchor.clamp(
            min=float(self.log_min_frames.item()),
            max=float(self.log_max_frames.item()),
        )
        return torch.exp(log_anchor) * mask

    def freeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze_parameters(self) -> None:
        for _, param in self.named_parameters():
            param.requires_grad_(True)

    def load_table_prior_file(self, path: str | Path) -> None:
        self.table_prior.load_prior_file(path)

    def load_checkpoint(self, path: str | Path, *, strict: bool = True) -> None:
        payload = torch.load(str(path), map_location="cpu")
        if isinstance(payload, Mapping):
            if isinstance(payload.get("state_dict"), Mapping):
                payload = payload["state_dict"]
            elif isinstance(payload.get("model"), Mapping):
                payload = payload["model"]
        if not isinstance(payload, Mapping):
            raise TypeError(f"Baseline checkpoint must be a mapping/state_dict, got {type(payload)!r}.")
        state_dict = {}
        own_keys = set(self.state_dict().keys())
        for key, value in payload.items():
            normalized = str(key)
            for prefix in ("rhythm_unit_frontend.baseline.", "baseline."):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix) :]
                    break
            if normalized in own_keys:
                state_dict[normalized] = value
        if not state_dict and all(str(key) in own_keys for key in payload.keys()):
            state_dict = dict(payload)
        if not state_dict:
            raise ValueError(
                "Baseline checkpoint does not contain ProtocolDurationBaseline weights. "
                "Expected keys under rhythm_unit_frontend.baseline.* or baseline.*."
            )
        self.load_state_dict(state_dict, strict=strict)


class DurationUnitFrontend(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        silent_token: int | None = None,
        separator_aware: bool = True,
        tail_open_units: int = 1,
        anchor_hidden_size: int = 128,
        anchor_min_frames: float = 1.0,
        anchor_max_frames: float = 12.0,
        phrase_boundary_threshold: float = 0.55,
    ) -> None:
        super().__init__()
        self.phrase_boundary_threshold = float(max(0.0, min(1.0, phrase_boundary_threshold)))
        self.base_frontend = RhythmUnitFrontend(
            silent_token=silent_token,
            separator_aware=separator_aware,
            tail_open_units=tail_open_units,
        )
        self.baseline = ProtocolDurationBaseline(
            vocab_size=vocab_size,
            hidden_size=anchor_hidden_size,
            min_frames=anchor_min_frames,
            max_frames=anchor_max_frames,
        )
        # Compatibility alias used by some legacy callers.
        self.anchor_net = self.baseline

    @staticmethod
    def _validate_precomputed_shapes(**named_tensors) -> None:
        expected_shape = None
        for name, value in named_tensors.items():
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor when provided, got {type(value)!r}")
            if value.dim() != 2:
                raise ValueError(f"{name} must be rank-2 [B, U], got shape={tuple(value.shape)}")
            if expected_shape is None:
                expected_shape = tuple(value.shape)
                continue
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"Precomputed source cache shape mismatch for {name}: expected {expected_shape}, got {tuple(value.shape)}"
                )

    def get_baseline_module(self) -> ProtocolDurationBaseline:
        return self.baseline

    def freeze_baseline(self) -> None:
        self.baseline.freeze_parameters()

    def unfreeze_baseline(self) -> None:
        self.baseline.unfreeze_parameters()

    def load_baseline_checkpoint(self, path: str | Path, *, strict: bool = True) -> None:
        self.baseline.load_checkpoint(path, strict=strict)

    def load_table_prior_file(self, path: str | Path) -> None:
        self.baseline.load_table_prior_file(path)

    def compute_baseline(
        self,
        content_units: torch.Tensor,
        unit_mask: torch.Tensor,
        *,
        stop_gradient: bool = False,
    ) -> torch.Tensor:
        anchor = self.baseline(
            content_units=content_units,
            unit_mask=unit_mask,
        )
        return anchor.detach() if stop_gradient else anchor

    def resolve_anchor_base_from_units(
        self,
        content_units: torch.Tensor,
        unit_mask: torch.Tensor,
        *,
        stop_gradient: bool = False,
    ) -> torch.Tensor:
        return self.compute_baseline(
            content_units=content_units,
            unit_mask=unit_mask,
            stop_gradient=stop_gradient,
        )

    def _resolve_anchor_base(
        self,
        *,
        base_batch,
        unit_anchor_base: torch.Tensor | None,
    ) -> torch.Tensor:
        if unit_anchor_base is not None:
            return unit_anchor_base.float() * base_batch.unit_mask.float()
        return self.compute_baseline(
            content_units=base_batch.content_units,
            unit_mask=base_batch.unit_mask,
        )

    def _convert_batch(
        self,
        base_batch,
        *,
        unit_anchor_base: torch.Tensor | None = None,
        source_boundary_cue: torch.Tensor | None = None,
        phrase_group_index: torch.Tensor | None = None,
        phrase_group_pos: torch.Tensor | None = None,
        phrase_final_mask: torch.Tensor | None = None,
    ) -> SourceUnitBatch:
        unit_mask = base_batch.unit_mask.float()
        sep_mask = base_batch.sep_hint.float() * unit_mask
        resolved_anchor_base = self._resolve_anchor_base(base_batch=base_batch, unit_anchor_base=unit_anchor_base)
        (
            resolved_boundary,
            resolved_phrase_index,
            resolved_phrase_pos,
            resolved_phrase_final,
        ) = self._resolve_phrase_sidecars(
            base_batch=base_batch,
            unit_mask=unit_mask,
            source_boundary_cue=source_boundary_cue,
            phrase_group_index=phrase_group_index,
            phrase_group_pos=phrase_group_pos,
            phrase_final_mask=phrase_final_mask,
        )
        return SourceUnitBatch(
            content_units=base_batch.content_units,
            source_duration_obs=base_batch.dur_anchor_src.float() * unit_mask,
            unit_anchor_base=resolved_anchor_base,
            unit_mask=unit_mask,
            sealed_mask=base_batch.sealed_mask.float() * unit_mask,
            sep_mask=sep_mask,
            source_boundary_cue=resolved_boundary,
            phrase_group_index=resolved_phrase_index,
            phrase_group_pos=resolved_phrase_pos,
            phrase_final_mask=resolved_phrase_final,
        )

    def _resolve_phrase_sidecars(
        self,
        *,
        base_batch,
        unit_mask: torch.Tensor,
        source_boundary_cue: torch.Tensor | None,
        phrase_group_index: torch.Tensor | None,
        phrase_group_pos: torch.Tensor | None,
        phrase_final_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total_units = int(base_batch.content_units.size(1))
        boundary = (
            source_boundary_cue.float() * unit_mask
            if isinstance(source_boundary_cue, torch.Tensor)
            else base_batch.boundary_confidence.float() * unit_mask
        )
        if (
            isinstance(phrase_group_index, torch.Tensor)
            and isinstance(phrase_group_pos, torch.Tensor)
            and isinstance(phrase_final_mask, torch.Tensor)
        ):
            return (
                boundary,
                phrase_group_index.long() * unit_mask.long(),
                phrase_group_pos.float() * unit_mask,
                phrase_final_mask.float() * unit_mask,
            )
        resolved_index = torch.zeros_like(base_batch.content_units, dtype=torch.long)
        resolved_pos = torch.zeros_like(boundary)
        resolved_final = torch.zeros_like(boundary)
        threshold = float(self.phrase_boundary_threshold)
        for batch_idx in range(base_batch.content_units.size(0)):
            visible = int(unit_mask[batch_idx].sum().item())
            if visible <= 0:
                continue
            boundary_row = boundary[batch_idx, :visible].float()
            sep_row = base_batch.sep_hint[batch_idx, :visible].float()
            break_mask = torch.maximum((boundary_row >= threshold).float(), sep_row)
            phrase_starts = [0]
            for idx in range(max(visible - 1, 0)):
                if float(break_mask[idx].item()) > 0.5:
                    phrase_starts.append(idx + 1)
                    resolved_final[batch_idx, idx] = 1.0
            resolved_final[batch_idx, visible - 1] = 1.0
            phrase_starts = sorted(set(int(x) for x in phrase_starts if 0 <= int(x) < visible))
            for group_id, start in enumerate(phrase_starts):
                end = phrase_starts[group_id + 1] if group_id + 1 < len(phrase_starts) else visible
                length = max(1, end - start)
                resolved_index[batch_idx, start:end] = group_id
                if length == 1:
                    resolved_pos[batch_idx, start] = 1.0
                else:
                    resolved_pos[batch_idx, start:end] = torch.linspace(
                        0.0,
                        1.0,
                        steps=length,
                        device=boundary.device,
                        dtype=boundary.dtype,
                    )
        return (
            boundary,
            resolved_index * unit_mask.long(),
            resolved_pos * unit_mask,
            resolved_final * unit_mask,
        )

    def from_content_tensor(
        self,
        content: torch.Tensor,
        *,
        content_lengths: torch.Tensor | None = None,
        mark_last_open: bool = True,
    ) -> SourceUnitBatch:
        base_batch = self.base_frontend.from_content_tensor(
            content,
            content_lengths=content_lengths,
            mark_last_open=mark_last_open,
        )
        return self._convert_batch(base_batch)

    def from_precomputed(
        self,
        *,
        content_units: torch.Tensor,
        source_duration_obs: torch.Tensor,
        unit_mask: torch.Tensor | None = None,
        sealed_mask: torch.Tensor | None = None,
        sep_mask: torch.Tensor | None = None,
        unit_anchor_base: torch.Tensor | None = None,
        source_boundary_cue: torch.Tensor | None = None,
        phrase_group_index: torch.Tensor | None = None,
        phrase_group_pos: torch.Tensor | None = None,
        phrase_final_mask: torch.Tensor | None = None,
    ) -> SourceUnitBatch:
        self._validate_precomputed_shapes(
            content_units=content_units,
            source_duration_obs=source_duration_obs,
            unit_mask=unit_mask,
            sealed_mask=sealed_mask,
            sep_mask=sep_mask,
            unit_anchor_base=unit_anchor_base,
            source_boundary_cue=source_boundary_cue,
            phrase_group_index=phrase_group_index,
            phrase_group_pos=phrase_group_pos,
            phrase_final_mask=phrase_final_mask,
        )
        base_batch = self.base_frontend.from_precomputed(
            content_units=content_units,
            dur_anchor_src=source_duration_obs,
            unit_mask=unit_mask,
            sealed_mask=sealed_mask,
            sep_hint=sep_mask,
        )
        return self._convert_batch(
            base_batch,
            unit_anchor_base=unit_anchor_base,
            source_boundary_cue=source_boundary_cue,
            phrase_group_index=phrase_group_index,
            phrase_group_pos=phrase_group_pos,
            phrase_final_mask=phrase_final_mask,
        )

    def init_stream_state(self, batch_size: int, *, device: torch.device | None = None):
        return self.base_frontend.init_stream_state(batch_size=batch_size, device=device)

    def step_content_tensor(
        self,
        content: torch.Tensor,
        state,
        *,
        content_lengths: torch.Tensor | None = None,
        mark_last_open: bool = True,
    ):
        base_batch, next_state = self.base_frontend.step_content_tensor(
            content,
            state,
            content_lengths=content_lengths,
            mark_last_open=mark_last_open,
        )
        return self._convert_batch(base_batch), next_state


NominalDurationBaseline = ProtocolDurationBaseline
AnchorNet = ProtocolDurationBaseline

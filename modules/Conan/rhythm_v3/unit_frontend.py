from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import SourceUnitBatch
from .unitizer import (
    StreamingRunLengthUnitizer,
    StreamingUnitizerRowState,
    StreamingUnitizerState,
    _estimate_boundary_confidence_tensor,
    build_compressed_sequence,
)


class RhythmUnitBatch:
    def __init__(
        self,
        *,
        content_units: torch.Tensor,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor,
        open_run_mask: torch.Tensor,
        sealed_mask: torch.Tensor,
        sep_hint: torch.Tensor,
        boundary_confidence: torch.Tensor,
    ) -> None:
        self.content_units = content_units
        self.dur_anchor_src = dur_anchor_src
        self.unit_mask = unit_mask
        self.open_run_mask = open_run_mask
        self.sealed_mask = sealed_mask
        self.sep_hint = sep_hint
        self.boundary_confidence = boundary_confidence


class RhythmUnitFrontend:
    """Local v3 copy of the prefix-safe token->unit frontend."""

    def __init__(
        self,
        *,
        silent_token: int | None = None,
        separator_aware: bool = True,
        tail_open_units: int = 1,
    ) -> None:
        self.silent_token = silent_token
        self.separator_aware = bool(separator_aware)
        self.tail_open_units = max(1, int(tail_open_units))
        self.unitizer = StreamingRunLengthUnitizer(
            silent_token=silent_token,
            separator_aware=separator_aware,
            tail_open_units=tail_open_units,
        )

    @staticmethod
    def _pad(seqs: list[torch.Tensor], pad_value: int = 0, *, dtype=torch.long) -> torch.Tensor:
        max_len = max((seq.numel() for seq in seqs), default=0)
        out = torch.full((len(seqs), max_len), pad_value, dtype=dtype)
        for idx, seq in enumerate(seqs):
            if seq.numel() > 0:
                out[idx, : seq.numel()] = seq.to(dtype=dtype)
        return out

    @staticmethod
    def _move_row_state(row: StreamingUnitizerRowState, device: torch.device) -> StreamingUnitizerRowState:
        return StreamingUnitizerRowState(
            units=row.units.to(device=device, dtype=torch.long),
            durations=row.durations.to(device=device, dtype=torch.long),
            sep_hint=row.sep_hint.to(device=device, dtype=torch.long),
            last_token=int(row.last_token),
            pending_separator=bool(row.pending_separator),
        )

    def _batch_from_compressed(self, compressed_list: list, *, device: torch.device | None = None) -> RhythmUnitBatch:
        unit_list = [torch.tensor(item.units, dtype=torch.long) for item in compressed_list]
        dur_list = [torch.tensor(item.durations, dtype=torch.long) for item in compressed_list]
        open_list = [torch.tensor(item.open_run_mask, dtype=torch.long) for item in compressed_list]
        sealed_list = [torch.tensor(item.sealed_mask, dtype=torch.long) for item in compressed_list]
        sep_list = [torch.tensor(item.sep_hint, dtype=torch.long) for item in compressed_list]
        boundary_list = [torch.tensor(item.boundary_confidence, dtype=torch.float32) for item in compressed_list]
        return self._batch_from_tensors(
            unit_list=unit_list,
            dur_list=dur_list,
            open_list=open_list,
            sealed_list=sealed_list,
            sep_list=sep_list,
            boundary_list=boundary_list,
            device=device,
        )

    def _batch_from_row_states(
        self,
        row_states: list[StreamingUnitizerRowState],
        *,
        mark_last_open: bool = True,
        device: torch.device | None = None,
    ) -> RhythmUnitBatch:
        unit_list: list[torch.Tensor] = []
        dur_list: list[torch.Tensor] = []
        open_list: list[torch.Tensor] = []
        sealed_list: list[torch.Tensor] = []
        sep_list: list[torch.Tensor] = []
        boundary_list: list[torch.Tensor] = []
        for row in row_states:
            units, durations, open_run_mask, sealed_mask, sep_hint, boundary_confidence = self.unitizer._export_row_tensors(
                row,
                mark_last_open=mark_last_open,
            )
            unit_list.append(units)
            dur_list.append(durations)
            open_list.append(open_run_mask)
            sealed_list.append(sealed_mask)
            sep_list.append(sep_hint)
            boundary_list.append(boundary_confidence)
        return self._batch_from_tensors(
            unit_list=unit_list,
            dur_list=dur_list,
            open_list=open_list,
            sealed_list=sealed_list,
            sep_list=sep_list,
            boundary_list=boundary_list,
            device=device,
        )

    def _batch_from_tensors(
        self,
        *,
        unit_list: list[torch.Tensor],
        dur_list: list[torch.Tensor],
        open_list: list[torch.Tensor],
        sealed_list: list[torch.Tensor],
        sep_list: list[torch.Tensor],
        boundary_list: list[torch.Tensor],
        device: torch.device | None = None,
    ) -> RhythmUnitBatch:
        mask_list = [torch.ones_like(unit_tensor, dtype=torch.float32) for unit_tensor in unit_list]
        content_units = self._pad(unit_list, pad_value=0, dtype=torch.long)
        dur_anchor_src = self._pad(dur_list, pad_value=0, dtype=torch.long)
        open_run_mask = self._pad(open_list, pad_value=0, dtype=torch.long)
        sealed_mask = self._pad(sealed_list, pad_value=0, dtype=torch.long)
        sep_hint = self._pad(sep_list, pad_value=0, dtype=torch.long)
        boundary_confidence = self._pad(boundary_list, pad_value=0, dtype=torch.float32)
        unit_mask = self._pad([m.long() for m in mask_list], pad_value=0, dtype=torch.long).float()

        if device is not None:
            content_units = content_units.to(device)
            dur_anchor_src = dur_anchor_src.to(device)
            open_run_mask = open_run_mask.to(device)
            sealed_mask = sealed_mask.to(device)
            sep_hint = sep_hint.to(device)
            boundary_confidence = boundary_confidence.to(device)
            unit_mask = unit_mask.to(device)

        return RhythmUnitBatch(
            content_units=content_units,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            open_run_mask=open_run_mask,
            sealed_mask=sealed_mask.float(),
            sep_hint=sep_hint,
            boundary_confidence=boundary_confidence,
        )

    def from_content_tensor(
        self,
        content: torch.Tensor,
        *,
        content_lengths: torch.Tensor | None = None,
        mark_last_open: bool = True,
    ) -> RhythmUnitBatch:
        if content.dim() != 2:
            raise ValueError(f"content must be rank-2 [B,T], got {tuple(content.shape)}")
        batch_size, total_steps = content.shape
        if content_lengths is None:
            content_lengths = torch.full(
                (batch_size,),
                int(total_steps),
                dtype=torch.long,
                device=content.device,
            )
        state = self.init_stream_state(batch_size, device=content.device)
        batch, _ = self.step_content_tensor(
            content,
            state,
            content_lengths=content_lengths,
            mark_last_open=mark_last_open,
        )
        return batch

    @staticmethod
    def _rebuild_boundary_confidence_from_cache(
        *,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor,
        open_run_mask: torch.Tensor,
        sep_hint: torch.Tensor,
    ) -> torch.Tensor:
        rebuilt_rows = []
        total_units = int(dur_anchor_src.size(1))
        for batch_idx in range(dur_anchor_src.size(0)):
            visible = int(unit_mask[batch_idx].sum().item())
            row = dur_anchor_src.new_zeros((total_units,), dtype=torch.float32)
            if visible > 0:
                rebuilt = _estimate_boundary_confidence_tensor(
                    dur_anchor_src[batch_idx, :visible],
                    sep_hint[batch_idx, :visible],
                    open_run_mask[batch_idx, :visible],
                ).to(device=dur_anchor_src.device, dtype=torch.float32)
                row[:visible] = rebuilt
            rebuilt_rows.append(row)
        return torch.stack(rebuilt_rows, dim=0) * unit_mask.float()

    def _rebuild_open_and_sealed_from_cache(
        self,
        *,
        unit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        open_run_mask = torch.zeros_like(unit_mask, dtype=torch.long)
        for batch_idx in range(unit_mask.size(0)):
            visible = int(unit_mask[batch_idx].sum().item())
            if visible <= 0:
                continue
            keep_open_from = max(0, visible - self.tail_open_units)
            open_run_mask[batch_idx, keep_open_from:visible] = 1
        sealed_mask = (1 - open_run_mask).clamp_min(0).float() * unit_mask.float()
        return open_run_mask, sealed_mask

    def from_precomputed(
        self,
        *,
        content_units: torch.Tensor,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor | None = None,
        open_run_mask: torch.Tensor | None = None,
        sealed_mask: torch.Tensor | None = None,
        sep_hint: torch.Tensor | None = None,
        boundary_confidence: torch.Tensor | None = None,
    ) -> RhythmUnitBatch:
        content_units = content_units.long()
        dur_anchor_src = dur_anchor_src.long()
        if unit_mask is None:
            unit_mask = dur_anchor_src.gt(0).float()
        else:
            unit_mask = unit_mask.float()
        if open_run_mask is None and sealed_mask is None:
            open_run_mask, sealed_mask = self._rebuild_open_and_sealed_from_cache(unit_mask=unit_mask)
        elif open_run_mask is None:
            sealed_mask = sealed_mask.float() * unit_mask.float()
            open_run_mask = ((sealed_mask <= 0.0).long() * unit_mask.long()).long()
        else:
            open_run_mask = open_run_mask.long() * unit_mask.long()
        if sep_hint is None:
            sep_hint = torch.zeros_like(content_units)
        if sealed_mask is None:
            sealed_mask = (1 - open_run_mask.long()).clamp_min(0).float() * unit_mask.float()
        else:
            sealed_mask = sealed_mask.float() * unit_mask.float()
        if boundary_confidence is None:
            boundary_confidence = self._rebuild_boundary_confidence_from_cache(
                dur_anchor_src=dur_anchor_src,
                unit_mask=unit_mask,
                open_run_mask=open_run_mask.long(),
                sep_hint=sep_hint.long(),
            )
        else:
            boundary_confidence = boundary_confidence.float() * unit_mask.float()
        return RhythmUnitBatch(
            content_units=content_units,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            open_run_mask=open_run_mask.long(),
            sealed_mask=sealed_mask,
            sep_hint=sep_hint.long(),
            boundary_confidence=boundary_confidence,
        )

    def init_stream_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
    ) -> StreamingUnitizerState:
        return self.unitizer.init_state(batch_size, device=device)

    def step_content_tensor(
        self,
        content: torch.Tensor,
        state: StreamingUnitizerState,
        *,
        content_lengths: torch.Tensor | None = None,
        mark_last_open: bool = True,
    ) -> tuple[RhythmUnitBatch, StreamingUnitizerState]:
        if content.dim() != 2:
            raise ValueError(f"content must be rank-2 [B,T], got {tuple(content.shape)}")
        batch_size, total_steps = content.shape
        if content_lengths is None:
            content_lengths = torch.full(
                (batch_size,),
                int(total_steps),
                dtype=torch.long,
                device=content.device,
            )
        next_rows: list[StreamingUnitizerRowState] = []
        for batch_idx in range(batch_size):
            valid_len = int(content_lengths[batch_idx].item())
            valid_len = max(0, min(valid_len, int(total_steps)))
            row_state = state.rows[batch_idx] if batch_idx < len(state.rows) else self.unitizer._empty_row_state(content.device)
            row_state = self._move_row_state(row_state, content.device)
            chunk = content[batch_idx, :valid_len].long()
            next_rows.append(self.unitizer.step_chunk_tensor(chunk, row_state))
        next_state = StreamingUnitizerState(rows=next_rows)
        batch = self._batch_from_row_states(next_rows, mark_last_open=mark_last_open, device=content.device)
        return batch, next_state


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

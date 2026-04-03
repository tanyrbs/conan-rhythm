from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def resolve_mel_frame_ms(hparams: Mapping[str, Any]) -> float:
    sample_rate = float(hparams["audio_sample_rate"])
    hop_size = float(hparams["hop_size"])
    if sample_rate <= 0.0:
        raise ValueError("audio_sample_rate must be positive.")
    return hop_size * 1000.0 / sample_rate


def resolve_chunk_frames(hparams: Mapping[str, Any]) -> int:
    chunk_size_ms = float(hparams["chunk_size"])
    mel_frame_ms = resolve_mel_frame_ms(hparams)
    return max(1, int(round(chunk_size_ms / mel_frame_ms)))


def resolve_vocoder_left_context_frames(source: Mapping[str, Any]) -> tuple[int, str]:
    for key in (
        "vocoder_left_context_frames",
        "streaming_vocoder_left_context_frames",
        "vocoder_stream_context",
    ):
        value = source.get(key, None)
        if value is None:
            continue
        try:
            return max(0, int(value)), key
        except (TypeError, ValueError):
            continue
    return 48, "default"


def acoustic_prefix_recompute_multiplier(num_chunks: int) -> float:
    chunks = max(0, int(num_chunks))
    if chunks <= 0:
        return 0.0
    return float(chunks + 1) / 2.0


estimate_prefix_recompute_multiplier = acoustic_prefix_recompute_multiplier


@dataclass(frozen=True)
class StreamingLayoutReport:
    audio_sample_rate: int
    hop_size: int
    mel_frame_ms: float
    chunk_frames: int
    chunk_ms: float
    right_context_frames: int
    right_context_ms: float
    algorithmic_first_packet_ms: float
    vocoder_left_context_frames: int
    vocoder_window_frames_steady: int
    vocoder_window_ms_steady: float
    vocoder_recompute_factor_steady: float
    acoustic_prefix_recompute: bool
    vocoder_native_streaming: bool
    full_end_to_end_stateful: bool

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def build_streaming_layout_report(
    hparams: Mapping[str, Any],
    *,
    vocoder_left_context_frames: int | None = None,
    vocoder_native_streaming: bool = False,
) -> StreamingLayoutReport:
    sample_rate = _safe_int(hparams.get("audio_sample_rate", 16000), 16000)
    hop_size = _safe_int(hparams.get("hop_size", 320), 320)
    mel_frame_ms = 1000.0 * float(hop_size) / float(sample_rate)
    chunk_frames = max(1, _safe_int(hparams.get("chunk_size", 20), 20) // 20)
    chunk_ms = float(chunk_frames) * mel_frame_ms
    right_context_frames = max(0, _safe_int(hparams.get("right_context", 0), 0))
    right_context_ms = float(right_context_frames) * mel_frame_ms
    if vocoder_left_context_frames is None:
        vocoder_left_context_frames, _ = resolve_vocoder_left_context_frames(hparams)
    vocoder_left_context_frames = max(0, int(vocoder_left_context_frames))
    vocoder_window_frames = chunk_frames if vocoder_native_streaming else vocoder_left_context_frames + chunk_frames
    vocoder_window_ms = float(vocoder_window_frames) * mel_frame_ms
    recompute_factor = 1.0 if vocoder_native_streaming else float(vocoder_window_frames) / float(chunk_frames)
    return StreamingLayoutReport(
        audio_sample_rate=sample_rate,
        hop_size=hop_size,
        mel_frame_ms=mel_frame_ms,
        chunk_frames=chunk_frames,
        chunk_ms=chunk_ms,
        right_context_frames=right_context_frames,
        right_context_ms=right_context_ms,
        algorithmic_first_packet_ms=chunk_ms + right_context_ms,
        vocoder_left_context_frames=vocoder_left_context_frames,
        vocoder_window_frames_steady=vocoder_window_frames,
        vocoder_window_ms_steady=vocoder_window_ms,
        vocoder_recompute_factor_steady=recompute_factor,
        acoustic_prefix_recompute=True,
        vocoder_native_streaming=bool(vocoder_native_streaming),
        full_end_to_end_stateful=False,
    )


def build_streaming_latency_report(
    hparams: Mapping[str, Any],
    *,
    duration_seconds: float | None = None,
    vocoder: Any | None = None,
) -> dict[str, Any]:
    supports_native_streaming = bool(
        getattr(vocoder, "supports_native_streaming", lambda: False)()
    ) if vocoder is not None else False
    left_context_frames, left_context_source = resolve_vocoder_left_context_frames(hparams)
    layout = build_streaming_layout_report(
        hparams,
        vocoder_left_context_frames=left_context_frames,
        vocoder_native_streaming=supports_native_streaming,
    )
    report: dict[str, Any] = {
        **layout.to_metadata(),
        "chunk_size_ms": float(hparams["chunk_size"]),
        "first_packet_algorithmic_latency_ms": float(layout.algorithmic_first_packet_ms),
        "acoustic_frontend_stateful_streaming": True,
        "acoustic_decoder_native_incremental": False,
        "vocoder_left_context_frames": int(left_context_frames),
        "vocoder_left_context_source": str(left_context_source),
        "vocoder_left_context_ms": float(left_context_frames * layout.mel_frame_ms),
        "vocoder_supports_native_streaming": bool(supports_native_streaming),
        "steady_state_vocoder_window_frames": int(layout.vocoder_window_frames_steady),
        "steady_state_vocoder_window_ms": float(layout.vocoder_window_ms_steady),
        "steady_state_vocoder_recompute_multiplier": float(layout.vocoder_recompute_factor_steady),
    }
    if duration_seconds is not None:
        total_frames = int(math.ceil(float(duration_seconds) * float(hparams["audio_sample_rate"]) / float(hparams["hop_size"])))
        num_chunks = int(math.ceil(float(total_frames) / float(max(1, layout.chunk_frames))))
        report.update(
            {
                "duration_seconds": float(duration_seconds),
                "estimated_total_mel_frames": int(total_frames),
                "estimated_num_chunks": int(num_chunks),
                "acoustic_prefix_recompute_multiplier": float(
                    acoustic_prefix_recompute_multiplier(num_chunks)
                ),
            }
        )
    return report


@dataclass
class PrefixCodeBuffer:
    total_capacity: int = 32

    def __post_init__(self):
        self.total_capacity = max(1, int(self.total_capacity))
        self._buffer: torch.Tensor | None = None
        self._length = 0

    @property
    def length(self) -> int:
        return int(self._length)

    def reset(self):
        self._buffer = None
        self._length = 0

    def current(self) -> torch.Tensor:
        if self._buffer is None:
            raise RuntimeError("PrefixCodeBuffer is empty.")
        return self._buffer[:, : self._length, ...]

    def append(self, values: torch.Tensor) -> torch.Tensor:
        if not isinstance(values, torch.Tensor):
            raise TypeError("PrefixCodeBuffer expects torch.Tensor chunks.")
        if values.dim() == 1:
            values = values.unsqueeze(0)
        if values.dim() < 2:
            raise ValueError(
                f"PrefixCodeBuffer expects [B, T, ...] or [B, T], got shape={tuple(values.shape)}"
            )
        batch = int(values.size(0))
        delta = int(values.size(1))
        if delta <= 0:
            if self._buffer is None:
                raise RuntimeError("Cannot append an empty first chunk to PrefixCodeBuffer.")
            return self.current()
        if self._buffer is None:
            shape = (batch, self.total_capacity) + tuple(values.shape[2:])
            self._buffer = values.new_empty(shape)
        elif int(self._buffer.size(0)) != batch:
            raise ValueError(
                f"PrefixCodeBuffer batch mismatch: existing={int(self._buffer.size(0))}, incoming={batch}"
            )
        required = self._length + delta
        if required > int(self._buffer.size(1)):
            raise ValueError(
                f"PrefixCodeBuffer capacity exceeded: required {required}, capacity {int(self._buffer.size(1))}."
            )
        self._buffer[:, self._length : required, ...] = values
        self._length = required
        return self.current()


@dataclass
class RollingMelContextBuffer:
    left_context_frames: int = 0

    def __post_init__(self):
        self.left_context_frames = max(0, int(self.left_context_frames))
        self._tail: torch.Tensor | None = None

    def reset(self):
        self._tail = None

    def build_window(self, mel_chunk: torch.Tensor) -> tuple[torch.Tensor, int]:
        if not isinstance(mel_chunk, torch.Tensor):
            raise TypeError("RollingMelContextBuffer expects torch.Tensor input.")
        if mel_chunk.dim() != 2:
            raise ValueError(
                f"RollingMelContextBuffer expects [T, C], got shape={tuple(mel_chunk.shape)}"
            )
        if mel_chunk.size(0) <= 0:
            if self._tail is None:
                return mel_chunk, 0
            return self._tail, min(self.left_context_frames, int(self._tail.size(0)))
        if self.left_context_frames <= 0 or self._tail is None or self._tail.size(0) <= 0:
            context = mel_chunk.new_empty((0, mel_chunk.size(1)))
        else:
            context = self._tail[-self.left_context_frames :]
        context_frames = int(context.size(0))
        window = torch.cat([context, mel_chunk], dim=0) if context_frames > 0 else mel_chunk
        if self.left_context_frames > 0:
            self._tail = window[-self.left_context_frames :].detach().clone()
        else:
            self._tail = mel_chunk.new_empty((0, mel_chunk.size(1)))
        return window, context_frames

    def push(self, mel_chunk: torch.Tensor) -> tuple[torch.Tensor, int]:
        return self.build_window(mel_chunk)


__all__ = [
    "PrefixCodeBuffer",
    "RollingMelContextBuffer",
    "StreamingLayoutReport",
    "acoustic_prefix_recompute_multiplier",
    "build_streaming_latency_report",
    "build_streaming_layout_report",
    "estimate_prefix_recompute_multiplier",
    "resolve_chunk_frames",
    "resolve_mel_frame_ms",
    "resolve_vocoder_left_context_frames",
]

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - hard dependency in this repo
    raise RuntimeError("plot_rhythm_diagnostics.py requires torch in the local Conan env") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.Conan.rhythm.reference_descriptor import RefRhythmDescriptor
from modules.Conan.rhythm.reference_encoder import (
    REF_RHYTHM_STATS_KEYS,
    REF_RHYTHM_TRACE_KEYS,
    ReferenceRhythmEncoder,
)
from utils.audio import librosa_wav2spec


AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
FEATURE_SUFFIXES = {".npz", ".pt", ".pth", ".json"}
REF_STATS_INDEX = {name: idx for idx, name in enumerate(REF_RHYTHM_STATS_KEYS)}
REF_TRACE_INDEX = {name: idx for idx, name in enumerate(REF_RHYTHM_TRACE_KEYS)}


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    try:
        return np.asarray(x)
    except Exception:
        return None


def _as_float_scalar(x: Any) -> Optional[float]:
    arr = _to_numpy(x)
    if arr is None:
        return None
    arr = np.asarray(arr).reshape(-1)
    if arr.size <= 0:
        return None
    value = arr[0]
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    try:
        return float(value)
    except Exception:
        return None


def _as_python_scalar(x: Any) -> Any:
    arr = _to_numpy(x)
    if arr is None:
        return x
    arr = np.asarray(arr).reshape(-1)
    if arr.size <= 0:
        return None
    value = arr[0]
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _squeeze_singletons(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    while arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return np.squeeze(arr)


def _maybe_matrix(arr: Any, *, preferred_last_dims: tuple[int, ...]) -> Optional[np.ndarray]:
    value = _to_numpy(arr)
    if value is None:
        return None
    value = _squeeze_singletons(value)
    if value.ndim != 2:
        return None
    if value.shape[-1] in preferred_last_dims:
        return value.astype(np.float32)
    if value.shape[0] in preferred_last_dims and value.shape[-1] not in preferred_last_dims:
        return value.T.astype(np.float32)
    if value.shape[-1] < value.shape[0] and value.shape[0] in preferred_last_dims:
        return value.T.astype(np.float32)
    return value.astype(np.float32)


def _maybe_trace_vector(arr: Any) -> Optional[np.ndarray]:
    value = _to_numpy(arr)
    if value is None:
        return None
    value = _squeeze_singletons(value)
    if value.ndim == 1:
        return value.astype(np.float32)
    if value.ndim == 2 and 1 in value.shape:
        return value.reshape(-1).astype(np.float32)
    return None


def _ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _safe_name(path: str) -> str:
    return Path(path.split("::", 1)[0]).stem


def _progress_axis(n: int) -> np.ndarray:
    if n <= 1:
        return np.array([0.0], dtype=np.float32)
    return np.linspace(0.0, 1.0, n, dtype=np.float32)


def _frame_axis(n: int, sample_rate: int, hop_size: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    return (np.arange(n, dtype=np.float32) * float(hop_size)) / float(sample_rate)


def _robust_ylim(
    x: np.ndarray,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    pad_ratio: float = 0.08,
) -> tuple[float, float]:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (-1.0, 1.0)
    lo = float(np.quantile(x, lower_q))
    hi = float(np.quantile(x, upper_q))
    if math.isclose(lo, hi):
        eps = max(abs(lo) * 0.1, 1e-3)
        return lo - eps, hi + eps
    pad = (hi - lo) * pad_ratio
    return lo - pad, hi + pad


def _maybe_resample_to_len(x: Optional[np.ndarray], tgt_len: int) -> Optional[np.ndarray]:
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if tgt_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    if len(x) == tgt_len:
        return x
    if len(x) <= 1:
        return np.full((tgt_len,), x[0] if len(x) else 0.0, dtype=np.float32)
    old_axis = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    new_axis = np.linspace(0.0, 1.0, tgt_len, dtype=np.float32)
    return np.interp(new_axis, old_axis, x).astype(np.float32)


@dataclass
class PlotConfig:
    sample_rate: int = 22050
    fft_size: int = 1024
    hop_size: int = 256
    win_length: int = 1024
    num_mels: int = 80
    fmin: int = 80
    fmax: int = -1
    trace_bins: int = 24
    smooth_kernel: int = 5
    trace_horizon: float = 0.35
    pause_energy_threshold_std: float = -0.5
    pause_delta_quantile: float = 0.35
    voiced_energy_threshold_std: float = -0.1
    boundary_quantile: float = 0.75
    compute_f0: bool = False
    audio_backfill: bool = True


@dataclass
class RhythmPayload:
    label: str
    source: str
    global_rate: Optional[float] = None
    pause_ratio: Optional[float] = None
    mean_pause_frames: Optional[float] = None
    mean_speech_frames: Optional[float] = None
    boundary_ratio: Optional[float] = None
    voiced_ratio: Optional[float] = None
    local_rate_trace: Optional[np.ndarray] = None
    boundary_trace: Optional[np.ndarray] = None
    ref_rhythm_stats: Optional[np.ndarray] = None
    ref_rhythm_trace: Optional[np.ndarray] = None
    planner_ref_stats: Optional[np.ndarray] = None
    planner_ref_trace: Optional[np.ndarray] = None
    mel: Optional[np.ndarray] = None
    f0: Optional[np.ndarray] = None
    energy: Optional[np.ndarray] = None
    energy_z: Optional[np.ndarray] = None
    delta: Optional[np.ndarray] = None
    pause_mask: Optional[np.ndarray] = None
    voiced: Optional[np.ndarray] = None
    local_rate_raw: Optional[np.ndarray] = None
    boundary_strength: Optional[np.ndarray] = None
    boundary_events: Optional[np.ndarray] = None
    progress: Optional[np.ndarray] = None
    progress_bins: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RhythmDiagnosticsExtractor:
    def __init__(self, cfg: PlotConfig) -> None:
        self.cfg = cfg
        self.encoder = ReferenceRhythmEncoder(
            trace_bins=cfg.trace_bins,
            smooth_kernel=cfg.smooth_kernel,
            trace_horizon=cfg.trace_horizon,
            pause_energy_threshold_std=cfg.pause_energy_threshold_std,
            pause_delta_quantile=cfg.pause_delta_quantile,
            voiced_energy_threshold_std=cfg.voiced_energy_threshold_std,
            boundary_quantile=cfg.boundary_quantile,
        ).eval()

    def _ensure_mel_bt80(self, mel: np.ndarray | torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(mel, dtype=torch.float32)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 3:
            raise ValueError(f"Expected mel rank-2 or rank-3, got shape={tuple(tensor.shape)}")
        if tensor.size(-1) == self.cfg.num_mels:
            return tensor
        if tensor.size(1) == self.cfg.num_mels:
            return tensor.transpose(1, 2)
        if tensor.size(1) <= tensor.size(2):
            return tensor.transpose(1, 2)
        return tensor

    def _masked_run_mean(self, mask: torch.Tensor) -> torch.Tensor:
        lengths = []
        for batch_mask in mask:
            values = batch_mask.tolist()
            runs = []
            run = 0
            target = int(values[0]) if len(values) > 0 else 0
            for value in values:
                value = int(value)
                if value == target:
                    run += 1
                else:
                    if target == 1:
                        runs.append(run)
                    target = value
                    run = 1
            if run > 0 and target == 1:
                runs.append(run)
            lengths.append(float(sum(runs) / max(len(runs), 1)) if runs else 0.0)
        return mask.new_tensor(lengths, dtype=torch.float32)

    def _resample_by_progress(self, feature_track: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        batch_size, total_frames, feat_dim = feature_track.shape
        trace = feature_track.new_zeros((batch_size, self.cfg.trace_bins, feat_dim))
        target_progress = torch.linspace(0.0, 1.0, self.cfg.trace_bins, device=feature_track.device)
        for batch_idx in range(batch_size):
            batch_progress = progress[batch_idx]
            batch_features = feature_track[batch_idx]
            for bin_idx, progress_value in enumerate(target_progress):
                right = int(torch.searchsorted(batch_progress, progress_value, right=False).item())
                if right <= 0:
                    trace[batch_idx, bin_idx] = batch_features[0]
                    continue
                if right >= total_frames:
                    trace[batch_idx, bin_idx] = batch_features[-1]
                    continue
                left = right - 1
                left_p = batch_progress[left]
                right_p = batch_progress[right]
                denom = (right_p - left_p).abs().clamp_min(1e-6)
                alpha = ((progress_value - left_p) / denom).clamp(0.0, 1.0)
                trace[batch_idx, bin_idx] = batch_features[left] * (1.0 - alpha) + batch_features[right] * alpha
        return trace

    def _compute_debug_from_mel(self, mel: np.ndarray | torch.Tensor) -> dict[str, np.ndarray]:
        ref_mel = self._ensure_mel_bt80(mel)
        energy = ref_mel.mean(dim=-1)
        energy_mean = energy.mean(dim=1, keepdim=True)
        energy_std = energy.std(dim=1, keepdim=True).clamp_min(1e-6)
        energy_z = (energy - energy_mean) / energy_std
        delta = torch.zeros_like(energy)
        delta[:, 1:] = (energy[:, 1:] - energy[:, :-1]).abs()
        delta_threshold = torch.quantile(delta, self.cfg.pause_delta_quantile, dim=1, keepdim=True)
        pause_mask = (energy_z <= self.cfg.pause_energy_threshold_std) & (delta <= delta_threshold)
        speech_mask = ~pause_mask
        voiced = energy_z.gt(self.cfg.voiced_energy_threshold_std).float()

        kernel = min(self.cfg.smooth_kernel, max(1, energy.size(1)))
        padding = kernel // 2
        local_rate = F.avg_pool1d(
            delta.unsqueeze(1),
            kernel_size=kernel,
            stride=1,
            padding=padding,
        ).squeeze(1)
        boundary_strength = F.avg_pool1d(
            delta.unsqueeze(1),
            kernel_size=min(kernel + 2, max(1, energy.size(1))),
            stride=1,
            padding=min(kernel + 2, max(1, energy.size(1))) // 2,
        ).squeeze(1)
        if local_rate.size(1) > energy.size(1):
            local_rate = local_rate[:, : energy.size(1)]
        if boundary_strength.size(1) > energy.size(1):
            boundary_strength = boundary_strength[:, : energy.size(1)]

        speech_progress = speech_mask.float().cumsum(dim=1)
        speech_total = speech_progress[:, -1:].clamp_min(1.0)
        progress = speech_progress / speech_total
        uniform = torch.linspace(0.0, 1.0, energy.size(1), device=energy.device)[None, :]
        segment_duration_bias = progress - uniform
        boundary_threshold = torch.quantile(
            boundary_strength,
            self.cfg.boundary_quantile,
            dim=1,
            keepdim=True,
        )
        boundary_events = boundary_strength >= boundary_threshold
        boundary_mean = boundary_strength.mean(dim=1, keepdim=True)
        boundary_std = boundary_strength.std(dim=1, keepdim=True).clamp_min(1e-6)
        boundary_strength_soft = torch.sigmoid((boundary_strength - boundary_mean) / boundary_std)
        feature_track = torch.stack(
            [
                pause_mask.float(),
                local_rate,
                boundary_strength_soft,
                segment_duration_bias,
                voiced,
            ],
            dim=-1,
        )
        trace = self._resample_by_progress(feature_track, progress)
        stats = torch.stack(
            [
                pause_mask.float().mean(dim=1),
                self._masked_run_mean(pause_mask.long()),
                self._masked_run_mean(speech_mask.long()),
                (local_rate[:, -1] - local_rate[:, 0]),
                boundary_events.float().mean(dim=1),
                voiced.mean(dim=1),
            ],
            dim=-1,
        )
        descriptor = RefRhythmDescriptor.from_stats_trace(stats, trace, include_sidecar=False)
        return {
            "energy": energy[0].detach().cpu().numpy().astype(np.float32),
            "energy_z": energy_z[0].detach().cpu().numpy().astype(np.float32),
            "delta": delta[0].detach().cpu().numpy().astype(np.float32),
            "pause_mask": pause_mask[0].float().detach().cpu().numpy().astype(np.float32),
            "voiced": voiced[0].detach().cpu().numpy().astype(np.float32),
            "local_rate_raw": local_rate[0].detach().cpu().numpy().astype(np.float32),
            "boundary_strength": boundary_strength[0].detach().cpu().numpy().astype(np.float32),
            "boundary_events": boundary_events[0].float().detach().cpu().numpy().astype(np.float32),
            "progress": progress[0].detach().cpu().numpy().astype(np.float32),
            "ref_rhythm_stats": stats[0].detach().cpu().numpy().astype(np.float32),
            "ref_rhythm_trace": trace[0].detach().cpu().numpy().astype(np.float32),
            "planner_ref_stats": descriptor["planner_ref_stats"][0].detach().cpu().numpy().astype(np.float32),
            "planner_ref_trace": descriptor["planner_ref_trace"][0].detach().cpu().numpy().astype(np.float32),
            "global_rate": float(descriptor["global_rate"][0, 0].detach().cpu().item()),
            "pause_ratio": float(descriptor["pause_ratio"][0, 0].detach().cpu().item()),
        }

    def _maybe_compute_f0(self, wav_path: str) -> Optional[np.ndarray]:
        if not self.cfg.compute_f0:
            return None
        import librosa

        wav, sr = librosa.load(wav_path, sr=self.cfg.sample_rate)
        f0 = librosa.yin(
            wav,
            fmin=max(40.0, float(self.cfg.fmin)),
            fmax=min(1100.0, float(self.cfg.sample_rate) / 2.0 - 1.0),
            sr=sr,
            frame_length=self.cfg.fft_size,
            hop_length=self.cfg.hop_size,
        )
        f0 = np.asarray(f0, dtype=np.float32)
        f0[~np.isfinite(f0)] = 0.0
        return f0

    def _payload_from_debug(
        self,
        *,
        label: str,
        source: str,
        mel_bt80: np.ndarray,
        debug: dict[str, Any],
        metadata: dict[str, Any],
    ) -> RhythmPayload:
        ref_stats = np.asarray(debug["ref_rhythm_stats"], dtype=np.float32)
        ref_trace = np.asarray(debug["ref_rhythm_trace"], dtype=np.float32)
        planner_stats = np.asarray(debug["planner_ref_stats"], dtype=np.float32)
        planner_trace = np.asarray(debug["planner_ref_trace"], dtype=np.float32)
        mean_pause_frames = float(ref_stats[REF_STATS_INDEX["mean_pause_frames"]])
        mean_speech_frames = float(ref_stats[REF_STATS_INDEX["mean_speech_frames"]])
        boundary_ratio = float(ref_stats[REF_STATS_INDEX["boundary_ratio"]])
        voiced_ratio = float(ref_stats[REF_STATS_INDEX["voiced_ratio"]])
        mel_np = np.asarray(mel_bt80, dtype=np.float32)
        if mel_np.ndim == 3:
            mel_np = mel_np[0]
        return RhythmPayload(
            label=label,
            source=source,
            global_rate=float(debug["global_rate"]),
            pause_ratio=float(debug["pause_ratio"]),
            mean_pause_frames=mean_pause_frames,
            mean_speech_frames=mean_speech_frames,
            boundary_ratio=boundary_ratio,
            voiced_ratio=voiced_ratio,
            local_rate_trace=planner_trace[:, 0].astype(np.float32),
            boundary_trace=planner_trace[:, 1].astype(np.float32),
            ref_rhythm_stats=ref_stats,
            ref_rhythm_trace=ref_trace,
            planner_ref_stats=planner_stats,
            planner_ref_trace=planner_trace,
            mel=mel_np.T.astype(np.float32),
            f0=metadata.get("f0"),
            energy=np.asarray(debug["energy"], dtype=np.float32),
            energy_z=np.asarray(debug["energy_z"], dtype=np.float32),
            delta=np.asarray(debug["delta"], dtype=np.float32),
            pause_mask=np.asarray(debug["pause_mask"], dtype=np.float32),
            voiced=np.asarray(debug["voiced"], dtype=np.float32),
            local_rate_raw=np.asarray(debug["local_rate_raw"], dtype=np.float32),
            boundary_strength=np.asarray(debug["boundary_strength"], dtype=np.float32),
            boundary_events=np.asarray(debug["boundary_events"], dtype=np.float32),
            progress=np.asarray(debug["progress"], dtype=np.float32),
            progress_bins=_progress_axis(planner_trace.shape[0]),
            metadata=metadata,
        )

    def from_audio(
        self,
        wav_path: str,
        *,
        label: Optional[str] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> RhythmPayload:
        wav_path = str(wav_path)
        spec = librosa_wav2spec(
            wav_path,
            fft_size=self.cfg.fft_size,
            hop_size=self.cfg.hop_size,
            win_length=self.cfg.win_length,
            num_mels=self.cfg.num_mels,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
            sample_rate=self.cfg.sample_rate,
        )
        mel = np.asarray(spec["mel"], dtype=np.float32)
        debug = self._compute_debug_from_mel(mel)
        metadata = dict(extra_metadata or {})
        metadata.setdefault("path", wav_path)
        metadata.setdefault("sample_rate", self.cfg.sample_rate)
        metadata.setdefault("hop_size", self.cfg.hop_size)
        metadata.setdefault("duration_sec", float(len(spec["wav"])) / float(self.cfg.sample_rate))
        metadata["f0"] = self._maybe_compute_f0(wav_path)
        return self._payload_from_debug(
            label=label or Path(wav_path).stem,
            source=wav_path,
            mel_bt80=mel,
            debug=debug,
            metadata=metadata,
        )

    def from_mapping(self, mapping: dict[str, Any], *, label: str, source: str) -> RhythmPayload:
        payload = RhythmPayload(label=label, source=source, metadata={})
        if isinstance(mapping, dict):
            for key in ("sample_id", "id", "utt", "speaker", "chapter", "path", "duration_sec"):
                if key in mapping:
                    payload.metadata[key] = _as_python_scalar(mapping[key])

        payload.global_rate = _as_float_scalar(mapping.get("global_rate"))
        payload.pause_ratio = _as_float_scalar(mapping.get("pause_ratio"))
        payload.mean_pause_frames = _as_float_scalar(mapping.get("mean_pause_frames"))
        payload.mean_speech_frames = _as_float_scalar(mapping.get("mean_speech_frames"))
        payload.boundary_ratio = _as_float_scalar(mapping.get("boundary_ratio"))
        payload.voiced_ratio = _as_float_scalar(mapping.get("voiced_ratio"))

        planner_stats = _maybe_trace_vector(mapping.get("planner_ref_stats"))
        if planner_stats is not None and planner_stats.size >= 2:
            payload.planner_ref_stats = planner_stats[:2].astype(np.float32)
            if payload.global_rate is None:
                payload.global_rate = float(planner_stats[0])
            if payload.pause_ratio is None:
                payload.pause_ratio = float(planner_stats[1])

        ref_stats = _maybe_trace_vector(mapping.get("ref_rhythm_stats"))
        if ref_stats is not None and ref_stats.size >= len(REF_RHYTHM_STATS_KEYS):
            ref_stats = ref_stats[: len(REF_RHYTHM_STATS_KEYS)].astype(np.float32)
            payload.ref_rhythm_stats = ref_stats
            payload.mean_pause_frames = float(ref_stats[REF_STATS_INDEX["mean_pause_frames"]])
            payload.mean_speech_frames = float(ref_stats[REF_STATS_INDEX["mean_speech_frames"]])
            payload.boundary_ratio = float(ref_stats[REF_STATS_INDEX["boundary_ratio"]])
            payload.voiced_ratio = float(ref_stats[REF_STATS_INDEX["voiced_ratio"]])
            if payload.pause_ratio is None:
                payload.pause_ratio = float(ref_stats[REF_STATS_INDEX["pause_ratio"]])
            if payload.global_rate is None:
                payload.global_rate = float(1.0 / max(ref_stats[REF_STATS_INDEX["mean_speech_frames"]], 1.0))
            if payload.planner_ref_stats is None:
                payload.planner_ref_stats = np.array([payload.global_rate, payload.pause_ratio], dtype=np.float32)

        planner_trace = _maybe_matrix(mapping.get("planner_ref_trace"), preferred_last_dims=(2,))
        if planner_trace is not None and planner_trace.shape[1] >= 2:
            planner_trace = planner_trace[:, :2].astype(np.float32)
            payload.planner_ref_trace = planner_trace
            payload.local_rate_trace = planner_trace[:, 0].astype(np.float32)
            payload.boundary_trace = planner_trace[:, 1].astype(np.float32)
            payload.progress_bins = _progress_axis(planner_trace.shape[0])

        ref_trace = _maybe_matrix(
            mapping.get("ref_rhythm_trace"),
            preferred_last_dims=(len(REF_RHYTHM_TRACE_KEYS),),
        )
        if ref_trace is not None and ref_trace.shape[1] >= len(REF_RHYTHM_TRACE_KEYS):
            ref_trace = ref_trace[:, : len(REF_RHYTHM_TRACE_KEYS)].astype(np.float32)
            payload.ref_rhythm_trace = ref_trace
            if payload.local_rate_trace is None:
                payload.local_rate_trace = ref_trace[:, REF_TRACE_INDEX["local_rate"]].astype(np.float32)
            if payload.boundary_trace is None:
                payload.boundary_trace = ref_trace[:, REF_TRACE_INDEX["boundary_strength"]].astype(np.float32)
            if payload.planner_ref_trace is None:
                payload.planner_ref_trace = np.stack([payload.local_rate_trace, payload.boundary_trace], axis=-1).astype(
                    np.float32
                )
            payload.progress_bins = _progress_axis(ref_trace.shape[0])

        mel = _maybe_matrix(mapping.get("mel"), preferred_last_dims=(self.cfg.num_mels,))
        if mel is None:
            mel = _maybe_matrix(mapping.get("mels"), preferred_last_dims=(self.cfg.num_mels,))
        if mel is not None:
            if mel.shape[-1] == self.cfg.num_mels:
                payload.mel = mel.T.astype(np.float32)
            else:
                payload.mel = mel.astype(np.float32)

        payload.f0 = _maybe_trace_vector(mapping.get("f0"))
        payload.energy = _maybe_trace_vector(mapping.get("energy"))
        payload.energy_z = _maybe_trace_vector(mapping.get("energy_z"))
        payload.delta = _maybe_trace_vector(mapping.get("delta"))
        payload.pause_mask = _maybe_trace_vector(mapping.get("pause_mask"))
        payload.voiced = _maybe_trace_vector(mapping.get("voiced"))
        payload.local_rate_raw = _maybe_trace_vector(mapping.get("local_rate_raw"))
        payload.boundary_strength = _maybe_trace_vector(mapping.get("boundary_strength"))
        payload.boundary_events = _maybe_trace_vector(mapping.get("boundary_events"))
        payload.progress = _maybe_trace_vector(mapping.get("progress"))
        progress_bins = _maybe_trace_vector(mapping.get("progress_bins"))
        if progress_bins is not None:
            payload.progress_bins = progress_bins.astype(np.float32)

        if payload.mean_speech_frames is None and payload.global_rate is not None and payload.global_rate > 0:
            payload.mean_speech_frames = float(1.0 / payload.global_rate)

        need_debug = (
            payload.mel is not None
            and (
                payload.ref_rhythm_stats is None
                or payload.ref_rhythm_trace is None
                or payload.energy_z is None
                or payload.delta is None
                or payload.pause_mask is None
                or payload.local_rate_raw is None
                or payload.boundary_strength is None
                or payload.boundary_events is None
            )
        )
        if need_debug:
            mel_bt80 = payload.mel.T.astype(np.float32)
            debug = self._compute_debug_from_mel(mel_bt80)
            merged = self._payload_from_debug(
                label=payload.label,
                source=payload.source,
                mel_bt80=mel_bt80,
                debug=debug,
                metadata=payload.metadata,
            )
            for field_name in (
                "global_rate",
                "pause_ratio",
                "mean_pause_frames",
                "mean_speech_frames",
                "boundary_ratio",
                "voiced_ratio",
                "local_rate_trace",
                "boundary_trace",
                "ref_rhythm_stats",
                "ref_rhythm_trace",
                "planner_ref_stats",
                "planner_ref_trace",
                "energy",
                "energy_z",
                "delta",
                "pause_mask",
                "voiced",
                "local_rate_raw",
                "boundary_strength",
                "boundary_events",
                "progress",
                "progress_bins",
            ):
                if getattr(payload, field_name) is None:
                    setattr(payload, field_name, getattr(merged, field_name))

        if payload.f0 is None and self.cfg.compute_f0 and payload.metadata.get("path"):
            path = Path(str(payload.metadata["path"]))
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                payload.f0 = self._maybe_compute_f0(str(path))

        if self.cfg.audio_backfill and payload.mel is None:
            audio_path = payload.metadata.get("path")
            if isinstance(audio_path, str):
                path = Path(audio_path)
                if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                    audio_payload = self.from_audio(audio_path, label=label, extra_metadata=payload.metadata)
                    for field_name in payload.__dataclass_fields__:
                        current = getattr(payload, field_name)
                        if current is None or (field_name == "metadata" and not current):
                            setattr(payload, field_name, getattr(audio_payload, field_name))
                    payload.metadata.update(audio_payload.metadata)

        return payload


def _parse_selector(spec: str) -> tuple[str, dict[str, str]]:
    if "::" not in spec:
        return spec, {}
    path, selector_text = spec.split("::", 1)
    selectors: dict[str, str] = {}
    for part in selector_text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid selector fragment: {part!r}")
        key, value = part.split("=", 1)
        selectors[key.strip()] = value.strip()
    return path, selectors


def _load_feature_file(path: str) -> Any:
    suffix = Path(path).suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu")
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported feature file: {path}")


def _bundle_records_from_obj(obj: Any) -> Optional[list[dict[str, Any]]]:
    if isinstance(obj, dict) and isinstance(obj.get("selected"), list):
        records = []
        for index, item in enumerate(obj["selected"]):
            if not isinstance(item, dict):
                continue
            merged = dict(item)
            if "progress_bins" not in merged and isinstance(obj.get("progress_bins"), list):
                merged["progress_bins"] = obj["progress_bins"]
            merged.setdefault("bundle_index", index)
            records.append(merged)
        return records
    if isinstance(obj, list) and all(isinstance(item, dict) for item in obj):
        return [dict(item) for item in obj]
    return None


def _select_bundle_record(records: list[dict[str, Any]], selectors: dict[str, str]) -> dict[str, Any]:
    if not records:
        raise ValueError("Bundle contains no selectable records")
    if not selectors:
        if len(records) == 1:
            return records[0]
        raise ValueError(
            "Bundle input contains multiple records; please use a selector like "
            "path/to/descriptors.json::sample_id=S01 or ::index=0"
        )
    if "index" in selectors:
        index = int(selectors["index"])
        if index < 0 or index >= len(records):
            raise IndexError(f"Bundle index out of range: {index}")
        return records[index]
    for key in ("sample_id", "id", "utt", "path"):
        if key in selectors:
            target = selectors[key]
            for record in records:
                if str(record.get(key)) == target:
                    return record
            raise KeyError(f"No bundle record with {key}={target!r}")
    raise ValueError(f"Unsupported selector keys: {sorted(selectors)}")


def load_payload_from_spec(spec: str, extractor: RhythmDiagnosticsExtractor) -> RhythmPayload:
    raw_path, selectors = _parse_selector(spec)
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(raw_path)
    if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
        extra = dict(selectors)
        return extractor.from_audio(str(path), label=selectors.get("label") or path.stem, extra_metadata=extra)
    if path.is_file() and path.suffix.lower() in FEATURE_SUFFIXES:
        obj = _load_feature_file(str(path))
        bundle_records = _bundle_records_from_obj(obj)
        if bundle_records is not None:
            mapping = _select_bundle_record(bundle_records, selectors)
            label = str(mapping.get("sample_id") or mapping.get("utt") or mapping.get("id") or path.stem)
            return extractor.from_mapping(mapping, label=label, source=spec)
        if not isinstance(obj, dict):
            raise ValueError(f"Unsupported top-level feature object for {path}: {type(obj)!r}")
        label = selectors.get("label") or path.stem
        return extractor.from_mapping(obj, label=label, source=spec)
    raise ValueError(f"Unsupported input spec: {spec}")


def collect_payloads(
    *,
    input_dir: Optional[str],
    inputs: Optional[Iterable[str]],
    extractor: RhythmDiagnosticsExtractor,
    recursive: bool,
    glob_pattern: str,
    limit: int,
) -> list[RhythmPayload]:
    payloads: list[RhythmPayload] = []
    if inputs:
        for spec in inputs:
            payloads.append(load_payload_from_spec(spec, extractor))
        return payloads[:limit] if limit > 0 else payloads
    if not input_dir:
        raise ValueError("Either --input_dir or --inputs must be provided")
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(input_dir)

    bundle_path = root / "descriptors.json"
    if bundle_path.is_file():
        obj = _load_feature_file(str(bundle_path))
        records = _bundle_records_from_obj(obj) or []
        for record in records:
            label = str(record.get("sample_id") or record.get("utt") or record.get("id") or record.get("bundle_index"))
            payloads.append(extractor.from_mapping(record, label=label, source=str(bundle_path)))
            if limit > 0 and len(payloads) >= limit:
                return payloads
        if payloads:
            return payloads

    iterator = root.rglob(glob_pattern) if recursive else root.glob(glob_pattern)
    for path in sorted(iterator):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in AUDIO_SUFFIXES or suffix in FEATURE_SUFFIXES:
            try:
                payloads.append(load_payload_from_spec(str(path), extractor))
            except Exception as exc:
                print(f"[WARN] skip {path}: {exc}")
            if limit > 0 and len(payloads) >= limit:
                break
    return payloads


def _warning_lines(payload: RhythmPayload) -> list[str]:
    warnings: list[str] = []
    if payload.pause_ratio is not None:
        if payload.pause_ratio < 0.02:
            warnings.append("pause_ratio very low -> pauses may be too sparse")
        elif payload.pause_ratio > 0.45:
            warnings.append("pause_ratio high -> pauses may be too heavy")
    if payload.local_rate_trace is not None:
        std = float(np.std(payload.local_rate_trace))
        if std < 0.01:
            warnings.append("local_rate_trace too flat -> local rhythm may be inactive")
        elif std > 0.35:
            warnings.append("local_rate_trace very spiky -> local rhythm may be too sharp")
    if payload.boundary_trace is not None:
        mean = float(np.mean(payload.boundary_trace))
        std = float(np.std(payload.boundary_trace))
        peaks = int(np.sum(payload.boundary_trace > (mean + std)))
        if peaks <= 1:
            warnings.append("boundary_trace peaks too few -> boundary cues may be weak")
        elif peaks > max(6, len(payload.boundary_trace) // 4):
            warnings.append("boundary_trace peaks too many -> boundary cues may be too dense")
    if not warnings:
        warnings = ["No obvious warning from simple heuristics"]
    return warnings


def plot_single_diagnostics(payload: RhythmPayload, *, cfg: PlotConfig, title: str, save_path: str) -> None:
    _ensure_dir(save_path)
    fig, axes = plt.subplots(
        nrows=6,
        ncols=1,
        figsize=(15, 15),
        gridspec_kw={"height_ratios": [3.6, 1.0, 1.2, 1.1, 1.3, 1.5]},
        constrained_layout=True,
    )

    mel = payload.mel
    frame_len = 0
    if mel is not None:
        frame_len = mel.shape[1]
    else:
        for candidate in (payload.energy, payload.energy_z, payload.delta, payload.local_rate_raw, payload.boundary_strength):
            if candidate is not None:
                frame_len = len(candidate)
                break
    frame_axis = _frame_axis(frame_len, cfg.sample_rate, cfg.hop_size)

    ax = axes[0]
    if mel is not None:
        extent = [0.0, float(frame_axis[-1]) if frame_len > 0 else 0.0, 0, mel.shape[0]]
        ax.imshow(mel, origin="lower", aspect="auto", interpolation="nearest", extent=extent, cmap="magma")
        ax.set_ylabel("Mel bins")
        ax.set_title(title)
    else:
        ax.text(0.5, 0.5, "No mel found", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)

    legend_handles = []
    ax2 = ax.twinx()
    if payload.f0 is not None and frame_len > 0:
        f0_plot = _maybe_resample_to_len(payload.f0, frame_len)
        h1, = ax2.plot(frame_axis, f0_plot, linewidth=0.9, color="tab:cyan", label="f0")
        legend_handles.append(h1)
    if payload.energy is not None and frame_len > 0:
        energy_plot = _maybe_resample_to_len(payload.energy, frame_len)
        h2, = ax2.plot(frame_axis, energy_plot, linewidth=1.0, alpha=0.9, color="tab:orange", label="energy")
        legend_handles.append(h2)
    if payload.pause_mask is not None and frame_len > 0:
        pause_plot = _maybe_resample_to_len(payload.pause_mask, frame_len)
        for idx in np.where(pause_plot > 0.5)[0]:
            left = frame_axis[idx]
            right = frame_axis[min(idx + 1, len(frame_axis) - 1)] if len(frame_axis) > 1 else left
            ax.axvspan(float(left), float(right), color="white", alpha=0.08, linewidth=0)
    if payload.boundary_events is not None and frame_len > 0:
        event_plot = _maybe_resample_to_len(payload.boundary_events, frame_len)
        event_positions = frame_axis[event_plot > 0.5]
        for xpos in event_positions:
            ax.axvline(float(xpos), color="cyan", alpha=0.2, linewidth=0.8)
    if legend_handles:
        ax2.legend(handles=legend_handles, loc="upper right")
    ax.set_xlabel("Time (s)")

    ax = axes[1]
    ax.axis("off")
    stats_lines = [
        f"global_rate      : {('N/A' if payload.global_rate is None else f'{payload.global_rate:.6f}')}",
        f"pause_ratio      : {('N/A' if payload.pause_ratio is None else f'{payload.pause_ratio:.4f}')}",
        f"mean_speech_fr   : {('N/A' if payload.mean_speech_frames is None else f'{payload.mean_speech_frames:.3f}')}",
        f"mean_pause_fr    : {('N/A' if payload.mean_pause_frames is None else f'{payload.mean_pause_frames:.3f}')}",
        f"boundary_ratio   : {('N/A' if payload.boundary_ratio is None else f'{payload.boundary_ratio:.4f}')}",
        f"voiced_ratio     : {('N/A' if payload.voiced_ratio is None else f'{payload.voiced_ratio:.4f}')}",
    ]
    meta_bits = []
    for key in ("sample_id", "speaker", "chapter", "utt", "path"):
        if payload.metadata.get(key) is not None:
            meta_bits.append(f"{key}: {payload.metadata[key]}")
    if meta_bits:
        stats_lines.extend(["", *meta_bits])
    ax.text(
        0.02,
        0.98,
        "\n".join(stats_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.95),
        transform=ax.transAxes,
    )

    ax = axes[2]
    if payload.energy_z is not None or payload.delta is not None:
        if payload.energy_z is not None:
            y = _maybe_resample_to_len(payload.energy_z, frame_len or len(payload.energy_z))
            x = _frame_axis(len(y), cfg.sample_rate, cfg.hop_size)
            ax.plot(x, y, linewidth=1.3, label="energy_z")
        if payload.delta is not None:
            y = _maybe_resample_to_len(payload.delta, frame_len or len(payload.delta))
            x = _frame_axis(len(y), cfg.sample_rate, cfg.hop_size)
            ax.plot(x, y, linewidth=1.2, label="delta")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Raw proxy")
        ax.set_xlabel("Time (s)")
    else:
        ax.text(0.5, 0.5, "No raw energy_z / delta found", ha="center", va="center", transform=ax.transAxes)

    ax = axes[3]
    if payload.pause_mask is not None or payload.voiced is not None:
        if payload.pause_mask is not None:
            y = _maybe_resample_to_len(payload.pause_mask, frame_len or len(payload.pause_mask))
            x = _frame_axis(len(y), cfg.sample_rate, cfg.hop_size)
            ax.step(x, y, where="mid", linewidth=1.2, label="pause_mask")
        if payload.voiced is not None:
            y = _maybe_resample_to_len(payload.voiced, frame_len or len(payload.voiced))
            x = _frame_axis(len(y), cfg.sample_rate, cfg.hop_size)
            ax.plot(x, y, linewidth=1.0, label="voiced", alpha=0.9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Mask / voiced")
        ax.set_xlabel("Time (s)")
    else:
        ax.text(0.5, 0.5, "No pause_mask / voiced found", ha="center", va="center", transform=ax.transAxes)

    ax = axes[4]
    plotted = False
    if payload.local_rate_raw is not None:
        y = _maybe_resample_to_len(payload.local_rate_raw, frame_len or len(payload.local_rate_raw))
        x = _frame_axis(len(y), cfg.sample_rate, cfg.hop_size)
        ax.plot(x, y, linewidth=1.4, label="local_rate_raw")
        plotted = True
    if payload.boundary_strength is not None:
        y = _maybe_resample_to_len(payload.boundary_strength, frame_len or len(payload.boundary_strength))
        x = _frame_axis(len(y), cfg.sample_rate, cfg.hop_size)
        ax.plot(x, y, linewidth=1.1, label="boundary_strength")
        plotted = True
    if payload.boundary_events is not None:
        y = _maybe_resample_to_len(payload.boundary_events, frame_len or len(payload.boundary_events))
        x = _frame_axis(len(y), cfg.sample_rate, cfg.hop_size)
        ax.scatter(x[y > 0.5], y[y > 0.5], s=12, color="tab:red", label="boundary_events")
        for xpos in x[y > 0.5]:
            ax.axvline(float(xpos), color="tab:red", alpha=0.12, linewidth=0.8)
        plotted = True
    if plotted:
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Raw local / boundary")
        ax.set_xlabel("Time (s)")
    else:
        ax.text(0.5, 0.5, "No raw local/boundary diagnostics found", ha="center", va="center", transform=ax.transAxes)

    ax = axes[5]
    progress = payload.progress_bins if payload.progress_bins is not None else (
        _progress_axis(len(payload.local_rate_trace)) if payload.local_rate_trace is not None else None
    )
    if payload.local_rate_trace is not None:
        ax.plot(progress, payload.local_rate_trace, linewidth=1.8, label="local_rate_trace")
    if payload.boundary_trace is not None:
        progress_b = payload.progress_bins if payload.progress_bins is not None else _progress_axis(len(payload.boundary_trace))
        ax.plot(progress_b, payload.boundary_trace, linewidth=1.8, label="boundary_trace")
        peak_idx = np.argsort(payload.boundary_trace)[-3:]
        peak_idx = np.sort(peak_idx)
        ax.scatter(progress_b[peak_idx], payload.boundary_trace[peak_idx], color="tab:red", s=20, zorder=3)
    if payload.local_rate_trace is not None or payload.boundary_trace is not None:
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Progress")
        ax.set_ylabel("Planner trace")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        series = []
        if payload.local_rate_trace is not None:
            series.append(payload.local_rate_trace)
        if payload.boundary_trace is not None:
            series.append(payload.boundary_trace)
        lo, hi = _robust_ylim(np.concatenate(series, axis=0))
        ax.set_ylim(lo, hi)
    else:
        ax.text(0.5, 0.5, "No planner traces found", ha="center", va="center", transform=ax.transAxes)

    warning_text = "\n".join(_warning_lines(payload))
    ax.text(
        1.01,
        0.98,
        warning_text,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.9),
    )

    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_triplet_compare(
    src_payload: RhythmPayload,
    ref_payload: RhythmPayload,
    out_payload: RhythmPayload,
    *,
    cfg: PlotConfig,
    title: str,
    save_path: str,
) -> None:
    _ensure_dir(save_path)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(title, fontsize=15)

    ax = axes[0, 0]
    ax.axis("off")
    rows = []
    for name, payload in (("source", src_payload), ("reference", ref_payload), ("output", out_payload)):
        rows.append(
            f"{name:<10} global_rate={('N/A' if payload.global_rate is None else f'{payload.global_rate:.6f}'):<12} "
            f"pause_ratio={('N/A' if payload.pause_ratio is None else f'{payload.pause_ratio:.4f}'):<10} "
            f"mean_speech_fr={('N/A' if payload.mean_speech_frames is None else f'{payload.mean_speech_frames:.2f}'):<8}"
        )
    ax.text(
        0.02,
        0.95,
        "\n".join(rows),
        va="top",
        ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.95),
        transform=ax.transAxes,
    )
    ax.set_title("Global scalars")

    ax = axes[0, 1]
    for name, payload in (("source", src_payload), ("reference", ref_payload), ("output", out_payload)):
        if payload.local_rate_trace is not None:
            x = payload.progress_bins if payload.progress_bins is not None else _progress_axis(len(payload.local_rate_trace))
            ax.plot(x, payload.local_rate_trace, linewidth=1.8, label=name)
    ax.set_title("local_rate_trace")
    ax.set_xlabel("Progress")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    for name, payload in (("source", src_payload), ("reference", ref_payload), ("output", out_payload)):
        if payload.boundary_trace is not None:
            x = payload.progress_bins if payload.progress_bins is not None else _progress_axis(len(payload.boundary_trace))
            ax.plot(x, payload.boundary_trace, linewidth=1.8, label=name)
    ax.set_title("boundary_trace")
    ax.set_xlabel("Progress")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    mel = out_payload.mel
    if mel is not None:
        frame_axis = _frame_axis(mel.shape[1], cfg.sample_rate, cfg.hop_size)
        extent = [0.0, float(frame_axis[-1]) if frame_axis.size > 0 else 0.0, 0, mel.shape[0]]
        ax.imshow(mel, origin="lower", aspect="auto", interpolation="nearest", extent=extent)
        ax.set_title("output mel")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mel bins")
    else:
        ax.text(0.5, 0.5, "No output mel found", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("output mel")

    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _payload_duration_sec(payload: RhythmPayload, sample_rate: int = 22050, hop_size: int = 256) -> Optional[float]:
    duration = payload.metadata.get("duration_sec")
    if duration is not None:
        try:
            return float(duration)
        except Exception:
            pass
    if payload.mel is not None and payload.mel.shape[1] > 0:
        return float(payload.mel.shape[1] * hop_size) / float(sample_rate)
    return None


def _plot_sorted_bar(
    ax: plt.Axes,
    items: list[RhythmPayload],
    *,
    metric_name: str,
    value_getter,
    color_map: str,
) -> None:
    filtered = [(item.label, value_getter(item)) for item in items if value_getter(item) is not None]
    if not filtered:
        ax.text(0.5, 0.5, f"No {metric_name}", ha="center", va="center", transform=ax.transAxes)
        return
    filtered = sorted(filtered, key=lambda x: x[1])
    labels = [x[0] for x in filtered]
    values = np.asarray([x[1] for x in filtered], dtype=np.float32)
    cmap = plt.get_cmap(color_map)
    if values.max() > values.min():
        norm = (values - values.min()) / (values.max() - values.min())
    else:
        norm = np.zeros_like(values)
    colors = [cmap(0.2 + 0.6 * float(v)) for v in norm]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor="black", alpha=0.9)
    ax.set_yticks(y, labels)
    ax.set_title(f"{metric_name} ranking")
    ax.grid(True, axis="x", alpha=0.25)
    for yi, value in enumerate(values):
        ax.text(float(value), yi, f" {value:.4f}", va="center", ha="left", fontsize=8)


def _plot_metric_heatmap(ax: plt.Axes, items: list[RhythmPayload], *, sample_rate: int, hop_size: int):
    labels = []
    raw_values = []
    metric_names = ["duration_s", "global_rate", "pause_ratio", "boundary_ratio", "voiced_ratio"]
    for item in items:
        duration = _payload_duration_sec(item, sample_rate=sample_rate, hop_size=hop_size)
        values = [duration, item.global_rate, item.pause_ratio, item.boundary_ratio, item.voiced_ratio]
        if any(v is None for v in values):
            continue
        labels.append(item.label)
        raw_values.append(np.asarray(values, dtype=np.float32))
    if not raw_values:
        ax.text(0.5, 0.5, "Need complete metric rows", ha="center", va="center", transform=ax.transAxes)
        return None
    raw_mat = np.stack(raw_values, axis=0)
    mean = raw_mat.mean(axis=0, keepdims=True)
    std = raw_mat.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    z = (raw_mat - mean) / std
    im = ax.imshow(z, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_title("metric overview (z-score)")
    ax.set_xticks(np.arange(len(metric_names)), metric_names, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    for i in range(raw_mat.shape[0]):
        for j in range(raw_mat.shape[1]):
            ax.text(j, i, f"{raw_mat[i, j]:.3f}", ha="center", va="center", fontsize=7, color="black")
    return im


def _plot_progress_small_multiples(items: list[RhythmPayload], *, save_prefix: str) -> None:
    valid = [x for x in items if x.local_rate_trace is not None and x.boundary_trace is not None]
    if not valid:
        return
    ordered = sorted(valid, key=lambda x: (_payload_duration_sec(x) or float("inf"), x.label))
    n = len(ordered)
    fig, axes = plt.subplots(
        nrows=n,
        ncols=2,
        figsize=(16, max(2.5 * n, 8)),
        sharex=True,
        constrained_layout=True,
    )
    if n == 1:
        axes = np.asarray([axes])
    local_all = np.concatenate([x.local_rate_trace for x in ordered], axis=0)
    boundary_all = np.concatenate([x.boundary_trace for x in ordered], axis=0)
    local_lo, local_hi = _robust_ylim(local_all)
    boundary_lo, boundary_hi = _robust_ylim(boundary_all, lower_q=0.0, upper_q=1.0, pad_ratio=0.05)
    for row_idx, item in enumerate(ordered):
        x = item.progress_bins if item.progress_bins is not None else _progress_axis(len(item.local_rate_trace))
        ax_local = axes[row_idx, 0]
        ax_local.plot(x, item.local_rate_trace, color="tab:blue", linewidth=2.0)
        ax_local.fill_between(x, 0.0, item.local_rate_trace, color="tab:blue", alpha=0.18)
        ax_local.set_ylim(local_lo, local_hi)
        ax_local.grid(True, alpha=0.22)
        if item.global_rate is not None and item.pause_ratio is not None:
            desc = (
                f"{item.label} | dur={(_payload_duration_sec(item) or float('nan')):.2f}s | "
                f"gr={item.global_rate:.4f} | pr={item.pause_ratio:.3f}"
            )
        else:
            desc = item.label
        ax_local.text(0.01, 0.88, desc, transform=ax_local.transAxes, fontsize=9, ha="left", va="top")
        if row_idx == 0:
            ax_local.set_title("local_rate_trace cards")
        if row_idx == n - 1:
            ax_local.set_xlabel("Progress")
        ax_local.set_ylabel("local")

        ax_boundary = axes[row_idx, 1]
        ax_boundary.plot(x, item.boundary_trace, color="tab:orange", linewidth=1.8)
        ax_boundary.fill_between(x, 0.0, item.boundary_trace, color="tab:orange", alpha=0.2)
        peak_idx = np.argsort(item.boundary_trace)[-3:]
        peak_idx = np.sort(peak_idx)
        ax_boundary.scatter(x[peak_idx], item.boundary_trace[peak_idx], color="tab:red", s=24, zorder=3)
        ax_boundary.set_ylim(boundary_lo, boundary_hi)
        ax_boundary.grid(True, alpha=0.22)
        if row_idx == 0:
            ax_boundary.set_title("boundary_trace cards")
        if row_idx == n - 1:
            ax_boundary.set_xlabel("Progress")
        ax_boundary.set_ylabel("boundary")
    fig.savefig(f"{save_prefix}_progress_cards.png", dpi=200)
    plt.close(fig)


def _plot_global_dashboard(items: list[RhythmPayload], *, save_prefix: str, sample_rate: int, hop_size: int) -> None:
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])

    _plot_sorted_bar(ax1, items, metric_name="global_rate", value_getter=lambda x: x.global_rate, color_map="Blues")
    _plot_sorted_bar(ax2, items, metric_name="pause_ratio", value_getter=lambda x: x.pause_ratio, color_map="Oranges")

    scatter_items = [
        x for x in items if x.mean_speech_frames is not None and x.pause_ratio is not None and x.global_rate is not None
    ]
    if scatter_items:
        xs = np.asarray([x.mean_speech_frames for x in scatter_items], dtype=np.float32)
        ys = np.asarray([x.pause_ratio for x in scatter_items], dtype=np.float32)
        cs = np.asarray([x.global_rate for x in scatter_items], dtype=np.float32)
        ss = np.asarray(
            [max(60.0, ((_payload_duration_sec(x, sample_rate=sample_rate, hop_size=hop_size) or 1.0) * 35.0)) for x in scatter_items],
            dtype=np.float32,
        )
        sc = ax3.scatter(xs, ys, c=cs, s=ss, cmap="viridis", alpha=0.9, edgecolors="black", linewidths=0.5)
        for item, xval, yval in zip(scatter_items, xs, ys):
            ax3.text(float(xval), float(yval), f" {item.label}", fontsize=8, va="center")
        ax3.set_title("mean_speech_frames vs pause_ratio")
        ax3.set_xlabel("mean_speech_frames")
        ax3.set_ylabel("pause_ratio")
        ax3.grid(True, alpha=0.25)
        cb = fig.colorbar(sc, ax=ax3, fraction=0.05, pad=0.03)
        cb.set_label("global_rate")
    else:
        ax3.text(0.5, 0.5, "Need mean_speech_frames + pause_ratio + global_rate", ha="center", va="center", transform=ax3.transAxes)

    heatmap = _plot_metric_heatmap(ax4, items, sample_rate=sample_rate, hop_size=hop_size)
    if heatmap is not None:
        fig.colorbar(heatmap, ax=ax4, fraction=0.05, pad=0.03, label="z-score")

    fig.savefig(f"{save_prefix}_global_dashboard.png", dpi=200)
    plt.close(fig)


def _plot_labeled_heatmap(
    items: list[RhythmPayload],
    *,
    trace_name: str,
    value_getter,
    sort_key,
    save_path: str,
    title: str,
) -> Optional[np.ndarray]:
    valid = [x for x in items if value_getter(x) is not None]
    if not valid:
        return None
    valid = sorted(valid, key=sort_key)
    max_len = max(len(value_getter(x)) for x in valid)
    mat = np.stack([_maybe_resample_to_len(value_getter(x), max_len) for x in valid], axis=0)
    labels = [x.label for x in valid]
    fig, ax = plt.subplots(figsize=(12, max(4.8, 0.48 * len(valid) + 2.5)), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Progress bins")
    ax.set_ylabel("Sample")
    ax.set_yticks(np.arange(len(labels)), labels)
    fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02, label=trace_name)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return mat


def _plot_mean_std_from_matrix(mat: Optional[np.ndarray], *, save_path: str, title: str) -> None:
    if mat is None:
        return
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    x = _progress_axis(mat.shape[1])
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    ax.plot(x, mean, linewidth=2.0)
    ax.fill_between(x, mean - std, mean + std, alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("Progress")
    ax.grid(True, alpha=0.25)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def export_audio_copies(
    items: list[RhythmPayload],
    *,
    export_audio_dir: str,
    sample_rate: int,
    hop_size: int,
) -> None:
    export_dir = Path(export_audio_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    playlist_paths = []
    for item in items:
        src = item.metadata.get("path")
        if not isinstance(src, str) or not Path(src).is_file():
            continue
        src_path = Path(src)
        target_name = f"{item.label}__{src_path.name}"
        dst_path = export_dir / target_name
        if not dst_path.exists() or src_path.stat().st_size != dst_path.stat().st_size:
            shutil.copy2(src_path, dst_path)
        playlist_paths.append(dst_path.name)
        manifest_rows.append(
            {
                "sample_id": item.label,
                "speaker": item.metadata.get("speaker", ""),
                "utt": item.metadata.get("utt", ""),
                "duration_sec": f"{(_payload_duration_sec(item, sample_rate=sample_rate, hop_size=hop_size) or 0.0):.4f}",
                "global_rate": "" if item.global_rate is None else f"{item.global_rate:.6f}",
                "pause_ratio": "" if item.pause_ratio is None else f"{item.pause_ratio:.6f}",
                "source_path": str(src_path),
                "exported_file": dst_path.name,
            }
        )

    if manifest_rows:
        manifest_path = export_dir / "audio_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id",
                    "speaker",
                    "utt",
                    "duration_sec",
                    "global_rate",
                    "pause_ratio",
                    "source_path",
                    "exported_file",
                ],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
        playlist_path = export_dir / "selected_samples.m3u"
        with playlist_path.open("w", encoding="utf-8", newline="\n") as f:
            for entry in playlist_paths:
                f.write(f"{entry}\n")


def emit_single_diagnostics(items: list[RhythmPayload], *, cfg: PlotConfig, output_dir: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        plot_single_diagnostics(item, cfg=cfg, title=item.label, save_path=str(out_dir / f"{item.label}.png"))


def plot_corpus_stats(
    items: list[RhythmPayload],
    *,
    save_prefix: str,
    cfg: PlotConfig,
    single_output_dir: Optional[str] = None,
    export_audio_dir: Optional[str] = None,
) -> None:
    if not items:
        raise RuntimeError("No payloads available for corpus plotting")
    out_dir = Path(save_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    global_rates = [x.global_rate for x in items if x.global_rate is not None]
    pause_ratios = [x.pause_ratio for x in items if x.pause_ratio is not None]
    mean_speech = [x.mean_speech_frames for x in items if x.mean_speech_frames is not None and x.pause_ratio is not None]
    paired_pause = [x.pause_ratio for x in items if x.mean_speech_frames is not None and x.pause_ratio is not None]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    if global_rates:
        axes[0].violinplot(global_rates, showmeans=True, showextrema=True)
        axes[0].set_title("global_rate distribution")
        axes[0].set_xticks([1])
        axes[0].set_xticklabels(["global_rate"])
        axes[0].grid(True, axis="y", alpha=0.25)
    else:
        axes[0].text(0.5, 0.5, "No global_rate", ha="center", va="center", transform=axes[0].transAxes)

    if pause_ratios:
        axes[1].violinplot(pause_ratios, showmeans=True, showextrema=True)
        axes[1].set_title("pause_ratio distribution")
        axes[1].set_xticks([1])
        axes[1].set_xticklabels(["pause_ratio"])
        axes[1].grid(True, axis="y", alpha=0.25)
    else:
        axes[1].text(0.5, 0.5, "No pause_ratio", ha="center", va="center", transform=axes[1].transAxes)

    if mean_speech and paired_pause:
        axes[2].scatter(mean_speech, paired_pause, s=26, alpha=0.85)
        axes[2].set_title("mean_speech_frames vs pause_ratio")
        axes[2].set_xlabel("mean_speech_frames")
        axes[2].set_ylabel("pause_ratio")
        axes[2].grid(True, alpha=0.25)
    else:
        axes[2].text(0.5, 0.5, "Need mean_speech_frames + pause_ratio", ha="center", va="center", transform=axes[2].transAxes)
    fig.savefig(f"{save_prefix}_global_stats.png", dpi=180)
    plt.close(fig)

    _plot_global_dashboard(items, save_prefix=save_prefix, sample_rate=cfg.sample_rate, hop_size=cfg.hop_size)
    _plot_progress_small_multiples(items, save_prefix=save_prefix)

    local_mat = _plot_labeled_heatmap(
        items,
        trace_name="local_rate_trace",
        value_getter=lambda x: x.local_rate_trace,
        sort_key=lambda x: float("inf") if x.global_rate is None else x.global_rate,
        save_path=f"{save_prefix}_local_rate_heatmap.png",
        title="local_rate_trace heatmap (sorted by global_rate)",
    )
    _plot_mean_std_from_matrix(
        local_mat,
        save_path=f"{save_prefix}_local_rate_mean_std.png",
        title="local_rate_trace mean +/- std",
    )

    boundary_mat = _plot_labeled_heatmap(
        items,
        trace_name="boundary_trace",
        value_getter=lambda x: x.boundary_trace,
        sort_key=lambda x: float("inf") if x.pause_ratio is None else x.pause_ratio,
        save_path=f"{save_prefix}_boundary_heatmap.png",
        title="boundary_trace heatmap (sorted by pause_ratio)",
    )
    _plot_mean_std_from_matrix(
        boundary_mat,
        save_path=f"{save_prefix}_boundary_mean_std.png",
        title="boundary_trace mean +/- std",
    )

    if single_output_dir:
        emit_single_diagnostics(items, cfg=cfg, output_dir=single_output_dir)
    if export_audio_dir:
        export_audio_copies(items, export_audio_dir=export_audio_dir, sample_rate=cfg.sample_rate, hop_size=cfg.hop_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot repo-aligned rhythm diagnostics. Supports explicit planner keys, "
            "maintained ref_rhythm_stats/ref_rhythm_trace, JSON bundles, and direct audio input."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common_plot_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--sample_rate", type=int, default=22050)
        p.add_argument("--fft_size", type=int, default=1024)
        p.add_argument("--hop_size", type=int, default=256)
        p.add_argument("--win_length", type=int, default=1024)
        p.add_argument("--num_mels", type=int, default=80)
        p.add_argument("--fmin", type=int, default=80)
        p.add_argument("--fmax", type=int, default=-1)
        p.add_argument("--trace_bins", type=int, default=24)
        p.add_argument("--smooth_kernel", type=int, default=5)
        p.add_argument("--trace_horizon", type=float, default=0.35)
        p.add_argument("--pause_energy_threshold_std", type=float, default=-0.5)
        p.add_argument("--pause_delta_quantile", type=float, default=0.35)
        p.add_argument("--voiced_energy_threshold_std", type=float, default=-0.1)
        p.add_argument("--boundary_quantile", type=float, default=0.75)
        p.add_argument("--with_f0", action="store_true", help="If direct audio is used, estimate f0 with librosa.yin.")
        p.add_argument(
            "--disable_audio_backfill",
            action="store_true",
            help="Do not auto-load bundle/feature record path back into raw audio when mel/raw diagnostics are missing.",
        )

    p1 = subparsers.add_parser("single", help="single-sample diagnostic panel")
    p1.add_argument("--input", required=True, help="audio / feature file / bundle selector, e.g. descriptors.json::sample_id=S01")
    p1.add_argument("--output", required=True, help="Output png path")
    add_common_plot_args(p1)

    p2 = subparsers.add_parser("compare", help="source/reference/output triplet comparison")
    p2.add_argument("--source", required=True)
    p2.add_argument("--reference", required=True)
    p2.add_argument("--output_input", required=True)
    p2.add_argument("--output_png", required=True)
    add_common_plot_args(p2)

    p3 = subparsers.add_parser("corpus", help="corpus-level dashboards / heatmaps")
    group = p3.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", help="Directory containing audio/features or a descriptors.json bundle")
    group.add_argument("--inputs", nargs="+", help="Explicit list of audio/features/bundle selectors")
    p3.add_argument("--save_prefix", required=True)
    p3.add_argument("--glob", default="*", help="Glob for directory scan; combine with --recursive for large trees")
    p3.add_argument("--recursive", action="store_true")
    p3.add_argument("--limit", type=int, default=-1, help="Optional cap on number of loaded items")
    p3.add_argument("--single_output_dir", default=None, help="Optional directory for regenerated per-sample diagnostic panels")
    p3.add_argument("--export_audio_dir", default=None, help="Optional directory to copy linked audio files plus audio manifest / m3u")
    add_common_plot_args(p3)
    return parser


def _cfg_from_args(args: argparse.Namespace) -> PlotConfig:
    return PlotConfig(
        sample_rate=args.sample_rate,
        fft_size=args.fft_size,
        hop_size=args.hop_size,
        win_length=args.win_length,
        num_mels=args.num_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        trace_bins=args.trace_bins,
        smooth_kernel=args.smooth_kernel,
        trace_horizon=args.trace_horizon,
        pause_energy_threshold_std=args.pause_energy_threshold_std,
        pause_delta_quantile=args.pause_delta_quantile,
        voiced_energy_threshold_std=args.voiced_energy_threshold_std,
        boundary_quantile=args.boundary_quantile,
        compute_f0=bool(args.with_f0),
        audio_backfill=not bool(args.disable_audio_backfill),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = _cfg_from_args(args)
    extractor = RhythmDiagnosticsExtractor(cfg)

    if args.mode == "single":
        payload = load_payload_from_spec(args.input, extractor)
        plot_single_diagnostics(payload, cfg=cfg, title=payload.label or _safe_name(args.input), save_path=args.output)
        print(f"[OK] saved: {args.output}")
        return

    if args.mode == "compare":
        src = load_payload_from_spec(args.source, extractor)
        ref = load_payload_from_spec(args.reference, extractor)
        out = load_payload_from_spec(args.output_input, extractor)
        title = f"{src.label} vs {ref.label} -> {out.label}"
        plot_triplet_compare(src, ref, out, cfg=cfg, title=title, save_path=args.output_png)
        print(f"[OK] saved: {args.output_png}")
        return

    if args.mode == "corpus":
        items = collect_payloads(
            input_dir=args.input_dir,
            inputs=args.inputs,
            extractor=extractor,
            recursive=bool(args.recursive),
            glob_pattern=args.glob,
            limit=int(args.limit),
        )
        if not items:
            raise RuntimeError("No valid items found for corpus plotting")
        plot_corpus_stats(
            items,
            save_prefix=args.save_prefix,
            cfg=cfg,
            single_output_dir=args.single_output_dir,
            export_audio_dir=args.export_audio_dir,
        )
        print(f"[OK] saved prefix: {args.save_prefix}")
        return

    raise RuntimeError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()

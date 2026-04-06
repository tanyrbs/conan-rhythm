#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.audio import librosa_wav2spec
from utils.commons.hparams import set_hparams


def _load_reference_encoder_class():
    module_path = REPO_ROOT / "modules" / "Conan" / "rhythm" / "reference_encoder.py"
    spec = importlib.util.spec_from_file_location("reference_encoder_standalone", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ReferenceRhythmEncoder


ReferenceRhythmEncoder = _load_reference_encoder_class()


@dataclass
class DescriptorSample:
    wav_path: Path
    rel_path: str
    duration_sec: float
    mel_frames: int
    global_rate: float
    pause_ratio: float
    local_rate_trace: np.ndarray
    boundary_trace: np.ndarray
    ref_rhythm_stats: np.ndarray
    ref_rhythm_trace: np.ndarray
    mel: np.ndarray
    energy: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Conan rhythm descriptors for manual inspection."
    )
    parser.add_argument(
        "--config",
        default="egs/conan_emformer_rhythm_v2_teacher_offline_train100_360.yaml",
        help="Config used to resolve audio/mel parameters.",
    )
    parser.add_argument(
        "--input_root",
        default="/root/autodl-tmp/data/LibriTTS/dev-clean",
        help="Root directory to scan for wav files.",
    )
    parser.add_argument(
        "--wav_glob",
        default="**/*.wav",
        help="Glob pattern relative to input_root.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write JSON/NPZ/PNG/CSV outputs.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of diverse samples to export.",
    )
    parser.add_argument(
        "--max_scan",
        type=int,
        default=240,
        help="Max wav files to scan before selecting diverse examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for deterministic scan ordering.",
    )
    parser.add_argument(
        "--copy_audio",
        action="store_true",
        help="Copy selected wav files into output_dir/audio.",
    )
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_hparams(config_path: str) -> dict:
    return set_hparams(config=config_path, print_hparams=False, global_hparams=False, reset=True)


def _build_descriptor(hparams: dict):
    return ReferenceRhythmEncoder(
        trace_bins=int(hparams.get("rhythm_trace_bins", 24)),
        trace_horizon=float(hparams.get("rhythm_trace_horizon", 0.35)),
        smooth_kernel=int(hparams.get("rhythm_trace_smooth_kernel", 5)),
    ).eval()


def _iter_wavs(root: Path, pattern: str, seed: int) -> list[Path]:
    wavs = sorted(root.glob(pattern))
    rng = np.random.default_rng(seed)
    if wavs:
        order = rng.permutation(len(wavs))
        wavs = [wavs[idx] for idx in order]
    return wavs


def _compute_descriptor_sample(
    wav_path: Path,
    *,
    root: Path,
    hparams: dict,
    descriptor: RefRhythmDescriptor,
) -> DescriptorSample:
    wav2spec = librosa_wav2spec(
        str(wav_path),
        fft_size=int(hparams["fft_size"]),
        hop_size=int(hparams["hop_size"]),
        win_length=int(hparams["win_size"]),
        num_mels=int(hparams["audio_num_mel_bins"]),
        fmin=float(hparams.get("fmin", 80)),
        fmax=float(hparams.get("fmax", -1)),
        sample_rate=int(hparams["audio_sample_rate"]),
        loud_norm=bool(hparams.get("loud_norm", False)),
        trim_long_sil=bool(hparams.get("trim_long_sil", False)),
    )
    mel = wav2spec["mel"].astype(np.float32)
    mel_tensor = torch.from_numpy(mel).unsqueeze(0)
    with torch.no_grad():
        packed = descriptor(mel_tensor)
    ref_rhythm_stats = packed["ref_rhythm_stats"]
    ref_rhythm_trace = packed["ref_rhythm_trace"]
    global_rate = torch.reciprocal(ref_rhythm_stats[:, 2:3].clamp_min(1.0))
    pause_ratio = ref_rhythm_stats[:, 0:1].clamp(0.0, 1.0)
    local_rate_trace = ref_rhythm_trace[:, :, 1:2]
    boundary_trace = ref_rhythm_trace[:, :, 2:3]
    energy = mel.mean(axis=1).astype(np.float32)
    duration_sec = float(mel.shape[0] * int(hparams["hop_size"]) / int(hparams["audio_sample_rate"]))
    return DescriptorSample(
        wav_path=wav_path,
        rel_path=str(wav_path.relative_to(root)),
        duration_sec=duration_sec,
        mel_frames=int(mel.shape[0]),
        global_rate=float(global_rate.squeeze().cpu().item()),
        pause_ratio=float(pause_ratio.squeeze().cpu().item()),
        local_rate_trace=local_rate_trace.squeeze(0).squeeze(-1).cpu().numpy().astype(np.float32),
        boundary_trace=boundary_trace.squeeze(0).squeeze(-1).cpu().numpy().astype(np.float32),
        ref_rhythm_stats=ref_rhythm_stats.squeeze(0).cpu().numpy().astype(np.float32),
        ref_rhythm_trace=ref_rhythm_trace.squeeze(0).cpu().numpy().astype(np.float32),
        mel=mel,
        energy=energy,
    )


def _feature_matrix(samples: list[DescriptorSample]) -> np.ndarray:
    features = []
    for sample in samples:
        features.append(
            [
                sample.duration_sec,
                sample.global_rate,
                sample.pause_ratio,
                float(np.std(sample.local_rate_trace)),
                float(np.mean(sample.boundary_trace)),
            ]
        )
    mat = np.asarray(features, dtype=np.float32)
    mean = mat.mean(axis=0, keepdims=True)
    std = mat.std(axis=0, keepdims=True)
    std = np.where(std < 1.0e-6, 1.0, std)
    return (mat - mean) / std


def _select_diverse_indices(samples: list[DescriptorSample], k: int) -> list[int]:
    if len(samples) <= k:
        return list(range(len(samples)))
    feats = _feature_matrix(samples)
    duration = np.asarray([sample.duration_sec for sample in samples], dtype=np.float32)
    selected = [int(np.argmin(duration))]
    max_duration_idx = int(np.argmax(duration))
    if max_duration_idx not in selected:
        selected.append(max_duration_idx)
    while len(selected) < k:
        best_idx = None
        best_score = -1.0
        for idx in range(len(samples)):
            if idx in selected:
                continue
            dists = np.linalg.norm(feats[idx][None, :] - feats[selected], axis=1)
            score = float(np.min(dists))
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(int(best_idx))
    return selected[:k]


def _progress_axis(length: int) -> np.ndarray:
    if length <= 1:
        return np.asarray([0.0], dtype=np.float32)
    return np.linspace(0.0, 1.0, length, dtype=np.float32)


def _plot_sample(sample: DescriptorSample, save_path: Path) -> None:
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 10),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.0, 1.3, 1.3]},
    )
    axes[0].imshow(sample.mel.T, origin="lower", aspect="auto", interpolation="nearest")
    axes[0].set_title(
        f"{sample.rel_path}\n"
        f"duration={sample.duration_sec:.2f}s  global_rate={sample.global_rate:.4f}  "
        f"pause_ratio={sample.pause_ratio:.4f}"
    )
    axes[0].set_ylabel("Mel")
    axes[1].plot(np.arange(sample.energy.shape[0]), sample.energy, linewidth=1.0)
    axes[1].set_ylabel("Energy")
    axes[1].grid(True, alpha=0.25)
    progress = _progress_axis(len(sample.local_rate_trace))
    axes[2].plot(progress, sample.local_rate_trace, linewidth=1.8)
    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_ylabel("local_rate_trace")
    axes[2].grid(True, alpha=0.25)
    axes[3].plot(progress, sample.boundary_trace, linewidth=1.8)
    axes[3].set_xlim(0.0, 1.0)
    axes[3].set_ylabel("boundary_trace")
    axes[3].set_xlabel("Progress")
    axes[3].grid(True, alpha=0.25)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _write_sample(sample: DescriptorSample, output_dir: Path, *, rank: int, copy_audio: bool) -> None:
    stem = f"{rank:02d}_{sample.wav_path.stem}"
    json_path = output_dir / f"{stem}.json"
    npz_path = output_dir / f"{stem}.npz"
    png_path = output_dir / f"{stem}.png"

    payload = {
        "rank": rank,
        "wav_path": str(sample.wav_path),
        "relative_path": sample.rel_path,
        "duration_sec": sample.duration_sec,
        "mel_frames": sample.mel_frames,
        "global_rate": sample.global_rate,
        "pause_ratio": sample.pause_ratio,
        "local_rate_trace": sample.local_rate_trace.tolist(),
        "boundary_trace": sample.boundary_trace.tolist(),
        "ref_rhythm_stats": sample.ref_rhythm_stats.tolist(),
        "ref_rhythm_trace": sample.ref_rhythm_trace.tolist(),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(
        npz_path,
        wav_path=str(sample.wav_path),
        duration_sec=np.asarray(sample.duration_sec, dtype=np.float32),
        ref_rhythm_stats=sample.ref_rhythm_stats,
        ref_rhythm_trace=sample.ref_rhythm_trace,
        global_rate=np.asarray(sample.global_rate, dtype=np.float32),
        pause_ratio=np.asarray(sample.pause_ratio, dtype=np.float32),
        local_rate_trace=sample.local_rate_trace,
        boundary_trace=sample.boundary_trace,
        mel=sample.mel,
        energy=sample.energy,
    )
    _plot_sample(sample, png_path)
    if copy_audio:
        audio_dir = output_dir / "audio"
        _ensure_dir(audio_dir)
        shutil.copy2(sample.wav_path, audio_dir / sample.wav_path.name)


def _write_summary(samples: Iterable[DescriptorSample], output_dir: Path) -> None:
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    rows = []
    for idx, sample in enumerate(samples, start=1):
        rows.append(
            {
                "rank": idx,
                "wav_path": str(sample.wav_path),
                "relative_path": sample.rel_path,
                "duration_sec": sample.duration_sec,
                "mel_frames": sample.mel_frames,
                "global_rate": sample.global_rate,
                "pause_ratio": sample.pause_ratio,
                "local_rate_std": float(np.std(sample.local_rate_trace)),
                "boundary_mean": float(np.mean(sample.boundary_trace)),
            }
        )
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    _ensure_dir(output_dir)
    hparams = _load_hparams(args.config)
    descriptor = _build_descriptor(hparams)
    input_root = Path(args.input_root)
    wavs = _iter_wavs(input_root, args.wav_glob, args.seed)
    if not wavs:
        raise RuntimeError(f"No wav files found under {input_root} with pattern {args.wav_glob}")
    scanned = []
    for wav_path in wavs[: max(1, int(args.max_scan))]:
        try:
            scanned.append(
                _compute_descriptor_sample(
                    wav_path,
                    root=input_root,
                    hparams=hparams,
                    descriptor=descriptor,
                )
            )
        except Exception as exc:
            print(f"[WARN] skip {wav_path}: {exc}")
    if not scanned:
        raise RuntimeError("No valid samples were processed.")
    selected_indices = _select_diverse_indices(scanned, max(1, int(args.num_samples)))
    selected = [scanned[idx] for idx in selected_indices]
    for rank, sample in enumerate(selected, start=1):
        _write_sample(sample, output_dir, rank=rank, copy_audio=bool(args.copy_audio))
    _write_summary(selected, output_dir)
    print(f"[OK] exported {len(selected)} samples to {output_dir}")


if __name__ == "__main__":
    main()

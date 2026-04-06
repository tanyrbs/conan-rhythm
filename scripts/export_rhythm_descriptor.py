#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample

from modules.Conan.rhythm.reference_descriptor import RefRhythmDescriptor
from modules.pe.rmvpe.spec import MelSpectrogram


AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus", ".aiff", ".aif", ".wv"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export rhythm descriptors (global_rate, pause_ratio, traces) from audio or mel features."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Audio (wav/flac/ogg/...) or mel (.npy/.npz/.pt) files.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("artifacts/rhythm_descriptor_exports"),
        help="Directory to store JSON summaries, numpy traces, and debug plots.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for mel extraction/descriptor.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate for audio inputs.")
    parser.add_argument("--n-mels", type=int, default=80, help="Number of mel channels for feature extraction.")
    parser.add_argument("--window-length", type=int, default=1024, help="STFT window length.")
    parser.add_argument("--hop-length", type=int, default=256, help="STFT hop length.")
    parser.add_argument("--fmin", type=float, default=20.0, help="Mel filter lower bound.")
    parser.add_argument("--fmax", type=float, default=8000.0, help="Mel filter upper bound.")
    parser.add_argument("--trace-bins", type=int, default=24, help="Resolution of the planner trace outputs.")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating PNG plots.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing exports for the same stem.")
    return parser.parse_args()


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class RhythmDescriptorExporter:
    def __init__(
        self,
        *,
        device: str,
        sample_rate: int,
        n_mels: int,
        window_length: int,
        hop_length: int,
        fmin: float,
        fmax: float,
        trace_bins: int,
    ) -> None:
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self._n_mels = n_mels
        self._hop_length = hop_length
        self.trace_bins = trace_bins
        self._mel_extractor = MelSpectrogram(
            n_mel_channels=n_mels,
            sampling_rate=sample_rate,
            win_length=window_length,
            hop_length=hop_length,
            n_fft=window_length,
            mel_fmin=fmin,
            mel_fmax=fmax,
        ).to(self.device)
        self._descriptor = RefRhythmDescriptor(trace_bins=trace_bins).to(self.device)
        self._resamplers: dict[tuple[int, int], Resample] = {}

    def _resampler_for(self, orig_sr: int) -> Resample:
        key = (orig_sr, self.sample_rate)
        if key not in self._resamplers:
            self._resamplers[key] = Resample(orig_sr, self.sample_rate)
        resampler = self._resamplers[key].to(self.device)
        self._resamplers[key] = resampler
        return resampler

    def _load_audio(self, path: Path) -> tuple[torch.Tensor, float]:
        waveform, sr = torchaudio.load(path)
        if waveform.numel() == 0:
            raise ValueError(f"Empty waveform in {path}")
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = self._resampler_for(sr)(waveform)
        return waveform.to(self.device), waveform.shape[-1] / self.sample_rate

    def _normalize_mel_tensor(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        if mel.dim() != 3:
            raise ValueError(f"Mel tensor must be rank 3, got {mel.shape}")
        if mel.shape[1] == self._n_mels and mel.shape[2] != self._n_mels:
            mel = mel.transpose(1, 2)
        elif mel.shape[2] != self._n_mels and mel.shape[1] < mel.shape[2]:
            mel = mel.transpose(1, 2)
        return mel.to(self.device)

    def _load_mel(self, path: Path) -> torch.Tensor:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            array = np.load(path)
        elif suffix == ".npz":
            archive = np.load(path)
            if "mel" in archive:
                array = archive["mel"]
            elif archive.files:
                array = archive[archive.files[0]]
            else:
                raise ValueError(f"No arrays found in {path}")
        elif suffix == ".pt":
            payload = torch.load(path)
            if isinstance(payload, dict) and "mel" in payload:
                array = payload["mel"]
            else:
                array = payload
        else:
            raise ValueError(f"Unsupported feature input: {path}")
        tensor = torch.as_tensor(array, dtype=torch.float32)
        return self._normalize_mel_tensor(tensor)

    def _audio_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mel = self._mel_extractor(waveform)
        return mel.transpose(1, 2)

    def describe(self, sources: Iterable[Path], output_root: Path, *, no_plot: bool, overwrite: bool) -> None:
        for source in sources:
            self._describe_single(source=source, output_root=output_root, no_plot=no_plot, overwrite=overwrite)

    def _describe_single(self, *, source: Path, output_root: Path, no_plot: bool, overwrite: bool) -> None:
        if not source.exists():
            raise FileNotFoundError(source)
        output_dir = output_root / source.stem
        if output_dir.exists():
            if overwrite:
                shutil.rmtree(output_dir)
            else:
                raise FileExistsError(
                    f"Exports for {source} already exist in {output_dir}; rerun with --overwrite to replace."
                )
        _ensure_output_dir(output_dir)

        if source.suffix.lower() in AUDIO_EXTS:
            waveform, duration = self._load_audio(source)
            mel = self._audio_to_mel(waveform)
            source_kind = "audio"
        else:
            mel = self._load_mel(source)
            duration = float(mel.shape[1]) * self._hop_length / self.sample_rate
            source_kind = "mel"

        descriptor = self._descriptor(mel)
        global_rate = descriptor["global_rate"].squeeze().cpu().item()
        pause_ratio = descriptor["pause_ratio"].squeeze().cpu().item()
        local_rate_trace = descriptor["local_rate_trace"].squeeze(-1).cpu().numpy()
        boundary_trace = descriptor["boundary_trace"].squeeze(-1).cpu().numpy()

        summary = {
            "source": str(source.resolve()),
            "type": source_kind,
            "global_rate": float(global_rate),
            "pause_ratio": float(pause_ratio),
            "frames": int(mel.shape[1]),
            "trace_bins": self.trace_bins,
            "duration_sec": float(duration),
        }
        with open(output_dir / "descriptor_summary.json", "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

        np.save(output_dir / "local_rate_trace.npy", local_rate_trace)
        np.save(output_dir / "boundary_trace.npy", boundary_trace)

        if not no_plot:
            fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
            axes[0].plot(local_rate_trace, label="local_rate_trace")
            axes[0].set_ylabel("rate")
            axes[1].plot(boundary_trace, label="boundary_trace", color="tab:orange")
            axes[1].set_ylabel("boundary")
            axes[-1].set_xlabel("trace bin")
            axes[0].set_title(
                f"{source.name} (global_rate={global_rate:.3f}, pause_ratio={pause_ratio:.3f})"
            )
            for ax in axes:
                ax.grid(True, linestyle=":", alpha=0.4)
            plt.tight_layout()
            fig.savefig(output_dir / "trace_plot.png")
            plt.close(fig)

        print(f"Exported descriptor for {source.name} -> {output_dir}")


def main() -> None:
    args = _parse_args()
    exporter = RhythmDescriptorExporter(
        device=args.device,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        window_length=args.window_length,
        hop_length=args.hop_length,
        fmin=args.fmin,
        fmax=args.fmax,
        trace_bins=args.trace_bins,
    )
    exporter.describe(
        sources=args.inputs,
        output_root=args.output_dir,
        no_plot=args.no_plot,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

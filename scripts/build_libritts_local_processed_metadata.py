from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _normalize_split_tag(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _build_item_name(split_tag: str, wav_path: Path) -> str:
    parts = wav_path.stem.split("_")
    if len(parts) >= 4:
        speaker, chapter = parts[0], parts[1]
        rest = "_".join(parts[2:])
        return f"{speaker}_{split_tag}_{chapter}_{rest}"
    parent_parts = wav_path.relative_to(wav_path.parents[2]).parts
    speaker = wav_path.parts[-3]
    return f"{speaker}_{split_tag}_{'_'.join(parent_parts).replace('.', '_')}"


def _scan_split(raw_root: Path, split_name: str) -> list[Path]:
    split_root = raw_root / split_name
    if not split_root.exists():
        raise FileNotFoundError(f"Missing raw split directory: {split_root}")
    return sorted(split_root.rglob("*.wav"))


def _select_candidates(
    *,
    wav_paths: list[Path],
    split_tag: str,
    limit: int,
    min_frames: int,
    max_frames: int,
    model,
    device: torch.device,
    librosa_wav2spec,
    hparams,
) -> tuple[list[dict], dict[str, int]]:
    selected: list[dict] = []
    speaker_ids: set[str] = set()
    stats = {"seen": 0, "kept": 0, "too_short": 0, "too_long": 0}
    for wav_path in tqdm(wav_paths, desc=f"metadata:{split_tag}"):
        if limit > 0 and len(selected) >= limit:
            break
        wav2spec = librosa_wav2spec(
            str(wav_path),
            fft_size=hparams["fft_size"],
            hop_size=hparams["hop_size"],
            win_length=hparams["win_size"],
            num_mels=hparams["audio_num_mel_bins"],
            fmin=hparams["fmin"],
            fmax=hparams["fmax"],
            sample_rate=hparams["audio_sample_rate"],
            loud_norm=hparams["loud_norm"],
        )
        mel = wav2spec["mel"]
        stats["seen"] += 1
        frames = int(mel.shape[0])
        if frames < min_frames:
            stats["too_short"] += 1
            continue
        if max_frames > 0 and frames > max_frames:
            stats["too_long"] += 1
            continue
        mel_tensor = torch.from_numpy(mel).unsqueeze(0).float().to(device)
        with torch.inference_mode():
            logits = model.inference(mel_tensor)
        tokens = logits.argmax(dim=-1).squeeze(0).detach().cpu().tolist()
        if len(tokens) != frames:
            tokens = tokens[:frames]
        if len(tokens) <= 0:
            continue
        split_item_name = _build_item_name(split_tag, wav_path)
        speaker_id = split_item_name.split("_", 1)[0]
        speaker_ids.add(speaker_id)
        selected.append(
            {
                "item_name": split_item_name,
                "wav_fn": str(wav_path),
                "hubert": " ".join(str(int(x)) for x in tokens),
                "duration": round(float(len(wav2spec["wav"])) / float(hparams["audio_sample_rate"]), 6),
                "split": split_tag,
            }
        )
        stats["kept"] += 1
    stats["num_speakers"] = len(speaker_ids)
    return selected, stats


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build processed metadata from a local LibriTTS tree.")
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--processed_data_dir", required=True)
    parser.add_argument("--config", default="egs/conan_emformer_rhythm_v2_teacher_offline.yaml")
    parser.add_argument(
        "--emformer_ckpt",
        default="checkpoints/Emformer/model_ckpt_steps_700000.ckpt",
    )
    parser.add_argument("--exp_name", default="build_libritts_local_processed_metadata")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument(
        "--train_split",
        action="append",
        default=None,
        help=(
            "Training split name. Repeat the flag or pass a comma-separated value to combine "
            "multiple training splits, e.g. --train_split train-clean-100 --train_split train-clean-360"
        ),
    )
    parser.add_argument("--valid_split", default="dev-clean")
    parser.add_argument("--test_split", default="test-clean")
    parser.add_argument("--train_limit", type=int, default=0, help="0 means no limit; use the full split.")
    parser.add_argument("--valid_limit", type=int, default=0, help="0 means no limit; use the full split.")
    parser.add_argument("--test_limit", type=int, default=0, help="0 means no limit; use the full split.")
    parser.add_argument("--min_mel_frames", type=int, default=32)
    parser.add_argument("--max_mel_frames", type=int, default=600)
    return parser


def _normalize_split_arg_list(values: list[str] | None, *, default: str) -> list[str]:
    raw_values = values if values else [default]
    normalized: list[str] = []
    for raw in raw_values:
        parts = [part.strip() for part in str(raw).split(",")]
        for part in parts:
            if part and part not in normalized:
                normalized.append(part)
    if not normalized:
        normalized.append(default)
    return normalized


def main() -> None:
    args = build_argparser().parse_args()

    from modules.Emformer.emformer import EmformerDistillModel
    from utils.audio import librosa_wav2spec
    from utils.commons.ckpt_utils import load_ckpt_emformer
    from utils.commons.hparams import hparams, set_hparams

    raw_root = Path(args.raw_root).resolve()
    processed_data_dir = Path(args.processed_data_dir).resolve()
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    set_hparams(
        config=args.config,
        exp_name=args.exp_name,
        hparams_str=(
            f"processed_data_dir='{processed_data_dir.as_posix()}',"
            "style=True,rhythm_minimal_style_only=True"
        ),
    )
    model = EmformerDistillModel(hparams)
    load_ckpt_emformer(model, args.emformer_ckpt)
    model.to(device)
    model.eval()

    train_splits = _normalize_split_arg_list(args.train_split, default="train-clean-100")
    split_specs = [
        (split_name, _normalize_split_tag(split_name), args.train_limit)
        for split_name in train_splits
    ] + [
        (args.valid_split, _normalize_split_tag(args.valid_split), args.valid_limit),
        (args.test_split, _normalize_split_tag(args.test_split), args.test_limit),
    ]

    items: list[dict] = []
    split_stats: dict[str, dict[str, int]] = {}
    speakers: set[str] = set()
    for raw_split_name, split_tag, limit in split_specs:
        wav_paths = _scan_split(raw_root, raw_split_name)
        split_items, stats = _select_candidates(
            wav_paths=wav_paths,
            split_tag=split_tag,
            limit=limit,
            min_frames=args.min_mel_frames,
            max_frames=args.max_mel_frames,
            model=model,
            device=device,
            librosa_wav2spec=librosa_wav2spec,
            hparams=hparams,
        )
        items.extend(split_items)
        split_stats[split_tag] = stats
        speakers.update(item["item_name"].split("_", 1)[0] for item in split_items)

    spker_map = {speaker: idx for idx, speaker in enumerate(sorted(speakers))}
    metadata_path = processed_data_dir / "metadata_vctk_librittsr_gt.json"
    metadata_json_path = processed_data_dir / "metadata.json"
    speaker_path = processed_data_dir / "spker_set.json"
    summary_path = processed_data_dir / "build_summary.json"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    with metadata_json_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    with speaker_path.open("w", encoding="utf-8") as f:
        json.dump(spker_map, f, ensure_ascii=False, indent=2)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "raw_root": str(raw_root),
                "processed_data_dir": str(processed_data_dir),
                "device": str(device),
                "config": args.config,
                "emformer_ckpt": args.emformer_ckpt,
                "limits": {
                    "train": args.train_limit,
                    "valid": args.valid_limit,
                    "test": args.test_limit,
                },
                "frame_filter": {
                    "min_mel_frames": args.min_mel_frames,
                    "max_mel_frames": args.max_mel_frames,
                },
                "train_splits": train_splits,
                "train_split_tags": [_normalize_split_tag(split) for split in train_splits],
                "split_tags": {
                    "train": _normalize_split_tag(train_splits[0]) if len(train_splits) == 1 else "train",
                    "valid": _normalize_split_tag(args.valid_split),
                    "test": _normalize_split_tag(args.test_split),
                },
                "split_stats": split_stats,
                "num_items": len(items),
                "num_speakers": len(spker_map),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[build-libritts-metadata] raw_root={raw_root}")
    print(f"[build-libritts-metadata] processed_data_dir={processed_data_dir}")
    print(f"[build-libritts-metadata] device={device}")
    print(f"[build-libritts-metadata] items={len(items)} speakers={len(spker_map)}")
    print(f"[build-libritts-metadata] metadata={metadata_path}")
    print(f"[build-libritts-metadata] speaker_map={speaker_path}")
    print(
        "[build-libritts-metadata] suggested overrides: "
        f"valid_prefixes=['_{_normalize_split_tag(args.valid_split)}_'],"
        f"test_prefixes=['_{_normalize_split_tag(args.test_split)}_']"
    )


if __name__ == "__main__":
    main()

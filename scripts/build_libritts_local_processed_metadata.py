from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEXT_SUFFIXES = (".normalized.txt", ".original.txt", ".txt")


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


def _read_text_sidecar(wav_path: Path) -> tuple[str, str, str]:
    normalized = ""
    original = ""
    fallback = ""
    for suffix in TEXT_SUFFIXES:
        candidate = wav_path.with_suffix(suffix)
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8").strip()
        if suffix == ".normalized.txt":
            normalized = text
        elif suffix == ".original.txt":
            original = text
        elif fallback == "":
            fallback = text
    text = normalized or original or fallback
    return text, normalized, original


def _scalar(value, default=None):
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value
    return value


def _build_sidecar_top1_content_record(wav_path: Path) -> dict:
    sidecar_path = wav_path.with_name(f"{wav_path.stem}_content_topk.npz")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Missing content_topk sidecar: {sidecar_path}")
    with np.load(sidecar_path) as payload:
        if "content_topk_ids" not in payload:
            raise KeyError(f"`content_topk_ids` missing in sidecar: {sidecar_path}")
        topk_ids = np.asarray(payload["content_topk_ids"])
        if topk_ids.ndim != 2 or topk_ids.shape[0] <= 0 or topk_ids.shape[1] <= 0:
            raise ValueError(f"Invalid top-k shape {topk_ids.shape} in sidecar: {sidecar_path}")
        top1_ids = topk_ids[:, 0].astype(np.int64, copy=False)
        frames = int(_scalar(payload.get("topk_frames"), default=top1_ids.shape[0]))
        audio_sample_rate = int(_scalar(payload.get("audio_sample_rate"), default=16000))
        hop_size = int(_scalar(payload.get("hop_size"), default=320))
        topk_k = int(_scalar(payload.get("topk_k"), default=topk_ids.shape[1]))
        duration = round(float(frames * hop_size) / float(audio_sample_rate), 6)
        return {
            "hubert": " ".join(str(int(x)) for x in top1_ids.tolist()),
            "frames": frames,
            "duration": duration,
            "content_source": "content_topk_sidecar_top1",
            "content_topk_path": str(sidecar_path),
            "content_topk_frames": frames,
            "content_topk_k": topk_k,
            "content_topk_schema_family": str(_scalar(payload.get("sidecar_schema_family"), default="")),
            "content_topk_schema_version": int(_scalar(payload.get("sidecar_schema_version"), default=0)),
            "content_topk_schema_hash": str(_scalar(payload.get("sidecar_schema_hash"), default="")),
        }


def _select_candidates(
    *,
    wav_paths: list[Path],
    raw_split_name: str,
    split_tag: str,
    canonical_split: str,
    limit: int,
    min_frames: int,
    max_frames: int,
    content_source: str,
    model,
    device: torch.device,
    librosa_wav2spec,
    hparams,
) -> tuple[list[dict], dict[str, int]]:
    selected: list[dict] = []
    speaker_ids: set[str] = set()
    stats = {
        "seen": 0,
        "kept": 0,
        "too_short": 0,
        "too_long": 0,
        "from_sidecar_top1": 0,
        "from_emformer": 0,
    }
    for wav_path in tqdm(wav_paths, desc=f"metadata:{split_tag}"):
        if limit > 0 and len(selected) >= limit:
            break
        text, normalized_text, original_text = _read_text_sidecar(wav_path)
        stats["seen"] += 1
        content_record = None
        if content_source in {"auto", "sidecar_top1"}:
            try:
                content_record = _build_sidecar_top1_content_record(wav_path)
            except FileNotFoundError:
                if content_source == "sidecar_top1":
                    raise
            except Exception as exc:
                raise RuntimeError(f"Failed to load top-k sidecar for {wav_path}: {exc}") from exc
        if content_record is None:
            if model is None or librosa_wav2spec is None or hparams is None:
                raise RuntimeError(
                    "Emformer fallback requested, but model/hparams are unavailable. "
                    "Use --content_source sidecar_top1 only when all sidecars exist."
                )
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
            frames = int(mel.shape[0])
            mel_tensor = torch.from_numpy(mel).unsqueeze(0).float().to(device)
            with torch.inference_mode():
                logits = model.inference(mel_tensor)
            tokens = logits.argmax(dim=-1).squeeze(0).detach().cpu().tolist()
            if len(tokens) != frames:
                tokens = tokens[:frames]
            if len(tokens) <= 0:
                continue
            content_record = {
                "hubert": " ".join(str(int(x)) for x in tokens),
                "frames": frames,
                "duration": round(float(len(wav2spec["wav"])) / float(hparams["audio_sample_rate"]), 6),
                "content_source": "emformer_argmax",
            }
        frames = int(content_record["frames"])
        if frames < min_frames:
            stats["too_short"] += 1
            continue
        if max_frames > 0 and frames > max_frames:
            stats["too_long"] += 1
            continue
        split_item_name = _build_item_name(split_tag, wav_path)
        speaker_id = split_item_name.split("_", 1)[0]
        speaker_ids.add(speaker_id)
        f0_path = wav_path.parent.parent / f"{wav_path.parent.name}_f0" / f"{wav_path.stem}_f0.npy"
        record = {
            "item_name": split_item_name,
            "wav_fn": str(wav_path),
            "hubert": str(content_record["hubert"]),
            "duration": float(content_record["duration"]),
            "split": canonical_split,
            "subset": raw_split_name,
            "source_subset": raw_split_name,
            "split_tag": split_tag,
            "speaker_id": str(speaker_id),
            "text": text,
            "normalized_text": normalized_text,
            "original_text": original_text,
            "f0_path": str(f0_path),
        }
        for key in (
            "content_source",
            "content_topk_path",
            "content_topk_frames",
            "content_topk_k",
            "content_topk_schema_family",
            "content_topk_schema_version",
            "content_topk_schema_hash",
        ):
            if key in content_record:
                record[key] = content_record[key]
        selected.append(record)
        stats["kept"] += 1
        if content_record["content_source"] == "content_topk_sidecar_top1":
            stats["from_sidecar_top1"] += 1
        else:
            stats["from_emformer"] += 1
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
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Only build train metadata; skip valid/test entirely. Useful when this processed/binary dir is used only via train_sets.",
    )
    parser.add_argument("--min_mel_frames", type=int, default=32)
    parser.add_argument("--max_mel_frames", type=int, default=600)
    parser.add_argument(
        "--content_source",
        choices=["auto", "sidecar_top1", "emformer"],
        default="auto",
        help=(
            "Metadata content source: prefer adjacent *_content_topk.npz top1 tokens, "
            "or fall back to Emformer inference."
        ),
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split each raw split into N shards by index for parallel extraction.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="0-based shard index within --num_shards.",
    )
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
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be a positive integer.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards.")

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

    model = None
    librosa_wav2spec = None
    hparams = None
    if args.content_source != "sidecar_top1":
        from modules.Emformer.emformer import EmformerDistillModel
        from utils.audio import librosa_wav2spec
        from utils.commons.ckpt_utils import load_ckpt_emformer
        from utils.commons.hparams import hparams, set_hparams

        set_hparams(
            config=args.config,
            exp_name=args.exp_name,
            hparams_str=(
                f"processed_data_dir='{processed_data_dir.as_posix()}',"
                "style=True,rhythm_minimal_style_only=True"
            ),
            reset=True,
        )
        model = EmformerDistillModel(hparams)
        load_ckpt_emformer(model, args.emformer_ckpt)
        model.to(device)
        model.eval()

    train_splits = _normalize_split_arg_list(args.train_split, default="train-clean-100")
    split_specs = [
        (split_name, _normalize_split_tag(split_name), "train", args.train_limit)
        for split_name in train_splits
    ]
    if not args.train_only:
        split_specs += [
            (args.valid_split, _normalize_split_tag(args.valid_split), "valid", args.valid_limit),
            (args.test_split, _normalize_split_tag(args.test_split), "test", args.test_limit),
        ]

    items: list[dict] = []
    split_stats: dict[str, dict[str, int]] = {}
    speakers: set[str] = set()
    split_item_names: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    for raw_split_name, split_tag, canonical_split, limit in split_specs:
        wav_paths = _scan_split(raw_root, raw_split_name)
        if args.num_shards > 1:
            wav_paths = wav_paths[args.shard_index::args.num_shards]
        split_items, stats = _select_candidates(
            wav_paths=wav_paths,
            raw_split_name=raw_split_name,
            split_tag=split_tag,
            canonical_split=canonical_split,
            limit=limit,
            min_frames=args.min_mel_frames,
            max_frames=args.max_mel_frames,
            content_source=args.content_source,
            model=model,
            device=device,
            librosa_wav2spec=librosa_wav2spec,
            hparams=hparams,
        )
        items.extend(split_items)
        split_stats[split_tag] = stats
        speakers.update(item["item_name"].split("_", 1)[0] for item in split_items)
        split_item_names[canonical_split].extend(item["item_name"] for item in split_items)

    spker_map = {speaker: idx for idx, speaker in enumerate(sorted(speakers))}
    metadata_path = processed_data_dir / "metadata_vctk_librittsr_gt.json"
    metadata_json_path = processed_data_dir / "metadata.json"
    speaker_path = processed_data_dir / "spker_set.json"
    summary_path = processed_data_dir / "build_summary.json"
    split_manifest_paths = {
        split_name: processed_data_dir / f"{split_name}_item_names.txt"
        for split_name in ("train", "valid", "test")
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    with metadata_json_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    with speaker_path.open("w", encoding="utf-8") as f:
        json.dump(spker_map, f, ensure_ascii=False, indent=2)
    for split_name, manifest_path in split_manifest_paths.items():
        with manifest_path.open("w", encoding="utf-8") as f:
            for item_name in split_item_names[split_name]:
                f.write(f"{item_name}\n")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "raw_root": str(raw_root),
                "processed_data_dir": str(processed_data_dir),
                "device": str(device),
                "content_source": args.content_source,
                "config": args.config,
                "emformer_ckpt": args.emformer_ckpt,
                "limits": {
                    "train": args.train_limit,
                    "valid": 0 if args.train_only else args.valid_limit,
                    "test": 0 if args.train_only else args.test_limit,
                },
                "frame_filter": {
                    "min_mel_frames": args.min_mel_frames,
                    "max_mel_frames": args.max_mel_frames,
                },
                "sharding": {
                    "num_shards": args.num_shards,
                    "shard_index": args.shard_index,
                },
                "train_splits": train_splits,
                "train_split_tags": [_normalize_split_tag(split) for split in train_splits],
                "canonical_splits": {
                    "train": train_splits,
                    "valid": [] if args.train_only else [args.valid_split],
                    "test": [] if args.train_only else [args.test_split],
                },
                "split_tags": {
                    "train": [_normalize_split_tag(split) for split in train_splits],
                    "valid": [] if args.train_only else [_normalize_split_tag(args.valid_split)],
                    "test": [] if args.train_only else [_normalize_split_tag(args.test_split)],
                },
                "split_stats": split_stats,
                "split_item_manifests": {
                    split_name: {
                        "path": str(path),
                        "count": len(split_item_names[split_name]),
                    }
                    for split_name, path in split_manifest_paths.items()
                },
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
    if args.num_shards > 1:
        print(f"[build-libritts-metadata] sharding={args.shard_index + 1}/{args.num_shards}")
    print(f"[build-libritts-metadata] items={len(items)} speakers={len(spker_map)}")
    print(f"[build-libritts-metadata] metadata={metadata_path}")
    print(f"[build-libritts-metadata] speaker_map={speaker_path}")
    print("[build-libritts-metadata] explicit metadata split labels use canonical train/valid/test.")
    print("[build-libritts-metadata] split item manifests:")
    for split_name, manifest_path in split_manifest_paths.items():
        print(f"  - {split_name}: {manifest_path} ({len(split_item_names[split_name])} items)")


if __name__ == "__main__":
    main()

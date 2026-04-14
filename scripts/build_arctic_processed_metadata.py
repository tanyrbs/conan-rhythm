from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CMU_SPEAKER_TO_DIR = {
    "bdl": "cmu_us_bdl_arctic",
    "slt": "cmu_us_slt_arctic",
}


@dataclass(frozen=True)
class RawItem:
    dataset: str
    speaker: str
    prompt_id: str
    wav_path: Path
    txt_raw: str
    txt_norm: str


def _parse_csv_arg(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _natural_key(text: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(text))]


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_cmu_transcripts(root: Path) -> dict[str, str]:
    transcript_path = root / "etc" / "txt.done.data"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Missing CMU transcript file: {transcript_path}")
    transcripts: dict[str, str] = {}
    pattern = re.compile(r'^\(\s*(\S+)\s+"(.*)"\s*\)$')
    for raw_line in transcript_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if not match:
            continue
        transcripts[match.group(1)] = match.group(2).strip()
    return transcripts


def _scan_cmu_root(root: Path, speaker: str) -> list[RawItem]:
    transcripts = _load_cmu_transcripts(root)
    wav_dir = root / "wav"
    if not wav_dir.exists():
        raise FileNotFoundError(f"Missing CMU wav directory: {wav_dir}")
    items: list[RawItem] = []
    for wav_path in sorted(wav_dir.glob("*.wav"), key=lambda p: _natural_key(p.stem)):
        prompt_id = wav_path.stem
        raw_text = transcripts.get(prompt_id, "").strip()
        text_norm = _normalize_text(raw_text)
        if not raw_text or not text_norm:
            continue
        items.append(
            RawItem(
                dataset="cmu_arctic",
                speaker=speaker,
                prompt_id=prompt_id,
                wav_path=wav_path.resolve(),
                txt_raw=raw_text,
                txt_norm=text_norm,
            )
        )
    return items


def _scan_l2_speaker(root: Path, speaker: str) -> list[RawItem]:
    speaker_root = root / speaker
    transcript_dir = speaker_root / "transcript"
    wav_dir = speaker_root / "wav"
    if not transcript_dir.exists():
        raise FileNotFoundError(f"Missing L2 transcript directory: {transcript_dir}")
    if not wav_dir.exists():
        raise FileNotFoundError(f"Missing L2 wav directory: {wav_dir}")
    items: list[RawItem] = []
    for txt_path in sorted(transcript_dir.glob("*.txt"), key=lambda p: _natural_key(p.stem)):
        prompt_id = txt_path.stem
        wav_path = wav_dir / f"{prompt_id}.wav"
        if not wav_path.exists():
            continue
        raw_text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        text_norm = _normalize_text(raw_text)
        if not raw_text or not text_norm:
            continue
        items.append(
            RawItem(
                dataset="l2_arctic",
                speaker=speaker.lower(),
                prompt_id=prompt_id,
                wav_path=wav_path.resolve(),
                txt_raw=raw_text,
                txt_norm=text_norm,
            )
        )
    return items


def _discover_l2_speakers(l2_root: Path) -> list[str]:
    speakers: list[str] = []
    for path in sorted(l2_root.iterdir(), key=lambda p: _natural_key(p.name)):
        if not path.is_dir():
            continue
        if path.name.lower() == "suitcase_corpus":
            continue
        if (path / "transcript").exists() and (path / "wav").exists():
            speakers.append(path.name)
    return speakers


def _select_prompt_splits(
    *,
    prompt_ids: list[str],
    valid_prompt_count: int,
    test_prompt_count: int,
    train_prompt_count: int,
) -> dict[str, set[str]]:
    if valid_prompt_count < 0 or test_prompt_count < 0 or train_prompt_count < 0:
        raise ValueError("Prompt counts must be >= 0.")
    total_reserved = valid_prompt_count + test_prompt_count
    if len(prompt_ids) <= total_reserved:
        raise ValueError(
            f"Not enough shared prompts to reserve valid/test splits: prompts={len(prompt_ids)} "
            f"valid={valid_prompt_count} test={test_prompt_count}"
        )
    valid_ids = prompt_ids[:valid_prompt_count]
    test_ids = prompt_ids[valid_prompt_count : valid_prompt_count + test_prompt_count]
    train_pool = prompt_ids[valid_prompt_count + test_prompt_count :]
    if train_prompt_count > 0:
        train_pool = train_pool[:train_prompt_count]
    if not train_pool:
        raise ValueError("Training split would be empty after prompt selection.")
    return {
        "train": set(train_pool),
        "valid": set(valid_ids),
        "test": set(test_ids),
    }


def _pick_majority_text_items(
    items_by_prompt: dict[str, list[RawItem]],
    *,
    min_shared_speakers: int,
) -> tuple[dict[str, list[RawItem]], dict[str, dict[str, int]]]:
    kept: dict[str, list[RawItem]] = {}
    stats: dict[str, dict[str, int]] = {}
    for prompt_id, items in items_by_prompt.items():
        counts = Counter(item.txt_norm for item in items)
        if not counts:
            continue
        majority_text, majority_count = counts.most_common(1)[0]
        filtered = [item for item in items if item.txt_norm == majority_text]
        if len(filtered) < min_shared_speakers:
            continue
        kept[prompt_id] = sorted(filtered, key=lambda item: _natural_key(item.speaker))
        stats[prompt_id] = {
            "raw_items": len(items),
            "kept_items": len(filtered),
            "majority_count": majority_count,
            "text_variants": len(counts),
        }
    return kept, stats


def _build_item_name(speaker: str, split: str, prompt_id: str) -> str:
    return f"{speaker}_{split}_{prompt_id}"


def _load_model_and_hparams(args):
    from modules.Emformer.emformer import EmformerDistillModel
    from utils.commons.ckpt_utils import load_ckpt_emformer
    from utils.commons.hparams import hparams, set_hparams

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
            f"processed_data_dir='{Path(args.processed_data_dir).resolve().as_posix()}',"
            "style=True,rhythm_minimal_style_only=True"
        ),
    )
    model = EmformerDistillModel(hparams)
    load_ckpt_emformer(model, args.emformer_ckpt)
    model.to(device)
    model.eval()
    return model, hparams, device


def _tokenize_items(
    *,
    split_to_raw_items: dict[str, list[RawItem]],
    model,
    hparams,
    device: torch.device,
    min_mel_frames: int,
    max_mel_frames: int,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    from utils.audio import librosa_wav2spec

    metadata_items: list[dict] = []
    stats: dict[str, dict[str, int]] = {
        split: {"seen": 0, "kept": 0, "too_short": 0, "too_long": 0, "empty_tokens": 0}
        for split in split_to_raw_items
    }
    for split, raw_items in split_to_raw_items.items():
        for raw_item in tqdm(raw_items, desc=f"hubert:{split}"):
            stats[split]["seen"] += 1
            wav2spec = librosa_wav2spec(
                str(raw_item.wav_path),
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
            if frames < min_mel_frames:
                stats[split]["too_short"] += 1
                continue
            if max_mel_frames > 0 and frames > max_mel_frames:
                stats[split]["too_long"] += 1
                continue
            mel_tensor = torch.from_numpy(mel).unsqueeze(0).float().to(device)
            with torch.inference_mode():
                logits = model.inference(mel_tensor)
            tokens = logits.argmax(dim=-1).squeeze(0).detach().cpu().tolist()
            tokens = tokens[:frames]
            if not tokens:
                stats[split]["empty_tokens"] += 1
                continue
            metadata_items.append(
                {
                    "item_name": _build_item_name(raw_item.speaker, split, raw_item.prompt_id),
                    "wav_fn": str(raw_item.wav_path),
                    "hubert": " ".join(str(int(token)) for token in tokens),
                    "duration": round(float(len(wav2spec["wav"])) / float(hparams["audio_sample_rate"]), 6),
                    "split": split,
                    "txt": raw_item.txt_norm,
                    "txt_raw": raw_item.txt_raw,
                    "prompt_id": raw_item.prompt_id,
                    "speaker": raw_item.speaker,
                    "dataset": raw_item.dataset,
                }
            )
            stats[split]["kept"] += 1
    metadata_items.sort(key=lambda item: _natural_key(item["item_name"]))
    return metadata_items, stats


def _target_priority(source_speaker: str, candidate_speaker: str):
    native = {"bdl", "slt"}
    if source_speaker == "bdl":
        rank = 0 if candidate_speaker == "slt" else (1 if candidate_speaker in native else 2)
    elif source_speaker == "slt":
        rank = 0 if candidate_speaker == "bdl" else (1 if candidate_speaker in native else 2)
    else:
        if candidate_speaker == "bdl":
            rank = 0
        elif candidate_speaker == "slt":
            rank = 1
        elif candidate_speaker in native:
            rank = 2
        else:
            rank = 3
    return (rank, _natural_key(candidate_speaker))


def _build_pair_manifest(metadata_items: list[dict]) -> tuple[dict[str, list[dict]], dict[str, dict[str, int]]]:
    split_items: dict[str, list[dict]] = defaultdict(list)
    for item in metadata_items:
        split_items[str(item["split"])].append(item)

    manifest: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    stats: dict[str, dict[str, int]] = {}
    for split, items in split_items.items():
        items_by_prompt: dict[str, list[dict]] = defaultdict(list)
        items_by_speaker: dict[str, list[dict]] = defaultdict(list)
        for item in items:
            items_by_prompt[str(item["prompt_id"])].append(item)
            items_by_speaker[str(item["speaker"])].append(item)
        for speaker_items in items_by_speaker.values():
            speaker_items.sort(key=lambda item: _natural_key(item["prompt_id"]))
        split_pairs: list[dict] = []
        dropped_ref = 0
        dropped_target = 0
        for item in sorted(items, key=lambda row: _natural_key(row["item_name"])):
            speaker = str(item["speaker"])
            prompt_id = str(item["prompt_id"])
            same_speaker_items = items_by_speaker[speaker]
            ref_candidates = [cand for cand in same_speaker_items if cand["item_name"] != item["item_name"]]
            if not ref_candidates:
                dropped_ref += 1
                continue
            ref_item = ref_candidates[0]
            target_candidates = [
                cand
                for cand in items_by_prompt[prompt_id]
                if cand["speaker"] != speaker and cand["txt"] == item["txt"]
            ]
            if not target_candidates:
                dropped_target += 1
                continue
            target_candidates.sort(key=lambda cand: _target_priority(speaker, str(cand["speaker"])))
            target_item = target_candidates[0]
            split_pairs.append(
                {
                    "source": item["item_name"],
                    "group_id": prompt_id,
                    "refs": [
                        {
                            "ref": ref_item["item_name"],
                            "target": target_item["item_name"],
                            "pair_rank": 0,
                            "pair_label": "same_text_diff_speaker",
                        }
                    ],
                }
            )
        manifest[split] = split_pairs
        stats[split] = {
            "items": len(items),
            "pairs": len(split_pairs),
            "dropped_missing_ref": dropped_ref,
            "dropped_missing_target": dropped_target,
            "speakers": len(items_by_speaker),
            "prompts": len(items_by_prompt),
        }
    return manifest, stats


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build processed metadata/pairs for CMU ARCTIC + L2-ARCTIC.")
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--processed_data_dir", required=True)
    parser.add_argument("--config", default="egs/conan_emformer_rhythm_v3.yaml")
    parser.add_argument("--emformer_ckpt", default="checkpoints/Emformer/model_ckpt_steps_700000.ckpt")
    parser.add_argument("--exp_name", default="build_arctic_processed_metadata")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--cmu_speakers", default="bdl,slt")
    parser.add_argument("--l2_speakers", default="")
    parser.add_argument("--exclude_l2_speakers", default="")
    parser.add_argument("--valid_prompt_count", type=int, default=16)
    parser.add_argument("--test_prompt_count", type=int, default=16)
    parser.add_argument("--train_prompt_count", type=int, default=0, help="0 means use all remaining prompts.")
    parser.add_argument("--min_shared_speakers", type=int, default=2)
    parser.add_argument("--min_mel_frames", type=int, default=32)
    parser.add_argument("--max_mel_frames", type=int, default=600)
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    raw_root = Path(args.raw_root).resolve()
    processed_data_dir = Path(args.processed_data_dir).resolve()
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    cmu_speakers = [speaker.lower() for speaker in _parse_csv_arg(args.cmu_speakers)]
    if not cmu_speakers:
        raise ValueError("At least one CMU speaker is required.")
    unknown_cmu = sorted(set(cmu_speakers) - set(CMU_SPEAKER_TO_DIR))
    if unknown_cmu:
        raise ValueError(f"Unsupported CMU speakers: {unknown_cmu}. Supported={sorted(CMU_SPEAKER_TO_DIR)}")

    l2_root = raw_root / "l2arctic_release_v5.0"
    all_l2_speakers = _discover_l2_speakers(l2_root)
    if args.l2_speakers:
        l2_speakers = _parse_csv_arg(args.l2_speakers)
    else:
        l2_speakers = all_l2_speakers
    excluded_l2 = {speaker.lower() for speaker in _parse_csv_arg(args.exclude_l2_speakers)}
    l2_speakers = [speaker for speaker in l2_speakers if speaker.lower() not in excluded_l2]
    missing_l2 = [speaker for speaker in l2_speakers if speaker not in all_l2_speakers]
    if missing_l2:
        raise ValueError(f"Unknown L2 speakers: {missing_l2}")

    raw_items: list[RawItem] = []
    for speaker in cmu_speakers:
        raw_items.extend(_scan_cmu_root(raw_root / CMU_SPEAKER_TO_DIR[speaker], speaker=speaker))
    for speaker in l2_speakers:
        raw_items.extend(_scan_l2_speaker(l2_root, speaker=speaker))
    if not raw_items:
        raise RuntimeError("No ARCTIC raw items were discovered.")

    items_by_prompt: dict[str, list[RawItem]] = defaultdict(list)
    for item in raw_items:
        items_by_prompt[item.prompt_id].append(item)
    prompt_items, prompt_stats = _pick_majority_text_items(
        items_by_prompt,
        min_shared_speakers=int(args.min_shared_speakers),
    )
    eligible_prompt_ids = sorted(prompt_items, key=_natural_key)
    if not eligible_prompt_ids:
        raise RuntimeError("No prompt ids satisfied the shared-speaker threshold after text normalization.")

    split_prompt_ids = _select_prompt_splits(
        prompt_ids=eligible_prompt_ids,
        valid_prompt_count=int(args.valid_prompt_count),
        test_prompt_count=int(args.test_prompt_count),
        train_prompt_count=int(args.train_prompt_count),
    )
    split_to_raw_items = {
        split: [
            item
            for prompt_id in sorted(prompt_ids, key=_natural_key)
            for item in prompt_items[prompt_id]
        ]
        for split, prompt_ids in split_prompt_ids.items()
    }

    model, hparams, device = _load_model_and_hparams(args)
    metadata_items, hubert_stats = _tokenize_items(
        split_to_raw_items=split_to_raw_items,
        model=model,
        hparams=hparams,
        device=device,
        min_mel_frames=int(args.min_mel_frames),
        max_mel_frames=int(args.max_mel_frames),
    )
    if not metadata_items:
        raise RuntimeError("No metadata items survived HuBERT extraction.")

    manifest, pair_stats = _build_pair_manifest(metadata_items)
    speakers = sorted({str(item["speaker"]) for item in metadata_items}, key=_natural_key)
    spker_map = {speaker: idx for idx, speaker in enumerate(speakers)}

    metadata_path = processed_data_dir / "metadata.json"
    metadata_alias_path = processed_data_dir / "metadata_vctk_librittsr_gt.json"
    speaker_path = processed_data_dir / "spker_set.json"
    summary_path = processed_data_dir / "build_summary.json"
    pair_manifest_path = processed_data_dir / "pairs.json"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_items, f, ensure_ascii=False, indent=2)
    with metadata_alias_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_items, f, ensure_ascii=False, indent=2)
    with speaker_path.open("w", encoding="utf-8") as f:
        json.dump(spker_map, f, ensure_ascii=False, indent=2)
    with pair_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "raw_root": str(raw_root),
                "processed_data_dir": str(processed_data_dir),
                "config": args.config,
                "emformer_ckpt": args.emformer_ckpt,
                "device": str(device),
                "speakers": speakers,
                "cmu_speakers": cmu_speakers,
                "l2_speakers": [speaker.lower() for speaker in l2_speakers],
                "num_items": len(metadata_items),
                "num_speakers": len(spker_map),
                "split_tags": {"train": "train", "valid": "valid", "test": "test"},
                "selected_prompt_ids": {
                    split: sorted(prompt_ids, key=_natural_key)
                    for split, prompt_ids in split_prompt_ids.items()
                },
                "prompt_stats": prompt_stats,
                "hubert_stats": hubert_stats,
                "pair_stats": pair_stats,
                "paths": {
                    "metadata": str(metadata_path),
                    "metadata_alias": str(metadata_alias_path),
                    "spker_set": str(speaker_path),
                    "pair_manifest": str(pair_manifest_path),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[build-arctic-metadata] processed_data_dir={processed_data_dir}")
    print(f"[build-arctic-metadata] device={device}")
    print(f"[build-arctic-metadata] speakers={len(spker_map)} items={len(metadata_items)}")
    for split in ("train", "valid", "test"):
        split_pairs = pair_stats.get(split, {})
        split_seen = hubert_stats.get(split, {})
        print(
            f"[build-arctic-metadata] split={split} "
            f"seen={split_seen.get('seen', 0)} kept={split_seen.get('kept', 0)} "
            f"pairs={split_pairs.get('pairs', 0)}"
        )
    print(f"[build-arctic-metadata] metadata={metadata_path}")
    print(f"[build-arctic-metadata] speaker_map={speaker_path}")
    print(f"[build-arctic-metadata] pairs={pair_manifest_path}")


if __name__ == "__main__":
    main()

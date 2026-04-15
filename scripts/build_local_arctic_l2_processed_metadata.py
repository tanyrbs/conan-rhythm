from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_TXT_DONE_RE = re.compile(r'^\(\s*([^\s]+)\s+"(.*)"\s*\)\s*$')


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build local ARCTIC/L2-ARCTIC processed metadata and pair manifests for "
            "rhythm_v3 quick validation."
        )
    )
    parser.add_argument("--data_root", required=True, help="Directory containing cmu_us_*_arctic and l2arctic_release_v5.0.")
    parser.add_argument("--processed_data_dir", required=True)
    parser.add_argument("--config", default="egs/conan_emformer_rhythm_v3.yaml")
    parser.add_argument(
        "--emformer_ckpt",
        default="checkpoints/Emformer/model_ckpt_steps_700000.ckpt",
    )
    parser.add_argument(
        "--exp_name",
        default="",
        help="Optional experiment name. Leave empty to avoid saving a checkpoint config during metadata build.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--train_utterances", type=int, default=48)
    parser.add_argument("--valid_utterances", type=int, default=8)
    parser.add_argument("--test_utterances", type=int, default=8)
    parser.add_argument("--ref_count_per_source", type=int, default=3)
    parser.add_argument("--min_mel_frames", type=int, default=32)
    parser.add_argument("--max_mel_frames", type=int, default=600)
    parser.add_argument(
        "--speaker_allowlist",
        default="",
        help="Optional comma-separated list of L2 speakers to include. Empty means all.",
    )
    parser.add_argument(
        "--max_l2_speakers",
        type=int,
        default=0,
        help="Optional cap on how many L2 speakers to include. 0 means all.",
    )
    parser.add_argument(
        "--native_target_speaker",
        choices=["bdl", "slt"],
        default="slt",
        help="Native paired-target speaker for L2 sources. The other native speaker is used when the source already matches this speaker.",
    )
    return parser.parse_args()


def _resolve_device(spec: str) -> torch.device:
    mode = str(spec or "auto").strip().lower()
    if mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    return torch.device("cpu")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _parse_txt_done(path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _TXT_DONE_RE.match(line)
        if match is None:
            continue
        utt_id = str(match.group(1)).strip()
        text = _normalize_text(match.group(2))
        if utt_id:
            prompts[utt_id] = text
    if not prompts:
        raise RuntimeError(f"No prompts parsed from {path}")
    return prompts


def _load_prompt_map(data_root: Path) -> dict[str, str]:
    native_roots = {
        "bdl": data_root / "cmu_us_bdl_arctic" / "etc" / "txt.done.data",
        "slt": data_root / "cmu_us_slt_arctic" / "etc" / "txt.done.data",
    }
    prompt_map: dict[str, str] | None = None
    for speaker, prompt_path in native_roots.items():
        if not prompt_path.exists():
            raise FileNotFoundError(f"Missing ARCTIC prompt file for {speaker}: {prompt_path}")
        speaker_map = _parse_txt_done(prompt_path)
        if prompt_map is None:
            prompt_map = speaker_map
            continue
        for utt_id, text in speaker_map.items():
            existing = prompt_map.get(utt_id)
            if existing is None:
                prompt_map[utt_id] = text
            elif existing != text:
                raise RuntimeError(
                    f"Prompt mismatch for {utt_id}: bdl/slt disagree between txt.done.data files."
                )
    assert prompt_map is not None
    return prompt_map


def _collect_native_wavs(data_root: Path) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for speaker in ("bdl", "slt"):
        wav_dir = data_root / f"cmu_us_{speaker}_arctic" / "wav"
        if not wav_dir.exists():
            raise FileNotFoundError(f"Missing native wav directory: {wav_dir}")
        out[speaker] = {path.stem: path for path in sorted(wav_dir.glob("*.wav"))}
    return out


def _resolve_l2_speakers(args: argparse.Namespace, l2_root: Path) -> list[str]:
    all_speakers = sorted(
        path.name
        for path in l2_root.iterdir()
        if path.is_dir() and path.name != "suitcase_corpus"
    )
    if args.speaker_allowlist.strip():
        allow = {part.strip() for part in args.speaker_allowlist.split(",") if part.strip()}
        all_speakers = [speaker for speaker in all_speakers if speaker in allow]
    if int(args.max_l2_speakers) > 0:
        all_speakers = all_speakers[: int(args.max_l2_speakers)]
    if not all_speakers:
        raise RuntimeError("No L2 speakers selected.")
    return all_speakers


def _collect_l2_wavs(args: argparse.Namespace, data_root: Path) -> dict[str, dict[str, Path]]:
    l2_root = data_root / "l2arctic_release_v5.0"
    if not l2_root.exists():
        raise FileNotFoundError(f"Missing L2-ARCTIC root: {l2_root}")
    out: dict[str, dict[str, Path]] = {}
    for speaker in _resolve_l2_speakers(args, l2_root):
        wav_dir = l2_root / speaker / speaker / "wav"
        if not wav_dir.exists():
            raise FileNotFoundError(f"Missing L2 wav directory: {wav_dir}")
        out[speaker] = {path.stem: path for path in sorted(wav_dir.glob("*.wav"))}
    return out


def _split_utterance_ids(prompt_map: dict[str, str], args: argparse.Namespace) -> dict[str, list[str]]:
    utterance_ids = sorted(prompt_map.keys())
    train_n = int(args.train_utterances)
    valid_n = int(args.valid_utterances)
    test_n = int(args.test_utterances)
    total_needed = train_n + valid_n + test_n
    if total_needed <= 0:
        raise RuntimeError("At least one utterance must be assigned across train/valid/test.")
    if total_needed > len(utterance_ids):
        raise RuntimeError(
            f"Requested {total_needed} utterances but only {len(utterance_ids)} canonical prompts exist."
        )
    return {
        "train": utterance_ids[:train_n],
        "valid": utterance_ids[train_n : train_n + valid_n],
        "test": utterance_ids[train_n + valid_n : train_n + valid_n + test_n],
    }


def _build_candidates(
    *,
    prompt_map: dict[str, str],
    split_to_utt_ids: dict[str, list[str]],
    native_wavs: dict[str, dict[str, Path]],
    l2_wavs: dict[str, dict[str, Path]],
) -> list[dict]:
    candidates: list[dict] = []
    all_speakers = {**native_wavs, **l2_wavs}
    for speaker, wav_map in sorted(all_speakers.items()):
        for split, utt_ids in split_to_utt_ids.items():
            for utt_id in utt_ids:
                wav_path = wav_map.get(utt_id)
                if wav_path is None:
                    continue
                text = prompt_map.get(utt_id)
                if not text:
                    continue
                candidates.append(
                    {
                        "item_name": f"{speaker}_{split}_{utt_id}",
                        "speaker": speaker,
                        "split": split,
                        "utt_id": utt_id,
                        "txt": text,
                        "wav_fn": str(wav_path),
                        "is_native": speaker in {"bdl", "slt"},
                    }
                )
    if not candidates:
        raise RuntimeError("No candidate wavs were found for the selected speakers and utterance splits.")
    return candidates


def _select_candidates(
    *,
    rows: list[dict],
    min_frames: int,
    max_frames: int,
    model,
    device: torch.device,
    librosa_wav2spec,
    hparams,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    selected: list[dict] = []
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"seen": 0, "kept": 0, "too_short": 0, "too_long": 0}
    )
    for row in tqdm(rows, desc="metadata:arctic_l2"):
        split = str(row["split"])
        wav2spec = librosa_wav2spec(
            row["wav_fn"],
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
        stats[split]["seen"] += 1
        frames = int(mel.shape[0])
        if frames < min_frames:
            stats[split]["too_short"] += 1
            continue
        if max_frames > 0 and frames > max_frames:
            stats[split]["too_long"] += 1
            continue
        mel_tensor = torch.from_numpy(mel).unsqueeze(0).float().to(device)
        with torch.inference_mode():
            logits = model.inference(mel_tensor)
        tokens = logits.argmax(dim=-1).squeeze(0).detach().cpu().tolist()
        if len(tokens) != frames:
            tokens = tokens[:frames]
        if not tokens:
            continue
        selected.append(
            {
                "item_name": row["item_name"],
                "speaker": row["speaker"],
                "split": split,
                "utt_id": row["utt_id"],
                "txt": row["txt"],
                "wav_fn": row["wav_fn"],
                "hubert": " ".join(str(int(token)) for token in tokens),
                "duration": round(float(len(wav2spec["wav"])) / float(hparams["audio_sample_rate"]), 6),
                "is_native": bool(row["is_native"]),
            }
        )
        stats[split]["kept"] += 1
    return selected, dict(stats)


def _resolve_target_speaker(source_speaker: str, native_target_speaker: str) -> str:
    if source_speaker == native_target_speaker:
        return "bdl" if native_target_speaker == "slt" else "slt"
    if source_speaker in {"bdl", "slt"}:
        return native_target_speaker
    return native_target_speaker


def _build_pair_manifest(
    *,
    items: list[dict],
    ref_count_per_source: int,
    native_target_speaker: str,
) -> tuple[dict[str, list[dict]], dict[str, int]]:
    by_split_speaker: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    by_split_speaker_utt: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
    for item in items:
        split = str(item["split"])
        speaker = str(item["speaker"])
        utt_id = str(item["utt_id"])
        by_split_speaker[split][speaker].append(item)
        by_split_speaker_utt[split][(speaker, utt_id)] = item
    for split in by_split_speaker:
        for speaker in by_split_speaker[split]:
            by_split_speaker[split][speaker].sort(key=lambda entry: str(entry["utt_id"]))

    manifest: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    stats = {"sources": 0, "kept": 0, "missing_target": 0, "missing_ref": 0}
    for split in ("train", "valid", "test"):
        speaker_groups = by_split_speaker.get(split, {})
        for speaker, source_items in sorted(speaker_groups.items()):
            for idx, source_item in enumerate(source_items):
                stats["sources"] += 1
                utt_id = str(source_item["utt_id"])
                target_speaker = _resolve_target_speaker(speaker, native_target_speaker)
                target_item = by_split_speaker_utt.get(split, {}).get((target_speaker, utt_id))
                if target_item is None:
                    stats["missing_target"] += 1
                    continue
                ref_candidates = [
                    candidate
                    for candidate in source_items
                    if str(candidate["item_name"]) != str(source_item["item_name"])
                ]
                if not ref_candidates:
                    stats["missing_ref"] += 1
                    continue
                refs: list[dict] = []
                count = min(int(ref_count_per_source), len(ref_candidates))
                for offset in range(count):
                    ref_item = ref_candidates[(idx + offset) % len(ref_candidates)]
                    refs.append(
                        {
                            "ref_item_name": str(ref_item["item_name"]),
                            "target_item_name": str(target_item["item_name"]),
                            "pair_rank": int(offset),
                            "pair_label": "same_speaker_diff_text",
                        }
                    )
                manifest[split].append(
                    {
                        "source_item_name": str(source_item["item_name"]),
                        "group_id": f"{speaker}_{utt_id}",
                        "refs": refs,
                    }
                )
                stats["kept"] += 1
    return manifest, stats


def main() -> None:
    args = _parse_args()

    from modules.Emformer.emformer import EmformerDistillModel
    from utils.audio import librosa_wav2spec
    from utils.commons.ckpt_utils import load_ckpt_emformer
    from utils.commons.hparams import hparams, set_hparams

    data_root = Path(args.data_root).resolve()
    processed_data_dir = Path(args.processed_data_dir).resolve()
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    prompt_map = _load_prompt_map(data_root)
    split_to_utt_ids = _split_utterance_ids(prompt_map, args)
    native_wavs = _collect_native_wavs(data_root)
    l2_wavs = _collect_l2_wavs(args, data_root)
    candidates = _build_candidates(
        prompt_map=prompt_map,
        split_to_utt_ids=split_to_utt_ids,
        native_wavs=native_wavs,
        l2_wavs=l2_wavs,
    )

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

    items, split_stats = _select_candidates(
        rows=candidates,
        min_frames=int(args.min_mel_frames),
        max_frames=int(args.max_mel_frames),
        model=model,
        device=device,
        librosa_wav2spec=librosa_wav2spec,
        hparams=hparams,
    )
    if not items:
        raise RuntimeError("No items survived tokenization and frame filtering.")

    pair_manifest, pair_stats = _build_pair_manifest(
        items=items,
        ref_count_per_source=int(args.ref_count_per_source),
        native_target_speaker=str(args.native_target_speaker),
    )

    speakers = sorted({str(item["speaker"]) for item in items})
    spker_map = {speaker: idx for idx, speaker in enumerate(speakers)}

    metadata_path = processed_data_dir / "metadata_vctk_librittsr_gt.json"
    metadata_json_path = processed_data_dir / "metadata.json"
    pair_path = processed_data_dir / "pairs.json"
    speaker_path = processed_data_dir / "spker_set.json"
    summary_path = processed_data_dir / "build_summary.json"

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(items, handle, ensure_ascii=False, indent=2)
    with metadata_json_path.open("w", encoding="utf-8") as handle:
        json.dump(items, handle, ensure_ascii=False, indent=2)
    with pair_path.open("w", encoding="utf-8") as handle:
        json.dump(pair_manifest, handle, ensure_ascii=False, indent=2)
    with speaker_path.open("w", encoding="utf-8") as handle:
        json.dump(spker_map, handle, ensure_ascii=False, indent=2)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "data_root": str(data_root),
                "processed_data_dir": str(processed_data_dir),
                "config": args.config,
                "emformer_ckpt": args.emformer_ckpt,
                "device": str(device),
                "native_target_speaker": args.native_target_speaker,
                "selected_l2_speakers": sorted(l2_wavs.keys()),
                "split_tags": {"train": "train", "valid": "valid", "test": "test"},
                "utterance_split_ids": split_to_utt_ids,
                "split_stats": split_stats,
                "pair_stats": pair_stats,
                "num_items": len(items),
                "num_speakers": len(spker_map),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[build-local-arctic-l2] data_root={data_root}")
    print(f"[build-local-arctic-l2] processed_data_dir={processed_data_dir}")
    print(f"[build-local-arctic-l2] device={device}")
    print(f"[build-local-arctic-l2] items={len(items)} speakers={len(spker_map)}")
    print(f"[build-local-arctic-l2] metadata={metadata_json_path}")
    print(f"[build-local-arctic-l2] pairs={pair_path}")
    print(f"[build-local-arctic-l2] speaker_map={speaker_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _speaker_id(item: dict) -> str:
    speaker_id = str(item.get("speaker_id", "")).strip()
    if speaker_id:
        return speaker_id
    return str(item["item_name"]).split("_", 1)[0]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge processed metadata shards into one processed dir.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shard_dir", action="append", required=True, help="Repeat for each shard directory.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    shard_dirs = [Path(x).resolve() for x in args.shard_dir]
    output_dir.mkdir(parents=True, exist_ok=True)

    all_items: list[dict] = []
    seen_item_names: set[str] = set()
    shard_summaries: list[dict] = []
    split_item_names: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    speakers: set[str] = set()

    for shard_dir in shard_dirs:
        metadata_path = shard_dir / "metadata.json"
        if not metadata_path.exists():
            alt_path = shard_dir / "metadata_vctk_librittsr_gt.json"
            if alt_path.exists():
                metadata_path = alt_path
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing shard metadata.json under {shard_dir}")
        items = _load_json(metadata_path)
        summary_path = shard_dir / "build_summary.json"
        if summary_path.exists():
            shard_summaries.append(_load_json(summary_path))
        else:
            shard_summaries.append({"processed_data_dir": str(shard_dir), "num_items": len(items)})
        for item in items:
            item_name = str(item["item_name"])
            if item_name in seen_item_names:
                raise ValueError(f"Duplicate item_name while merging shards: {item_name}")
            seen_item_names.add(item_name)
            all_items.append(item)
            split_name = str(item.get("split", "")).strip().lower()
            if split_name not in split_item_names:
                raise ValueError(f"Unexpected split '{split_name}' in item {item_name}")
            split_item_names[split_name].append(item_name)
            speakers.add(_speaker_id(item))

    split_order = {"train": 0, "valid": 1, "test": 2}
    all_items.sort(key=lambda item: (split_order.get(str(item.get("split", "")), 99), str(item["item_name"])))
    for split_name in split_item_names:
        split_item_names[split_name].sort()

    spker_map = {speaker: idx for idx, speaker in enumerate(sorted(speakers))}
    metadata_path = output_dir / "metadata_vctk_librittsr_gt.json"
    metadata_json_path = output_dir / "metadata.json"
    speaker_path = output_dir / "spker_set.json"
    summary_path = output_dir / "build_summary.json"
    split_manifest_paths = {
        split_name: output_dir / f"{split_name}_item_names.txt"
        for split_name in ("train", "valid", "test")
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    with metadata_json_path.open("w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    with speaker_path.open("w", encoding="utf-8") as f:
        json.dump(spker_map, f, ensure_ascii=False, indent=2)
    for split_name, manifest_path in split_manifest_paths.items():
        with manifest_path.open("w", encoding="utf-8") as f:
            for item_name in split_item_names[split_name]:
                f.write(f"{item_name}\n")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output_dir": str(output_dir),
                "num_items": len(all_items),
                "num_speakers": len(spker_map),
                "split_item_manifests": {
                    split_name: {
                        "path": str(path),
                        "count": len(split_item_names[split_name]),
                    }
                    for split_name, path in split_manifest_paths.items()
                },
                "shard_dirs": [str(x) for x in shard_dirs],
                "shard_summaries": shard_summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[merge-processed-metadata-shards] output_dir={output_dir}")
    print(f"[merge-processed-metadata-shards] shard_count={len(shard_dirs)}")
    print(f"[merge-processed-metadata-shards] items={len(all_items)} speakers={len(spker_map)}")
    for split_name, manifest_path in split_manifest_paths.items():
        print(
            f"  - {split_name}: {manifest_path} ({len(split_item_names[split_name])} items)"
        )


if __name__ == "__main__":
    main()

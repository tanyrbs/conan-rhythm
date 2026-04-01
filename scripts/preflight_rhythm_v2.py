from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.Conan.rhythm.supervision import (
    RHYTHM_CACHE_VERSION,
    RHYTHM_GUIDANCE_SURFACE_NAME,
    RHYTHM_REFERENCE_MODE_STATIC_REF_FULL,
    RHYTHM_RETIMED_SOURCE_GUIDANCE,
    RHYTHM_RETIMED_SOURCE_TEACHER,
    RHYTHM_TEACHER_SURFACE_NAME,
    RHYTHM_TRACE_HOP_MS,
    RHYTHM_UNIT_HOP_MS,
)
from utils.commons.hparams import set_hparams
from utils.commons.indexed_datasets import IndexedDataset
from tasks.Conan.dataset import ConanDataset
from tasks.Conan.Conan import ConanTask
import torch


CORE_RHYTHM_FIELDS = {
    "content_units",
    "dur_anchor_src",
    "open_run_mask",
    "sealed_mask",
    "sep_hint",
    "boundary_confidence",
    "ref_rhythm_stats",
    "ref_rhythm_trace",
}
CACHED_ONLY_META_FIELD_GROUPS = [
    ("rhythm_cache_version",),
    ("rhythm_unit_hop_ms",),
    ("rhythm_trace_hop_ms",),
    ("rhythm_trace_bins",),
    ("rhythm_trace_horizon",),
    ("rhythm_slow_topk",),
    ("rhythm_selector_cell_size",),
    ("rhythm_source_phrase_threshold",),
    ("rhythm_reference_mode_id",),
    ("rhythm_guidance_surface_name",),
]

GUIDANCE_FIELD_GROUPS = [
    ("rhythm_speech_exec_tgt",),
    ("rhythm_blank_exec_tgt", "rhythm_pause_exec_tgt"),
    ("rhythm_speech_budget_tgt",),
    ("rhythm_blank_budget_tgt", "rhythm_pause_budget_tgt"),
    ("rhythm_target_confidence",),
    ("rhythm_guidance_confidence",),
    ("rhythm_guidance_surface_name",),
]

TEACHER_FIELD_GROUPS = [
    ("rhythm_teacher_speech_exec_tgt",),
    ("rhythm_teacher_blank_exec_tgt", "rhythm_teacher_pause_exec_tgt"),
    ("rhythm_teacher_speech_budget_tgt",),
    ("rhythm_teacher_blank_budget_tgt", "rhythm_teacher_pause_budget_tgt"),
    ("rhythm_teacher_allocation_tgt",),
    ("rhythm_teacher_prefix_clock_tgt",),
    ("rhythm_teacher_prefix_backlog_tgt",),
    ("rhythm_teacher_confidence",),
    ("rhythm_teacher_surface_name",),
]

RETIMED_FIELD_GROUPS = [
    ("rhythm_retimed_mel_tgt",),
    ("rhythm_retimed_mel_len",),
    ("rhythm_retimed_frame_weight",),
    ("rhythm_retimed_target_confidence",),
    ("rhythm_retimed_target_source_id",),
    ("rhythm_retimed_target_surface_name",),
]


def _normalize_surface(surface: str) -> str:
    aliases = {
        "cache_teacher": "teacher",
        "offline": "teacher",
        "offline_teacher": "teacher",
        "teacher_surface": "teacher",
        "guidance_surface": "guidance",
        "self": "guidance",
    }
    return aliases.get(surface, surface)


def _normalize_distill(surface: str) -> str:
    aliases = {
        "off": "none",
        "disable": "none",
        "disabled": "none",
        "false": "none",
        "cache_teacher": "cache",
        "cached_teacher": "cache",
        "full_context": "offline",
        "shared_offline": "offline",
        "algo": "algorithmic",
        "teacher": "cache",
    }
    return aliases.get(surface, surface)


def _extract_scalar(value):
    arr = np.asarray(value)
    if arr.size <= 0:
        raise RuntimeError("Encountered empty scalar field while validating rhythm cache metadata.")
    return arr.reshape(-1)[0]


def _expected_cache_contract(hp: dict) -> dict[str, int | float]:
    return {
        "rhythm_cache_version": int(hp.get("rhythm_cache_version", RHYTHM_CACHE_VERSION)),
        "rhythm_unit_hop_ms": int(hp.get("rhythm_unit_hop_ms", RHYTHM_UNIT_HOP_MS)),
        "rhythm_trace_hop_ms": int(hp.get("rhythm_trace_hop_ms", RHYTHM_TRACE_HOP_MS)),
        "rhythm_trace_bins": int(hp.get("rhythm_trace_bins", 24)),
        "rhythm_trace_horizon": float(hp.get("rhythm_trace_horizon", 0.35)),
        "rhythm_slow_topk": int(hp.get("rhythm_slow_topk", 6)),
        "rhythm_selector_cell_size": int(hp.get("rhythm_selector_cell_size", 3)),
        "rhythm_source_phrase_threshold": float(hp.get("rhythm_source_phrase_threshold", 0.55)),
        "rhythm_reference_mode_id": int(
            hp.get("rhythm_reference_mode_id", RHYTHM_REFERENCE_MODE_STATIC_REF_FULL)
        ),
    }


def _detect_stage(hp: dict, config_path: str) -> str:
    distill = _normalize_distill(str(hp.get("rhythm_distill_surface", "auto") or "auto").strip().lower())
    if (
        bool(hp.get("rhythm_apply_train_override", False))
        or bool(hp.get("rhythm_apply_valid_override", False))
        or bool(hp.get("rhythm_require_retimed_cache", False))
        or "retimed_train" in str(config_path).lower()
    ):
        return "retimed_train"
    if (
        float(hp.get("lambda_rhythm_distill", 0.0)) > 0.0
        or bool(hp.get("rhythm_enable_dual_mode_teacher", False))
        or distill in {"cache", "offline", "algorithmic"}
        or "dual_mode_kd" in str(config_path).lower()
    ):
        return "dual_mode_kd"
    if (
        bool(hp.get("rhythm_schedule_only_stage", False))
        or bool(hp.get("rhythm_optimize_module_only", False))
        or "schedule_only" in str(config_path).lower()
    ):
        return "schedule_only"
    return "transitional"


def _validate_stage_contract(hp: dict, *, config_path: str) -> tuple[str, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    stage = _detect_stage(hp, config_path)
    target_mode = str(hp.get("rhythm_dataset_target_mode", "prefer_cache") or "prefer_cache").strip().lower()
    primary = _normalize_surface(str(hp.get("rhythm_primary_target_surface", "guidance") or "guidance").strip().lower())
    distill = _normalize_distill(str(hp.get("rhythm_distill_surface", "auto") or "auto").strip().lower())
    require_cached_teacher = bool(hp.get("rhythm_require_cached_teacher", False))
    require_retimed_cache = bool(hp.get("rhythm_require_retimed_cache", False))
    enable_dual = bool(hp.get("rhythm_enable_dual_mode_teacher", False))
    schedule_only = bool(hp.get("rhythm_schedule_only_stage", False))
    optimize_module_only = bool(hp.get("rhythm_optimize_module_only", False))
    lambda_distill = float(hp.get("lambda_rhythm_distill", 0.0))
    apply_train = bool(hp.get("rhythm_apply_train_override", False))
    apply_valid = bool(hp.get("rhythm_apply_valid_override", False))
    use_retimed_target = bool(hp.get("rhythm_use_retimed_target_if_available", False))
    retimed_source = str(hp.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower()
    binarize_teacher = bool(hp.get("rhythm_binarize_teacher_targets", False))
    binarize_retimed = bool(hp.get("rhythm_binarize_retimed_mel_targets", False))

    if stage == "schedule_only":
        if target_mode != "cached_only":
            errors.append("Formal schedule-only stage should use rhythm_dataset_target_mode: cached_only.")
        if primary != "teacher":
            errors.append("Formal schedule-only stage should use rhythm_primary_target_surface: teacher.")
        if not require_cached_teacher:
            errors.append("Formal schedule-only stage should require cached teacher surfaces.")
        if not schedule_only:
            errors.append("Formal schedule-only stage should keep rhythm_schedule_only_stage: true.")
        if not optimize_module_only:
            errors.append("Formal schedule-only stage should keep rhythm_optimize_module_only: true.")
        if enable_dual:
            errors.append("Stage-1 schedule-only should not enable rhythm_enable_dual_mode_teacher.")
        if lambda_distill > 0.0 or distill != "none":
            errors.append("Stage-1 schedule-only should keep distillation disabled.")
        if apply_train or apply_valid:
            errors.append("Stage-1 schedule-only should not enable train/valid retimed rendering.")
    elif stage == "dual_mode_kd":
        if target_mode != "cached_only":
            errors.append("Formal dual-mode KD stage should use rhythm_dataset_target_mode: cached_only.")
        if primary != "teacher":
            errors.append("Formal dual-mode KD stage should use rhythm_primary_target_surface: teacher.")
        if distill != "offline":
            errors.append("Formal dual-mode KD stage should use rhythm_distill_surface: offline.")
        if lambda_distill <= 0.0:
            errors.append("Formal dual-mode KD stage should keep lambda_rhythm_distill > 0.")
        if not enable_dual:
            errors.append("Formal dual-mode KD stage requires rhythm_enable_dual_mode_teacher: true.")
        if not require_cached_teacher:
            errors.append("Formal dual-mode KD stage should require cached teacher surfaces.")
        if apply_train or apply_valid:
            errors.append("Dual-mode KD stage should not enable train/valid retimed rendering; that belongs to stage-3.")
        if not schedule_only:
            warnings.append("Dual-mode KD no longer keeps rhythm_schedule_only_stage: true; this deviates from the maintained stage-2 path.")
        if not optimize_module_only:
            warnings.append("Dual-mode KD no longer keeps rhythm_optimize_module_only: true; this deviates from the maintained stage-2 path.")
    elif stage == "retimed_train":
        if target_mode != "cached_only":
            errors.append("Formal retimed-train stage should use rhythm_dataset_target_mode: cached_only.")
        if primary != "teacher":
            errors.append("Formal retimed-train stage should use rhythm_primary_target_surface: teacher.")
        if distill != "offline":
            errors.append("Formal retimed-train stage should use rhythm_distill_surface: offline.")
        if not enable_dual:
            errors.append("Formal retimed-train stage requires rhythm_enable_dual_mode_teacher: true.")
        if not require_cached_teacher:
            errors.append("Formal retimed-train stage should require cached teacher surfaces.")
        if not require_retimed_cache:
            errors.append("Formal retimed-train stage should require cached retimed mel targets.")
        if not use_retimed_target:
            errors.append("Formal retimed-train stage requires rhythm_use_retimed_target_if_available: true.")
        if not apply_train or not apply_valid:
            errors.append("Formal retimed-train stage should enable both train/valid retimed rendering.")
        if schedule_only:
            errors.append("Formal retimed-train stage should set rhythm_schedule_only_stage: false.")
        if optimize_module_only:
            errors.append("Formal retimed-train stage should set rhythm_optimize_module_only: false.")
        if retimed_source != "teacher":
            errors.append("Formal retimed-train stage should use rhythm_binarize_retimed_mel_source: teacher.")
        if not binarize_teacher:
            errors.append("Formal retimed-train stage requires rhythm_binarize_teacher_targets: true.")
        if not binarize_retimed:
            errors.append("Formal retimed-train stage requires rhythm_binarize_retimed_mel_targets: true.")
    else:
        warnings.append(
            "This config resolves to a transitional/prefer_cache path, not the maintained formal chain "
            "(schedule_only -> dual_mode_kd -> retimed_train)."
        )
    return stage, errors, warnings


def _expected_fields(hp: dict) -> tuple[list[tuple[str, ...]], list[str], list[str]]:
    expected_groups: list[tuple[str, ...]] = [(key,) for key in sorted(CORE_RHYTHM_FIELDS)]
    errors: list[str] = []
    warnings: list[str] = []

    primary = _normalize_surface(str(hp.get("rhythm_primary_target_surface", "guidance") or "guidance").strip().lower())
    distill = _normalize_distill(str(hp.get("rhythm_distill_surface", "auto") or "auto").strip().lower())
    cached_only = str(hp.get("rhythm_dataset_target_mode", "prefer_cache") or "prefer_cache").strip().lower() == "cached_only"

    if cached_only:
        expected_groups.extend(CACHED_ONLY_META_FIELD_GROUPS)
    if primary == "guidance":
        expected_groups.extend(GUIDANCE_FIELD_GROUPS)
    if (
        primary == "teacher"
        or bool(hp.get("rhythm_require_cached_teacher", False))
        or bool(hp.get("rhythm_binarize_teacher_targets", False))
        or str(hp.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower() == "teacher"
    ):
        expected_groups.extend(TEACHER_FIELD_GROUPS)
    if (
        bool(hp.get("rhythm_require_retimed_cache", False))
        or bool(hp.get("rhythm_apply_train_override", False))
        or bool(hp.get("rhythm_apply_valid_override", False))
    ):
        expected_groups.extend(RETIMED_FIELD_GROUPS)

    if float(hp.get("lambda_rhythm_distill", 0.0)) > 0.0:
        if distill == "none":
            errors.append("lambda_rhythm_distill > 0 but rhythm_distill_surface disables distillation.")
        if distill == "offline" and not bool(hp.get("rhythm_enable_dual_mode_teacher", False)):
            errors.append("Offline distillation requires rhythm_enable_dual_mode_teacher: true.")
        if distill == "cache":
            expected_groups.extend(TEACHER_FIELD_GROUPS)

    if bool(hp.get("rhythm_apply_train_override", False)) and not bool(hp.get("rhythm_use_retimed_target_if_available", False)):
        errors.append("Train-time retimed rendering requires rhythm_use_retimed_target_if_available: true.")

    if bool(hp.get("rhythm_schedule_only_stage", False)) and bool(hp.get("rhythm_apply_train_override", False)):
        errors.append("Schedule-only stage should not enable train-time retimed rendering.")

    if cached_only and int(hp.get("rhythm_cache_version", -1)) <= 0:
        errors.append("cached_only requires a positive rhythm_cache_version.")

    if primary == "teacher" and not bool(hp.get("rhythm_binarize_teacher_targets", False)):
        warnings.append("Primary surface is teacher but rhythm_binarize_teacher_targets is false.")

    dedup = []
    seen = set()
    for group in expected_groups:
        if group in seen:
            continue
        seen.add(group)
        dedup.append(group)
    return dedup, errors, warnings


def _open_dataset(path_prefix: str):
    idx_path = f"{path_prefix}.idx"
    data_path = f"{path_prefix}.data"
    if not os.path.exists(idx_path) or not os.path.exists(data_path):
        return None
    return IndexedDataset(path_prefix)


def _collect_presence(ds: IndexedDataset, limit: int) -> tuple[list[dict], Counter, list[str]]:
    items: list[dict] = []
    counts: Counter[str] = Counter()
    mismatches: list[str] = []
    num_items = min(len(ds), max(1, limit))
    for idx in range(num_items):
        item = ds[idx]
        items.append(item)
        for key in item.keys():
            counts[key] += 1
        if "rhythm_cache_version" in item:
            try:
                version = int(item["rhythm_cache_version"][0]) if hasattr(item["rhythm_cache_version"], "__len__") else int(item["rhythm_cache_version"])
                counts[f"__cache_version__{version}"] += 1
            except Exception:
                mismatches.append(f"item[{idx}] has unreadable rhythm_cache_version")
    return items, counts, mismatches


def _validate_inspected_items(items: list[dict], hp: dict, *, split: str) -> list[str]:
    errors: list[str] = []
    cached_only = str(hp.get("rhythm_dataset_target_mode", "prefer_cache") or "prefer_cache").strip().lower() == "cached_only"
    need_teacher = (
        _normalize_surface(str(hp.get("rhythm_primary_target_surface", "guidance") or "guidance").strip().lower()) == "teacher"
        or bool(hp.get("rhythm_require_cached_teacher", False))
        or str(hp.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower() == "teacher"
    )
    need_retimed = bool(hp.get("rhythm_require_retimed_cache", False)) or bool(hp.get("rhythm_apply_train_override", False)) or bool(hp.get("rhythm_apply_valid_override", False))
    expected_meta = _expected_cache_contract(hp)
    for idx, item in enumerate(items):
        item_name = str(item.get("item_name", f"{split}[{idx}]"))
        if cached_only:
            for key, expected in expected_meta.items():
                if key not in item:
                    errors.append(f"{item_name}: missing cached_only contract key '{key}'.")
                    continue
                found = _extract_scalar(item[key])
                if isinstance(expected, float):
                    if abs(float(found) - expected) > 1e-5:
                        errors.append(
                            f"{item_name}: cache contract mismatch for {key}, found={float(found):.6f}, expected={expected:.6f}."
                        )
                elif int(found) != expected:
                    errors.append(
                        f"{item_name}: cache contract mismatch for {key}, found={int(found)}, expected={expected}."
                    )
            if "rhythm_guidance_surface_name" in item:
                guidance_surface = str(_extract_scalar(item["rhythm_guidance_surface_name"]))
                if guidance_surface != RHYTHM_GUIDANCE_SURFACE_NAME:
                    errors.append(
                        f"{item_name}: rhythm_guidance_surface_name mismatch, found={guidance_surface}, expected={RHYTHM_GUIDANCE_SURFACE_NAME}."
                    )
        if need_teacher:
            if "rhythm_teacher_surface_name" not in item:
                errors.append(f"{item_name}: missing rhythm_teacher_surface_name for teacher-backed training.")
            else:
                teacher_surface = str(_extract_scalar(item["rhythm_teacher_surface_name"]))
                if teacher_surface != RHYTHM_TEACHER_SURFACE_NAME:
                    errors.append(
                        f"{item_name}: rhythm_teacher_surface_name mismatch, found={teacher_surface}, expected={RHYTHM_TEACHER_SURFACE_NAME}."
                    )
        if need_retimed:
            if "rhythm_retimed_target_source_id" not in item:
                errors.append(f"{item_name}: missing rhythm_retimed_target_source_id for retimed training.")
            else:
                retimed_source = str(hp.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower()
                expected_source_id = RHYTHM_RETIMED_SOURCE_TEACHER if retimed_source == "teacher" else RHYTHM_RETIMED_SOURCE_GUIDANCE
                found_source_id = int(_extract_scalar(item["rhythm_retimed_target_source_id"]))
                if found_source_id != expected_source_id:
                    errors.append(
                        f"{item_name}: rhythm_retimed_target_source_id mismatch, found={found_source_id}, expected={expected_source_id}."
                    )
            if "rhythm_retimed_target_surface_name" not in item:
                errors.append(f"{item_name}: missing rhythm_retimed_target_surface_name for retimed training.")
            else:
                expected_surface = RHYTHM_TEACHER_SURFACE_NAME if str(hp.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower() == "teacher" else RHYTHM_GUIDANCE_SURFACE_NAME
                found_surface = str(_extract_scalar(item["rhythm_retimed_target_surface_name"]))
                if found_surface != expected_surface:
                    errors.append(
                        f"{item_name}: rhythm_retimed_target_surface_name mismatch, found={found_surface}, expected={expected_surface}."
                    )
    return errors


def _run_dataset_and_model_dry_run(split: str, *, run_model: bool) -> list[str]:
    errors: list[str] = []
    try:
        ds = ConanDataset(prefix=split, shuffle=False)
    except Exception as exc:
        return [f"Failed to build ConanDataset for split '{split}': {exc}"]
    filtered_len = len(ds)
    print(f"[preflight] dataset_split={split} filtered_items={filtered_len}")
    if filtered_len <= 0:
        return [f"Split '{split}' is empty after ConanDataset filtering."]
    try:
        batch = ds.collater([ds[0]])
    except Exception as exc:
        return [f"Failed to collate split '{split}' sample: {exc}"]
    if not run_model:
        return errors
    try:
        task = ConanTask()
        task.build_tts_model()
        task.global_step = 0
        with torch.no_grad():
            losses, output = task.run_model(batch, infer=False)
        if "mel_out" not in output:
            errors.append(f"Model dry-run for split '{split}' did not produce mel_out.")
        if "rhythm_execution" not in output:
            errors.append(f"Model dry-run for split '{split}' did not produce rhythm_execution.")
        print(f"[preflight] model_dry_run split={split} mel_out={tuple(output['mel_out'].shape)} "
              f"units={tuple(output['speech_duration_exec'].shape) if 'speech_duration_exec' in output else 'n/a'} "
              f"loss_keys={sorted(losses.keys())[:8]}")
    except Exception as exc:
        errors.append(f"Model dry-run failed for split '{split}': {exc}")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Preflight check for Rhythm V2 staged training.")
    parser.add_argument("--config", required=True, help="YAML config to validate.")
    parser.add_argument("--exp_name", default="", help="Optional temporary exp name for hparams loading.")
    parser.add_argument("--hparams", default="", help="Extra hparam overrides.")
    parser.add_argument("--binary_data_dir", default="", help="Optional override for binary_data_dir.")
    parser.add_argument("--inspect_items", type=int, default=8, help="How many items to inspect per split.")
    parser.add_argument("--splits", nargs="*", default=["train", "valid"], help="Dataset prefixes to inspect.")
    parser.add_argument("--model_dry_run", action="store_true", help="Also build ConanDataset/ConanTask and run one no-grad batch.")
    args = parser.parse_args()

    hparams_str = args.hparams
    if args.binary_data_dir:
        override = f"binary_data_dir='{args.binary_data_dir}'"
        hparams_str = override if not hparams_str else f"{hparams_str},{override}"
    hp = set_hparams(config=args.config, exp_name=args.exp_name, hparams_str=hparams_str, global_hparams=True, print_hparams=False)

    required_groups, errors, warnings = _expected_fields(hp)
    stage, stage_errors, stage_warnings = _validate_stage_contract(hp, config_path=args.config)
    errors.extend(stage_errors)
    warnings.extend(stage_warnings)
    binary_dir = hp.get("binary_data_dir", "")
    if not binary_dir:
        errors.append("binary_data_dir is empty.")
    elif not os.path.isdir(binary_dir):
        errors.append(f"binary_data_dir does not exist: {binary_dir}")

    print(f"[preflight] config={args.config}")
    print(f"[preflight] stage={stage}")
    print(f"[preflight] binary_data_dir={binary_dir}")
    printable_groups = [" | ".join(group) for group in required_groups]
    print(f"[preflight] required_field_group_count={len(required_groups)}")
    print(f"[preflight] required_field_groups={printable_groups}")

    for split in args.splits:
        split_path = os.path.join(binary_dir, split)
        ds = _open_dataset(split_path)
        if ds is None:
            errors.append(f"Missing indexed dataset for split '{split}' at {split_path}.data/.idx")
            continue
        if len(ds) <= 0:
            errors.append(f"Indexed dataset for split '{split}' is empty at {split_path}.")
            continue
        items, counts, mismatches = _collect_presence(ds, args.inspect_items)
        inspected = min(len(ds), args.inspect_items)
        print(f"[preflight] split={split} items={len(ds)} inspected={inspected}")
        for group in required_groups:
            have = 0
            for item in items:
                if any(key in item for key in group):
                    have += 1
            label = " | ".join(group)
            print(f"  - {label}: {have}/{inspected}")
            if have < inspected:
                errors.append(f"Split '{split}' is missing required field group '{label}' in {inspected - have} inspected items.")
        expected_version = int(hp.get("rhythm_cache_version", -1))
        seen_versions = [name for name in counts if name.startswith("__cache_version__")]
        if seen_versions:
            print(f"  - cache_versions_seen={[name.replace('__cache_version__', '') for name in seen_versions]}")
            if counts.get(f"__cache_version__{expected_version}", 0) < min(len(ds), args.inspect_items):
                errors.append(
                    f"Split '{split}' has cache version mismatch against expected rhythm_cache_version={expected_version}."
                )
        errors.extend(_validate_inspected_items(items, hp, split=split))
        for mismatch in mismatches:
            errors.append(mismatch)
        errors.extend(_run_dataset_and_model_dry_run(split, run_model=args.model_dry_run))

    if warnings:
        print("[preflight] warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print("[preflight] FAILED")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    print("[preflight] OK")


if __name__ == "__main__":
    main()

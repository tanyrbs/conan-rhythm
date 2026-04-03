from __future__ import annotations

"""Shared preflight implementation extracted from scripts/preflight_rhythm_v2.py."""

import argparse
import os
import sys
from collections import Counter

from modules.Conan.rhythm.policy import (
    resolve_runtime_offline_teacher_enable as resolve_policy_runtime_offline_teacher_enable,
)
from modules.Conan.rhythm.stages import (
    resolve_runtime_dual_mode_teacher_enable,
    resolve_teacher_as_main,
)
from modules.Conan.rhythm.surface_metadata import (
    compatible_rhythm_cache_versions,
    materialize_rhythm_cache_compat_fields,
)
from tasks.Conan.rhythm.config_contract import (
    collect_config_contract_evaluation,
    collect_cache_contract_report,
)
from utils.commons.hparams import set_hparams
from utils.commons.indexed_datasets import IndexedDataset


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
        item = materialize_rhythm_cache_compat_fields(ds[idx])
        items.append(item)
        for key in item.keys():
            counts[key] += 1
        if "rhythm_cache_version" in item:
            try:
                raw_value = item["rhythm_cache_version"]
                version = int(raw_value[0]) if hasattr(raw_value, "__len__") else int(raw_value)
                counts[f"__cache_version__{version}"] += 1
            except Exception:
                mismatches.append(f"item[{idx}] has unreadable rhythm_cache_version")
    return items, counts, mismatches


def _run_dataset_and_model_dry_run(split: str, *, context, run_model: bool) -> list[str]:
    errors: list[str] = []
    try:
        from tasks.Conan.dataset import ConanDataset
    except Exception as exc:
        return [f"Failed to import ConanDataset for split '{split}': {exc}"]
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
        import torch
        from tasks.Conan.Conan import ConanTask

        task = ConanTask()
        task.build_tts_model()
        task.global_step = 0
        with torch.no_grad():
            losses, output = task.run_model(batch, infer=False)
        if "mel_out" not in output:
            errors.append(f"Model dry-run for split '{split}' did not produce mel_out.")
        if "rhythm_execution" not in output:
            errors.append(f"Model dry-run for split '{split}' did not produce rhythm_execution.")

        stage = context.stage
        hp = context.hparams
        runtime_teacher_enabled = resolve_policy_runtime_offline_teacher_enable(hp, stage=stage)
        dual_mode_enabled = resolve_runtime_dual_mode_teacher_enable(hp, stage=stage, infer=False)
        teacher_as_main = resolve_teacher_as_main(hp, stage=stage, infer=False)
        reported_stage = str(output.get("rhythm_stage", "") or "").strip().lower()
        if reported_stage and reported_stage != stage:
            errors.append(
                f"Model dry-run for split '{split}' reported rhythm_stage={reported_stage}, expected {stage}."
            )
        offline_execution = output.get("rhythm_offline_execution")
        teacher_as_main_runtime = bool(output.get("rhythm_teacher_as_main", 0.0))
        if teacher_as_main and not teacher_as_main_runtime:
            errors.append(
                f"Model dry-run for split '{split}' expected rhythm_teacher_as_main=true in stage {stage}."
            )
        if not teacher_as_main and teacher_as_main_runtime:
            errors.append(
                f"Model dry-run for split '{split}' unexpectedly reported rhythm_teacher_as_main in stage {stage}."
            )
        if stage == "teacher_offline" and offline_execution is not None:
            errors.append(
                f"Model dry-run for split '{split}' should not emit rhythm_offline_execution in teacher_offline stage."
            )
        if not dual_mode_enabled and offline_execution is not None:
            errors.append(
                f"Model dry-run for split '{split}' unexpectedly produced rhythm_offline_execution while "
                "runtime dual-mode teacher resolves disabled."
            )
        if dual_mode_enabled and offline_execution is None:
            errors.append(
                f"Model dry-run for split '{split}' expected rhythm_offline_execution in dual-mode stage but found none."
            )
        if not runtime_teacher_enabled and offline_execution is not None:
            errors.append(
                f"Model dry-run for split '{split}' unexpectedly produced rhythm_offline_execution while "
                "runtime offline teacher branch resolves disabled."
            )
        if stage == "legacy_schedule_only" and offline_execution is not None:
            errors.append(
                f"Model dry-run for split '{split}' unexpectedly produced rhythm_offline_execution in schedule-only stage."
            )
        apply_split = bool(hp.get("rhythm_apply_train_override", False)) if split == "train" else bool(
            hp.get("rhythm_apply_valid_override", False)
        )
        retimed_start = int(hp.get("rhythm_retimed_target_start_steps", 0) or 0)
        if apply_split and retimed_start <= 0:
            if not bool(output.get("acoustic_target_is_retimed", False)):
                errors.append(
                    f"Model dry-run for split '{split}' expected retimed acoustic target but got source-aligned target."
                )
            acoustic_target_source = str(output.get("acoustic_target_source", "") or "").strip().lower()
            if not acoustic_target_source:
                errors.append(
                    f"Model dry-run for split '{split}' expected acoustic_target_source when retimed stage is active."
                )
            elif acoustic_target_source in {"source", "source_aligned"}:
                errors.append(
                    f"Model dry-run for split '{split}' got acoustic_target_source={acoustic_target_source}, "
                    "but retimed stage expects cached/online/hybrid retimed target routing."
                )
        print(
            f"[preflight] model_dry_run split={split} mel_out={tuple(output['mel_out'].shape)} "
            f"units={tuple(output['speech_duration_exec'].shape) if 'speech_duration_exec' in output else 'n/a'} "
            f"loss_keys={sorted(losses.keys())[:8]}"
        )
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
    hp = set_hparams(
        config=args.config,
        exp_name=args.exp_name,
        hparams_str=hparams_str,
        global_hparams=True,
        print_hparams=False,
    )

    evaluation = collect_config_contract_evaluation(
        hp,
        config_path=args.config,
        model_dry_run=args.model_dry_run,
    )
    context = evaluation.context
    contract_report = evaluation.report
    required_groups = contract_report.required_field_groups
    errors = list(contract_report.errors)
    warnings = list(contract_report.warnings)

    binary_dir = hp.get("binary_data_dir", "")
    if not binary_dir:
        errors.append("binary_data_dir is empty.")
    elif not os.path.isdir(binary_dir):
        errors.append(f"binary_data_dir does not exist: {binary_dir}")
    processed_dir = hp.get("processed_data_dir", "")
    if not processed_dir:
        errors.append("processed_data_dir is empty.")
    elif not os.path.isdir(processed_dir):
        errors.append(f"processed_data_dir does not exist: {processed_dir}")

    print(f"[preflight] config={args.config}")
    print(f"[preflight] profile={context.profile}")
    print(f"[preflight] stage={context.stage}")
    print(f"[preflight] binary_data_dir={binary_dir}")
    print(f"[preflight] processed_data_dir={processed_dir}")
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
            have = sum(1 for item in items if any(key in item for key in group))
            print(f"  - {' | '.join(group)}: {have}/{inspected}")

        expected_version = int(hp.get("rhythm_cache_version", -1))
        seen_versions = [name for name in counts if name.startswith("__cache_version__")]
        if seen_versions:
            print(f"  - cache_versions_seen={[name.replace('__cache_version__', '') for name in seen_versions]}")
            compatible_versions = compatible_rhythm_cache_versions(expected_version)
            compatible_seen = sum(
                counts.get(f"__cache_version__{version}", 0) for version in compatible_versions
            )
            if compatible_seen < inspected:
                errors.append(
                    f"Split '{split}' has cache version mismatch against expected rhythm_cache_version={expected_version} "
                    f"(accepted compatible versions: {compatible_versions})."
                )

        split_contract = collect_cache_contract_report(context, items, split=split)
        errors.extend(split_contract.errors)
        warnings.extend(split_contract.warnings)
        errors.extend(mismatches)

        if args.model_dry_run:
            errors.extend(_run_dataset_and_model_dry_run(split, context=context, run_model=True))

    dedup_warnings = list(dict.fromkeys(warnings))
    dedup_errors = list(dict.fromkeys(errors))
    if dedup_warnings:
        print("[preflight] warnings:")
        for warning in dedup_warnings:
            print(f"  - {warning}")

    if dedup_errors:
        print("[preflight] FAILED")
        for error in dedup_errors:
            print(f"  - {error}")
        sys.exit(1)
    print("[preflight] OK")


if __name__ == "__main__":
    main()

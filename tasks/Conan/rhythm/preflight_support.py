from __future__ import annotations

"""Shared preflight implementation extracted from scripts/preflight_rhythm_v2.py."""

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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


@dataclass
class SplitInspectionReport:
    split: str
    split_path: str
    dataset_len: int = 0
    inspected_items: int = 0
    dataset_ready: bool = False
    field_group_hits: list[tuple[tuple[str, ...], int]] = field(default_factory=list)
    cache_versions_seen: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreflightRunContext:
    args: argparse.Namespace
    hparams: dict[str, Any]
    context: Any
    required_groups: list[tuple[str, ...]]
    binary_dir: str
    processed_dir: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    split_reports: list[SplitInspectionReport] = field(default_factory=list)


def _open_dataset(path_prefix: str):
    idx_path = f"{path_prefix}.idx"
    data_path = f"{path_prefix}.data"
    if not os.path.exists(idx_path) or not os.path.exists(data_path):
        return None
    return IndexedDataset(path_prefix)


def _inspect_indexed_split_files(path_prefix: str) -> list[str]:
    issues: list[str] = []
    idx_path = f"{path_prefix}.idx"
    data_path = f"{path_prefix}.data"
    lengths_path = f"{path_prefix}_lengths.npy"
    if not os.path.exists(idx_path) or not os.path.exists(data_path):
        return issues
    if os.path.getsize(data_path) <= 0:
        issues.append(f"Indexed dataset data file is empty at {data_path}.")
    if os.path.getsize(idx_path) <= 0:
        issues.append(f"Indexed dataset index file is empty at {idx_path}.")
    if not os.path.exists(lengths_path):
        issues.append(f"Missing lengths file for split at {lengths_path}.")
    return issues


def _inspect_indexed_split_arrays(path_prefix: str, *, dataset_len: int) -> list[str]:
    issues: list[str] = []
    lengths_path = f"{path_prefix}_lengths.npy"
    if os.path.exists(lengths_path):
        try:
            lengths = np.load(lengths_path, allow_pickle=False)
            found = int(np.asarray(lengths).reshape(-1).shape[0])
            if found != int(dataset_len):
                issues.append(
                    f"Indexed dataset lengths mismatch at {lengths_path}: found={found}, expected={int(dataset_len)}."
                )
        except Exception as exc:
            issues.append(f"Failed to read lengths file at {lengths_path}: {exc}")
    spk_ids_path = f"{path_prefix}_spk_ids.npy"
    if os.path.exists(spk_ids_path):
        try:
            spk_ids = np.load(spk_ids_path, allow_pickle=False)
            found = int(np.asarray(spk_ids).reshape(-1).shape[0])
            if found != int(dataset_len):
                issues.append(
                    f"Indexed dataset speaker-id mismatch at {spk_ids_path}: found={found}, expected={int(dataset_len)}."
                )
        except Exception as exc:
            issues.append(f"Failed to read speaker-id file at {spk_ids_path}: {exc}")
    return issues


def _inspect_pitch_feature_readiness(
    items: list[dict],
    *,
    split: str,
    use_pitch_embed: bool,
) -> list[str]:
    if not use_pitch_embed or not items:
        return []
    issues: list[str] = []
    for idx, item in enumerate(items):
        item_name = str(item.get("item_name", f"{split}[{idx}]"))
        f0 = item.get("f0")
        if f0 is None:
            issues.append(
                f"Split '{split}' item '{item_name}' is missing non-empty f0 while use_pitch_embed=true."
            )
            continue
        try:
            f0_arr = np.asarray(f0)
        except Exception as exc:
            issues.append(
                f"Split '{split}' item '{item_name}' has unreadable f0 while use_pitch_embed=true: {exc}"
            )
            continue
        if int(f0_arr.size) <= 0:
            issues.append(
                f"Split '{split}' item '{item_name}' is missing non-empty f0 while use_pitch_embed=true."
            )
    return issues


def _inspect_processed_data_dir(processed_dir: str) -> list[str]:
    issues: list[str] = []
    if not processed_dir:
        issues.append(
            "processed_data_dir is empty. cached-only preflight mainly validates binary cache readiness; "
            "processed_data_dir is only required for metadata/export/integration workflows."
        )
    elif not os.path.isdir(processed_dir):
        issues.append(
            f"processed_data_dir does not exist: {processed_dir}. cached-only preflight mainly validates binary cache "
            "readiness; processed_data_dir is only required for metadata/export/integration workflows."
        )
    return issues


def _build_cache_shape_contract(hparams):
    from tasks.Conan.rhythm.dataset_contracts import RhythmDatasetCacheContract
    from tasks.Conan.rhythm.dataset_mixin import RhythmConanDatasetMixin

    owner = type(
        "_PreflightRhythmCacheOwner",
        (),
        {
            "hparams": hparams,
            "_RHYTHM_SOURCE_CACHE_KEYS": RhythmConanDatasetMixin._RHYTHM_SOURCE_CACHE_KEYS,
            "_RHYTHM_REF_DEBUG_CACHE_KEYS": RhythmConanDatasetMixin._RHYTHM_REF_DEBUG_CACHE_KEYS,
            "_RHYTHM_REF_PLANNER_DEBUG_CACHE_KEYS": RhythmConanDatasetMixin._RHYTHM_REF_PLANNER_DEBUG_CACHE_KEYS,
        },
    )()
    return RhythmDatasetCacheContract(owner)


def _validate_item_shape_contracts(items: list[dict], *, context, split: str) -> list[str]:
    errors: list[str] = []
    contract = _build_cache_shape_contract(context.hparams)
    source_keys = tuple(contract.owner._RHYTHM_SOURCE_CACHE_KEYS)
    for idx, item in enumerate(items):
        item_name = str(item.get("item_name", f"{split}[{idx}]"))
        try:
            if all(key in item for key in source_keys):
                contract.validate_source_cache_shapes(item, item_name=item_name)
            if "ref_rhythm_stats" in item and "ref_rhythm_trace" in item:
                contract.validate_reference_conditioning_shapes(item, item_name=item_name)
            if "dur_anchor_src" in item:
                expected_units = int(np.asarray(item["dur_anchor_src"]).reshape(-1).shape[0])
                contract.validate_target_shapes(item, item_name=item_name, expected_units=expected_units)
        except Exception as exc:
            errors.append(str(exc))
    return errors


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
        batch = ds.collater([ds[idx] for idx in range(min(len(ds), 4))])
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


def _parse_preflight_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight check for Rhythm V2 staged training.")
    parser.add_argument("--config", required=True, help="YAML config to validate.")
    parser.add_argument("--exp_name", default="", help="Optional temporary exp name for hparams loading.")
    parser.add_argument("--hparams", default="", help="Extra hparam overrides.")
    parser.add_argument("--binary_data_dir", default="", help="Optional override for binary_data_dir.")
    parser.add_argument("--processed_data_dir", default="", help="Optional override for processed_data_dir.")
    parser.add_argument("--inspect_items", type=int, default=8, help="How many items to inspect per split.")
    parser.add_argument("--splits", nargs="*", default=["train", "valid"], help="Dataset prefixes to inspect.")
    parser.add_argument(
        "--model_dry_run",
        action="store_true",
        help="Also build ConanDataset/ConanTask and run one no-grad batch.",
    )
    return parser.parse_args(argv)


def _compose_hparams_override(args: argparse.Namespace) -> str:
    overrides: list[str] = []
    if args.binary_data_dir:
        overrides.append(f"binary_data_dir='{args.binary_data_dir}'")
    if args.processed_data_dir:
        overrides.append(f"processed_data_dir='{args.processed_data_dir}'")
    if not args.hparams:
        return ",".join(overrides)
    if not overrides:
        return args.hparams
    return f"{args.hparams},{','.join(overrides)}"


def _prepare_preflight_runtime(args: argparse.Namespace) -> PreflightRunContext:
    hp = set_hparams(
        config=args.config,
        exp_name=args.exp_name,
        hparams_str=_compose_hparams_override(args),
        global_hparams=True,
        print_hparams=False,
        reset=True,
    )
    evaluation = collect_config_contract_evaluation(
        hp,
        config_path=args.config,
        model_dry_run=args.model_dry_run,
    )
    contract_report = evaluation.report
    return PreflightRunContext(
        args=args,
        hparams=hp,
        context=evaluation.context,
        required_groups=[tuple(group) for group in contract_report.required_field_groups],
        binary_dir=hp.get("binary_data_dir", ""),
        processed_dir=hp.get("processed_data_dir", ""),
        errors=list(contract_report.errors),
        warnings=list(contract_report.warnings),
    )


def _inspect_split_data_staging(run_ctx: PreflightRunContext, split: str) -> SplitInspectionReport:
    report = SplitInspectionReport(
        split=split,
        split_path=os.path.join(run_ctx.binary_dir, split),
    )
    report.errors.extend(_inspect_indexed_split_files(report.split_path))
    ds = _open_dataset(report.split_path)
    if ds is None:
        report.errors.append(
            f"Missing indexed dataset for split '{split}' at {report.split_path}.data/.idx"
        )
        return report
    report.dataset_len = len(ds)
    if report.dataset_len <= 0:
        report.errors.append(f"Indexed dataset for split '{split}' is empty at {report.split_path}.")
        return report
    report.dataset_ready = True
    report.errors.extend(_inspect_indexed_split_arrays(report.split_path, dataset_len=report.dataset_len))

    items, counts, mismatches = _collect_presence(ds, run_ctx.args.inspect_items)
    report.inspected_items = min(report.dataset_len, run_ctx.args.inspect_items)
    report.field_group_hits = [
        (group, sum(1 for item in items if any(key in item for key in group)))
        for group in run_ctx.required_groups
    ]

    expected_version = int(run_ctx.hparams.get("rhythm_cache_version", -1))
    report.cache_versions_seen = [
        name.replace("__cache_version__", "")
        for name in counts
        if name.startswith("__cache_version__")
    ]
    if report.cache_versions_seen:
        compatible_versions = compatible_rhythm_cache_versions(expected_version)
        compatible_seen = sum(
            counts.get(f"__cache_version__{version}", 0) for version in compatible_versions
        )
        if compatible_seen < report.inspected_items:
            report.errors.append(
                f"Split '{split}' has cache version mismatch against expected rhythm_cache_version={expected_version} "
                f"(accepted compatible versions: {compatible_versions})."
            )

    split_contract = collect_cache_contract_report(run_ctx.context, items, split=split)
    report.errors.extend(split_contract.errors)
    report.warnings.extend(split_contract.warnings)
    report.errors.extend(mismatches)
    report.errors.extend(_validate_item_shape_contracts(items, context=run_ctx.context, split=split))
    report.errors.extend(
        _inspect_pitch_feature_readiness(
            items,
            split=split,
            use_pitch_embed=bool(run_ctx.hparams.get("use_pitch_embed", False)),
        )
    )
    return report


def _collect_data_staging_checks(run_ctx: PreflightRunContext) -> None:
    if not run_ctx.binary_dir:
        run_ctx.errors.append("binary_data_dir is empty.")
    elif not os.path.isdir(run_ctx.binary_dir):
        run_ctx.errors.append(f"binary_data_dir does not exist: {run_ctx.binary_dir}")
    run_ctx.warnings.extend(_inspect_processed_data_dir(run_ctx.processed_dir))

    for split in run_ctx.args.splits:
        report = _inspect_split_data_staging(run_ctx, split)
        run_ctx.split_reports.append(report)
        run_ctx.errors.extend(report.errors)
        run_ctx.warnings.extend(report.warnings)


def _collect_control_preview_checks(run_ctx: PreflightRunContext) -> None:
    if not run_ctx.args.model_dry_run:
        return
    for report in run_ctx.split_reports:
        if not report.dataset_ready:
            continue
        preview_errors = _run_dataset_and_model_dry_run(
            report.split,
            context=run_ctx.context,
            run_model=True,
        )
        report.errors.extend(preview_errors)
        run_ctx.errors.extend(preview_errors)


def run_preflight(args: argparse.Namespace) -> PreflightRunContext:
    run_ctx = _prepare_preflight_runtime(args)
    _collect_data_staging_checks(run_ctx)
    _collect_control_preview_checks(run_ctx)
    return run_ctx


def _emit_preflight_header(run_ctx: PreflightRunContext) -> None:
    printable_groups = [" | ".join(group) for group in run_ctx.required_groups]
    print(f"[preflight] config={run_ctx.args.config}")
    print(f"[preflight] profile={run_ctx.context.profile}")
    print(f"[preflight] stage={run_ctx.context.stage}")
    print(f"[preflight] binary_data_dir={run_ctx.binary_dir}")
    print(f"[preflight] processed_data_dir={run_ctx.processed_dir}")
    print(f"[preflight] required_field_group_count={len(run_ctx.required_groups)}")
    print(f"[preflight] required_field_groups={printable_groups}")


def _emit_split_report(report: SplitInspectionReport) -> None:
    if report.dataset_len <= 0:
        return
    print(f"[preflight] split={report.split} items={report.dataset_len} inspected={report.inspected_items}")
    for group, have in report.field_group_hits:
        print(f"  - {' | '.join(group)}: {have}/{report.inspected_items}")
    if report.cache_versions_seen:
        print(f"  - cache_versions_seen={report.cache_versions_seen}")


def _dedupe_findings(findings: list[str]) -> list[str]:
    return list(dict.fromkeys(findings))


def _emit_preflight_summary(run_ctx: PreflightRunContext) -> None:
    dedup_warnings = _dedupe_findings(run_ctx.warnings)
    dedup_errors = _dedupe_findings(run_ctx.errors)
    if dedup_warnings:
        print("[preflight] warnings:")
        for warning in dedup_warnings:
            print(f"  - {warning}")
    if dedup_errors:
        print("[preflight] FAILED")
        for error in dedup_errors:
            print(f"  - {error}")
        return
    print("[preflight] OK")


def main(argv: list[str] | None = None):
    run_ctx = _prepare_preflight_runtime(_parse_preflight_args(argv))
    _emit_preflight_header(run_ctx)
    _collect_data_staging_checks(run_ctx)
    for report in run_ctx.split_reports:
        _emit_split_report(report)
    _collect_control_preview_checks(run_ctx)
    _emit_preflight_summary(run_ctx)
    if _dedupe_findings(run_ctx.errors):
        sys.exit(1)


if __name__ == "__main__":
    main()

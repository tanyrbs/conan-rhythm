from __future__ import annotations

from copy import deepcopy
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from modules.Conan.rhythm_v3.contracts import (
    DurationExecution,
    DurationRuntimeState,
    PromptConditioningEvidence,
    ReferenceDurationMemory,
    SourceUnitBatch,
    StructuredDurationOperatorMemory,
)
from utils.plot.rhythm_v3_viz import (
    RhythmV3DebugRecord,
    attach_projection_debug,
    build_debug_records_from_batch,
    build_monotonicity_table,
    build_prefix_silence_review_table,
    build_projection_debug_payload,
    build_ref_crop_table,
    record_summary,
    save_debug_records,
    summarize_falsification_ladder,
)
from scripts.rhythm_v3_debug_records import (
    _warn_sparse_review_metadata,
    build_gate_status,
    collect_gate_issues,
    main as debug_records_main,
)
from utils.plot.rhythm_v3_viz.review import compute_g, compute_source_global_rate_for_analysis


def test_projection_debug_payload_exposes_alignment_trace():
    payload = build_projection_debug_payload(
        source_units=np.asarray([3, 7, 9], dtype=np.int64),
        source_durations=np.asarray([2.0, 3.0, 1.0], dtype=np.float32),
        source_silence_mask=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        target_units=np.asarray([3, 7, 9], dtype=np.int64),
        target_durations=np.asarray([2.0, 4.0, 1.0], dtype=np.float32),
        target_valid_mask=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        target_speech_mask=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
    )
    assert payload["unit_duration_tgt"].shape == (3,)
    assert payload["unit_duration_proj_raw_tgt"].shape == (3,)
    assert payload["unit_alignment_mode_id_tgt"].shape == (1,)
    assert payload["unit_alignment_assigned_source_debug"].shape == (3,)
    assert payload["unit_alignment_source_valid_run_index_debug"].shape == (3,)
    assert payload["unit_alignment_unmatched_speech_ratio_tgt"].shape == (1,)
    assert payload["unit_alignment_mean_local_confidence_speech_tgt"].shape == (1,)
    assert payload["unit_alignment_mean_coarse_confidence_speech_tgt"].shape == (1,)
    assert payload["paired_target_content_units_debug"].dtype == np.int64


def test_attach_projection_debug_keeps_record_outside_dataset_mixin():
    source_record = build_debug_records_from_batch(
        sample={
            "item_name": np.asarray(["demo"], dtype=object),
            "content_units": torch.tensor([[3, 7, 9]], dtype=torch.long),
            "dur_anchor_src": torch.tensor([[2.0, 3.0, 1.0]], dtype=torch.float32),
            "source_silence_mask": torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
            "unit_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        }
    )[0]
    enriched = attach_projection_debug(
        source_record,
        target_units=np.asarray([3, 7, 9], dtype=np.int64),
        target_durations=np.asarray([2.0, 4.0, 1.0], dtype=np.float32),
        target_valid_mask=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        target_speech_mask=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
    )
    assert enriched.unit_duration_tgt is not None
    assert enriched.unit_duration_proj_raw_tgt is not None
    assert enriched.unit_alignment_mode_id_tgt is not None
    assert enriched.unit_alignment_assigned_source_debug is not None
    assert enriched.paired_target_duration_obs_debug is not None


def test_build_debug_records_from_mode_id_only_uses_generic_continuous_alignment_kind():
    records = build_debug_records_from_batch(
        sample={
            "item_name": np.asarray(["demo_mode_only"], dtype=object),
            "content_units": torch.tensor([[3, 7]], dtype=torch.long),
            "dur_anchor_src": torch.tensor([[2.0, 3.0]], dtype=torch.float32),
            "source_silence_mask": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            "unit_mask": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            "unit_duration_tgt": torch.tensor([[2.0, 3.0]], dtype=torch.float32),
            "unit_duration_proj_raw_tgt": torch.tensor([[2.0, 3.0]], dtype=torch.float32),
            "unit_alignment_mode_id_tgt": torch.tensor([[1]], dtype=torch.long),
        }
    )
    assert len(records) == 1
    assert records[0].metadata["alignment_kind"] == "continuous"
    assert int(np.asarray(records[0].unit_alignment_mode_id_tgt).reshape(-1)[0]) == 1


def test_build_debug_records_and_summary_from_runtime_objects():
    source_batch = SourceUnitBatch(
        content_units=torch.tensor([[5, 6, 8]], dtype=torch.long),
        source_duration_obs=torch.tensor([[2.0, 4.0, 1.0]], dtype=torch.float32),
        unit_anchor_base=torch.tensor([[2.0, 4.0, 1.0]], dtype=torch.float32),
        unit_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        sealed_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        sep_mask=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
        source_silence_mask=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        source_boundary_cue=torch.tensor([[0.1, 0.8, 0.2]], dtype=torch.float32),
        source_run_stability=torch.tensor([[1.0, 0.9, 0.7]], dtype=torch.float32),
    )
    ref_memory = ReferenceDurationMemory(
        global_rate=torch.tensor([[0.2]], dtype=torch.float32),
        operator=StructuredDurationOperatorMemory(operator_coeff=torch.zeros((1, 8), dtype=torch.float32)),
        prompt=PromptConditioningEvidence(
            prompt_log_duration=torch.log(torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32)),
            prompt_log_residual=torch.tensor([[0.1, -0.1, 0.0]], dtype=torch.float32),
        ),
        prompt_valid_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        prompt_speech_mask=torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
    )
    execution = DurationExecution(
        unit_logstretch=torch.tensor([[0.1, -0.05, 0.0]], dtype=torch.float32),
        unit_duration_exec=torch.tensor([[2.2, 3.8, 1.0]], dtype=torch.float32),
        basis_activation=torch.zeros((1, 3, 1), dtype=torch.float32),
        commit_mask=torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
        next_state=DurationRuntimeState(
            committed_units=torch.tensor([2], dtype=torch.long),
            rounding_residual=torch.tensor([0.0], dtype=torch.float32),
            prefix_unit_offset=torch.tensor([[0.0, 0.1, 0.1]], dtype=torch.float32),
        ),
        global_bias_scalar=torch.tensor([[0.05]], dtype=torch.float32),
        global_shift_analytic=torch.tensor([[0.12, 0.02, 0.0]], dtype=torch.float32),
        coarse_logstretch=torch.tensor([[0.17, 0.07, 0.0]], dtype=torch.float32),
        coarse_correction=torch.tensor([[0.05, 0.05, 0.05]], dtype=torch.float32),
        local_residual=torch.tensor([[-0.07, -0.12, 0.0]], dtype=torch.float32),
        source_rate_seq=torch.tensor([[0.08, 0.18, 0.18]], dtype=torch.float32),
        prefix_unit_offset=torch.tensor([[0.0, 0.1, 0.1]], dtype=torch.float32),
        projector_rounding_residual=torch.tensor([[0.03]], dtype=torch.float32),
    )
    sample = {
        "item_name": np.asarray(["demo_case"], dtype=object),
        "content_units": source_batch.content_units,
        "dur_anchor_src": source_batch.source_duration_obs,
        "source_silence_mask": source_batch.source_silence_mask,
        "unit_mask": source_batch.unit_mask,
        "prompt_content_units": torch.tensor([[9, 9, 8]], dtype=torch.long),
        "prompt_duration_obs": torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32),
        "prompt_valid_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        "prompt_speech_mask": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
        "unit_duration_tgt": torch.tensor([[2.4, 3.5, 1.0]], dtype=torch.float32),
        "unit_confidence_local_tgt": torch.tensor([[1.0, 0.8, 0.0]], dtype=torch.float32),
        "unit_confidence_coarse_tgt": torch.tensor([[1.0, 0.7, 0.2]], dtype=torch.float32),
        "unit_confidence_tgt": torch.tensor([[1.0, 0.7, 0.2]], dtype=torch.float32),
    }
    records = build_debug_records_from_batch(
        sample=sample,
        model_output={
            "rhythm_unit_batch": source_batch,
            "rhythm_ref_conditioning": ref_memory,
            "rhythm_execution": execution,
            "rhythm_v3_g_variant": "raw_median",
            "rhythm_debug_g_support_count": torch.tensor([[2.0]], dtype=torch.float32),
            "rhythm_debug_g_speech_count": torch.tensor([[2.0]], dtype=torch.float32),
            "rhythm_debug_g_valid_count": torch.tensor([[3.0]], dtype=torch.float32),
            "rhythm_debug_g_support_ratio_vs_speech": torch.tensor([[1.0]], dtype=torch.float32),
            "rhythm_debug_g_support_ratio_vs_valid": torch.tensor([[2.0 / 3.0]], dtype=torch.float32),
            "rhythm_debug_g_valid": torch.tensor([[1.0]], dtype=torch.float32),
            "rhythm_debug_g_drop_edge_runs": torch.tensor([[1.0]], dtype=torch.float32),
            "rhythm_debug_g_strict_speech_only": torch.tensor([[1.0]], dtype=torch.float32),
        },
        metadata={"phase": "unit-test"},
    )
    assert len(records) == 1
    assert records[0].projector_rounding_residual is not None
    assert records[0].metadata["g_variant"] == "raw_median"
    assert records[0].metadata["g_support_count"] == 2.0
    summary = record_summary(records[0])
    assert summary["item_name"] == "demo_case"
    assert summary["reference_seconds"] > 0.0
    assert np.isfinite(summary["oracle_bias"])
    assert np.isfinite(summary["predicted_bias"])
    assert summary["src_spk"] == "demo"
    assert summary["g_variant"] == "raw_median"
    assert np.isfinite(summary["g_ref"])
    assert np.isfinite(summary["g_src_utt"])
    assert np.isfinite(summary["delta_g"])
    assert summary["g_compute_status"] == "ok"
    assert summary["g_src_compute_status"] == "ok"
    assert summary["gate0_row_dropped"] == 0.0
    assert summary["gate0_drop_reason"] == "ok"
    assert summary["g_support_count"] == 2.0
    assert np.isfinite(summary["g_support_ratio_vs_valid"])
    assert np.isfinite(summary["c_star"])
    assert np.isfinite(summary["tempo_src"])
    assert np.isfinite(summary["tempo_out"])
    assert np.isfinite(summary["tempo_delta"])
    assert np.isfinite(summary["analytic_gap_abs_mean"])
    assert np.isfinite(summary["coarse_bias_abs_mean"])
    assert np.isfinite(summary["local_residual_abs_mean"])
    assert np.isfinite(summary["cumulative_drift"])


def test_review_tables_and_gate_ladder_use_unified_falsification_fields():
    base = RhythmV3DebugRecord.from_mapping(
        {
            "item_name": "srcA_0001",
            "metadata": {
                "src_id": "srcA",
                "pair_id": "pairA",
                "sample_id": "streamA",
                "eval_mode": "analytic",
                "same_text_reference": 0.0,
                "same_text_target": 1.0,
                "ref_prompt_id": "ref_0002",
                "src_prompt_id": "srcA_0001",
                "ref_spk": "srcA",
                "src_spk": "srcA",
            },
            "source_content_units": np.asarray([5, 6, 8], dtype=np.int64),
            "source_duration_obs": np.asarray([2.0, 4.0, 1.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            "unit_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "sealed_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "sep_mask": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            "prompt_content_units": np.asarray([9, 9, 8], dtype=np.int64),
            "prompt_duration_obs": np.asarray([3.0, 2.0, 1.0], dtype=np.float32),
            "prompt_valid_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "prompt_speech_mask": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
            "unit_duration_tgt": np.asarray([2.4, 3.5, 1.0], dtype=np.float32),
            "unit_confidence_tgt": np.asarray([1.0, 0.7, 0.2], dtype=np.float32),
            "unit_confidence_coarse_tgt": np.asarray([1.0, 0.7, 0.2], dtype=np.float32),
            "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.0], dtype=np.float32),
            "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.85], dtype=np.float32),
            "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.85], dtype=np.float32),
            "global_rate": 0.25,
            "source_rate_seq": np.asarray([0.08, 0.18, 0.18], dtype=np.float32),
            "global_shift_analytic": np.asarray([0.17, 0.07, 0.07], dtype=np.float32),
            "global_bias_scalar": 0.05,
            "coarse_correction": np.asarray([0.05, 0.05, 0.05], dtype=np.float32),
            "local_residual": np.asarray([-0.02, -0.04, 0.0], dtype=np.float32),
            "unit_logstretch": np.asarray([0.15, 0.01, 0.0], dtype=np.float32),
            "unit_duration_exec": np.asarray([2.8, 4.6, 1.0], dtype=np.float32),
            "unit_duration_raw": np.asarray([2.7, 4.5, 1.0], dtype=np.float32),
            "commit_mask": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
            "prefix_unit_offset": np.asarray([0.0, 0.08, 0.08], dtype=np.float32),
            "projector_budget_hit_pos": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "projector_budget_hit_neg": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        }
    )

    def clone_record(**updates):
        payload = deepcopy(base.to_dict())
        metadata = dict(payload.get("metadata") or {})
        metadata.update(updates.pop("metadata", {}))
        payload["metadata"] = metadata
        payload.update(updates)
        return RhythmV3DebugRecord.from_mapping(payload)

    slow = clone_record(
        metadata={"ref_bin": "slow"},
        prompt_duration_obs=np.asarray([4.5, 3.0, 1.0], dtype=np.float32),
        global_rate=0.45,
        unit_duration_exec=np.asarray([3.4, 5.1, 1.0], dtype=np.float32),
        unit_duration_raw=np.asarray([3.3, 5.0, 1.0], dtype=np.float32),
    )
    mid = clone_record(metadata={"ref_bin": "mid"})
    fast = clone_record(
        metadata={"ref_bin": "fast"},
        prompt_duration_obs=np.asarray([2.0, 1.5, 1.0], dtype=np.float32),
        global_rate=0.05,
        unit_duration_exec=np.asarray([1.8, 3.1, 1.0], dtype=np.float32),
        unit_duration_raw=np.asarray([1.9, 3.0, 1.0], dtype=np.float32),
    )
    prefix_short = clone_record(
        metadata={"sample_id": "stream_eval", "prefix_ratio": 0.5},
        unit_logstretch=np.asarray([0.08, -0.01, 0.0], dtype=np.float32),
        unit_duration_exec=np.asarray([2.5, 4.2, 1.0], dtype=np.float32),
        commit_mask=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        prefix_unit_offset=np.asarray([0.0, 0.04, 0.04], dtype=np.float32),
    )
    prefix_long = clone_record(
        metadata={"sample_id": "stream_eval", "prefix_ratio": 1.0},
        unit_logstretch=np.asarray([0.15, 0.01, 0.0], dtype=np.float32),
        unit_duration_exec=np.asarray([2.8, 4.6, 1.0], dtype=np.float32),
        commit_mask=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        prefix_unit_offset=np.asarray([0.0, 0.08, 0.08], dtype=np.float32),
    )

    records = [slow, mid, fast, prefix_short, prefix_long]
    ref_crop_df = build_ref_crop_table(records, drop_edge_runs=1)
    monotonicity_df = build_monotonicity_table(records, drop_edge_runs=1)
    prefix_silence_df = build_prefix_silence_review_table(records)
    ladder_df = summarize_falsification_ladder(ref_crop_df, monotonicity_df, prefix_silence_df)

    assert "g_src_utt" in ref_crop_df.columns
    assert "g_compute_status" in ref_crop_df.columns
    assert "g_src_compute_status" in ref_crop_df.columns
    assert "gate0_row_dropped" in ref_crop_df.columns
    assert "g_src_prefix_mean" in ref_crop_df.columns
    assert "same_text_reference" in ref_crop_df.columns
    assert "same_speaker_reference" in ref_crop_df.columns
    assert np.isfinite(ref_crop_df["delta_g"]).any()
    summary = record_summary(mid)
    assert np.isfinite(summary["alignment_unmatched_speech_ratio"])
    assert np.isfinite(summary["alignment_mean_coarse_confidence_speech"])
    assert summary["gate0_drop_reason"] == "ok"
    assert set(monotonicity_df["ref_bin"].tolist()) >= {"slow", "mid", "fast"}
    assert "tempo_delta" in monotonicity_df.columns
    assert np.isfinite(monotonicity_df["mono_triplet_ok"]).any()
    assert "budget_hit_rate" in prefix_silence_df.columns
    assert "cumulative_drift" in prefix_silence_df.columns
    assert ladder_df.shape[0] == 1
    assert ladder_df.iloc[0]["eval_mode"] == "analytic"
    assert np.isfinite(ladder_df.iloc[0]["monotonicity_rate"])
    assert "tempo_transfer_slope" in ladder_df.columns
    assert "negative_control_gap" in ladder_df.columns


def test_falsification_ladder_reports_negative_control_gap():
    monotonicity_df = pd.DataFrame(
        [
            {
                "src_id": "src_a",
                "pair_id": "real_triplet",
                "triplet_id": "src_a|analytic|real|real_triplet",
                "eval_mode": "analytic",
                "ref_condition": "real",
                "ref_bin": "slow",
                "delta_g": -0.4,
                "tempo_delta": -0.3,
                "mono_triplet_ok": 1.0,
            },
            {
                "src_id": "src_a",
                "pair_id": "real_triplet",
                "triplet_id": "src_a|analytic|real|real_triplet",
                "eval_mode": "analytic",
                "ref_condition": "real",
                "ref_bin": "mid",
                "delta_g": 0.0,
                "tempo_delta": 0.0,
                "mono_triplet_ok": 1.0,
            },
            {
                "src_id": "src_a",
                "pair_id": "real_triplet",
                "triplet_id": "src_a|analytic|real|real_triplet",
                "eval_mode": "analytic",
                "ref_condition": "real",
                "ref_bin": "fast",
                "delta_g": 0.4,
                "tempo_delta": 0.35,
                "mono_triplet_ok": 1.0,
            },
            {
                "src_id": "src_a",
                "pair_id": "neg_triplet",
                "triplet_id": "src_a|analytic|random_ref|neg_triplet",
                "eval_mode": "analytic",
                "ref_condition": "random_ref",
                "ref_bin": "slow",
                "delta_g": -0.4,
                "tempo_delta": -0.05,
                "mono_triplet_ok": 0.0,
            },
            {
                "src_id": "src_a",
                "pair_id": "neg_triplet",
                "triplet_id": "src_a|analytic|random_ref|neg_triplet",
                "eval_mode": "analytic",
                "ref_condition": "random_ref",
                "ref_bin": "mid",
                "delta_g": 0.0,
                "tempo_delta": 0.0,
                "mono_triplet_ok": 0.0,
            },
            {
                "src_id": "src_a",
                "pair_id": "neg_triplet",
                "triplet_id": "src_a|analytic|random_ref|neg_triplet",
                "eval_mode": "analytic",
                "ref_condition": "random_ref",
                "ref_bin": "fast",
                "delta_g": 0.4,
                "tempo_delta": 0.05,
                "mono_triplet_ok": 0.0,
            },
        ]
    )
    ladder_df = summarize_falsification_ladder(
        pd.DataFrame(columns=["eval_mode"]),
        monotonicity_df,
        pd.DataFrame(columns=["eval_mode"]),
    )
    assert "negative_control_gap" in ladder_df.columns
    assert float(ladder_df.iloc[0]["negative_control_gap"]) > 0.0
    assert int(ladder_df.iloc[0]["n_negative_controls"]) == 3


def test_compute_g_and_source_global_rate_report_failure_status_without_silent_drop():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = compute_g(None, speech_mask=np.asarray([1.0, 1.0], dtype=np.float32))
    assert np.isnan(value)
    assert any("compute_g failed" in str(item.message) for item in caught)

    value_with_status, status = compute_g(
        None,
        speech_mask=np.asarray([1.0, 1.0], dtype=np.float32),
        return_status=True,
    )
    assert np.isnan(value_with_status)
    assert status == "missing:duration_obs"

    g_src, g_src_status = compute_source_global_rate_for_analysis(
        source_duration_obs=np.asarray([2.0, 3.0], dtype=np.float32),
        source_speech_mask=None,
        return_status=True,
    )
    assert np.isnan(g_src)
    assert g_src_status == "missing:source_speech_mask"


def test_compute_source_global_rate_for_analysis_strict_mode_requires_explicit_speech_mask():
    g_src, g_src_status = compute_source_global_rate_for_analysis(
        source_duration_obs=np.asarray([2.0, 3.0], dtype=np.float32),
        source_speech_mask=None,
        return_status=True,
        require_explicit_speech_mask=True,
    )
    assert np.isnan(g_src)
    assert g_src_status == "missing:source_speech_mask"


def test_warn_sparse_review_metadata_accepts_continuous_viterbi_alignment_kind(capsys):
    rows = []
    for eval_mode in ("analytic", "coarse_only", "learned"):
        for ref_bin in ("slow", "mid", "fast"):
            rows.append(
                {
                    "pair_id": f"{eval_mode}_{ref_bin}",
                    "same_text_reference": 0.0,
                    "same_text_target": 1.0,
                    "lexical_mismatch": 1.0,
                    "ref_len_sec": 4.0,
                    "speech_ratio": 0.8,
                    "alignment_kind": "continuous_viterbi_v1",
                    "target_duration_surface": "projection_raw",
                    "g_support_count": 4.0,
                    "g_support_ratio_vs_speech": 1.0,
                    "g_support_ratio_vs_valid": 1.0,
                    "g_valid": 1.0,
                    "g_trim_ratio": 0.2,
                    "prompt_global_weight_present": 1.0,
                    "prompt_unit_log_prior_present": 0.0,
                    "alignment_unmatched_speech_ratio": 0.0,
                    "alignment_mean_local_confidence_speech": 0.9,
                    "alignment_mean_coarse_confidence_speech": 0.9,
                    "projector_boundary_hit_rate": 0.0,
                    "projector_boundary_decay_rate": 0.0,
                    "g_compute_status": "ok",
                    "g_src_compute_status": "ok",
                    "gate0_row_dropped": 0.0,
                    "gate0_drop_reason": "ok",
                    "eval_mode": eval_mode,
                    "src_id": "src_demo",
                    "ref_bin": ref_bin,
                    "ref_condition": "real",
                }
            )
    rows.append(
        {
            **rows[0],
            "pair_id": "analytic_random_ref",
            "ref_condition": "random_ref",
            "ref_bin": "mid",
        }
    )
    frame = pd.DataFrame(rows)
    _warn_sparse_review_metadata(frame)
    captured = capsys.readouterr()
    assert "continuous_precomputed_coverage" not in captured.err
    assert "continuous_alignment_coverage" not in captured.err


def test_build_ref_crop_table_strict_gate_requires_explicit_prompt_speech_mask():
    record = RhythmV3DebugRecord.from_mapping(
        {
            "item_name": "src_strict_gate",
            "metadata": {
                "src_id": "src_strict_gate",
                "pair_id": "pair_strict_gate",
                "sample_id": "sample_strict_gate",
                "eval_mode": "analytic",
                "ref_bin": "mid",
                "ref_condition": "real",
            },
            "source_content_units": np.asarray([5, 6, 8], dtype=np.int64),
            "source_duration_obs": np.asarray([2.0, 4.0, 1.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            "unit_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "prompt_content_units": np.asarray([9, 9, 8], dtype=np.int64),
            "prompt_duration_obs": np.asarray([3.0, 2.0, 1.0], dtype=np.float32),
            "prompt_valid_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "unit_duration_tgt": np.asarray([2.4, 3.5, 1.0], dtype=np.float32),
            "unit_confidence_tgt": np.asarray([1.0, 0.7, 0.2], dtype=np.float32),
            "unit_logstretch": np.asarray([0.15, 0.01, 0.0], dtype=np.float32),
            "unit_duration_exec": np.asarray([2.8, 4.6, 1.0], dtype=np.float32),
            "source_rate_seq": np.asarray([0.08, 0.18, 0.18], dtype=np.float32),
        }
    )
    frame = build_ref_crop_table([record], require_explicit_speech_mask=True)
    assert frame.shape[0] == 1
    assert float(frame.iloc[0]["prompt_speech_mask_explicit"]) == 0.0
    assert frame.iloc[0]["gate0_drop_reason"] == "g_ref:missing:source_speech_mask"
    assert float(frame.iloc[0]["gate0_row_dropped"]) == 1.0


def test_collect_gate_issues_uses_domain_validity_not_only_finite():
    frame = pd.DataFrame(
        [
            {
                "pair_id": "pair_gate",
                "same_text_reference": 0.0,
                "same_text_target": 1.0,
                "lexical_mismatch": 1.0,
                "ref_len_sec": 4.0,
                "speech_ratio": 0.45,
                "alignment_kind": "continuous_viterbi_v1",
                "target_duration_surface": "projection_raw",
                "g_support_count": 2.0,
                "g_support_ratio_vs_speech": 1.0,
                "g_support_ratio_vs_valid": 1.0,
                "g_valid": 1.0,
                "g_domain_valid": 0.0,
                "g_trim_ratio": 0.2,
                "prompt_global_weight_present": 1.0,
                "prompt_unit_log_prior_present": 0.0,
                "alignment_unmatched_speech_ratio": 0.0,
                "alignment_mean_local_confidence_speech": 0.9,
                "alignment_mean_coarse_confidence_speech": 0.9,
                "projector_boundary_hit_rate": 0.0,
                "projector_boundary_decay_rate": 0.0,
                "g_compute_status": "ok",
                "g_src_compute_status": "ok",
                "gate0_row_dropped": 1.0,
                "gate0_drop_reason": "g_domain:invalid",
                "eval_mode": "analytic",
                "src_id": "src_demo",
                "ref_bin": "mid",
                "ref_condition": "real",
            }
        ]
    )
    issues = collect_gate_issues(frame)
    assert any("g_domain_valid_mean=0.000<0.950" == issue for issue in issues)
    status = build_gate_status(frame)
    assert status["gate0_pass"] is False
    assert "g_domain_valid_mean=0.000<0.950" in status["warnings"]


def test_debug_records_strict_gates_fail_nonzero_and_write_gate_status(tmp_path, monkeypatch):
    record = RhythmV3DebugRecord.from_mapping(
        {
            "item_name": "src_cli_gate",
            "metadata": {
                "src_id": "src_cli_gate",
                "pair_id": "pair_cli_gate",
                "sample_id": "sample_cli_gate",
                "eval_mode": "analytic",
                "ref_bin": "mid",
                "ref_condition": "real",
                "same_text_reference": 0.0,
                "same_text_target": 1.0,
            },
            "source_content_units": np.asarray([5, 6, 8], dtype=np.int64),
            "source_duration_obs": np.asarray([2.0, 4.0, 1.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            "unit_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "prompt_content_units": np.asarray([9, 9, 8], dtype=np.int64),
            "prompt_duration_obs": np.asarray([3.0, 2.0, 1.0], dtype=np.float32),
            "prompt_valid_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "unit_duration_tgt": np.asarray([2.4, 3.5, 1.0], dtype=np.float32),
            "unit_confidence_tgt": np.asarray([1.0, 0.7, 0.2], dtype=np.float32),
            "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.0], dtype=np.float32),
            "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.85], dtype=np.float32),
            "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.85], dtype=np.float32),
            "unit_logstretch": np.asarray([0.15, 0.01, 0.0], dtype=np.float32),
            "unit_duration_exec": np.asarray([2.8, 4.6, 1.0], dtype=np.float32),
            "source_rate_seq": np.asarray([0.08, 0.18, 0.18], dtype=np.float32),
            "prefix_unit_offset": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        }
    )
    bundle_path = tmp_path / "debug_bundle.pt"
    save_debug_records([record], bundle_path)
    output_path = tmp_path / "summary.csv"
    gate_status_path = tmp_path / "gate_status.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rhythm_v3_debug_records.py",
            "--input",
            str(bundle_path),
            "--output",
            str(output_path),
            "--gate-status-json",
            str(gate_status_path),
            "--strict-gates",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        debug_records_main()
    assert exc_info.value.code != 0
    assert gate_status_path.exists()
    payload = gate_status_path.read_text(encoding="utf-8")
    assert "gate0_pass" in payload


def test_debug_records_cli_strict_gates_fail_on_missing_negative_control(tmp_path, monkeypatch, capsys):
    import scripts.rhythm_v3_debug_records as cli

    row = {
        "item_name": "demo_case",
        "src_id": "src_demo",
        "sample_id": "sample_demo",
        "pair_id": "pair_demo",
        "eval_mode": "analytic",
        "ref_bin": "mid",
        "ref_condition": "real",
        "same_text_reference": 0.0,
        "same_text_target": 1.0,
        "lexical_mismatch": 1.0,
        "ref_len_sec": 4.0,
        "speech_ratio": 0.8,
        "alignment_kind": "continuous_viterbi_v1",
        "target_duration_surface": "projection_raw",
        "g_support_count": 4.0,
        "g_support_ratio_vs_speech": 1.0,
        "g_support_ratio_vs_valid": 1.0,
        "g_valid": 1.0,
        "g_trim_ratio": 0.2,
        "prompt_global_weight_present": 1.0,
        "prompt_unit_log_prior_present": 0.0,
        "alignment_unmatched_speech_ratio": 0.0,
        "alignment_mean_local_confidence_speech": 0.9,
        "alignment_mean_coarse_confidence_speech": 0.9,
        "projector_boundary_hit_rate": 0.0,
        "projector_boundary_decay_rate": 0.0,
        "gate0_row_dropped": 0.0,
        "gate0_drop_reason": "ok",
        "g_compute_status": "ok",
        "g_src_compute_status": "ok",
    }

    monkeypatch.setattr(cli, "load_debug_records", lambda raw: [object()])
    monkeypatch.setattr(cli, "record_summary", lambda record, **kwargs: dict(row))
    monkeypatch.setattr(cli, "build_prefix_silence_review_table", lambda records: pd.DataFrame())
    monkeypatch.setattr(cli, "build_monotonicity_table", lambda records, **kwargs: pd.DataFrame())

    output = tmp_path / "summary.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rhythm_v3_debug_records.py",
            "--input",
            "dummy_bundle.npz",
            "--output",
            str(output),
            "--strict-gates",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    captured = capsys.readouterr()
    assert excinfo.value.code != 0
    assert "unrecognized arguments" not in captured.err
    assert (
        "strict gate failure" in captured.err.lower()
        or "negative_control_reference" in captured.err
        or "missing_controls" in captured.err
    )

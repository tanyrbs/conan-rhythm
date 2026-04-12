from __future__ import annotations

from copy import deepcopy

import numpy as np
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
    summarize_falsification_ladder,
)


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
    assert payload["unit_alignment_assigned_source_debug"].shape == (3,)
    assert payload["unit_alignment_source_valid_run_index_debug"].shape == (3,)
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
    assert enriched.unit_alignment_assigned_source_debug is not None
    assert enriched.paired_target_duration_obs_debug is not None


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
        },
        metadata={"phase": "unit-test"},
    )
    assert len(records) == 1
    assert records[0].projector_rounding_residual is not None
    assert records[0].metadata["g_variant"] == "raw_median"
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
    assert np.isfinite(summary["c_star"])
    assert np.isfinite(summary["tempo_src"])
    assert np.isfinite(summary["tempo_out"])
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
    assert "g_src_prefix_mean" in ref_crop_df.columns
    assert "same_text_reference" in ref_crop_df.columns
    assert np.isfinite(ref_crop_df["delta_g"]).any()
    assert set(monotonicity_df["ref_bin"].tolist()) >= {"slow", "mid", "fast"}
    assert np.isfinite(monotonicity_df["mono_triplet_ok"]).any()
    assert "budget_hit_rate" in prefix_silence_df.columns
    assert "cumulative_drift" in prefix_silence_df.columns
    assert ladder_df.shape[0] == 1
    assert ladder_df.iloc[0]["eval_mode"] == "analytic"
    assert np.isfinite(ladder_df.iloc[0]["monotonicity_rate"])

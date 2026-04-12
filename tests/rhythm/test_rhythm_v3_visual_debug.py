from __future__ import annotations

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
    attach_projection_debug,
    build_debug_records_from_batch,
    build_projection_debug_payload,
    record_summary,
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
        },
        metadata={"phase": "unit-test"},
    )
    assert len(records) == 1
    summary = record_summary(records[0])
    assert summary["item_name"] == "demo_case"
    assert summary["reference_seconds"] > 0.0
    assert np.isfinite(summary["oracle_bias"])
    assert np.isfinite(summary["predicted_bias"])

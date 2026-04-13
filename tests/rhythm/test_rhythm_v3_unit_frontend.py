from __future__ import annotations

import torch

from modules.Conan.rhythm.supervision import build_source_rhythm_cache
from modules.Conan.rhythm_v3.unit_frontend import DurationUnitFrontend, ProtocolDurationBaseline
from modules.Conan.rhythm_v3.unitizer import StreamingRunLengthUnitizer


def test_protocol_duration_baseline_table_prior_file_offsets_nominal_anchor(tmp_path):
    baseline = ProtocolDurationBaseline(
        vocab_size=4,
        hidden_size=8,
        min_frames=1.0,
        max_frames=4.0,
    )
    for param in baseline.local_trunk.parameters():
        param.data.zero_()
    units = torch.tensor([[1, 2]], dtype=torch.long)
    mask = torch.ones((1, 2), dtype=torch.float32)
    before = baseline(units, mask)
    prior_path = tmp_path / "baseline_prior.pt"
    torch.save({"anchor_prior_frames": torch.tensor([2.0, 4.0, 1.0, 2.0])}, prior_path)
    baseline.load_table_prior_file(prior_path)
    after = baseline(units, mask)
    assert torch.all(after[:, 0] > before[:, 0])
    assert torch.all(after[:, 1] < before[:, 1])


def test_duration_unit_frontend_baseline_freeze_and_unfreeze_cover_table_plus_local_protocol():
    frontend = DurationUnitFrontend(vocab_size=8)
    frontend.unfreeze_baseline()
    assert all(param.requires_grad for param in frontend.baseline.local_trunk.parameters())
    assert hasattr(frontend.baseline, "table_prior")
    assert not hasattr(frontend.baseline, "struct_prior")
    assert not hasattr(frontend.baseline, "struct_scale")
    frontend.freeze_baseline()
    assert all(not param.requires_grad for param in frontend.baseline.parameters())


def test_duration_unit_frontend_emits_explicit_silence_runs_into_source_batch():
    frontend = DurationUnitFrontend(
        vocab_size=64,
        silent_token=57,
        emit_silence_runs=True,
    )
    batch = frontend.from_content_tensor(
        torch.tensor([[1, 1, 57, 57, 2, 2, 57]], dtype=torch.long),
        mark_last_open=False,
    )
    assert torch.equal(batch.content_units, torch.tensor([[1, 57, 2, 57]], dtype=torch.long))
    assert torch.allclose(batch.source_duration_obs, torch.tensor([[2.0, 2.0, 2.0, 1.0]], dtype=torch.float32))
    assert torch.allclose(batch.source_silence_mask, torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float32))


def test_duration_unit_frontend_from_precomputed_preserves_float_source_duration_obs():
    frontend = DurationUnitFrontend(vocab_size=64, silent_token=57, emit_silence_runs=True)
    batch = frontend.from_precomputed(
        content_units=torch.tensor([[1, 57, 2]], dtype=torch.long),
        source_duration_obs=torch.tensor([[1.5, 2.25, 3.75]], dtype=torch.float32),
        unit_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        source_silence_mask=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
    )
    assert batch.source_duration_obs.dtype == torch.float32
    assert torch.allclose(batch.source_duration_obs, torch.tensor([[1.5, 2.25, 3.75]], dtype=torch.float32))


def test_duration_unit_frontend_from_precomputed_preserves_source_run_stability():
    frontend = DurationUnitFrontend(vocab_size=64, silent_token=57, emit_silence_runs=True)
    batch = frontend.from_precomputed(
        content_units=torch.tensor([[1, 57, 2]], dtype=torch.long),
        source_duration_obs=torch.tensor([[1.5, 2.25, 3.75]], dtype=torch.float32),
        unit_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        source_silence_mask=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
        source_run_stability=torch.tensor([[0.9, 0.2, 0.8]], dtype=torch.float32),
    )
    assert torch.allclose(batch.source_run_stability, torch.tensor([[0.9, 0.2, 0.8]], dtype=torch.float32))


def test_duration_unit_frontend_simple_global_stats_omits_rate_log_base():
    frontend = DurationUnitFrontend(
        vocab_size=64,
        silent_token=57,
        emit_silence_runs=True,
        simple_global_stats=True,
    )
    batch = frontend.from_content_tensor(
        torch.tensor([[1, 1, 57, 2, 2]], dtype=torch.long),
        mark_last_open=False,
    )
    assert batch.unit_rate_log_base is None


def test_shared_source_cache_builder_can_materialize_explicit_silence_runs():
    cache = build_source_rhythm_cache(
        [1, 1, 57, 57, 2, 57, 57, 3],
        silent_token=57,
        emit_silence_runs=True,
    )
    assert cache["content_units"].tolist() == [1, 57, 2, 57, 3]
    assert cache["dur_anchor_src"].tolist() == [2, 2, 1, 2, 1]
    assert cache["source_silence_mask"].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]
    assert "source_run_stability" in cache
    assert len(cache["source_run_stability"].tolist()) == len(cache["content_units"].tolist())


def test_duration_unit_frontend_marks_short_open_runs_less_stable_than_long_sealed_runs():
    frontend = DurationUnitFrontend(
        vocab_size=64,
        silent_token=57,
        emit_silence_runs=True,
        tail_open_units=1,
        debounce_min_run_frames=2,
    )
    batch = frontend.from_content_tensor(
        torch.tensor([[1, 1, 2]], dtype=torch.long),
        mark_last_open=True,
    )
    assert batch.source_run_stability.shape == batch.source_duration_obs.shape
    assert float(batch.source_run_stability[0, 0].item()) > float(batch.source_run_stability[0, 1].item())


def test_streaming_unitizer_tensor_export_matches_python_export():
    unitizer = StreamingRunLengthUnitizer(
        silent_token=57,
        separator_aware=True,
        tail_open_units=2,
        emit_silence_runs=True,
        debounce_min_run_frames=2,
    )
    tokens = [1, 1, 57, 57, 2, 2, 57]
    python_seq = unitizer.compress(tokens, mark_last_open=True)
    tensor_seq = unitizer.compress_tensor(tokens, mark_last_open=True)
    assert tensor_seq.to_python() == python_seq
    python_seq_closed = unitizer.compress(tokens, mark_last_open=False)
    tensor_seq_closed = unitizer.compress_tensor(tokens, mark_last_open=False)
    assert tensor_seq_closed.to_python() == python_seq_closed

    frontend = DurationUnitFrontend(
        vocab_size=64,
        silent_token=57,
        emit_silence_runs=True,
        tail_open_units=2,
        debounce_min_run_frames=2,
    )
    batch_from_python = frontend.base_frontend._batch_from_compressed([python_seq])
    batch_from_tensor = frontend.base_frontend._batch_from_compressed([tensor_seq])
    assert torch.equal(batch_from_python.content_units, batch_from_tensor.content_units)
    assert torch.allclose(batch_from_python.dur_anchor_src, batch_from_tensor.dur_anchor_src)
    assert torch.allclose(batch_from_python.silence_mask, batch_from_tensor.silence_mask)
    assert torch.allclose(batch_from_python.run_stability, batch_from_tensor.run_stability)



def test_shared_source_cache_builder_debounces_short_silence_flicker_in_offline_path():
    cache = build_source_rhythm_cache(
        [1, 57, 1],
        silent_token=57,
        emit_silence_runs=True,
        debounce_min_run_frames=2,
    )
    assert cache["content_units"].tolist() == [1]
    assert cache["dur_anchor_src"].tolist() == [3]
    assert cache["source_silence_mask"].tolist() == [0.0]


def test_shared_source_cache_builder_suppresses_micro_silence_between_different_units():
    cache = build_source_rhythm_cache(
        [1, 57, 2],
        silent_token=57,
        emit_silence_runs=True,
        debounce_min_run_frames=1,
    )
    assert cache["content_units"].tolist() == [1, 2]
    assert cache["dur_anchor_src"].tolist() == [2, 1]
    assert cache["source_silence_mask"].tolist() == [0.0, 0.0]


def test_duration_unit_frontend_stream_state_debounces_short_tail_silence_flicker():
    frontend = DurationUnitFrontend(
        vocab_size=64,
        silent_token=57,
        emit_silence_runs=True,
        tail_open_units=2,
        debounce_min_run_frames=2,
    )
    state = frontend.init_stream_state(batch_size=1)
    _, state = frontend.step_content_tensor(
        torch.tensor([[1]], dtype=torch.long),
        state,
        content_lengths=torch.tensor([1], dtype=torch.long),
        mark_last_open=True,
    )
    batch, state = frontend.step_content_tensor(
        torch.tensor([[57, 1]], dtype=torch.long),
        state,
        content_lengths=torch.tensor([2], dtype=torch.long),
        mark_last_open=True,
    )
    assert torch.equal(state.rows[0].units.cpu(), torch.tensor([1], dtype=torch.long))
    assert torch.equal(state.rows[0].durations.cpu(), torch.tensor([3], dtype=torch.long))
    assert torch.equal(batch.content_units, torch.tensor([[1]], dtype=torch.long))
    assert torch.allclose(batch.source_duration_obs, torch.tensor([[3.0]], dtype=torch.float32))
    assert torch.allclose(batch.source_silence_mask, torch.tensor([[0.0]], dtype=torch.float32))


def test_duration_unit_frontend_matches_offline_source_cache_after_debounce():
    tokens = torch.tensor([[1, 57, 1, 2]], dtype=torch.long)
    frontend = DurationUnitFrontend(
        vocab_size=64,
        silent_token=57,
        emit_silence_runs=True,
        tail_open_units=2,
        debounce_min_run_frames=2,
    )
    batch = frontend.from_content_tensor(tokens, mark_last_open=False)
    cache = build_source_rhythm_cache(
        [1, 57, 1, 2],
        silent_token=57,
        emit_silence_runs=True,
        tail_open_units=2,
        debounce_min_run_frames=2,
    )
    assert batch.content_units[0, :2].tolist() == cache["content_units"].tolist()
    assert batch.source_duration_obs[0, :2].tolist() == cache["dur_anchor_src"].tolist()
    assert batch.source_silence_mask[0, :2].tolist() == cache["source_silence_mask"].tolist()


def test_duration_unit_frontend_can_materialize_existing_stream_state_without_new_tokens():
    frontend = DurationUnitFrontend(
        vocab_size=64,
        silent_token=57,
        emit_silence_runs=True,
        tail_open_units=2,
        debounce_min_run_frames=2,
    )
    state = frontend.init_stream_state(batch_size=1)
    _, state = frontend.step_content_tensor(
        torch.tensor([[1, 57, 1, 2]], dtype=torch.long),
        state,
        content_lengths=torch.tensor([4], dtype=torch.long),
        mark_last_open=True,
    )
    batch = frontend.materialize_stream_state(
        state,
        mark_last_open=True,
    )
    assert torch.equal(batch.content_units, torch.tensor([[1, 2]], dtype=torch.long))
    assert torch.allclose(batch.source_duration_obs, torch.tensor([[3.0, 1.0]], dtype=torch.float32))
    assert torch.allclose(batch.source_silence_mask, torch.tensor([[0.0, 0.0]], dtype=torch.float32))

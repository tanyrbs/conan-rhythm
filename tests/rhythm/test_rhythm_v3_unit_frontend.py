from __future__ import annotations

import torch

from modules.Conan.rhythm.supervision import build_source_rhythm_cache
from modules.Conan.rhythm_v3.unit_frontend import DurationUnitFrontend, ProtocolDurationBaseline


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


def test_shared_source_cache_builder_can_materialize_explicit_silence_runs():
    cache = build_source_rhythm_cache(
        [1, 1, 57, 57, 2, 57, 57, 3],
        silent_token=57,
        emit_silence_runs=True,
    )
    assert cache["content_units"].tolist() == [1, 57, 2, 57, 3]
    assert cache["dur_anchor_src"].tolist() == [2, 2, 1, 2, 1]
    assert cache["source_silence_mask"].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]

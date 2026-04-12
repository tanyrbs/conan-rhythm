from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from inference.Conan import StreamingVoiceConversion


ROOT = Path(__file__).resolve().parents[2]


def test_run_voice_conversion_uses_current_streaming_inference():
    source = (ROOT / "inference" / "run_voice_conversion.py").read_text(encoding="utf-8")
    assert "from inference.Conan import StreamingVoiceConversion" in source
    assert "Conan_previous" not in source


def test_streaming_inference_extracts_prompt_units_for_v3_by_default():
    source = (ROOT / "inference" / "Conan.py").read_text(encoding="utf-8")
    assert "def _extract_prompt_unit_conditioning" in source
    assert "rhythm_ref_conditioning = self._extract_prompt_unit_conditioning(" in source
    assert "prepared_spk_embed=prepared_spk_embed" in source
    assert '"prompt_content_units"' in source
    assert '"prompt_duration_obs"' in source
    assert '"prompt_unit_mask"' in source
    assert '"prompt_source_boundary_cue"' in source
    assert '"prompt_global_weight"' in source


def test_streaming_inference_reports_content_history_windowing_metadata():
    source = (ROOT / "inference" / "Conan.py").read_text(encoding="utf-8")
    assert '"content_history_windowing_enabled"' in source
    assert '"content_history_left_context_tokens"' in source


def test_streaming_inference_uses_incremental_rhythm_frontend_cache_updates():
    source = (ROOT / "inference" / "Conan.py").read_text(encoding="utf-8")
    assert "rhythm_frontend.step_content_tensor(" in source
    assert "export_duration_v3_source_cache(unit_batch)" in source


def test_streaming_inference_can_return_rhythm_debug_bundle():
    source = (ROOT / "inference" / "Conan.py").read_text(encoding="utf-8")
    assert "return_debug_bundle: bool = False" in source
    assert "self.last_rhythm_debug_bundle = None" in source
    assert "build_debug_record(" in source
    assert "commit_frontier must be monotone non-decreasing" in source
    assert "Committed prefix was rewritten" in source
    assert "requires cached prompt source_silence_mask" in source
    assert "rhythm_allow_eos_tail_flush_fallback" in source


def test_run_streaming_latency_report_defaults_to_v3_config():
    source = (ROOT / "inference" / "run_streaming_latency_report.py").read_text(encoding="utf-8")
    assert 'default="egs/conan_emformer_rhythm_v3.yaml"' in source


def test_v3_preflight_and_smoke_scripts_exist():
    preflight = (ROOT / "scripts" / "preflight_rhythm_v3.py").read_text(encoding="utf-8")
    smoke = (ROOT / "scripts" / "smoke_test_rhythm_v3.py").read_text(encoding="utf-8")
    assert "tasks.Conan.rhythm.preflight_support" in preflight
    assert "ConanDurationAdapter" in smoke
    assert "rhythm_v3_rate_mode" in smoke


def test_streaming_inference_monotone_frontier_guard_raises_on_rollback():
    with pytest.raises(RuntimeError, match="monotone non-decreasing"):
        StreamingVoiceConversion._assert_monotone_committed_frontier(
            previous_frontier=4,
            current_frontier=3,
        )


def test_streaming_inference_rewrite_guard_rejects_committed_prefix_changes():
    prev_state = SimpleNamespace(
        cached_duration_exec=torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float32),
        committed_units=torch.tensor([2], dtype=torch.long),
    )
    next_state = SimpleNamespace(
        cached_duration_exec=torch.tensor([[2.0, 9.0, 4.0]], dtype=torch.float32),
        committed_units=torch.tensor([2], dtype=torch.long),
    )
    with pytest.raises(RuntimeError, match="Committed prefix was rewritten"):
        StreamingVoiceConversion._assert_committed_prefix_not_rewritten(
            prev_state=prev_state,
            next_state=next_state,
        )


def test_streaming_inference_rewrite_guard_allows_open_tail_updates():
    prev_state = SimpleNamespace(
        cached_duration_exec=torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float32),
        committed_units=torch.tensor([2], dtype=torch.long),
    )
    next_state = SimpleNamespace(
        cached_duration_exec=torch.tensor([[2.0, 3.0, 9.0]], dtype=torch.float32),
        committed_units=torch.tensor([2], dtype=torch.long),
    )
    StreamingVoiceConversion._assert_committed_prefix_not_rewritten(
        prev_state=prev_state,
        next_state=next_state,
    )


def test_streaming_inference_eos_tail_policy_returns_tail_when_fallback_enabled():
    mel = torch.randn(5, 80)
    tail, unresolved = StreamingVoiceConversion._resolve_uncommitted_eos_tail(
        last_mel_out=mel,
        prev_committed_len=3,
        allow_tail_flush=True,
    )
    assert unresolved == 2
    assert tail.shape == (2, 80)


def test_streaming_inference_eos_tail_policy_rejects_tail_when_strict():
    mel = torch.randn(4, 80)
    with pytest.raises(RuntimeError, match="strict committed-only EOS"):
        StreamingVoiceConversion._resolve_uncommitted_eos_tail(
            last_mel_out=mel,
            prev_committed_len=1,
            allow_tail_flush=False,
        )

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from inference import Conan as conan_module
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
    assert "ref_source_id=inp.get(\"ref_wav\")" in source
    assert '"prompt_content_units"' in source
    assert '"prompt_duration_obs"' in source
    assert '"prompt_unit_mask"' in source
    assert '"prompt_source_boundary_cue"' in source
    assert '"prompt_global_weight"' in source
    assert '"prompt_unit_log_prior_present"' in source


def test_streaming_inference_prompt_cache_key_prefers_file_backed_fast_path():
    source = (ROOT / "inference" / "Conan.py").read_text(encoding="utf-8")
    assert "ref_source_id: str | None = None" in source
    assert "os.path.isfile(ref_path)" in source
    assert "os.stat(ref_path)" in source


def test_streaming_inference_reports_content_history_windowing_metadata():
    source = (ROOT / "inference" / "Conan.py").read_text(encoding="utf-8")
    assert '"content_history_windowing_enabled"' in source
    assert '"content_history_left_context_tokens"' in source
    assert '"rhythm_prefix_budget_abs_p95"' in source
    assert '"rhythm_boundary_decay_applied_rate"' in source


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


def test_v3_preflight_script_exists_and_v3_smoke_script_is_retired():
    preflight = (ROOT / "scripts" / "preflight_rhythm_v3.py").read_text(encoding="utf-8")
    assert "tasks.Conan.rhythm.preflight_support" in preflight
    assert not (ROOT / "scripts" / "smoke_test_rhythm_v3.py").exists()


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


def test_streaming_inference_runtime_summary_helpers_report_prefix_budget_and_boundary_decay():
    prefix_offset = torch.tensor([[0.0, -0.2, 0.4, 0.1]], dtype=torch.float32)
    boundary_decay = torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float32)
    p95 = StreamingVoiceConversion._summarize_prefix_budget_abs_p95(prefix_offset)
    decay_rate = StreamingVoiceConversion._summarize_boundary_decay_applied_rate(boundary_decay)
    assert p95 == pytest.approx(float(torch.quantile(prefix_offset.abs().reshape(-1), 0.95).item()))
    assert decay_rate == pytest.approx(0.5)


def test_streaming_inference_prompt_global_weight_rejects_shape_mismatch_by_default():
    with pytest.raises(RuntimeError, match="prompt_run_stability shape mismatch"):
        StreamingVoiceConversion._build_prompt_global_weight(
            prompt_speech_mask=torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
            prompt_run_stability=torch.tensor([[1.0, 0.2]], dtype=torch.float32),
        )


def test_streaming_inference_prompt_global_weight_can_repair_shape_when_enabled():
    weight = StreamingVoiceConversion._build_prompt_global_weight(
        prompt_speech_mask=torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
        prompt_run_stability=torch.tensor([[1.0, 0.2]], dtype=torch.float32),
        allow_shape_repair=True,
    )
    assert torch.allclose(weight, torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32))


def test_streaming_inference_frontend_signature_includes_global_stat_knobs(monkeypatch):
    monkeypatch.setattr(StreamingVoiceConversion, "_build_model", lambda self: object())
    monkeypatch.setattr(
        StreamingVoiceConversion,
        "_build_vocoder",
        lambda self: SimpleNamespace(supports_native_streaming=lambda: False),
    )
    monkeypatch.setattr(StreamingVoiceConversion, "_build_emformer", lambda self: object())
    monkeypatch.setattr(StreamingVoiceConversion, "_vocoder_warm_zero", lambda self: None)
    monkeypatch.setattr(conan_module, "resolve_vocoder_left_context_frames", lambda hp: (0, "none"))
    monkeypatch.setattr(
        conan_module,
        "build_streaming_layout_report",
        lambda *args, **kwargs: SimpleNamespace(to_metadata=lambda: {}),
    )
    monkeypatch.setattr(conan_module, "build_streaming_latency_report", lambda *args, **kwargs: {})

    base_hparams = {
        "work_dir": str(ROOT),
        "vocoder": "dummy",
        "chunk_size": 160,
        "silent_token": 57,
        "rhythm_separator_aware": True,
        "rhythm_tail_open_units": 1,
        "rhythm_v3_emit_silence_runs": True,
        "rhythm_v3_debounce_min_run_frames": 2,
        "rhythm_source_phrase_threshold": 0.55,
        "rhythm_v3_summary_pool_speech_only": True,
    }
    raw = StreamingVoiceConversion(
        {
            **base_hparams,
            "rhythm_v3_g_variant": "raw_median",
            "rhythm_v3_drop_edge_runs_for_g": 0,
            "rhythm_v3_unit_prior_path": None,
        }
    )
    unit_norm = StreamingVoiceConversion(
        {
            **base_hparams,
            "rhythm_v3_g_variant": "unit_norm",
            "rhythm_v3_drop_edge_runs_for_g": 1,
            "rhythm_v3_unit_prior_path": str(ROOT / "tests" / "rhythm" / "fixtures" / "unit_prior_demo.npz"),
            "rhythm_v3_summary_pool_speech_only": False,
        }
    )
    assert raw.rhythm_v3_frontend_signature != unit_norm.rhythm_v3_frontend_signature

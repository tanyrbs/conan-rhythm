from __future__ import annotations

from pathlib import Path


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


def test_streaming_inference_reports_content_history_windowing_metadata():
    source = (ROOT / "inference" / "Conan.py").read_text(encoding="utf-8")
    assert '"content_history_windowing_enabled"' in source
    assert '"content_history_left_context_tokens"' in source


def test_run_streaming_latency_report_defaults_to_v3_config():
    source = (ROOT / "inference" / "run_streaming_latency_report.py").read_text(encoding="utf-8")
    assert 'default="egs/conan_emformer_rhythm_v3.yaml"' in source


def test_v3_preflight_and_smoke_scripts_exist():
    preflight = (ROOT / "scripts" / "preflight_rhythm_v3.py").read_text(encoding="utf-8")
    smoke = (ROOT / "scripts" / "smoke_test_rhythm_v3.py").read_text(encoding="utf-8")
    assert "tasks.Conan.rhythm.preflight_support" in preflight
    assert "ConanDurationAdapter" in smoke
    assert "rhythm_v3_rate_mode" in smoke

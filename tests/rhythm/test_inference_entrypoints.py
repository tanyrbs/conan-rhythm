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
    assert "rhythm_ref_conditioning = self._extract_prompt_unit_conditioning(ref_mel_batch)" in source
    assert '"prompt_content_units"' in source
    assert '"prompt_duration_obs"' in source
    assert '"prompt_unit_mask"' in source
    assert '"prompt_source_boundary_cue"' not in source


def test_run_streaming_latency_report_defaults_to_v3_config():
    source = (ROOT / "inference" / "run_streaming_latency_report.py").read_text(encoding="utf-8")
    assert 'default="egs/conan_emformer_rhythm_v3.yaml"' in source

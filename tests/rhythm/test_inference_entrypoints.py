from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_run_voice_conversion_uses_current_streaming_inference():
    source = (ROOT / "inference" / "run_voice_conversion.py").read_text(encoding="utf-8")
    assert "from inference.Conan import StreamingVoiceConversion" in source
    assert "Conan_previous" not in source


def test_run_streaming_latency_report_defaults_to_v3_config():
    source = (ROOT / "inference" / "run_streaming_latency_report.py").read_text(encoding="utf-8")
    assert 'default="egs/conan_emformer_rhythm_v3.yaml"' in source

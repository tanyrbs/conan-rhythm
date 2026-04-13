from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
    assert "ref_cache_id=effective_ref_cache_id" in source
    assert '"prompt_content_units"' in source
    assert '"prompt_duration_obs"' in source
    assert '"prompt_unit_mask"' in source
    assert '"prompt_source_boundary_cue"' in source
    assert '"prompt_global_weight"' in source
    assert '"prompt_unit_log_prior_present"' in source


def test_streaming_inference_prompt_cache_key_prefers_file_backed_fast_path():
    source = (ROOT / "inference" / "Conan.py").read_text(encoding="utf-8")
    assert "ref_source_id: str | None = None" in source
    assert "ref_cache_id: str | None = None" in source
    assert "os.path.isfile(ref_path)" in source
    assert "os.stat(ref_path)" in source


def test_streaming_inference_prompt_cache_key_prefers_explicit_ref_cache_id(monkeypatch):
    monkeypatch.setattr(conan_module.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(
        conan_module.os,
        "stat",
        lambda _path: SimpleNamespace(st_mtime=1.0, st_mtime_ns=10, st_size=123),
    )
    mel_a = torch.randn(1, 4, 8)
    mel_b = torch.randn(1, 6, 8)
    key_a = StreamingVoiceConversion._make_prompt_conditioning_cache_key(
        mel_a,
        frontend_signature="sig",
        ref_source_id="ref_a.wav",
        ref_cache_id="shared-ref",
    )
    key_b = StreamingVoiceConversion._make_prompt_conditioning_cache_key(
        mel_b,
        frontend_signature="sig",
        ref_source_id="ref_b.wav",
        ref_cache_id="shared-ref",
    )
    key_c = StreamingVoiceConversion._make_prompt_conditioning_cache_key(
        mel_b,
        frontend_signature="sig",
        ref_source_id="ref_b.wav",
        ref_cache_id="other-ref",
    )
    assert key_a == key_b
    assert key_a != key_c


def test_streaming_inference_reports_content_history_windowing_metadata():
    source = (ROOT / "inference" / "Conan.py").read_text(encoding="utf-8")
    assert '"content_history_windowing_enabled"' in source
    assert '"content_history_left_context_tokens"' in source
    assert '"rhythm_prefix_budget_abs_p95"' in source
    assert '"rhythm_boundary_decay_applied_rate"' in source
    assert '"prompt_global_weight_present"' in source
    assert '"prompt_unit_log_prior_present"' in source
    assert '"rhythm_v3_g_variant"' in source
    assert '"rhythm_eos_tail_flush_severity"' in source


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
    monkeypatch.setattr(
        conan_module,
        "build_duration_v3_frontend_signature",
        lambda *args, **kwargs: repr(sorted(kwargs.items())),
    )

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


def test_streaming_inference_infer_once_reports_prompt_sidecars_and_eos_severity(monkeypatch):
    svc = StreamingVoiceConversion.__new__(StreamingVoiceConversion)
    svc.device = torch.device("cpu")
    svc.hparams = {
        "hop_size": 160,
        "audio_sample_rate": 16000,
        "right_context": 0,
        "chunk_size": 2,
        "style": False,
        "rhythm_allow_eos_tail_flush_fallback": True,
        "rhythm_v3_g_variant": "weighted_median",
    }
    svc.vocoder = SimpleNamespace(
        reset_stream=lambda: None,
        spec2wav=lambda arr: np.zeros(int(arr.shape[0]), dtype=np.float32),
    )
    svc.vocoder_left_context_frames = 0
    svc.vocoder_left_context_source = "none"
    svc.vocoder_native_streaming_capable = False
    svc._vocoder_warm_zero = lambda: None
    svc._wav_to_mel = lambda _path: np.zeros((4, 80), dtype=np.float32)
    svc._prepare_spk_embed = lambda ref_mel_batch, spk_embed=None: torch.zeros((1, 1, 4), dtype=torch.float32)
    svc._resolve_content_window_left_tokens = lambda: 0
    svc._compute_content_window_start = lambda **kwargs: 0
    svc._render_vocoder_chunk = lambda mel_chunk, mel_context_buffer=None: np.zeros(
        int(mel_chunk.shape[0]), dtype=np.float32
    )
    svc._resolve_committed_token_frontier_from_cache = lambda **kwargs: None
    svc.get_streaming_runtime_metadata = lambda duration_seconds=None: {"runtime_base": duration_seconds}
    svc.last_rhythm_debug_bundle = None
    svc.runtime_metadata = {}
    svc.last_inference_metadata = {}

    captured = {}

    def fake_extract_prompt_unit_conditioning(
        ref_mel_batch,
        prepared_spk_embed=None,
        ref_source_id=None,
        ref_cache_id=None,
    ):
        captured["ref_source_id"] = ref_source_id
        captured["ref_cache_id"] = ref_cache_id
        return {
            "prompt_global_weight_present": torch.ones((1, 1), dtype=torch.float32),
            "prompt_unit_log_prior_present": torch.zeros((1, 1), dtype=torch.float32),
        }

    svc._extract_prompt_unit_conditioning = fake_extract_prompt_unit_conditioning

    class _DummyFrontend:
        def step_content_tensor(self, new_codes, state=None, content_lengths=None, mark_last_open=True):
            return new_codes, state

    class _DummyEmformerInner:
        def infer(self, chunk, lengths, state):
            return torch.zeros((1, chunk.size(1)), dtype=torch.long), None, state

    class _DummyModel:
        rhythm_enabled = True
        rhythm_enable_v2 = False
        rhythm_enable_v3 = True
        rhythm_minimal_style_only = False
        rhythm_unit_frontend = None

        def __call__(
            self,
            content,
            spk_embed,
            target,
            ref,
            f0,
            uv,
            infer,
            global_steps,
            content_lengths,
            rhythm_state,
            rhythm_ref_conditioning,
            rhythm_source_cache,
            decoder_cache,
        ):
            return {
                "mel_out": torch.zeros((1, 3, 80), dtype=torch.float32),
                "rhythm_ref_conditioning": rhythm_ref_conditioning,
                "rhythm_state_next": rhythm_state,
                "decoder_cache": decoder_cache,
            }

    svc.model = _DummyModel()
    svc.emformer = SimpleNamespace(
        emformer=_DummyEmformerInner(),
        mode="unit",
        proj=lambda x: x,
        proj1=lambda x: x,
    )

    monkeypatch.setattr(conan_module, "resolve_chunk_frames", lambda _hp: 2)
    monkeypatch.setattr(conan_module, "resolve_mel_frame_ms", lambda _hp: 10.0)
    monkeypatch.setattr(conan_module, "PrefixCodeBuffer", lambda total_capacity: SimpleNamespace(append=lambda x: x))
    monkeypatch.setattr(
        conan_module,
        "RollingMelContextBuffer",
        lambda left_context_frames: SimpleNamespace(left_context_frames=left_context_frames),
    )
    monkeypatch.setattr(
        conan_module,
        "extract_incremental_committed_mel",
        lambda out, prev_committed_len, batch_index=0: (
            out["mel_out"][0].new_zeros((0, out["mel_out"][0].shape[-1])),
            0,
        ),
    )

    _, _, metadata = svc.infer_once(
        {"src_wav": "src.wav", "ref_wav": "ref.wav"},
        return_metadata=True,
        ref_cache_id="manual-ref-id",
    )
    assert captured == {"ref_source_id": "ref.wav", "ref_cache_id": "manual-ref-id"}
    assert metadata["prompt_global_weight_present"] is True
    assert metadata["prompt_unit_log_prior_present"] is False
    assert metadata["rhythm_v3_g_variant"] == "weighted_median"
    assert metadata["rhythm_uncommitted_eos_tail_frames"] == 3
    assert metadata["rhythm_eos_tail_flush_severity"] == "warning"

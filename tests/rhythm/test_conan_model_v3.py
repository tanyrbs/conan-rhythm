from __future__ import annotations

from unittest import mock

import pytest
import torch
import torch.nn as nn

from modules.Conan.Conan import Conan, _resolve_content_vocab_size


class _DummyV3Adapter:
    def __call__(
        self,
        *,
        ret,
        content,
        ref,
        target,
        f0,
        uv,
        infer,
        global_steps,
        content_embed,
        tgt_nonpadding,
        content_lengths,
        rhythm_state,
        rhythm_ref_conditioning,
        rhythm_apply_override,
        rhythm_runtime_overrides,
        rhythm_source_cache,
        rhythm_offline_source_cache,
        speech_state_fn,
        ref_lengths=None,
    ):
        del (
            ref,
            target,
            f0,
            uv,
            infer,
            global_steps,
            content_lengths,
            rhythm_state,
            rhythm_apply_override,
            rhythm_runtime_overrides,
            rhythm_source_cache,
            rhythm_offline_source_cache,
            speech_state_fn,
            ref_lengths,
        )
        assert isinstance(rhythm_ref_conditioning, dict)
        assert set(("prompt_content_units", "prompt_duration_obs")).issubset(rhythm_ref_conditioning.keys())
        assert "prompt_unit_mask" in rhythm_ref_conditioning
        ret["rhythm_version"] = "v3"
        ret["rhythm_execution"] = object()
        ret["speech_duration_exec"] = torch.ones((content.size(0), content.size(1)), dtype=content_embed.dtype)
        return content_embed, tgt_nonpadding, None, None


def _build_dummy_model():
    model = Conan.__new__(Conan)
    nn.Module.__init__(model)
    model.rhythm_enabled = True
    model.rhythm_enable_v3 = True
    model.rhythm_minimal_style_only = True
    model.rhythm_adapter = _DummyV3Adapter()
    model.out_dims = 80
    model.hparams = {
        "style": False,
        "use_pitch_embed": False,
    }
    return model


def _build_canonical_ref_conditioning():
    return {
        "prompt_content_units": torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
        "prompt_duration_obs": torch.tensor([[3.0, 4.0, 2.0, 0.0]], dtype=torch.float32),
        "prompt_unit_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
    }


def test_conan_forward_v3_short_circuits_acoustic_train_path():
    model = _build_dummy_model()
    content = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    target = torch.randn(1, 4, 80)
    content_embed = torch.randn(1, 4, 32)
    tgt_nonpadding = torch.ones(1, 4, 1)
    with mock.patch(
        "modules.Conan.Conan.prepare_content_inputs",
        return_value=(content_embed, tgt_nonpadding),
    ):
        with mock.patch(
            "modules.Conan.Conan.run_acoustic_path",
            side_effect=AssertionError("acoustic path should be skipped"),
        ):
            output = Conan.forward(
                model,
                content,
                target=target,
                ref=None,
                infer=False,
                global_steps=0,
                content_lengths=torch.tensor([4]),
                rhythm_ref_conditioning=_build_canonical_ref_conditioning(),
                disable_acoustic_train_path=True,
            )
    assert output["rhythm_version"] == "v3"
    assert output["rhythm_disable_acoustic_train_path"] == 1.0
    assert torch.allclose(output["mel_out"], target)


def test_conan_forward_v3_infer_does_not_short_circuit_acoustic_path():
    model = _build_dummy_model()
    content = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    content_embed = torch.randn(1, 4, 32)
    tgt_nonpadding = torch.ones(1, 4, 1)

    def _fake_run_acoustic_path(*args, **kwargs):
        ret = kwargs["ret"]
        ret["acoustic_called"] = 1.0
        ret["mel_out"] = torch.zeros((1, 4, 80))
        return ret

    with mock.patch(
        "modules.Conan.Conan.prepare_content_inputs",
        return_value=(content_embed, tgt_nonpadding),
    ):
        with mock.patch("modules.Conan.Conan.run_acoustic_path", side_effect=_fake_run_acoustic_path):
            output = Conan.forward(
                model,
                content,
                target=None,
                ref=None,
                infer=True,
                global_steps=0,
                content_lengths=torch.tensor([4]),
                rhythm_ref_conditioning=_build_canonical_ref_conditioning(),
                disable_acoustic_train_path=True,
            )
    assert output["rhythm_version"] == "v3"
    assert output["acoustic_called"] == 1.0


def test_conan_prepare_rhythm_reference_rejects_v3_proxy_path():
    model = _build_dummy_model()
    ref = torch.randn(1, 6, 80)
    ref_lengths = torch.tensor([6], dtype=torch.long)
    with pytest.raises(RuntimeError, match="no longer supports mel-proxy reference preparation"):
        Conan.prepare_rhythm_reference(model, ref, ref_lengths=ref_lengths)


def test_resolve_content_vocab_size_does_not_fall_back_to_embedding_dim():
    with pytest.raises(ValueError, match="Unable to resolve content vocabulary size"):
        _resolve_content_vocab_size({"content_embedding_dim": 256})


def test_resolve_content_vocab_size_accepts_explicit_unit_count_alias():
    assert _resolve_content_vocab_size({"n_content_units": 321}) == 321

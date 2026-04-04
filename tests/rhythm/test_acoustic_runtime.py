import unittest

import torch

from modules.Conan.acoustic_runtime import run_acoustic_path


class _DummyModel:
    def __init__(self, *, use_pitch_embed: bool) -> None:
        self.hparams = {"use_pitch_embed": use_pitch_embed, "style": False}
        self.rhythm_minimal_style_only = False
        self.out_dims = 4
        self.forward_pitch_called = False

    def forward_pitch(self, decoder_inp, f0, uv, ret, **kwargs):
        self.forward_pitch_called = True
        return torch.ones_like(decoder_inp)

    def forward_decoder(self, decoder_inp, tgt_nonpadding, ret, infer, **kwargs):
        return decoder_inp * tgt_nonpadding


class AcousticRuntimeTests(unittest.TestCase):
    def test_run_acoustic_path_skips_pitch_branch_when_pitch_embed_disabled(self):
        model = _DummyModel(use_pitch_embed=False)
        ret = {}
        content_embed = torch.randn(2, 5, 3)
        tgt_nonpadding = torch.ones(2, 5, 1)
        spk_embed = torch.randn(2, 3)

        out = run_acoustic_path(
            model,
            ret=ret,
            content_embed=content_embed,
            tgt_nonpadding=tgt_nonpadding,
            spk_embed=spk_embed,
            ref=None,
            f0=None,
            uv=None,
            infer=False,
            global_steps=0,
            forward_kwargs={},
        )

        expected = content_embed + spk_embed.unsqueeze(1)
        self.assertFalse(model.forward_pitch_called)
        self.assertEqual(float(out.get("rhythm_pitch_embed_disabled", 0.0)), 1.0)
        self.assertTrue(torch.allclose(out["decoder_inp"], expected))

    def test_run_acoustic_path_uses_pitch_branch_when_enabled(self):
        model = _DummyModel(use_pitch_embed=True)
        ret = {}
        content_embed = torch.randn(1, 4, 2)
        tgt_nonpadding = torch.ones(1, 4, 1)
        spk_embed = torch.randn(1, 2)

        out = run_acoustic_path(
            model,
            ret=ret,
            content_embed=content_embed,
            tgt_nonpadding=tgt_nonpadding,
            spk_embed=spk_embed,
            ref=None,
            f0=None,
            uv=None,
            infer=False,
            global_steps=0,
            forward_kwargs={},
        )

        expected = content_embed + spk_embed.unsqueeze(1) + 1.0
        self.assertTrue(model.forward_pitch_called)
        self.assertTrue(torch.allclose(out["decoder_inp"], expected))


if __name__ == "__main__":
    unittest.main()

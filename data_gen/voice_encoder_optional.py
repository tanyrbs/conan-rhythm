from __future__ import annotations

import importlib


def require_voice_encoder_cls():
    try:
        module = importlib.import_module("resemblyzer")
    except Exception as exc:
        raise ImportError(
            "Resemblyzer is required when binarization_args.with_spk_embed=true. "
            "Install resemblyzer or disable with_spk_embed."
        ) from exc
    voice_encoder_cls = getattr(module, "VoiceEncoder", None)
    if voice_encoder_cls is None:
        raise ImportError(
            "resemblyzer.VoiceEncoder is unavailable. "
            "Install a compatible resemblyzer build or disable with_spk_embed."
        )
    return voice_encoder_cls


def build_voice_encoder(*, prefer_cuda: bool = True):
    encoder = require_voice_encoder_cls()()
    if not prefer_cuda:
        return encoder
    try:
        import torch

        if torch.cuda.is_available() and hasattr(encoder, "cuda"):
            encoder = encoder.cuda()
    except Exception:
        pass
    return encoder


__all__ = ["build_voice_encoder", "require_voice_encoder_cls"]

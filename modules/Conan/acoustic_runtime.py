from __future__ import annotations

import torch

from modules.Conan.rhythm.bridge import build_content_nonpadding


def prepare_content_inputs(model, *, content, content_lengths, ret: dict):
    ret["content"] = content
    ret["content_base"] = content
    tgt_nonpadding = build_content_nonpadding(
        content,
        content_lengths=content_lengths,
    )[:, :, None]
    content_embed = model._unit_speech_state_fn(content)
    ret["content_embed_proj"] = content_embed
    return content_embed, tgt_nonpadding


def resolve_style_embed(model, *, spk_embed, ref, ret: dict):
    if spk_embed is not None:
        if spk_embed.dim() == 2:
            spk_embed = spk_embed.unsqueeze(1)
        elif spk_embed.dim() != 3:
            raise ValueError(f"spk_embed must have shape [B, H] or [B, 1, H], got {tuple(spk_embed.shape)}.")
        ret["style_embed"] = style_embed = spk_embed
        return style_embed
    if ref is None:
        raise ValueError(
            "When spk_embed is None, need target tensor to extract speaker embedding."
        )
    ret["style_embed"] = style_embed = model.encode_spk_embed(
        ref.transpose(1, 2)
    ).transpose(1, 2)
    return style_embed


def maybe_short_circuit_acoustic_train(
    model,
    *,
    ret: dict,
    infer,
    target,
    f0,
    content,
    content_embed,
    tgt_nonpadding,
    disable_acoustic_train_path,
) -> bool:
    if not disable_acoustic_train_path or bool(infer):
        return False
    ret["tgt_nonpadding"] = tgt_nonpadding
    ret["rhythm_disable_acoustic_train_path"] = 1.0
    retimed_target = ret.get("rhythm_online_retimed_mel_tgt")
    if retimed_target is not None:
        ret["mel_out"] = retimed_target
    elif target is not None:
        ret["mel_out"] = target
    else:
        mel_len = (
            int(tgt_nonpadding.squeeze(-1).sum(dim=1).max().item())
            if tgt_nonpadding.numel() > 0
            else int(content.size(1))
        )
        ret["mel_out"] = content_embed.new_zeros((content.size(0), mel_len, model.out_dims))
    if f0 is not None:
        ret["f0_denorm_pred"] = f0.float()
    return True


def build_pitch_input(model, *, content_embed, style_embed, ref, ret: dict, infer, global_steps):
    pitch_inp = content_embed + style_embed
    if model.hparams["style"] and not model.rhythm_minimal_style_only:
        if ref is None:
            ret["rhythm_prosody_skipped_no_ref"] = 1.0
        else:
            prosody = model.get_prosody(pitch_inp, ref, ret, infer, global_steps)
            pitch_inp = pitch_inp + prosody
    ret["pitch_embed"] = pitch_inp
    return pitch_inp


def run_acoustic_path(
    model,
    *,
    ret: dict,
    content_embed: torch.Tensor,
    tgt_nonpadding: torch.Tensor,
    spk_embed,
    ref,
    f0,
    uv,
    infer,
    global_steps,
    forward_kwargs: dict,
):
    style_embed = resolve_style_embed(
        model,
        spk_embed=spk_embed,
        ref=ref,
        ret=ret,
    )
    pitch_inp = build_pitch_input(
        model,
        content_embed=content_embed,
        style_embed=style_embed,
        ref=ref,
        ret=ret,
        infer=infer,
        global_steps=global_steps,
    )
    if infer:
        f0, uv = None, None
    pitch_embed_out = model.forward_pitch(pitch_inp, f0, uv, ret, **forward_kwargs)
    ret["decoder_inp"] = decoder_inp = pitch_inp + pitch_embed_out
    ret["mel_out"] = model.forward_decoder(
        decoder_inp,
        tgt_nonpadding,
        ret,
        infer=infer,
        **forward_kwargs,
    )
    ret["tgt_nonpadding"] = tgt_nonpadding
    return ret

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.audio.pitch.utils import denorm_f0, f0_to_coarse

from modules.Conan.pitch_utils import (
    apply_silent_content_to_uv,
    build_midi_norm_f0_bounds,
    f0_minmax_denorm,
    f0_minmax_norm,
    infer_uv_from_logits,
    pack_flow_f0_target,
)


class ConanPitchMixin:
    def _resolve_pitch_handler(self):
        handler_map = {
            "diff": self.add_diff_pitch,
            "gmdiff": self.add_gmdiff_pitch,
            "flow": self.add_flow_pitch,
            "orig": self.add_orig_pitch,
        }
        f0_gen_type = self.hparams["f0_gen"]
        if f0_gen_type not in handler_map:
            raise ValueError(f"Unknown f0_gen type: {f0_gen_type}")
        return handler_map[f0_gen_type]

    def _predict_uv_logits(self, decoder_inp, ret):
        ret["uv_pred"] = uv_pred = self.uv_predictor(decoder_inp)
        return uv_pred

    def forward_pitch(self, decoder_inp, f0, uv, ret, **kwargs):
        pitch_pred_inp = decoder_inp
        if self.hparams["predictor_grad"] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + self.hparams["predictor_grad"] * (
                pitch_pred_inp - pitch_pred_inp.detach()
            )

        f0_out, uv_out = self._resolve_pitch_handler()(
            pitch_pred_inp,
            f0,
            uv,
            ret,
            **kwargs,
        )
        f0_denorm = denorm_f0(f0_out, uv_out)
        pitch = f0_to_coarse(f0_denorm)
        ret["f0_denorm_pred"] = f0_denorm
        return self.pitch_embed(pitch)

    def add_orig_pitch(self, decoder_inp, f0, uv, ret, encoder_out=None, **kwargs):
        infer = f0 is None
        uv_pred = self._predict_uv_logits(decoder_inp, ret)

        if infer:
            uv = infer_uv_from_logits(
                uv_pred,
                content=ret.get("content"),
                silent_token=self.hparams.get("silent_token"),
            )
            f0 = uv_pred[:, :, 1]
            ret["fdiff"] = 0.0
        else:
            nonpadding = (uv == 0).float()
            f0_pred = uv_pred[:, :, 1]
            denom = nonpadding.sum().clamp_min(1.0)
            ret["fdiff"] = (
                (F.mse_loss(f0_pred, f0, reduction="none") * nonpadding).sum()
                / denom
                * self.hparams["lambda_f0"]
            )
        return f0, uv

    def add_diff_pitch(self, decoder_inp, f0, uv, ret, encoder_out=None, **kwargs):
        infer = f0 is None
        uv_pred = self._predict_uv_logits(decoder_inp, ret)
        mel2ph = kwargs.get("mel2ph")

        if infer:
            uv = infer_uv_from_logits(
                uv_pred,
                content=ret.get("content"),
                silent_token=self.hparams.get("silent_token"),
            )
            midi_notes = kwargs.get("midi_notes")
            if midi_notes is None:
                raise ValueError("add_diff_pitch requires midi_notes during inference.")
            midi_notes = midi_notes.transpose(-1, -2)
            uv = uv.clone()
            uv[midi_notes[:, 0, :] == 0] = 1
            lower_norm_f0, upper_norm_f0 = build_midi_norm_f0_bounds(
                midi_notes,
                strict_upper=True,
            )
            f0 = self.f0_gen(
                decoder_inp.transpose(-1, -2),
                None,
                None,
                ret,
                infer,
                dyn_clip=[lower_norm_f0, upper_norm_f0],
            )[:, :, 0]
            f0 = f0_minmax_denorm(f0)
            ret["fdiff"] = 0.0
        else:
            nonpadding = (mel2ph > 0).float() if mel2ph is not None else (uv == 0).float()
            norm_f0 = f0_minmax_norm(f0, strict_upper=True)
            ret["fdiff"] = self.f0_gen(
                decoder_inp.transpose(-1, -2),
                norm_f0,
                nonpadding.unsqueeze(dim=1),
                ret,
                infer,
            )
        return f0, uv

    def add_flow_pitch(self, decoder_inp, f0, uv, ret, encoder_out=None, **kwargs):
        infer = f0 is None
        uv_pred = self._predict_uv_logits(decoder_inp, ret)
        initial_noise = kwargs.get("initial_noise")

        if infer:
            if uv is None:
                uv = infer_uv_from_logits(
                    uv_pred,
                    content=ret.get("content"),
                    silent_token=self.hparams.get("silent_token"),
                )
            f0_pred_norm = self.f0_gen(
                decoder_inp.transpose(1, 2),
                None,
                None,
                ret,
                infer=True,
                initial_noise=initial_noise,
            )
            f0_out = f0_minmax_denorm(f0_pred_norm, uv)
            ret["pflow"] = 0.0
            uv_out = uv
        else:
            nonpadding = (uv == 0).float()
            norm_f0 = f0_minmax_norm(f0, uv)
            ret["pflow"] = self.f0_gen(
                decoder_inp.transpose(1, 2),
                pack_flow_f0_target(norm_f0),
                nonpadding.unsqueeze(1),
                ret,
                infer=False,
            )
            f0_out = f0
            uv_out = uv

        return f0_out, uv_out

    def add_gmdiff_pitch(self, decoder_inp, f0, uv, ret, encoder_out=None, **kwargs):
        infer = f0 is None
        mel2ph = kwargs.get("mel2ph")

        if infer:
            midi_notes = kwargs.get("midi_notes")
            if midi_notes is None:
                raise ValueError("add_gmdiff_pitch requires midi_notes during inference.")
            midi_notes = midi_notes.transpose(-1, -2)
            lower_norm_f0, upper_norm_f0 = build_midi_norm_f0_bounds(
                midi_notes,
                strict_upper=True,
            )
            pitch_pred = self.f0_gen(
                decoder_inp.transpose(-1, -2),
                None,
                None,
                None,
                ret,
                infer,
                dyn_clip=[lower_norm_f0, upper_norm_f0],
            )
            f0 = pitch_pred[:, :, 0]
            uv = pitch_pred[:, :, 1]
            uv = uv.clone()
            uv[midi_notes[:, 0, :] == 0] = 1
            uv = apply_silent_content_to_uv(
                uv,
                content=ret.get("content"),
                silent_token=self.hparams.get("silent_token"),
            )
            f0 = f0_minmax_denorm(f0)
            ret["gdiff"] = 0.0
            ret["mdiff"] = 0.0
        else:
            nonpadding = (mel2ph > 0).float() if mel2ph is not None else (uv == 0).float()
            norm_f0 = f0_minmax_norm(f0, strict_upper=True)
            ret["mdiff"], ret["gdiff"], ret["nll"] = self.f0_gen(
                decoder_inp.transpose(-1, -2),
                norm_f0.unsqueeze(dim=1),
                uv,
                nonpadding,
                ret,
                infer,
            )
        return f0, uv

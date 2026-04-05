import torch
import torch.nn as nn

from modules.Conan.diff.net import DiffNet, F0DiffNet, OriDiffNet, CausalConv1d
from modules.Conan.acoustic_runtime import (
    maybe_short_circuit_acoustic_train,
    prepare_content_inputs,
    run_acoustic_path,
)
from modules.Conan.pitch_mixin import ConanPitchMixin
from modules.Conan.prosody_util import ProsodyAligner, LocalStyleAdaptor
from modules.Conan.rhythm.runtime_adapter import ConanRhythmAdapter
from modules.Conan.rhythm.factory import resolve_content_vocab_size
from modules.commons.conv import ConvBlocks, CausalFM
from modules.commons.nar_tts_modules import PitchPredictor
from modules.commons.transformer import SinusoidalPositionalEmbedding
from modules.tts.fs import FastSpeech
from utils.commons.hparams import hparams

Flow_DECODERS = {
    "wavenet": lambda hp: DiffNet(hp["audio_num_mel_bins"]),
    "orig": lambda hp: OriDiffNet(hp["audio_num_mel_bins"]),
    "conv": lambda hp: CausalFM(
        hp["hidden_size"],
        hp["hidden_size"],
        hp["dec_dilations"],
        hp["dec_kernel_size"],
        layers_in_block=hp["layers_in_block"],
        norm_type=hp["enc_dec_norm"],
        dropout=hp["dropout"],
        post_net_kernel=hp.get("dec_post_net_kernel", 3),
    ),
}

DEFAULT_MAX_SOURCE_POSITIONS = 2000
DEFAULT_MAX_TARGET_POSITIONS = 2000


def _require_flow_mel():
    try:
        from modules.Conan.flow.flow import FlowMel
    except ModuleNotFoundError as exc:
        missing_name = str(getattr(exc, "name", "") or "")
        exc_message = str(exc)
        if missing_name == "torchdyn" or missing_name.startswith("torchdyn."):
            raise ImportError(
                "FlowMel / ConanPostnet requires torchdyn. Install the flow dependencies or disable the flow "
                "postnet path before importing it."
            ) from exc
        if "torchdyn" in exc_message:
            raise ImportError(
                "FlowMel / ConanPostnet requires torchdyn. Install the flow dependencies or disable the flow "
                "postnet path before importing it."
            ) from exc
        raise
    return FlowMel


def _require_reflow_f0():
    try:
        from modules.Conan.flow.flow_f0 import ReflowF0
    except ModuleNotFoundError as exc:
        missing_name = str(getattr(exc, "name", "") or "")
        exc_message = str(exc)
        if missing_name == "torchdyn" or missing_name.startswith("torchdyn."):
            raise ImportError(
                "Flow-based F0 generation requires torchdyn. Install the flow dependencies or set f0_gen != 'flow'."
            ) from exc
        if "torchdyn" in exc_message:
            raise ImportError(
                "Flow-based F0 generation requires torchdyn. Install the flow dependencies or set f0_gen != 'flow'."
            ) from exc
        raise
    return ReflowF0


class Conan(ConanPitchMixin, FastSpeech):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__(dict_size, hparams, out_dims)

        hidden_size = hparams["hidden_size"]
        kernel_size = hparams["kernel_size"]
        content_vocab_size = resolve_content_vocab_size(hparams)
        self._init_content_backbone(
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            content_vocab_size=content_vocab_size,
        )
        self._init_rhythm_components(hparams=hparams, hidden_size=hidden_size)
        self._init_style_components(hparams=hparams)
        self._init_pitch_components(hparams=hparams)

    def _init_content_backbone(self, *, hidden_size: int, kernel_size: int, content_vocab_size: int) -> None:
        self.content_embedding = nn.Embedding(content_vocab_size, hidden_size)
        self.content_proj = nn.Sequential(
            CausalConv1d(hidden_size, hidden_size, kernel_size=kernel_size, dilation=1),
            nn.LeakyReLU(),
        )
        self.global_conv_in = nn.Conv1d(80, hidden_size, 1)
        self.global_encoder = ConvBlocks(
            hidden_size,
            hidden_size,
            None,
            kernel_size=31,
            layers_in_block=2,
            is_BTC=False,
            num_layers=5,
        )

    def _init_rhythm_components(self, *, hparams, hidden_size: int) -> None:
        self.rhythm_enable_v2 = bool(hparams.get("rhythm_enable_v2", False))
        self.rhythm_minimal_style_only = bool(hparams.get("rhythm_minimal_style_only", False))
        if not self.rhythm_enable_v2:
            return
        self.rhythm_adapter = ConanRhythmAdapter(hparams, hidden_size)
        self.rhythm_unit_frontend = self.rhythm_adapter.unit_frontend
        self.rhythm_module = self.rhythm_adapter.module
        self.rhythm_pause_state = self.rhythm_adapter.pause_state
        self.rhythm_render_phase_mlp = self.rhythm_adapter.render_phase_mlp
        self.rhythm_render_phase_gain = self.rhythm_adapter.render_phase_gain

    def _init_style_components(self, *, hparams) -> None:
        if not hparams["style"] or self.rhythm_minimal_style_only:
            return
        self.padding_idx = 0
        self.prosody_extractor = LocalStyleAdaptor(
            self.hidden_size, hparams["nVQ"], self.padding_idx
        )
        self.l1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.align = ProsodyAligner(num_layers=2)
        self.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        self.embed_positions = SinusoidalPositionalEmbedding(
            self.hidden_size,
            self.padding_idx,
            init_size=self.max_source_positions + self.padding_idx + 1,
        )

    def _build_uv_predictor(self, *, hparams) -> PitchPredictor:
        return PitchPredictor(
            self.hidden_size,
            n_chans=128,
            n_layers=5,
            dropout_rate=0.1,
            odim=2,
            kernel_size=hparams["predictor_kernel"],
        )

    def _init_pitch_components(self, *, hparams) -> None:
        self.uv_predictor = self._build_uv_predictor(hparams=hparams)
        if hparams["f0_gen"] != "flow":
            return
        self.pitch_flownet = F0DiffNet(in_dims=1)
        ReflowF0 = _require_reflow_f0()
        self.f0_gen = ReflowF0(
            out_dims=1,
            denoise_fn=self.pitch_flownet,
            timesteps=hparams["f0_timesteps"],
            f0_K_step=hparams["f0_K_step"],
        )

    def _unit_speech_state_fn(self, unit_ids):
        unit_embed = self.content_embedding(unit_ids)
        unit_embed = self.content_proj(unit_embed.transpose(1, 2)).transpose(1, 2)
        return unit_embed

    def _rhythm_render_frame_state_post(
        self,
        frame_states: torch.Tensor,
        frame_phase_features: torch.Tensor,
        blank_mask: torch.Tensor,
        total_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.rhythm_enable_v2:
            return frame_states * total_mask.unsqueeze(-1)
        return self.rhythm_adapter.render_frame_state_post(
            frame_states=frame_states,
            frame_phase_features=frame_phase_features,
            blank_mask=blank_mask,
            total_mask=total_mask,
        )

    def _resolve_rhythm_pause_topk_ratio(self, *, infer: bool, global_steps: int) -> float | None:
        if not self.rhythm_enable_v2:
            return None
        return self.rhythm_adapter.resolve_pause_topk_ratio(
            infer=infer,
            global_steps=global_steps,
        )

    def _resolve_rhythm_source_boundary_scale(self, *, infer: bool, global_steps: int, teacher: bool = False) -> float | None:
        if not self.rhythm_enable_v2:
            return None
        return self.rhythm_adapter.resolve_source_boundary_scale(
            infer=infer,
            global_steps=global_steps,
            teacher=teacher,
        )

    def _run_rhythm_stage(
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
        kwargs,
    ):
        if not self.rhythm_enable_v2:
            return content_embed, tgt_nonpadding, f0, uv
        return self.rhythm_adapter(
            ret=ret,
            content=content,
            ref=ref,
            target=target,
            f0=f0,
            uv=uv,
            infer=bool(infer),
            global_steps=int(global_steps),
            content_embed=content_embed,
            tgt_nonpadding=tgt_nonpadding,
            content_lengths=content_lengths,
            rhythm_state=rhythm_state,
            rhythm_ref_conditioning=rhythm_ref_conditioning,
            rhythm_apply_override=rhythm_apply_override,
            rhythm_runtime_overrides=rhythm_runtime_overrides,
            rhythm_source_cache=kwargs.get("rhythm_source_cache"),
            rhythm_offline_source_cache=kwargs.get("rhythm_offline_source_cache"),
            speech_state_fn=self._unit_speech_state_fn,
        )

    def forward(
        self,
        content,
        spk_embed=None,
        target=None,
        ref=None,
        f0=None,
        uv=None,
        infer=False,
        global_steps=0,
        **kwargs,
    ):
        ret = {}
        content_lengths = kwargs.get("content_lengths")
        rhythm_state = kwargs.get("rhythm_state")
        rhythm_ref_conditioning = kwargs.get("rhythm_ref_conditioning")
        rhythm_apply_override = kwargs.get("rhythm_apply_override")
        rhythm_runtime_overrides = kwargs.get("rhythm_runtime_overrides")
        content_embed, tgt_nonpadding = prepare_content_inputs(
            self,
            content=content,
            content_lengths=content_lengths,
            ret=ret,
        )
        content_embed, tgt_nonpadding, f0, uv = self._run_rhythm_stage(
            ret=ret,
            content=content,
            ref=ref,
            target=target,
            f0=f0,
            uv=uv,
            infer=infer,
            global_steps=global_steps,
            content_embed=content_embed,
            tgt_nonpadding=tgt_nonpadding,
            content_lengths=content_lengths,
            rhythm_state=rhythm_state,
            rhythm_ref_conditioning=rhythm_ref_conditioning,
            rhythm_apply_override=rhythm_apply_override,
            rhythm_runtime_overrides=rhythm_runtime_overrides,
            kwargs=kwargs,
        )

        disable_acoustic_train_path = bool(kwargs.get("disable_acoustic_train_path", False))
        if maybe_short_circuit_acoustic_train(
            self,
            ret=ret,
            infer=infer,
            target=target,
            f0=f0,
            content=content,
            content_embed=content_embed,
            tgt_nonpadding=tgt_nonpadding,
            disable_acoustic_train_path=disable_acoustic_train_path,
        ):
            return ret

        return run_acoustic_path(
            self,
            ret=ret,
            content_embed=content_embed,
            tgt_nonpadding=tgt_nonpadding,
            spk_embed=spk_embed,
            ref=ref,
            f0=f0,
            uv=uv,
            infer=infer,
            global_steps=global_steps,
            forward_kwargs=kwargs,
        )

    def encode_spk_embed(self, x):
        in_nonpadding = (x.abs().sum(dim=-2) > 0).float()[:, None, :]
        # forward encoder
        x_global = self.global_conv_in(x) * in_nonpadding
        global_z_e_x = (
            self.global_encoder(x_global, nonpadding=in_nonpadding) * in_nonpadding
        )
        # group by hidden to phoneme-level
        global_z_e_x = self.temporal_avg_pool(
            x=global_z_e_x, mask=(in_nonpadding == 0)
        )  # (B, C, T) -> (B, C, 1)
        spk_embed = global_z_e_x
        return spk_embed    

    def temporal_avg_pool(self, x, mask=None):
        len_ = (~mask).sum(dim=-1).unsqueeze(-1)
        x = x.masked_fill(mask, 0)
        x = x.sum(dim=-1).unsqueeze(-1)
        out = torch.div(x, len_)
        return out

    def get_prosody(self, encoder_out, ref_mels, ret, infer=False, global_steps=0):
        # get VQ prosody

        if "ref_upsample" not in ret or ret["ref_upsample"] is None:
            B, T, _ = ref_mels.shape  # get batch and frame length
            device = ref_mels.device
            base_ids = (
                torch.arange(T, device=device) // 4 + 1
            )  # [T] → 1,1,1,1,2,2,2,2,…
            ret["ref_upsample"] = base_ids.unsqueeze(0).expand(B, -1)  # [B, T]

        if global_steps > hparams["vq_start"] or infer:
            prosody_embedding, loss, ppl = self.prosody_extractor(
                ref_mels, ret["ref_upsample"], no_vq=False
            )
            ret["vq_loss"] = loss
            ret["ppl"] = ppl
        else:
            prosody_embedding = self.prosody_extractor(
                ref_mels, ret["ref_upsample"], no_vq=True
            )

        # add positional embedding
        positions = self.embed_positions(prosody_embedding[:, :, 0])
        prosody_embedding = self.l1(torch.cat([prosody_embedding, positions], dim=-1))

        # style-to-content attention
        src_key_padding_mask = encoder_out[:, :, 0].eq(self.padding_idx)
        prosody_key_padding_mask = prosody_embedding[:, :, 0].eq(self.padding_idx)
        # assert False,f'encoder_out:{encoder_out.size()},prosody_embedding:{prosody_embedding.size()},src_key_padding_mask:{src_key_padding_mask.size()},prosody_key_padding_mask:{prosody_key_padding_mask.size()}'
        if global_steps < hparams["forcing"]:
            output, guided_loss, attn_emo = self.align(
                encoder_out.transpose(0, 1),
                prosody_embedding.transpose(0, 1),
                src_key_padding_mask,
                prosody_key_padding_mask,
                forcing=True,
            )
        else:
            output, guided_loss, attn_emo = self.align(
                encoder_out.transpose(0, 1),
                prosody_embedding.transpose(0, 1),
                src_key_padding_mask,
                prosody_key_padding_mask,
                forcing=False,
            )

        ret["gloss"] = guided_loss
        ret["attn"] = attn_emo
        return output.transpose(0, 1)

    def forward_decoder(self, decoder_inp, tgt_nonpadding, ret, infer, **kwargs):
        x = decoder_inp  # [B, T, H]
        x = self.decoder(x)
        x = self.mel_out(x)
        # return x* tgt_nonpadding
        return x


class ConanPostnet(nn.Module):
    def __init__(self):
        super().__init__()
        cond_hs = 80 + hparams["hidden_size"]

        self.ln_proj = nn.Linear(cond_hs, hparams["hidden_size"])
        FlowMel = _require_flow_mel()
        self.postflow = FlowMel(
            out_dims=80,
            denoise_fn=Flow_DECODERS[hparams["flow_decoder_type"]](hparams),
            timesteps=hparams["timesteps"],
            K_step=hparams["K_step"],
            loss_type=hparams["flow_loss_type"],
            spec_min=hparams["spec_min"],
            spec_max=hparams["spec_max"],
        )

    def forward(self, tgt_mels, infer, ret, cfg=False, cfg_scale=1.0, noise=None):
        g = self.get_condition(ret)
        x_recon = ret["mel_out"]
        ucond = None
        if cfg and infer:
            B = g.shape[0]
            B_f = B // 2
            if tgt_mels != None:
                tgt_mels = tgt_mels[:B_f]
            x_recon = x_recon[:B_f]
            ucond = g[B_f:]
            g = g[:B_f]
        self.postflow(g, tgt_mels, x_recon, ret, infer, ucond, noise, cfg_scale)

    def get_condition(self, ret):
        x_recon = ret["mel_out"]
        decoder_inp = ret["decoder_inp"]
        g = x_recon.detach()
        B, T, _ = g.shape
        g = torch.cat([g, decoder_inp], dim=-1)
        g = self.ln_proj(g)
        return g


# class TechSinger(RFSinger):
    # def __init__(self, dict_size, hparams, out_dims=None):
    #     super().__init__(dict_size, hparams, out_dims)

    #     cond_hs = 80 + hparams["hidden_size"]
    #     self.ln_proj = nn.Linear(cond_hs, hparams["hidden_size"])
    #     self.postflow = FlowMel(
    #         out_dims=80,
    #         denoise_fn=Flow_DECODERS[hparams["flow_decoder_type"]](hparams),
    #         timesteps=hparams["timesteps"],
    #         K_step=hparams["K_step"],
    #         loss_type=hparams["flow_loss_type"],
    #         spec_min=hparams["spec_min"],
    #         spec_max=hparams["spec_max"],
    #     )

    # def forward(
    #     self,
    #     txt_tokens,
    #     mel2ph=None,
    #     spk_id=None,
    #     f0=None,
    #     uv=None,
    #     note=None,
    #     note_dur=None,
    #     note_type=None,
    #     mix=None,
    #     falsetto=None,
    #     breathy=None,
    #     bubble=None,
    #     strong=None,
    #     weak=None,
    #     pharyngeal=None,
    #     vibrato=None,
    #     glissando=None,
    #     target=None,
    #     cfg=False,
    #     cfg_scale=1.0,
    #     infer=False,
    # ):
    #     ret = {}
    #     encoder_out = self.encoder(txt_tokens)  # [B, T, C]
    #     note_out = self.note_encoder(note, note_dur, note_type)
    #     encoder_out = encoder_out + note_out
    #     src_nonpadding = (txt_tokens > 0).float()[:, :, None]
    #     ret["spk_embed"] = style_embed = self.forward_style_embed(None, spk_id)
    #     tech = self.tech_encoder(
    #         mix, falsetto, breathy, bubble, strong, weak, pharyngeal, vibrato, glissando
    #     )
    #     # add dur
    #     dur_inp = (encoder_out + style_embed) * src_nonpadding
    #     mel2ph = self.forward_dur(dur_inp, mel2ph, txt_tokens, ret)
    #     tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
    #     decoder_inp = expand_states(encoder_out, mel2ph)
    #     in_nonpadding = (mel2ph > 0).float()[:, :, None]

    #     ret["tech"] = tech = expand_states(tech, mel2ph) * tgt_nonpadding
    #     ret["mel2ph"] = mel2ph

    #     # add pitch embed
    #     midi_notes = None
    #     pitch_inp = (decoder_inp + style_embed + tech) * tgt_nonpadding
    #     if infer:
    #         f0, uv = None, None
    #         midi_notes = expand_states(note[:, :, None], mel2ph)
    #     decoder_inp = decoder_inp + self.forward_pitch(
    #         pitch_inp, f0, uv, mel2ph, ret, encoder_out, midi_notes=midi_notes
    #     )
    #     # decoder input
    #     ret["decoder_inp"] = decoder_inp = (
    #         decoder_inp + style_embed + tech
    #     ) * tgt_nonpadding
    #     ret["coarse_mel_out"] = self.forward_decoder(
    #         decoder_inp, tgt_nonpadding, ret, infer=infer
    #     )
    #     ret["tgt_nonpadding"] = tgt_nonpadding

    #     self.forward_post(target, infer, ret, cfg=cfg, cfg_scale=cfg_scale)
    #     return ret

    # def forward_post(self, tgt_mels, infer, ret, cfg=False, cfg_scale=1.0, noise=None):
    #     g = self.get_condition(ret)
    #     x_recon = ret["coarse_mel_out"]
    #     ucond = None
    #     if cfg and infer:
    #         B = g.shape[0]
    #         B_f = B // 2
    #         if tgt_mels != None:
    #             tgt_mels = tgt_mels[:B_f]
    #         x_recon = x_recon[:B_f]
    #         ucond = g[B_f:]
    #         g = g[:B_f]
    #     self.postflow(g, tgt_mels, x_recon, ret, infer, ucond, noise, cfg_scale)

    # def get_condition(self, ret):
    #     x_recon = ret["coarse_mel_out"]
    #     decoder_inp = ret["decoder_inp"]
    #     g = x_recon.detach()
    #     B, T, _ = g.shape
    #     g = torch.cat([g, decoder_inp], dim=-1)
    #     g = self.ln_proj(g)
    #     return g

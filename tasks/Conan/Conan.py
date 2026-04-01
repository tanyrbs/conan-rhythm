from modules.Conan.Conan import Conan, ConanPostnet
from tasks.Conan.base_gen_task import AuxDecoderMIDITask, f0_to_figure
from utils.commons.hparams import hparams
import torch
from utils.commons.ckpt_utils import load_ckpt
from tasks.Conan.dataset import ConanDataset
import torch.nn.functional as F
from utils.commons.tensor_utils import tensors_to_scalars
from utils.audio.pitch.utils import denorm_f0
import math
from modules.tts.iclspeech.multi_window_disc import Discriminator
import torch.nn as nn
import random
from tasks.Conan.rhythm.losses import RhythmLossTargets, build_rhythm_loss_dict
from tasks.Conan.rhythm.metrics import build_rhythm_metric_dict, build_streaming_chunk_metrics
from tasks.Conan.rhythm.streaming_eval import run_chunkwise_streaming_inference
from modules.Conan.rhythm.bridge import resolve_rhythm_apply_mode
from utils.nn.seq_utils import weights_nonzero_speech
from utils.metrics.ssim import ssim


class ConanEmbTask(AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = ConanDataset

    def build_tts_model(self):
        # dict_size = len(self.token_encoder)
        self.model = Conan(0, hparams)
        self.gen_params = [p for p in self.model.parameters() if p.requires_grad]

    def run_model(self, sample):
        with torch.no_grad():
            ref = sample['mels']
            output = self.model.encode_spk_embed(ref.transpose(1, 2)).squeeze(2)
        return {}, {"style_embed": output}


class ConanTask(AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = ConanDataset
        self.mse_loss_fn = torch.nn.MSELoss()
        self.build_disc_model()

    def build_tts_model(self):
        # dict_size = len(self.token_encoder)
        self.model = Conan(0, hparams)
        self.gen_params = [p for p in self.model.parameters() if p.requires_grad]

    def build_disc_model(self):
        disc_win_num = hparams['disc_win_num']
        h = hparams['mel_disc_hidden_size']
        self.mel_disc = Discriminator(
            time_lengths=[32, 64, 128][:disc_win_num],
            freq_length=80, hidden_size=h, kernel=(3, 3)
        )
        self.disc_params = list(self.mel_disc.parameters())

    @staticmethod
    def _parse_optional_bool(value):
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"", "none", "null", "auto", "default"}:
            return None
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Unsupported optional bool value: {value}")

    def _resolve_rhythm_apply_override(self, *, infer: bool, test: bool, explicit=None):
        explicit_value = self._parse_optional_bool(explicit)
        if explicit_value is not None:
            enabled = explicit_value
        else:
            stage = "test" if test else ("valid" if infer else "train")
            enabled = self._parse_optional_bool(hparams.get(f"rhythm_apply_{stage}_override", None))
        if enabled is None:
            return None
        stage = "test" if test else ("valid" if infer else "train")
        start_step = int(hparams.get(f"rhythm_{stage}_render_start_steps", 0) or 0)
        if enabled and stage in {"train", "valid"} and int(self.global_step) < start_step:
            return False
        retimed_target_start = int(hparams.get("rhythm_retimed_target_start_steps", 0) or 0)
        if (
            enabled
            and stage in {"train", "valid"}
            and bool(hparams.get("rhythm_use_retimed_target_if_available", False))
            and int(self.global_step) < retimed_target_start
        ):
            return False
        return enabled

    @staticmethod
    def _resolve_rhythm_target_mode() -> str:
        mode = str(hparams.get("rhythm_dataset_target_mode", "prefer_cache") or "prefer_cache").strip().lower()
        aliases = {
            "auto": "prefer_cache",
            "offline": "cached_only",
            "offline_only": "cached_only",
            "never": "cached_only",
            "runtime": "runtime_only",
            "always": "runtime_only",
        }
        return aliases.get(mode, mode)

    def _resolve_acoustic_target(self, sample, *, apply_rhythm_render: bool, infer: bool, test: bool):
        target = sample["mels"]
        is_retimed = False
        start_step = int(hparams.get("rhythm_retimed_target_start_steps", 0) or 0)
        if (
            bool(apply_rhythm_render)
            and bool(hparams.get("rhythm_use_retimed_target_if_available", False))
            and int(self.global_step) >= start_step
        ):
            retimed = sample.get("rhythm_retimed_mel_tgt")
            if retimed is not None:
                target = retimed
                is_retimed = True
            else:
                require_retimed = bool(hparams.get("rhythm_require_retimed_cache", False))
                if require_retimed or (not test and self._resolve_rhythm_target_mode() == "cached_only"):
                    stage = "test" if test else ("valid" if infer else "train")
                    raise RuntimeError(
                        "Rhythm retimed target is required for the active render path "
                        f"({stage}) but rhythm_retimed_mel_tgt is missing. Re-binarize the dataset."
                    )
        return target, is_retimed

    def _resolve_acoustic_weight(self, sample, *, acoustic_target_is_retimed: bool):
        if not acoustic_target_is_retimed:
            return None
        frame_weight = sample.get("rhythm_retimed_frame_weight")
        confidence = sample.get("rhythm_retimed_target_confidence")
        if confidence is None:
            return frame_weight
        confidence = confidence.float().clamp_min(float(hparams.get("rhythm_retimed_confidence_floor", 0.05)))
        if frame_weight is None:
            return confidence
        return frame_weight.float() * confidence

    @staticmethod
    def _resample_sequence_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.size(1) == target_len:
            return x
        if target_len <= 0:
            return x[:, :0]
        if x.size(1) <= 0:
            return x.new_zeros((x.size(0), target_len, x.size(2)))
        if x.dim() != 3:
            raise ValueError(f"Expected rank-3 tensor [B,T,C], got {tuple(x.shape)}")
        resized = F.interpolate(
            x.transpose(1, 2),
            size=target_len,
            mode='linear',
            align_corners=False,
        )
        return resized.transpose(1, 2)

    @staticmethod
    def _resample_weight_length(weight: torch.Tensor, target_len: int) -> torch.Tensor:
        if weight.size(1) == target_len:
            return weight
        if target_len <= 0:
            return weight[:, :0]
        if weight.size(1) <= 0:
            return weight.new_zeros((weight.size(0), target_len))
        resized = F.interpolate(
            weight.unsqueeze(1),
            size=target_len,
            mode='linear',
            align_corners=False,
        )
        return resized.squeeze(1)

    def _align_acoustic_target_to_output(self, mel_out, acoustic_target, acoustic_weight):
        if mel_out.size(1) == acoustic_target.size(1):
            return mel_out, acoustic_target, acoustic_weight
        if bool(hparams.get("rhythm_resample_retimed_target_to_output", True)):
            acoustic_target = self._resample_sequence_length(acoustic_target, mel_out.size(1))
            if acoustic_weight is not None:
                acoustic_weight = self._resample_weight_length(acoustic_weight.float(), mel_out.size(1))
            return mel_out, acoustic_target, acoustic_weight
        target_len = min(int(mel_out.size(1)), int(acoustic_target.size(1)))
        acoustic_target = acoustic_target[:, :target_len]
        mel_out = mel_out[:, :target_len]
        if acoustic_weight is not None:
            acoustic_weight = acoustic_weight[:, :target_len]
        return mel_out, acoustic_target, acoustic_weight

    @staticmethod
    def _expand_frame_weight(weight: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if weight is None:
            return weights_nonzero_speech(target)
        weight = weight.float()
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        while weight.dim() < target.dim():
            weight = weight.unsqueeze(-1)
        return weight

    def _weighted_l1_loss(self, decoder_output, target, frame_weight):
        loss = F.l1_loss(decoder_output, target, reduction='none')
        weights = self._expand_frame_weight(frame_weight, target)
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)

    def _weighted_mse_loss(self, decoder_output, target, frame_weight):
        loss = F.mse_loss(decoder_output, target, reduction='none')
        weights = self._expand_frame_weight(frame_weight, target)
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)

    def _weighted_ssim_loss(self, decoder_output, target, frame_weight, bias=6.0):
        decoder_output = decoder_output[:, None] + bias
        target = target[:, None] + bias
        ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
        weights = self._expand_frame_weight(frame_weight, target[:, 0])
        return (ssim_loss * weights).sum() / weights.sum().clamp_min(1.0)

    def _add_acoustic_loss(self, mel_out, target, losses, *, frame_weight=None):
        if frame_weight is None:
            self.add_mel_loss(mel_out, target, losses)
            return
        for loss_name, lambd in self.mel_losses.items():
            if loss_name == "l1":
                loss = self._weighted_l1_loss(mel_out, target, frame_weight)
            elif loss_name == "mse":
                loss = self._weighted_mse_loss(mel_out, target, frame_weight)
            elif loss_name == "ssim":
                loss = self._weighted_ssim_loss(mel_out, target, frame_weight)
            else:
                loss = getattr(self, f'{loss_name}_loss')(mel_out, target)
            losses[loss_name] = loss * lambd

    def drop_multi(self, tech, drop_p):
        if torch.rand(1) < drop_p:
            tech = torch.ones_like(tech, dtype=tech.dtype) * 2
        elif torch.rand(1) < drop_p:
            random_tech = torch.rand_like(tech, dtype=torch.float32)
            tech[random_tech < drop_p] = 2
        return tech
            
    def run_model(self, sample, infer=False, test=False, **kwargs):
        # txt_tokens = sample["txt_tokens"]
        # mel2ph = sample["mel2ph"]
        # spk_id = sample["spk_ids"]
        content = sample["content"]
        if 'spk_embed' in sample:
            spk_embed = sample["spk_embed"]
        else:
            spk_embed=None
        f0, uv = sample.get("f0", None), sample.get("uv", None)
        # notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]

        target = sample["mels"]
        if test:
            self.global_step = 200000
        use_reference = (
            test
            or self.global_step >= hparams["random_speaker_steps"]
            or bool(hparams.get("rhythm_force_reference_conditioning", False))
        )
        if use_reference:
            ref = sample['ref_mels']
        else:
            ref = target
        rhythm_apply_override = self._resolve_rhythm_apply_override(
            infer=infer,
            test=test,
            explicit=kwargs.get("rhythm_apply_override"),
        )
        apply_rhythm_render = resolve_rhythm_apply_mode(
            hparams,
            infer=infer,
            override=rhythm_apply_override,
        )
        acoustic_target, acoustic_target_is_retimed = self._resolve_acoustic_target(
            sample,
            apply_rhythm_render=bool(apply_rhythm_render),
            infer=infer,
            test=test,
        )
        acoustic_weight = self._resolve_acoustic_weight(
            sample,
            acoustic_target_is_retimed=acoustic_target_is_retimed,
        )
        rhythm_ref_conditioning = kwargs.get("rhythm_ref_conditioning")
        if rhythm_ref_conditioning is None:
            ref_stats = sample.get("ref_rhythm_stats")
            ref_trace = sample.get("ref_rhythm_trace")
            if ref_stats is not None and ref_trace is not None:
                rhythm_ref_conditioning = {
                    "ref_rhythm_stats": ref_stats,
                    "ref_rhythm_trace": ref_trace,
                }
        # assert False, f'content: {content.shape}, target: {target.shape},spk_embed: {spk_embed.shape}'
        # if not infer:
        #     tech_drop = {
        #         'mix': 0.1,
        #         'falsetto': 0.1,
        #         'breathy': 0.1,
        #         'bubble': 0.1,
        #         'strong': 0.1,
        #         'weak': 0.1,
        #         'glissando': 0.1,
        #         'pharyngeal': 0.1,
        #         'vibrato': 0.1,
        #     }
        #     for tech, drop_p in tech_drop.items():
        #         sample[tech] = self.drop_multi(sample[tech], drop_p)
        
        # mix, falsetto, breathy=sample['mix'], sample['falsetto'], sample['breathy']
        # bubble,strong,weak=sample['bubble'],sample['strong'],sample['weak']
        # pharyngeal, vibrato, glissando = sample['pharyngeal'], sample['vibrato'], sample['glissando']
        output = self.model(content,spk_embed=spk_embed, target=target,ref=ref,
                            f0=f0, uv=uv,
                            infer=infer, global_steps=self.global_step,
                            content_lengths=sample.get("mel_lengths"),
                            ref_lengths=sample.get("ref_mel_lengths"),
                            rhythm_apply_override=rhythm_apply_override,
                            rhythm_state=kwargs.get("rhythm_state"),
                            rhythm_ref_conditioning=rhythm_ref_conditioning)
        
        losses = {}
        output["acoustic_target_mel"] = acoustic_target
        output["acoustic_target_is_retimed"] = bool(acoustic_target_is_retimed)
        output["acoustic_target_weight"] = acoustic_weight
        if acoustic_target_is_retimed:
            mel_out_aligned, acoustic_target, acoustic_weight = self._align_acoustic_target_to_output(
                output['mel_out'],
                acoustic_target,
                acoustic_weight,
            )
            output['mel_out'] = mel_out_aligned
            output["acoustic_target_mel"] = acoustic_target
            output["acoustic_target_weight"] = acoustic_weight
        
        if not test:
            self._add_acoustic_loss(output['mel_out'], acoustic_target, losses, frame_weight=acoustic_weight)
            # self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
            self.add_pitch_loss(output, sample, losses)
            self.add_rhythm_loss(output, sample, losses)
            if hparams['style']:
                if self.global_step > hparams['forcing'] and self.global_step < hparams['random_speaker_steps']:
                    losses['gloss'] = output['gloss']
                if self.global_step > hparams['vq_start']:
                    losses['vq_loss'] = output['vq_loss']
                    losses['ppl'] = output['ppl']
        
        return losses, output

    def add_pitch_loss(self, output, sample, losses):
        # mel2ph = sample['mel2ph']  # [B, T_s]
        content = sample['content']
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (content != -1).float()
        if hparams["f0_gen"] == "diff":
            losses["fdiff"] = output["fdiff"]
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding.sum() * hparams['lambda_uv']
        elif hparams["f0_gen"] == "flow":
            losses["pflow"] = output["pflow"]
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding.sum() * hparams['lambda_uv']
        elif hparams["f0_gen"] == "gmdiff":
            losses["gdiff"] = output["gdiff"]
            losses["mdiff"] = output["mdiff"]
        elif hparams["f0_gen"] == "orig":
            losses["fdiff"] = output["fdiff"]
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding.sum() * hparams['lambda_uv']

    def _build_rhythm_loss_targets(self, output, sample):
        if "rhythm_execution" not in output or output["rhythm_execution"] is None:
            return None
        runtime_teacher = output.get("rhythm_offline_execution")
        algorithmic_teacher = output.get("rhythm_algorithmic_teacher")
        explicit_keys = [
            "rhythm_speech_exec_tgt",
            "rhythm_pause_exec_tgt",
            "rhythm_speech_budget_tgt",
            "rhythm_pause_budget_tgt",
        ]
        if all(key in sample for key in explicit_keys):
            guidance_speech = sample.get("rhythm_guidance_speech_tgt")
            guidance_pause = sample.get("rhythm_guidance_pause_tgt")
            distill_speech = sample.get("rhythm_teacher_speech_exec_tgt")
            distill_pause = sample.get("rhythm_teacher_pause_exec_tgt")
            distill_speech_budget = sample.get("rhythm_teacher_speech_budget_tgt")
            distill_pause_budget = sample.get("rhythm_teacher_pause_budget_tgt")
            distill_allocation = None
            distill_confidence = sample.get("rhythm_teacher_confidence")
            if distill_speech is None and runtime_teacher is not None:
                distill_speech = runtime_teacher.speech_duration_exec.detach()
                distill_pause = runtime_teacher.pause_after_exec.detach()
                distill_speech_budget = runtime_teacher.planner.speech_budget_win.detach()
                distill_pause_budget = runtime_teacher.planner.pause_budget_win.detach()
                distill_confidence = torch.ones_like(distill_speech_budget)
            if distill_speech is None and algorithmic_teacher is not None:
                distill_speech = algorithmic_teacher.speech_exec_tgt.detach()
                distill_pause = algorithmic_teacher.pause_exec_tgt.detach()
                distill_speech_budget = algorithmic_teacher.speech_budget_tgt.detach()
                distill_pause_budget = algorithmic_teacher.pause_budget_tgt.detach()
                distill_allocation = algorithmic_teacher.allocation_tgt.detach()
                distill_confidence = algorithmic_teacher.confidence.detach()
            elif distill_speech is not None and distill_pause is not None:
                distill_allocation = (distill_speech.float() + distill_pause.float())
            return RhythmLossTargets(
                speech_exec_tgt=sample["rhythm_speech_exec_tgt"],
                pause_exec_tgt=sample["rhythm_pause_exec_tgt"],
                speech_budget_tgt=sample["rhythm_speech_budget_tgt"],
                pause_budget_tgt=sample["rhythm_pause_budget_tgt"],
                unit_mask=output["rhythm_unit_batch"].unit_mask,
                plan_local_weight=float(hparams.get("rhythm_plan_local_weight", 0.5)),
                plan_cum_weight=float(hparams.get("rhythm_plan_cum_weight", 1.0)),
                sample_confidence=sample.get("rhythm_target_confidence"),
                guidance_speech_tgt=guidance_speech,
                guidance_pause_tgt=guidance_pause,
                guidance_confidence=sample.get("rhythm_guidance_confidence"),
                distill_speech_tgt=distill_speech,
                distill_pause_tgt=distill_pause,
                distill_speech_budget_tgt=distill_speech_budget,
                distill_pause_budget_tgt=distill_pause_budget,
                distill_allocation_tgt=distill_allocation,
                distill_confidence=distill_confidence,
            )
        if not hparams.get("rhythm_train_identity_fallback", False):
            return None
        unit_batch = output.get("rhythm_unit_batch")
        if unit_batch is None:
            return None
        unit_mask = unit_batch.unit_mask.float()
        speech_exec_tgt = unit_batch.dur_anchor_src.float() * unit_mask
        pause_exec_tgt = torch.zeros_like(speech_exec_tgt)
        speech_budget_tgt = speech_exec_tgt.sum(dim=1, keepdim=True)
        pause_budget_tgt = torch.zeros_like(speech_budget_tgt)
        return RhythmLossTargets(
            speech_exec_tgt=speech_exec_tgt,
            pause_exec_tgt=pause_exec_tgt,
            speech_budget_tgt=speech_budget_tgt,
            pause_budget_tgt=pause_budget_tgt,
            unit_mask=unit_mask,
            plan_local_weight=float(hparams.get("rhythm_plan_local_weight", 0.5)),
            plan_cum_weight=float(hparams.get("rhythm_plan_cum_weight", 1.0)),
            sample_confidence=torch.ones((unit_mask.size(0), 1), device=unit_mask.device),
        )

    def add_rhythm_loss(self, output, sample, losses):
        targets = self._build_rhythm_loss_targets(output, sample)
        if targets is None:
            return
        rhythm_losses = build_rhythm_loss_dict(output["rhythm_execution"], targets)
        losses["rhythm_exec_speech"] = rhythm_losses["rhythm_exec_speech"] * hparams.get("lambda_rhythm_exec_speech", 1.0)
        losses["rhythm_exec_pause"] = rhythm_losses["rhythm_exec_pause"] * hparams.get("lambda_rhythm_exec_pause", 1.0)
        losses["rhythm_budget"] = rhythm_losses["rhythm_budget"] * hparams.get("lambda_rhythm_budget", 0.25)
        losses["rhythm_plan"] = rhythm_losses["rhythm_plan"] * hparams.get("lambda_rhythm_plan", 0.0)
        losses["rhythm_guidance"] = rhythm_losses["rhythm_guidance"] * hparams.get("lambda_rhythm_guidance", 0.0)
        losses["rhythm_distill"] = rhythm_losses["rhythm_distill"] * hparams.get("lambda_rhythm_distill", 0.0)

            
    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output = {}
        loss_weights = {}
        disc_start = self.global_step >= hparams["disc_start_steps"] and hparams['lambda_mel_adv'] > 0
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            loss_output, model_out = self.run_model(sample, infer=False)
            self.model_out_gt = self.model_out = \
                {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
            if disc_start:
                mel_p = model_out['mel_out']
                if hasattr(self.model, 'out2mel'):
                    mel_p = self.model.out2mel(mel_p)
                o_ = self.mel_disc(mel_p)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    loss_output['a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                    loss_weights['a'] = hparams['lambda_mel_adv']
                if pc_ is not None:
                    loss_output['ac'] = self.mse_loss_fn(pc_, pc_.new_ones(pc_.size()))
                    loss_weights['ac'] = hparams['lambda_mel_adv']
        else:
            #######################
            #    Discriminator    #
            #######################
            if disc_start and self.global_step % hparams['disc_interval'] == 0:
                model_out = self.model_out_gt
                mel_g = sample['mels']
                mel_p = model_out['mel_out']
                o = self.mel_disc(mel_g)
                p, pc = o['y'], o['y_c']
                o_ = self.mel_disc(mel_p)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    loss_output["r"] = self.mse_loss_fn(p, p.new_ones(p.size()))
                    loss_output["f"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
                if pc_ is not None:
                    loss_output["rc"] = self.mse_loss_fn(pc, pc.new_ones(pc.size()))
                    loss_output["fc"] = self.mse_loss_fn(pc_, pc_.new_zeros(pc_.size()))
            if len(loss_output) == 0:
                return None
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['content'].size()[0]
        return total_loss, loss_output


    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'], model_out = self.run_model(sample, infer=True)
        outputs['rhythm_metrics'] = tensors_to_scalars(build_rhythm_metric_dict(model_out, sample))
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            sr = hparams["audio_sample_rate"]
            gt_f0 = denorm_f0(sample['f0'], sample["uv"])
            acoustic_target = model_out.get("acoustic_target_mel", sample["mels"])
            acoustic_target_is_retimed = bool(model_out.get("acoustic_target_is_retimed", False))
            if acoustic_target_is_retimed:
                wav_gt = self.vocoder.spec2wav(acoustic_target[0].cpu().numpy())
            else:
                wav_gt = self.vocoder.spec2wav(acoustic_target[0].cpu().numpy(), f0=gt_f0[0].cpu().numpy())
            self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sr)

            wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu().numpy(), f0=model_out["f0_denorm_pred"][0].cpu().numpy())
            self.logger.add_audio(f'wav_pred_{batch_idx}', wav_pred, self.global_step, sr)
            self.plot_mel(batch_idx, acoustic_target, model_out['mel_out'][0], f'mel_{batch_idx}')
            self.logger.add_figure(
                f'f0_{batch_idx}',
                f0_to_figure(None if acoustic_target_is_retimed else gt_f0[0], None, model_out["f0_denorm_pred"][0]),
                self.global_step)
        return outputs
    
    def test_step(self, sample, batch_idx):
        if "ref_mels" not in sample or sample["ref_mels"] is None:
            sample["ref_mels"] = sample["mels"]
        if hparams.get("rhythm_test_chunkwise", True):
            stream_result = run_chunkwise_streaming_inference(
                self,
                sample,
                tokens_per_chunk=int(hparams.get("rhythm_test_tokens_per_chunk", 4)),
            )
            outputs = stream_result.final_output
            mel_pred = stream_result.mel_pred
            stream_metrics = build_streaming_chunk_metrics(stream_result)
        else:
            outputs = self.run_model(sample, infer=True, test=True)[1]
            mel_pred = outputs['mel_out'][0]
            stream_metrics = {}
        item_name = sample['item_name'][0]
        base_fn = f'{item_name.replace(" ", "_")}[P]'

        # Pass through vocoder at once
        wav_pred = self.vocoder.spec2wav(mel_pred.cpu().numpy())

        # Optional: save gt (keep consistent with original implementation)
        if hparams.get('save_gt', False):
            mel_gt = sample['mels'][0].cpu().numpy()
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.saving_result_pool.add_job(
                self.save_result,
                args=[wav_gt, mel_gt,
                    base_fn.replace('[P]', '[G]'),
                    self.gen_dir, None, None,
                    denorm_f0(sample['f0'], sample['uv'])[0].cpu().numpy(),
                    None, None]
            )

        # Save prediction
        self.saving_result_pool.add_job(
            self.save_result,
            args=[wav_pred, mel_pred.cpu().numpy(),
                base_fn, self.gen_dir, None, None,
                denorm_f0(sample['f0'], sample['uv'])[0].cpu().numpy(),
                outputs.get('f0_denorm_pred')[0].cpu().numpy()
                if outputs.get('f0_denorm_pred') is not None else None,
                None]
        )

        result = {}
        result.update(tensors_to_scalars(build_rhythm_metric_dict(outputs, sample)))
        result.update(stream_metrics)
        return result


    def build_optimizer(self, model):
        
        optimizer_gen = torch.optim.AdamW(
            self.gen_params,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])

        optimizer_disc = torch.optim.AdamW(
            self.disc_params,
            lr=hparams['disc_lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            **hparams["discriminator_optimizer_params"]) if len(self.disc_params) > 0 else None

        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return [
            super().build_scheduler( optimizer[0]), # Generator Scheduler
            torch.optim.lr_scheduler.StepLR(optimizer=optimizer[1], # Discriminator Scheduler
                **hparams["discriminator_scheduler_params"]),
        ]

    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.gen_params, hparams['clip_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.disc_params, hparams["clip_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler[0].step(self.global_step // hparams['accumulate_grad_batches'])
            self.scheduler[1].step(self.global_step // hparams['accumulate_grad_batches'])

def self_clone(x):
    if x == None:
        return None
    y = x.clone()
    result = torch.cat((x, y), dim=0)
    return result

class VCPostnetTask(ConanTask):
    def __init__(self):
        super(VCPostnetTask, self).__init__()
        self.drop_prob=hparams['drop_tech_prob']

    def build_model(self):
        self.build_pretrain_model()
        self.model = ConanPostnet()

    def build_pretrain_model(self):
        dict_size = 0
        self.pretrain = Conan(dict_size, hparams)
        from utils.commons.ckpt_utils import load_ckpt
        load_ckpt(self.pretrain, hparams['fs2_ckpt_dir'], 'model', strict=True) 
        for k, v in self.pretrain.named_parameters():
            v.requires_grad = False    
    
    def run_model(self, sample, infer=False, noise=None,test=False):
        content = sample["content"]
        # spk_embed = sample["spk_embed"]
        spk_embed=None
        f0, uv = sample["f0"], sample["uv"]
        # notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        
        target = sample["mels"]
        ref=sample['ref_mels']
        cfg = False
        output = self.pretrain(content,spk_embed=spk_embed, target=target,ref=ref,
                f0=f0, uv=uv,
                infer=infer)

        self.model(target, infer, output, cfg, cfg_scale=hparams['cfg_scale'],  noise=noise)
        losses = {}
        losses["flow"] = output["flow"]
        return losses, output

    def build_optimizer(self, model):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams['lr'],
            betas=(0.9, 0.98),
            eps=1e-9)
        return self.optimizer

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def on_before_optimization(self, opt_idx):
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

# class TechSingerTask(RFSingerTask):

#     def build_tts_model(self):
#         dict_size = len(self.token_encoder)
#         self.model = Conan(dict_size, hparams)
#         self.gen_params = [p for p in self.model.parameters() if p.requires_grad]
            
#     def run_model(self, sample, infer=False):
#         txt_tokens = sample["txt_tokens"]
#         mel2ph = sample["mel2ph"]
#         spk_id = sample["spk_ids"]
#         f0, uv = sample["f0"], sample["uv"]
#         notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        
#         target = sample["mels"]
#         cfg = False
        
#         if infer:
#             cfg = True
#             mix, falsetto, breathy = sample['mix'],sample['falsetto'],sample['breathy']
#             bubble,strong,weak=sample['bubble'],sample['strong'],sample['weak']
#             pharyngeal,vibrato,glissando = sample['pharyngeal'],sample['vibrato'],sample['glissando']
#             umix, ufalsetto, ubreathy = torch.ones_like(mix, dtype=mix.dtype) * 2, torch.ones_like(falsetto, dtype=falsetto.dtype) * 2, torch.ones_like(breathy, dtype=breathy.dtype) * 2
#             ububble, ustrong, uweak = torch.ones_like(mix, dtype=mix.dtype) * 2, torch.ones_like(strong, dtype=strong.dtype) * 2, torch.ones_like(weak, dtype=weak.dtype) * 2
#             upharyngeal, uvibrato, uglissando = torch.ones_like(bubble, dtype=bubble.dtype) * 2, torch.ones_like(vibrato, dtype=vibrato.dtype) * 2, torch.ones_like(glissando, dtype=glissando.dtype) * 2
#             mix = torch.cat((mix, umix), dim=0)
#             falsetto = torch.cat((falsetto, ufalsetto), dim=0)
#             breathy = torch.cat((breathy, ubreathy), dim=0)
#             bubble = torch.cat((bubble, ububble), dim=0)
#             strong = torch.cat((strong, ustrong), dim=0)
#             weak = torch.cat((weak, uweak), dim=0)
#             pharyngeal = torch.cat((pharyngeal, upharyngeal), dim=0)
#             vibrato = torch.cat((vibrato, uvibrato), dim=0)
#             glissando = torch.cat((glissando, uglissando), dim=0)
            
#             txt_tokens = self_clone(txt_tokens)
#             mel2ph = self_clone(mel2ph)
#             spk_id = self_clone(spk_id)
#             f0 = self_clone(f0)
#             uv = self_clone(uv)
#             notes = self_clone(notes)
#             note_durs = self_clone(note_durs)
#             note_types = self_clone(note_types)
            
#             output = self.model(txt_tokens, mel2ph=mel2ph, spk_id=spk_id, f0=f0, uv=uv, 
#                                 note=notes, note_dur=note_durs, note_type=note_types,
#                                 mix=mix, falsetto=falsetto, breathy=breathy,
#                                 bubble=bubble, strong=strong, weak=weak,
#                                 pharyngeal=pharyngeal, vibrato=vibrato, glissando=glissando, target=target, cfg=cfg, cfg_scale=1.0,
#                                 infer=infer)
#         else:
#             tech_drop = {
#                 'mix': 0.1,
#                 'falsetto': 0.1,
#                 'breathy': 0.1,
#                 'bubble': 0.1,
#                 'strong': 0.1,
#                 'weak': 0.1,
#                 'glissando': 0.1,
#                 'pharyngeal': 0.1,
#                 'vibrato': 0.1,
#             }
#             for tech, drop_p in tech_drop.items():
#                 sample[tech] = self.drop_multi(sample[tech], drop_p)
#             mix, falsetto, breathy = sample['mix'],sample['falsetto'],sample['breathy']
#             bubble,strong,weak=sample['bubble'],sample['strong'],sample['weak']
#             pharyngeal,vibrato,glissando = sample['pharyngeal'],sample['vibrato'],sample['glissando']
#             output = self.model(txt_tokens, mel2ph=mel2ph, spk_id=spk_id, f0=f0, uv=uv, 
#                                 note=notes, note_dur=note_durs, note_type=note_types,
#                                 mix=mix, falsetto=falsetto, breathy=breathy,
#                                 bubble=bubble, strong=strong, weak=weak,
#                                 pharyngeal=pharyngeal, vibrato=vibrato, glissando=glissando, target=target, cfg=cfg,
#                                 infer=infer)
        
#         losses = {}
#         if not infer:
#             self.add_mel_loss(output['coarse_mel_out'], target, losses)
#             self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
#             self.add_pitch_loss(output, sample, losses)
#         losses["flow"] = output["flow"]
#         return losses, output

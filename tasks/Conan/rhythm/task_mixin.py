"""Rhythm-heavy Conan task logic extracted from tasks/Conan/Conan.py.

The public ConanTask class remains stable, while this mixin isolates rhythm
contracts, runtime target handling, and validation helpers for easier
debugging and focused inspection.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tasks.Conan.base_gen_task import f0_to_figure
from tasks.Conan.rhythm.loss_routing import route_conan_optimizer_losses, update_public_loss_aliases
from tasks.Conan.rhythm.losses import RhythmLossTargets, build_rhythm_loss_dict
from tasks.Conan.rhythm.metrics import build_rhythm_metric_dict, build_streaming_chunk_metrics
from tasks.Conan.rhythm.runtime_modes import (
    build_rhythm_ref_conditioning,
    collect_planner_runtime_outputs,
    resolve_acoustic_target_post_model as resolve_task_acoustic_target_post_model,
    resolve_task_runtime_state,
)
from tasks.Conan.rhythm.streaming_eval import run_chunkwise_streaming_inference
from tasks.Conan.rhythm.task_config import (
    parse_task_optional_bool,
    resolve_task_distill_surface,
    resolve_task_pause_boundary_weight,
    resolve_task_primary_target_surface,
    resolve_task_retimed_target_mode,
    resolve_task_target_mode,
    validate_rhythm_training_hparams,
)
from tasks.Conan.rhythm.targets import (
    DistillConfidenceBundle,
    RhythmTargetBuildConfig,
    build_identity_rhythm_loss_targets,
    build_rhythm_loss_targets_from_sample,
    scale_rhythm_loss_terms,
)
from modules.Conan.rhythm.policy import (
    resolve_apply_override,
    resolve_cumplan_lambda,
    should_optimize_render_params,
    use_strict_mainline,
)
from modules.Conan.rhythm.source_boundary import resolve_boundary_score_unit
from modules.Conan.rhythm.stages import (
    resolve_runtime_dual_mode_teacher_enable,
)
from utils.audio.pitch.utils import denorm_f0
from utils.commons.hparams import hparams
from utils.commons.tensor_utils import tensors_to_scalars
from utils.metrics.ssim import ssim
from utils.nn.seq_utils import weights_nonzero_speech


class RhythmConanTaskMixin:
    @staticmethod
    def _validate_rhythm_training_hparams():
        validate_rhythm_training_hparams(hparams)

    def _collect_rhythm_gen_params(self):
        if self.model is None or not getattr(self.model, "rhythm_enable_v2", False):
            return []
        params = []
        if getattr(self.model, "rhythm_module", None) is not None:
            for name, param in self.model.rhythm_module.named_parameters():
                if self._should_skip_rhythm_named_param(name):
                    continue
                params.append(param)
        if bool(hparams.get("rhythm_optimize_pause_state", False)) and getattr(self.model, "rhythm_pause_state", None) is not None:
            params.append(self.model.rhythm_pause_state)
        optimize_render_params = should_optimize_render_params(hparams)
        if optimize_render_params and getattr(self.model, "rhythm_render_phase_mlp", None) is not None:
            params.extend(list(self.model.rhythm_render_phase_mlp.parameters()))
        if optimize_render_params and getattr(self.model, "rhythm_render_phase_gain", None) is not None:
            params.append(self.model.rhythm_render_phase_gain)
        dedup = []
        seen = set()
        for param in params:
            if param is None or not getattr(param, "requires_grad", False):
                continue
            key = id(param)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(param)
        return dedup

    @staticmethod
    def _should_skip_rhythm_named_param(name: str) -> bool:
        if bool(hparams.get("rhythm_train_offline_confidence_heads", False)):
            return False
        normalized = str(name or "")
        return (
            "offline_teacher.confidence_trunk" in normalized
            or "offline_teacher.confidence_heads" in normalized
        )

    def _collect_offline_teacher_gen_params(self):
        if self.model is None or not getattr(self.model, "rhythm_enable_v2", False):
            return []
        rhythm_module = getattr(self.model, "rhythm_module", None)
        if rhythm_module is None:
            return []
        params = []
        if getattr(rhythm_module, "unit_embedding", None) is not None:
            params.extend(list(rhythm_module.unit_embedding.parameters()))
        if getattr(rhythm_module, "reference_descriptor", None) is not None:
            params.extend(list(rhythm_module.reference_descriptor.parameters()))
        if getattr(rhythm_module, "offline_teacher", None) is not None:
            for name, param in rhythm_module.offline_teacher.named_parameters():
                if self._should_skip_rhythm_named_param(f"offline_teacher.{name}"):
                    continue
                params.append(param)
        dedup = []
        seen = set()
        for param in params:
            if param is None or not getattr(param, "requires_grad", False):
                continue
            key = id(param)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(param)
        return dedup

    @staticmethod
    def _get_rhythm_cumplan_lambda() -> float:
        return resolve_cumplan_lambda(hparams)

    @staticmethod
    def _resolve_rhythm_pause_boundary_weight() -> float:
        return resolve_task_pause_boundary_weight(hparams)

    @staticmethod
    def _parse_optional_bool(value):
        return parse_task_optional_bool(value)

    def _resolve_rhythm_apply_override(self, *, infer: bool, test: bool, explicit=None, current_step=None):
        effective_step = int(self.global_step if current_step is None else current_step)
        return resolve_apply_override(
            hparams,
            infer=infer,
            test=test,
            explicit=explicit,
            current_step=effective_step,
        )

    @staticmethod
    def _resolve_rhythm_target_mode() -> str:
        return resolve_task_target_mode(hparams)

    @staticmethod
    def _resolve_rhythm_primary_target_surface() -> str:
        return resolve_task_primary_target_surface(hparams)

    @staticmethod
    def _resolve_rhythm_distill_surface() -> str:
        return resolve_task_distill_surface(hparams)

    @staticmethod
    def _use_rhythm_strict_mainline() -> bool:
        return use_strict_mainline(hparams)

    @staticmethod
    def _use_runtime_dual_mode_teacher() -> bool:
        return resolve_runtime_dual_mode_teacher_enable(hparams, infer=False)

    @staticmethod
    def _collect_rhythm_source_cache(sample, *, prefix: str = ""):
        keys = (
            "content_units",
            "dur_anchor_src",
            "open_run_mask",
            "sealed_mask",
            "sep_hint",
            "boundary_confidence",
        )
        payload = {}
        for key in keys:
            sample_key = f"{prefix}{key}"
            value = sample.get(sample_key)
            if value is not None:
                payload[key] = value
        return payload or None

    @staticmethod
    def _slice_rhythm_surface_to_student(
        *,
        speech_exec,
        pause_exec,
        student_units: int,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor,
    ):
        speech_exec = speech_exec[:, :student_units]
        pause_exec = pause_exec[:, :student_units]
        speech_budget = speech_exec.float().sum(dim=1, keepdim=True)
        pause_budget = pause_exec.float().sum(dim=1, keepdim=True)
        allocation = (speech_exec.float() + pause_exec.float()) * unit_mask[:, :student_units].float()
        prefix_clock, prefix_backlog = RhythmConanTaskMixin._build_prefix_carry_from_exec(
            speech_exec,
            pause_exec,
            dur_anchor_src[:, :student_units],
            unit_mask[:, :student_units],
        )
        return (
            speech_exec,
            pause_exec,
            speech_budget,
            pause_budget,
            allocation,
            prefix_clock,
            prefix_backlog,
        )

    @staticmethod
    def _build_prefix_carry_from_exec(speech_exec, pause_exec, dur_anchor_src, unit_mask):
        unit_mask = unit_mask.float()
        prefix_clock = torch.cumsum(
            ((speech_exec.float() + pause_exec.float()) - dur_anchor_src.float()) * unit_mask,
            dim=1,
        ) * unit_mask
        prefix_backlog = prefix_clock.clamp_min(0.0) * unit_mask
        return prefix_clock, prefix_backlog

    @staticmethod
    def _normalize_distill_confidence(distill_confidence, *, batch_size: int, device: torch.device):
        floor = float(hparams.get("rhythm_distill_confidence_floor", 0.05))
        power = float(hparams.get("rhythm_distill_confidence_power", 1.0))
        if distill_confidence is None:
            confidence = torch.ones((batch_size, 1), device=device)
        else:
            confidence = distill_confidence.detach().float().reshape(batch_size, -1)[:, :1].to(device=device)
        confidence = confidence.clamp(min=floor, max=1.0)
        if abs(power - 1.0) > 1e-8:
            confidence = confidence.pow(power)
        return confidence

    @staticmethod
    def _normalize_component_distill_confidence(
        component_confidence,
        *,
        fallback_confidence: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ):
        if component_confidence is None:
            return fallback_confidence
        return RhythmConanTaskMixin._normalize_distill_confidence(
            component_confidence,
            batch_size=batch_size,
            device=device,
        )

    @staticmethod
    def _build_rhythm_target_build_config() -> RhythmTargetBuildConfig:
        return RhythmTargetBuildConfig(
            primary_target_surface=RhythmConanTaskMixin._resolve_rhythm_primary_target_surface(),
            distill_surface=RhythmConanTaskMixin._resolve_rhythm_distill_surface(),
            lambda_guidance=float(hparams.get("lambda_rhythm_guidance", 0.0) or 0.0),
            lambda_distill=float(hparams.get("lambda_rhythm_distill", 0.0) or 0.0),
            distill_budget_weight=float(hparams.get("rhythm_distill_budget_weight", 0.5)),
            distill_allocation_weight=float(hparams.get("rhythm_distill_allocation_weight", 0.5)),
            distill_prefix_weight=float(hparams.get("rhythm_distill_prefix_weight", 0.25)),
            distill_speech_shape_weight=float(hparams.get("rhythm_distill_speech_shape_weight", 0.0)),
            distill_pause_shape_weight=float(hparams.get("rhythm_distill_pause_shape_weight", 0.0)),
            plan_local_weight=float(hparams.get("rhythm_plan_local_weight", 0.5)),
            plan_cum_weight=float(hparams.get("rhythm_plan_cum_weight", 1.0)),
            pause_boundary_weight=RhythmConanTaskMixin._resolve_rhythm_pause_boundary_weight(),
            feasible_debt_weight=float(hparams.get("rhythm_feasible_debt_weight", 0.05)),
        )

    @staticmethod
    def _build_runtime_distill_confidence_bundle(output) -> DistillConfidenceBundle:
        return DistillConfidenceBundle(
            shared=output.get("rhythm_offline_confidence"),
            exec=output.get("rhythm_offline_confidence_exec"),
            budget=output.get("rhythm_offline_confidence_budget"),
            prefix=output.get("rhythm_offline_confidence_prefix"),
            allocation=output.get("rhythm_offline_confidence_allocation"),
        )

    @staticmethod
    def _resolve_retimed_target_mode() -> str:
        return resolve_task_retimed_target_mode(hparams)

    def _resolve_acoustic_target_post_model(
        self,
        sample,
        model_out,
        *,
        apply_rhythm_render: bool,
        infer: bool,
        test: bool,
        current_step=None,
    ):
        return resolve_task_acoustic_target_post_model(
            sample,
            model_out,
            hparams=hparams,
            global_step=int(self.global_step),
            apply_rhythm_render=apply_rhythm_render,
            infer=infer,
            test=test,
            current_step=current_step,
        )

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

    def _run_offline_teacher_model(self, sample, *, infer: bool, test: bool, **kwargs):
        if infer or test:
            return None
        if not getattr(self.model, "rhythm_enable_v2", False):
            return None
        rhythm_module = getattr(self.model, "rhythm_module", None)
        if rhythm_module is None or getattr(rhythm_module, "offline_teacher", None) is None:
            raise RuntimeError(
                "rhythm_teacher_only_stage requires the learned offline teacher runtime branch to be instantiated."
            )
        source_cache = self._collect_rhythm_source_cache(sample)
        if source_cache is None:
            raise RuntimeError("rhythm_teacher_only_stage requires cached source-unit fields in the batch.")
        unit_batch = self.model.rhythm_unit_frontend.from_precomputed(
            content_units=source_cache["content_units"],
            dur_anchor_src=source_cache["dur_anchor_src"],
            unit_mask=source_cache.get("unit_mask"),
            open_run_mask=source_cache.get("open_run_mask"),
            sealed_mask=source_cache.get("sealed_mask"),
            sep_hint=source_cache.get("sep_hint"),
            boundary_confidence=source_cache.get("boundary_confidence"),
        )
        rhythm_ref_conditioning = build_rhythm_ref_conditioning(
            sample,
            explicit=kwargs.get("rhythm_ref_conditioning"),
        )
        if rhythm_ref_conditioning is None:
            raise RuntimeError("rhythm_teacher_only_stage requires ref_rhythm_stats and ref_rhythm_trace.")
        teacher_scale = self.model._resolve_rhythm_source_boundary_scale(
            infer=False,
            global_steps=int(self.global_step),
            teacher=True,
        )
        pause_ratio = self.model._resolve_rhythm_pause_topk_ratio(
            infer=False,
            global_steps=int(self.global_step),
        )
        execution, confidence = rhythm_module.forward_teacher(
            content_units=unit_batch.content_units,
            dur_anchor_src=unit_batch.dur_anchor_src,
            ref_conditioning=rhythm_ref_conditioning,
            unit_mask=unit_batch.unit_mask,
            open_run_mask=torch.zeros_like(unit_batch.content_units),
            sealed_mask=torch.ones_like(unit_batch.unit_mask).float(),
            sep_hint=unit_batch.sep_hint,
            boundary_confidence=unit_batch.boundary_confidence,
            projector_pause_topk_ratio_override=pause_ratio,
            source_boundary_scale_override=teacher_scale,
        )
        output = {
            "rhythm_execution": execution,
            "rhythm_unit_batch": unit_batch,
            "disable_acoustic_train_path": 1.0,
            "rhythm_schedule_only_stage": 0.0,
            "rhythm_teacher_only_stage": 1.0,
            "rhythm_offline_confidence": confidence.get("overall") if isinstance(confidence, dict) else None,
            "rhythm_offline_confidence_exec": confidence.get("exec") if isinstance(confidence, dict) else None,
            "rhythm_offline_confidence_budget": confidence.get("budget") if isinstance(confidence, dict) else None,
            "rhythm_offline_confidence_prefix": confidence.get("prefix") if isinstance(confidence, dict) else None,
            "rhythm_offline_confidence_allocation": confidence.get("allocation") if isinstance(confidence, dict) else None,
        }
        output.update(collect_planner_runtime_outputs(execution))
        losses = {}
        self.add_rhythm_loss(output, sample, losses)
        route_conan_optimizer_losses(
            losses,
            mel_loss_names=tuple(self.mel_losses.keys()),
            hparams=hparams,
            schedule_only_stage=False,
        )
        update_public_loss_aliases(losses, mel_loss_names=tuple(self.mel_losses.keys()))
        return losses, output

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
        runtime_state = resolve_task_runtime_state(
            hparams,
            global_step=int(self.global_step),
            infer=infer,
            test=test,
            explicit_apply_override=kwargs.get("rhythm_apply_override"),
            has_f0=f0 is not None,
            has_uv=uv is not None,
        )
        effective_global_step = runtime_state.effective_global_step
        ref = sample["ref_mels"] if runtime_state.use_reference else target
        rhythm_apply_override = runtime_state.rhythm_apply_override
        apply_rhythm_render = runtime_state.apply_rhythm_render
        retimed_stage_active = runtime_state.retimed_stage_active
        disable_source_pitch_supervision = runtime_state.disable_source_pitch_supervision
        if disable_source_pitch_supervision and retimed_stage_active and not self._warned_retimed_pitch_supervision:
            print("| Rhythm V2: retimed canvas active, disabling source-aligned pitch supervision for this run.")
            self._warned_retimed_pitch_supervision = True
        rhythm_ref_conditioning = build_rhythm_ref_conditioning(
            sample,
            explicit=kwargs.get("rhythm_ref_conditioning"),
        )
        stage = runtime_state.stage
        teacher_as_main = runtime_state.teacher_as_main
        if stage == "teacher_offline" and not teacher_as_main:
            teacher_only_result = self._run_offline_teacher_model(
                sample,
                infer=infer,
                test=test,
                rhythm_ref_conditioning=rhythm_ref_conditioning,
            )
            if teacher_only_result is not None:
                return teacher_only_result
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
        disable_acoustic_train_path = runtime_state.disable_acoustic_train_path
        disable_source_pitch_supervision = bool(disable_source_pitch_supervision or disable_acoustic_train_path)
        if disable_acoustic_train_path and rhythm_ref_conditioning is not None:
            ref = None
        runtime_offline_source_cache = None
        if self._use_runtime_dual_mode_teacher() and not bool(infer):
            runtime_offline_source_cache = self._collect_rhythm_source_cache(sample, prefix="rhythm_offline_")
        output = self.model(content,spk_embed=spk_embed, target=target,ref=ref,
                            f0=None if disable_source_pitch_supervision else f0,
                            uv=None if disable_source_pitch_supervision else uv,
                            infer=infer, global_steps=effective_global_step,
                            content_lengths=sample.get("mel_lengths"),
                            ref_lengths=sample.get("ref_mel_lengths"),
                            rhythm_apply_override=rhythm_apply_override,
                            rhythm_state=kwargs.get("rhythm_state"),
                            rhythm_ref_conditioning=rhythm_ref_conditioning,
                            rhythm_source_cache=self._collect_rhythm_source_cache(sample),
                            rhythm_offline_source_cache=runtime_offline_source_cache,
                            disable_acoustic_train_path=disable_acoustic_train_path)
        acoustic_target, acoustic_target_is_retimed, acoustic_weight, acoustic_target_source = self._resolve_acoustic_target_post_model(
            sample,
            output,
            apply_rhythm_render=bool(apply_rhythm_render),
            infer=infer,
            test=test,
            current_step=effective_global_step,
        )
        output.update(collect_planner_runtime_outputs(output.get("rhythm_execution")))
        
        losses = {}
        output["acoustic_target_mel"] = acoustic_target
        output["acoustic_target_is_retimed"] = bool(acoustic_target_is_retimed)
        output["acoustic_target_weight"] = acoustic_weight
        output["acoustic_target_source"] = acoustic_target_source
        output["rhythm_pitch_supervision_disabled"] = float(disable_source_pitch_supervision)
        output["disable_acoustic_train_path"] = float(disable_acoustic_train_path)
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
            schedule_only_stage = stage == "legacy_schedule_only"
            output["rhythm_schedule_only_stage"] = float(schedule_only_stage)
            output["rhythm_stage"] = stage
            if not schedule_only_stage and not disable_acoustic_train_path:
                self._add_acoustic_loss(output['mel_out'], acoustic_target, losses, frame_weight=acoustic_weight)
                # self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
                self.add_pitch_loss(output, sample, losses)
            self.add_rhythm_loss(output, sample, losses)
            if (
                hparams['style']
                and not schedule_only_stage
                and not getattr(self.model, "rhythm_minimal_style_only", False)
            ):
                if (
                    self.global_step > hparams['forcing']
                    and self.global_step < hparams['random_speaker_steps']
                    and 'gloss' in output
                ):
                    losses['gloss'] = output['gloss']
                if self.global_step > hparams['vq_start'] and 'vq_loss' in output and 'ppl' in output:
                    losses['vq_loss'] = output['vq_loss']
                    losses['ppl'] = output['ppl']
            route_conan_optimizer_losses(
                losses,
                mel_loss_names=tuple(self.mel_losses.keys()),
                hparams=hparams,
                schedule_only_stage=schedule_only_stage,
            )
            update_public_loss_aliases(losses, mel_loss_names=tuple(self.mel_losses.keys()))
        
        return losses, output

    def add_pitch_loss(self, output, sample, losses):
        # mel2ph = sample['mel2ph']  # [B, T_s]
        if bool(output.get("rhythm_pitch_supervision_disabled", False)):
            return
        content = output.get("content", sample['content'])
        f0 = output.get("retimed_f0_tgt", sample.get('f0'))
        uv = output.get("retimed_uv_tgt", sample.get('uv'))
        if f0 is None or uv is None:
            return
        nonpadding = output.get("tgt_nonpadding")
        if nonpadding is None:
            lengths = sample.get("mel_lengths")
            if lengths is not None:
                steps = torch.arange(content.size(1), device=content.device)[None, :]
                nonpadding = (steps < lengths[:, None].long()).float()
            else:
                nonpadding = (content != 0).float()
        else:
            if nonpadding.dim() == 3:
                nonpadding = nonpadding.squeeze(-1)
            nonpadding = nonpadding.float()
        nonpadding = nonpadding.to(device=uv.device)
        nonpadding_sum = nonpadding.sum().clamp_min(1.0)
        if hparams["f0_gen"] == "diff":
            losses["fdiff"] = output["fdiff"]
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding_sum * hparams['lambda_uv']
        elif hparams["f0_gen"] == "flow":
            losses["pflow"] = output["pflow"]
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding_sum * hparams['lambda_uv']
        elif hparams["f0_gen"] == "gmdiff":
            losses["gdiff"] = output["gdiff"]
            losses["mdiff"] = output["mdiff"]
        elif hparams["f0_gen"] == "orig":
            losses["fdiff"] = output["fdiff"]
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding_sum * hparams['lambda_uv']

    @staticmethod
    def _slice_runtime_teacher_execution(execution, *, teacher_units: int):
        if execution is None:
            return None
        current_units = int(execution.speech_duration_exec.size(1))
        if teacher_units >= current_units:
            return execution
        planner = getattr(execution, "planner", None)
        speech_exec = execution.speech_duration_exec[:, :teacher_units]
        blank_exec = getattr(execution, "blank_duration_exec", execution.pause_after_exec)[:, :teacher_units]
        pause_exec = execution.pause_after_exec[:, :teacher_units]
        speech_budget = speech_exec.float().sum(dim=1, keepdim=True)
        pause_budget = blank_exec.float().sum(dim=1, keepdim=True)
        zero_delta = speech_budget.new_zeros(speech_budget.shape)
        boundary_score = resolve_boundary_score_unit(planner, fallback=speech_exec.new_zeros(speech_exec.shape))
        if torch.is_tensor(boundary_score):
            boundary_score = boundary_score[:, :teacher_units]
        planner_view = SimpleNamespace(
            speech_budget_win=speech_budget,
            pause_budget_win=pause_budget,
            blank_budget_win=pause_budget,
            boundary_score_unit=boundary_score,
            boundary_latent=boundary_score,
            source_boundary_cue=(
                planner.source_boundary_cue[:, :teacher_units]
                if planner is not None and torch.is_tensor(getattr(planner, "source_boundary_cue", None))
                else getattr(planner, "source_boundary_cue", None)
                if planner is not None
                else None
            ),
            feasible_speech_budget_delta=zero_delta,
            feasible_pause_budget_delta=zero_delta,
            feasible_total_budget_delta=zero_delta,
        )
        return SimpleNamespace(
            speech_duration_exec=speech_exec,
            blank_duration_exec=blank_exec,
            pause_after_exec=pause_exec,
            planner=planner_view,
        )

    def _build_runtime_teacher_supervision_targets(self, output, sample):
        runtime_teacher = output.get("rhythm_offline_execution")
        offline_unit_batch = output.get("rhythm_offline_unit_batch")
        unit_batch = output.get("rhythm_unit_batch")
        if runtime_teacher is None:
            return None
        if offline_unit_batch is not None and all(
            key in sample
            for key in (
                "rhythm_offline_teacher_speech_exec_tgt",
                "rhythm_offline_teacher_pause_exec_tgt",
            )
        ):
            batch_for_targets = offline_unit_batch
            speech_exec_key = "rhythm_offline_teacher_speech_exec_tgt"
            pause_exec_key = "rhythm_offline_teacher_pause_exec_tgt"
            speech_budget_key = "rhythm_offline_teacher_speech_budget_tgt"
            pause_budget_key = "rhythm_offline_teacher_pause_budget_tgt"
        else:
            batch_for_targets = unit_batch if unit_batch is not None else offline_unit_batch
            speech_exec_key = "rhythm_teacher_speech_exec_tgt"
            pause_exec_key = (
                "rhythm_teacher_pause_exec_tgt"
                if "rhythm_teacher_pause_exec_tgt" in sample
                else "rhythm_teacher_blank_exec_tgt"
            )
            speech_budget_key = "rhythm_teacher_speech_budget_tgt"
            pause_budget_key = (
                "rhythm_teacher_pause_budget_tgt"
                if "rhythm_teacher_pause_budget_tgt" in sample
                else "rhythm_teacher_blank_budget_tgt"
            )
        if batch_for_targets is None or not all(key in sample for key in (speech_exec_key, pause_exec_key)):
            return None
        teacher_units = min(
            int(runtime_teacher.speech_duration_exec.size(1)),
            int(batch_for_targets.dur_anchor_src.size(1)),
            int(sample[speech_exec_key].size(1)),
            int(sample[pause_exec_key].size(1)),
        )
        if teacher_units <= 0:
            return None
        unit_mask = batch_for_targets.unit_mask
        if unit_mask is None:
            unit_mask = batch_for_targets.dur_anchor_src.gt(0).float()
        speech_exec_tgt = sample[speech_exec_key][:, :teacher_units].float()
        pause_exec_tgt = sample[pause_exec_key][:, :teacher_units].float()
        speech_budget_tgt = sample.get(speech_budget_key)
        pause_budget_tgt = sample.get(pause_budget_key)
        if speech_budget_tgt is None:
            speech_budget_tgt = speech_exec_tgt.sum(dim=1, keepdim=True)
        else:
            speech_budget_tgt = speech_budget_tgt[:, :1].float()
        if pause_budget_tgt is None:
            pause_budget_tgt = pause_exec_tgt.sum(dim=1, keepdim=True)
        else:
            pause_budget_tgt = pause_budget_tgt[:, :1].float()
        sample_confidence = sample.get(
            "rhythm_offline_teacher_confidence",
            sample.get("rhythm_teacher_confidence", sample.get("rhythm_target_confidence")),
        )
        if isinstance(sample_confidence, torch.Tensor):
            sample_confidence = sample_confidence.detach()
        teacher_execution = self._slice_runtime_teacher_execution(runtime_teacher, teacher_units=teacher_units)
        targets = RhythmLossTargets(
            speech_exec_tgt=speech_exec_tgt,
            pause_exec_tgt=pause_exec_tgt,
            speech_budget_tgt=speech_budget_tgt,
            pause_budget_tgt=pause_budget_tgt,
            unit_mask=unit_mask[:, :teacher_units],
            dur_anchor_src=batch_for_targets.dur_anchor_src[:, :teacher_units],
            plan_local_weight=float(hparams.get("rhythm_plan_local_weight", 0.5)),
            plan_cum_weight=float(hparams.get("rhythm_plan_cum_weight", 1.0)),
            sample_confidence=sample_confidence,
            pause_boundary_weight=self._resolve_rhythm_pause_boundary_weight(),
            feasible_debt_weight=float(hparams.get("rhythm_feasible_debt_weight", 0.05)),
        )
        return teacher_execution, targets

    def _build_rhythm_mainline_targets(self, output, sample):
        unit_batch = output.get("rhythm_unit_batch")
        if unit_batch is None:
            return None
        return build_rhythm_loss_targets_from_sample(
            sample=sample,
            unit_batch=unit_batch,
            config=self._build_rhythm_target_build_config(),
            normalize_distill_confidence=self._normalize_distill_confidence,
            normalize_component_confidence=self._normalize_component_distill_confidence,
            build_prefix_carry_from_exec=self._build_prefix_carry_from_exec,
            slice_rhythm_surface_to_student=self._slice_rhythm_surface_to_student,
        )

    def _build_rhythm_loss_targets(self, output, sample):
        if "rhythm_execution" not in output or output["rhythm_execution"] is None:
            return None
        unit_batch = output.get("rhythm_unit_batch")
        if unit_batch is None:
            return None
        runtime_teacher = output.get("rhythm_offline_execution")
        algorithmic_teacher = output.get("rhythm_algorithmic_teacher")
        targets = build_rhythm_loss_targets_from_sample(
            sample=sample,
            unit_batch=unit_batch,
            config=self._build_rhythm_target_build_config(),
            runtime_teacher=runtime_teacher,
            algorithmic_teacher=algorithmic_teacher,
            offline_confidences=self._build_runtime_distill_confidence_bundle(output),
            normalize_distill_confidence=self._normalize_distill_confidence,
            normalize_component_confidence=self._normalize_component_distill_confidence,
            build_prefix_carry_from_exec=self._build_prefix_carry_from_exec,
            slice_rhythm_surface_to_student=self._slice_rhythm_surface_to_student,
        )
        if targets is not None:
            return targets
        if not hparams.get("rhythm_train_identity_fallback", False):
            return None
        return build_identity_rhythm_loss_targets(
            unit_batch=unit_batch,
            config=self._build_rhythm_target_build_config(),
        )

    def add_rhythm_loss(self, output, sample, losses):
        targets = self._build_rhythm_loss_targets(output, sample)
        if targets is None:
            return
        rhythm_execution = output["rhythm_execution"]
        if rhythm_execution is not None and "rhythm_pause_exec_surrogate_used" not in output:
            output["rhythm_pause_exec_surrogate_used"] = 0.0
        rhythm_losses = build_rhythm_loss_dict(rhythm_execution, targets)
        losses.update(
            scale_rhythm_loss_terms(
                rhythm_losses,
                hparams=hparams,
                cumplan_lambda=self._get_rhythm_cumplan_lambda(),
            )
        )

        runtime_teacher = output.get("rhythm_offline_execution")
        lambda_teacher_aux = float(hparams.get("lambda_rhythm_teacher_aux", 0.0) or 0.0)
        if runtime_teacher is not None and lambda_teacher_aux > 0.0:
            teacher_bundle = self._build_runtime_teacher_supervision_targets(output, sample)
            if teacher_bundle is not None:
                teacher_execution, teacher_targets = teacher_bundle
                teacher_losses = build_rhythm_loss_dict(teacher_execution, teacher_targets)
                teacher_exec = (
                    teacher_losses["rhythm_exec_speech"] * hparams.get("lambda_rhythm_exec_speech", 1.0)
                    + teacher_losses["rhythm_exec_pause"] * hparams.get("lambda_rhythm_exec_pause", 1.0)
                )
                teacher_state = (
                    teacher_losses["rhythm_budget"] * hparams.get("lambda_rhythm_budget", 0.25)
                    + teacher_losses["rhythm_cumplan"] * self._get_rhythm_cumplan_lambda()
                )
                teacher_aux = teacher_exec + teacher_state
                losses["rhythm_distill"] = losses["rhythm_distill"] + lambda_teacher_aux * teacher_aux
                losses["rhythm_teacher_aux_exec"] = teacher_exec.detach()
                losses["rhythm_teacher_aux_state"] = teacher_state.detach()
                losses["rhythm_teacher_aux"] = (lambda_teacher_aux * teacher_aux).detach()

    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output = {}
        loss_weights = {}
        disc_start = (
            self.global_step >= hparams["disc_start_steps"]
            and hparams['lambda_mel_adv'] > 0
            and self.mel_disc is not None
        )
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            loss_output, model_out = self.run_model(sample, infer=False)
            self.model_out_gt = self.model_out = \
                {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
            adv_disabled = bool(
                model_out.get("acoustic_target_is_retimed", False)
                and hparams.get("rhythm_disable_mel_adv_when_retimed", True)
            )
            self._disc_skip_for_retimed = adv_disabled
            if disc_start and not adv_disabled:
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
                adv_disabled = bool(getattr(self, "_disc_skip_for_retimed", False))
                if not adv_disabled:
                    mel_g = model_out.get('acoustic_target_mel', sample['mels'])
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
            gt_f0 = (
                denorm_f0(sample['f0'], sample["uv"])
                if sample.get('f0') is not None and sample.get('uv') is not None
                else None
            )
            pred_f0 = model_out.get("f0_denorm_pred")
            acoustic_target = model_out.get("acoustic_target_mel", sample["mels"])
            acoustic_target_is_retimed = bool(model_out.get("acoustic_target_is_retimed", False))
            if acoustic_target_is_retimed or gt_f0 is None:
                wav_gt = self.vocoder.spec2wav(acoustic_target[0].cpu().numpy())
            else:
                wav_gt = self.vocoder.spec2wav(acoustic_target[0].cpu().numpy(), f0=gt_f0[0].cpu().numpy())
            self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sr)

            if pred_f0 is not None:
                wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu().numpy(), f0=pred_f0[0].cpu().numpy())
            else:
                wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu().numpy())
            self.logger.add_audio(f'wav_pred_{batch_idx}', wav_pred, self.global_step, sr)
            self.plot_mel(batch_idx, acoustic_target, model_out['mel_out'][0], f'mel_{batch_idx}')
            if gt_f0 is not None or pred_f0 is not None:
                self.logger.add_figure(
                    f'f0_{batch_idx}',
                    f0_to_figure(
                        None if acoustic_target_is_retimed or gt_f0 is None else gt_f0[0],
                        None,
                        pred_f0[0] if pred_f0 is not None else None,
                    ),
                    self.global_step,
                )
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
        gt_f0_np = (
            denorm_f0(sample['f0'], sample['uv'])[0].cpu().numpy()
            if sample.get('f0') is not None and sample.get('uv') is not None
            else None
        )
        pred_f0_np = outputs.get('f0_denorm_pred')[0].cpu().numpy() if outputs.get('f0_denorm_pred') is not None else None
        if hparams.get('save_gt', False):
            mel_gt = sample['mels'][0].cpu().numpy()
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.saving_result_pool.add_job(
                self.save_result,
                args=[wav_gt, mel_gt,
                    base_fn.replace('[P]', '[G]'),
                    self.gen_dir, None, None,
                    gt_f0_np,
                    None, None]
            )

        # Save prediction
        self.saving_result_pool.add_job(
            self.save_result,
            args=[wav_pred, mel_pred.cpu().numpy(),
                base_fn, self.gen_dir, None, None,
                gt_f0_np,
                pred_f0_np,
                None]
        )

        result = {}
        result.update(tensors_to_scalars(build_rhythm_metric_dict(outputs, sample)))
        result.update(stream_metrics)
        return result


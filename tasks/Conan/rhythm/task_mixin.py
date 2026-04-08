"""Rhythm-heavy Conan task logic extracted from tasks/Conan/Conan.py.

The public ConanTask class remains stable, while this mixin isolates rhythm
contracts, runtime target handling, and validation helpers for easier
debugging and focused inspection.
"""

import torch
import torch.nn.functional as F

from tasks.Conan.rhythm.acoustic_loss_utils import (
    expand_frame_weight,
    reduce_weighted_elementwise_loss,
)
from tasks.Conan.rhythm.losses import build_rhythm_loss_dict
from tasks.Conan.rhythm.loss_routing import compute_reporting_total_loss
from tasks.Conan.rhythm.metrics import build_rhythm_metric_dict, build_streaming_chunk_metrics
from tasks.Conan.rhythm.loss_balance import AdaptiveRhythmLossBalancer
from tasks.Conan.rhythm.plot_utils import f0_to_figure
from tasks.Conan.rhythm.distill_confidence import (
    build_runtime_distill_confidence_bundle as build_runtime_distill_confidence_bundle_helper,
    normalize_component_distill_confidence as normalize_component_distill_confidence_helper,
    normalize_distill_confidence as normalize_distill_confidence_helper,
)
from tasks.Conan.rhythm.runtime_teacher_supervision import (
    build_runtime_teacher_supervision_targets as build_runtime_teacher_supervision_targets_helper,
)
from tasks.Conan.rhythm.teacher_aux import build_runtime_teacher_aux_loss_dict
from tasks.Conan.rhythm.runtime_modes import (
    build_rhythm_ref_conditioning,
    collect_planner_runtime_outputs,
    resolve_acoustic_target_post_model as resolve_task_acoustic_target_post_model,
    resolve_task_runtime_state,
)
from tasks.Conan.rhythm.reference_regularization import (
    build_predicted_compact_reference_descriptor,
    build_target_compact_reference_descriptor,
    compute_descriptor_consistency_loss,
    compute_group_reference_contrastive_loss,
)
from tasks.Conan.rhythm.streaming_eval import run_chunkwise_streaming_inference
from tasks.Conan.rhythm.task_runtime_support import RhythmTaskRuntimeSupport
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
    RhythmTargetBuildConfig,
    build_identity_rhythm_loss_targets,
    build_rhythm_loss_targets_from_sample,
    scale_rhythm_loss_terms,
)
from modules.Conan.rhythm.policy import (
    resolve_apply_override,
    resolve_prefix_state_lambda,
    should_optimize_render_params,
    use_strict_mainline,
)
from modules.Conan.rhythm.prefix_state import build_prefix_state_from_exec_torch
from modules.Conan.rhythm.stages import (
    resolve_runtime_dual_mode_teacher_enable,
)
from utils.audio.pitch.utils import denorm_f0
from utils.commons.hparams import hparams
from utils.commons.tensor_utils import tensors_to_scalars
from utils.metrics.ssim import ssim


class RhythmConanTaskMixin:
    def _task_runtime_support(self) -> RhythmTaskRuntimeSupport:
        helper = getattr(self, "_cached_rhythm_task_runtime_support", None)
        if helper is None:
            helper = RhythmTaskRuntimeSupport(self)
            self._cached_rhythm_task_runtime_support = helper
        return helper

    def _rhythm_loss_balancer(self) -> AdaptiveRhythmLossBalancer:
        signature = (
            str(hparams.get("rhythm_loss_balance_mode", "none") or "none").strip().lower(),
            float(hparams.get("rhythm_loss_balance_beta", 0.98)),
            float(hparams.get("rhythm_loss_balance_alpha", 0.50)),
            int(hparams.get("rhythm_loss_balance_warmup_steps", 0) or 0),
            float(hparams.get("rhythm_loss_balance_min_scale", 0.50)),
            float(hparams.get("rhythm_loss_balance_max_scale", 2.00)),
            float(hparams.get("rhythm_loss_balance_eps", 1.0e-6)),
        )
        helper = getattr(self, "_cached_rhythm_loss_balancer", None)
        if helper is None or getattr(self, "_cached_rhythm_loss_balancer_signature", None) != signature:
            helper = AdaptiveRhythmLossBalancer.from_hparams(hparams)
            self._cached_rhythm_loss_balancer = helper
            self._cached_rhythm_loss_balancer_signature = signature
        return helper

    @staticmethod
    def _collect_runtime_observability_outputs(model_out: dict) -> dict:
        if model_out is None:
            return {}
        observability = {}
        for metric_key, field_name in (
            ("rhythm_metric_disable_acoustic_train_path", "disable_acoustic_train_path"),
            ("rhythm_metric_module_only_objective", "rhythm_module_only_objective"),
            ("rhythm_metric_skip_acoustic_objective", "rhythm_skip_acoustic_objective"),
            ("rhythm_metric_pitch_supervision_disabled", "rhythm_pitch_supervision_disabled"),
            ("rhythm_metric_missing_retimed_pitch_target", "rhythm_missing_retimed_pitch_target"),
            ("rhythm_metric_acoustic_target_is_retimed", "acoustic_target_is_retimed"),
            ("rhythm_metric_acoustic_target_length_delta_before_align", "acoustic_target_length_delta_before_align"),
            (
                "rhythm_metric_acoustic_target_length_mismatch_abs_before_align",
                "acoustic_target_length_mismatch_abs_before_align",
            ),
            ("rhythm_metric_acoustic_target_resampled_to_output", "acoustic_target_resampled_to_output"),
            ("rhythm_metric_acoustic_target_trimmed_to_output", "acoustic_target_trimmed_to_output"),
            (
                "rhythm_metric_retimed_pitch_target_length_mismatch_abs_before_align",
                "retimed_pitch_target_length_mismatch_abs_before_align",
            ),
            ("rhythm_metric_retimed_pitch_target_resampled_to_output", "retimed_pitch_target_resampled_to_output"),
            ("rhythm_metric_retimed_pitch_target_trimmed_to_output", "retimed_pitch_target_trimmed_to_output"),
            ("rhythm_metric_stage3_acoustic_loss_scale", "rhythm_stage3_acoustic_loss_scale"),
            ("rhythm_metric_retimed_acoustic_loss_scale", "rhythm_retimed_acoustic_loss_scale"),
            ("rhythm_metric_stage3_pitch_loss_scale", "rhythm_stage3_pitch_loss_scale"),
            ("rhythm_metric_retimed_pitch_loss_scale", "rhythm_retimed_pitch_loss_scale"),
            ("rhythm_metric_projector_pause_soft_selection_active", "rhythm_projector_pause_soft_selection_active"),
            ("rhythm_metric_projector_force_full_commit", "rhythm_projector_force_full_commit"),
            (
                "rhythm_metric_teacher_projector_force_full_commit",
                "rhythm_teacher_projector_force_full_commit",
            ),
            (
                "rhythm_metric_teacher_projector_soft_pause_selection",
                "rhythm_teacher_projector_soft_pause_selection",
            ),
        ):
            value = model_out.get(field_name)
            if isinstance(value, torch.Tensor):
                if value.numel() <= 0:
                    continue
                observability[metric_key] = value.detach().float().mean()
            elif isinstance(value, (bool, int, float)):
                observability[metric_key] = float(value)
        return observability

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
        return self._task_runtime_support().dedup_trainable_params(params)

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
        return self._task_runtime_support().dedup_trainable_params(params)

    @staticmethod
    def _get_rhythm_prefix_state_lambda() -> float:
        return resolve_prefix_state_lambda(hparams)

    @staticmethod
    def _get_rhythm_cumplan_lambda() -> float:
        return RhythmConanTaskMixin._get_rhythm_prefix_state_lambda()

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
        prefix_clock, prefix_backlog = build_prefix_state_from_exec_torch(
            speech_exec=speech_exec,
            pause_exec=pause_exec,
            dur_anchor_src=dur_anchor_src[:, :student_units],
            unit_mask=unit_mask[:, :student_units],
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
    def _normalize_distill_confidence(distill_confidence, *, batch_size: int, device: torch.device):
        return normalize_distill_confidence_helper(
            distill_confidence,
            batch_size=batch_size,
            device=device,
            floor=float(hparams.get("rhythm_distill_confidence_floor", 0.05)),
            power=float(hparams.get("rhythm_distill_confidence_power", 1.0)),
        )

    @staticmethod
    def _normalize_component_distill_confidence(
        component_confidence,
        *,
        fallback_confidence: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ):
        return normalize_component_distill_confidence_helper(
            component_confidence,
            fallback_confidence=fallback_confidence,
            batch_size=batch_size,
            device=device,
            floor=float(hparams.get("rhythm_distill_confidence_floor", 0.05)),
            power=float(hparams.get("rhythm_distill_confidence_power", 1.0)),
            preserve_zeros=True,
        )

    def _build_rhythm_target_build_config(self) -> RhythmTargetBuildConfig:
        return self._task_runtime_support().build_rhythm_target_build_config()

    def _maybe_balance_rhythm_loss_terms(self, losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._rhythm_loss_balancer().apply(
            losses,
            global_step=int(self.global_step),
            training=bool(self.training and torch.is_grad_enabled()),
        )

    @staticmethod
    def _resolve_rhythm_plan_weights() -> tuple[float, float]:
        lambda_plan = float(hparams.get("lambda_rhythm_plan", 0.0) or 0.0)
        if lambda_plan <= 0.0:
            return 0.0, 0.0
        return (
            float(hparams.get("rhythm_plan_local_weight", 0.5)),
            float(hparams.get("rhythm_plan_cum_weight", 1.0)),
        )

    @staticmethod
    def _build_reference_descriptor_bundle(output, sample):
        target = build_target_compact_reference_descriptor(sample)
        if target is None:
            return None, None
        pred = build_predicted_compact_reference_descriptor(
            output,
            trace_bins=int(target.local_rate_trace.size(1)),
        )
        return pred, target

    def _add_reference_descriptor_regularization(self, output, sample, losses):
        lambda_stats = float(hparams.get("lambda_rhythm_ref_descriptor_stats", 0.0) or 0.0)
        lambda_trace = float(hparams.get("lambda_rhythm_ref_descriptor_trace", 0.0) or 0.0)
        lambda_group = float(hparams.get("lambda_rhythm_ref_group_contrastive", 0.0) or 0.0)
        if lambda_stats <= 0.0 and lambda_trace <= 0.0 and lambda_group <= 0.0:
            return
        pred, target = self._build_reference_descriptor_bundle(output, sample)
        if pred is None or target is None:
            return
        descriptor_losses = compute_descriptor_consistency_loss(
            pred,
            target,
            local_weight=float(hparams.get("rhythm_ref_descriptor_local_weight", 1.0)),
            boundary_weight=float(hparams.get("rhythm_ref_descriptor_boundary_weight", 1.0)),
        )
        if lambda_stats > 0.0:
            losses["L_ref_desc_stats"] = descriptor_losses["stats"] * lambda_stats
        if lambda_trace > 0.0:
            losses["L_ref_desc_trace"] = (
                descriptor_losses["local_trace"]
                + float(hparams.get("rhythm_ref_descriptor_boundary_loss_weight", 1.0))
                * descriptor_losses["boundary_trace"]
            ) * lambda_trace
        if lambda_group > 0.0 and "rhythm_pair_group_id" in sample:
            group_loss = compute_group_reference_contrastive_loss(
                pred,
                target,
                sample["rhythm_pair_group_id"],
                temperature=float(hparams.get("rhythm_ref_group_contrastive_temperature", 0.10)),
                gap_floor=float(hparams.get("rhythm_ref_group_contrastive_gap_floor", 0.10)),
                min_scale=float(hparams.get("rhythm_ref_group_contrastive_min_scale", 0.50)),
                gap_power=float(hparams.get("rhythm_ref_group_contrastive_gap_power", 1.0)),
            )
            if group_loss is not None:
                losses["L_ref_group_contrast"] = group_loss * lambda_group

    @staticmethod
    def _build_runtime_distill_confidence_bundle(output):
        return build_runtime_distill_confidence_bundle_helper(output)

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
    def _expand_frame_weight(weight: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
        weights = expand_frame_weight(weight, target)
        if weights.shape != target.shape:
            weights = torch.broadcast_to(weights, target.shape)
        return weights

    def _weighted_l1_loss(self, decoder_output, target, frame_weight):
        loss = F.l1_loss(decoder_output, target, reduction='none')
        return reduce_weighted_elementwise_loss(
            loss,
            frame_weight=frame_weight,
            target=target,
        )

    def _weighted_mse_loss(self, decoder_output, target, frame_weight):
        loss = F.mse_loss(decoder_output, target, reduction='none')
        return reduce_weighted_elementwise_loss(
            loss,
            frame_weight=frame_weight,
            target=target,
        )

    def _weighted_ssim_loss(self, decoder_output, target, frame_weight, bias=6.0):
        decoder_output = decoder_output[:, None] + bias
        target = target[:, None] + bias
        ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
        return reduce_weighted_elementwise_loss(
            ssim_loss,
            frame_weight=frame_weight,
            target=target[:, 0],
        )

    @staticmethod
    def _resolve_stage3_acoustic_hparam(name: str, legacy_name: str, default):
        if name in hparams:
            return hparams.get(name)
        if legacy_name in hparams:
            return hparams.get(legacy_name)
        return default

    def _resolve_stage3_acoustic_loss_scale(
        self,
        *,
        stage: str,
        retimed_stage_active: bool,
        acoustic_target_is_retimed: bool,
        infer: bool,
        test: bool,
    ) -> float:
        """Resolve a true scalar on the stage-3 retimed acoustic objective.

        Frame weights already normalize by their full broadcasted mass, so they
        mainly reshape trust across frames and do not reliably lower the total
        objective magnitude. Stage-3 stabilization therefore needs an explicit
        scalar multiplier on the final acoustic loss.
        """
        if infer or test:
            return 1.0
        if stage != "student_retimed":
            return 1.0
        if not bool(retimed_stage_active) or not bool(acoustic_target_is_retimed):
            return 1.0
        end_scale = float(
            self._resolve_stage3_acoustic_hparam(
                "rhythm_stage3_acoustic_weight_end",
                "rhythm_retimed_acoustic_loss_scale",
                1.0,
            )
        )
        start_scale = float(
            self._resolve_stage3_acoustic_hparam(
                "rhythm_stage3_acoustic_weight_start",
                "rhythm_retimed_acoustic_loss_scale_start",
                end_scale,
            )
        )
        warmup_steps = int(
            self._resolve_stage3_acoustic_hparam(
                "rhythm_stage3_acoustic_ramp_steps",
                "rhythm_retimed_acoustic_loss_scale_warmup_steps",
                0,
            )
            or 0
        )
        if warmup_steps <= 0:
            return max(0.0, end_scale)
        # Anchor the ramp to the moment stage-3 retimed training becomes active,
        # rather than to the absolute global step. This preserves the intended
        # curriculum when student_retimed warm-starts from a large stage-2
        # checkpoint step.
        anchor_step = getattr(self, "_rhythm_stage3_acoustic_ramp_anchor_step", None)
        anchor_stage = getattr(self, "_rhythm_stage3_acoustic_ramp_anchor_stage", None)
        current_step = int(self.global_step)
        if anchor_step is None or anchor_stage != stage or current_step < int(anchor_step):
            anchor_step = current_step
            self._rhythm_stage3_acoustic_ramp_anchor_step = anchor_step
            self._rhythm_stage3_acoustic_ramp_anchor_stage = stage
        progress_steps = max(0, current_step - int(anchor_step))
        progress = min(max(progress_steps / float(warmup_steps), 0.0), 1.0)
        scale = start_scale + (end_scale - start_scale) * progress
        return max(0.0, float(scale))

    def _resolve_acoustic_loss_scale(self, output, *, infer: bool, test: bool) -> float:
        return self._resolve_stage3_acoustic_loss_scale(
            stage=str(output.get("rhythm_stage", "")),
            retimed_stage_active=bool(output.get("acoustic_target_is_retimed", False)),
            acoustic_target_is_retimed=bool(output.get("acoustic_target_is_retimed", False)),
            infer=infer,
            test=test,
        )

    def _add_acoustic_loss(self, mel_out, target, losses, *, frame_weight=None, loss_scale: float = 1.0):
        loss_scale = float(max(0.0, loss_scale))
        if frame_weight is None:
            mel_losses = {}
            self.add_mel_loss(mel_out, target, mel_losses)
            for loss_name, loss_value in mel_losses.items():
                losses[loss_name] = loss_value * loss_scale
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
            losses[loss_name] = loss * lambd * loss_scale

    def drop_multi(self, tech, drop_p):
        if torch.rand(1) < drop_p:
            tech = torch.ones_like(tech, dtype=tech.dtype) * 2
        elif torch.rand(1) < drop_p:
            random_tech = torch.rand_like(tech, dtype=torch.float32)
            tech[random_tech < drop_p] = 2
        return tech

    def _run_offline_teacher_model(self, sample, *, infer: bool, test: bool, **kwargs):
        if test:
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
            infer=bool(infer),
            global_steps=int(self.global_step),
            teacher=True,
        )
        pause_ratio = self.model._resolve_rhythm_pause_topk_ratio(
            infer=bool(infer),
            global_steps=int(self.global_step),
        )
        teacher_force_full_commit = bool(hparams.get("rhythm_teacher_projector_force_full_commit", True))
        teacher_soft_pause_selection = hparams.get("rhythm_teacher_projector_soft_pause_selection", None)
        if teacher_soft_pause_selection is not None:
            teacher_soft_pause_selection = bool(teacher_soft_pause_selection)
        execution, confidence = rhythm_module.forward_teacher(
            content_units=unit_batch.content_units,
            dur_anchor_src=unit_batch.dur_anchor_src,
            ref_conditioning=rhythm_ref_conditioning,
            unit_mask=unit_batch.unit_mask,
            sep_hint=unit_batch.sep_hint,
            boundary_confidence=unit_batch.boundary_confidence,
            projector_pause_topk_ratio_override=pause_ratio,
            source_boundary_scale_override=teacher_scale,
            projector_force_full_commit=teacher_force_full_commit,
            projector_soft_pause_selection=teacher_soft_pause_selection,
        )
        output = {
            "rhythm_execution": execution,
            "rhythm_unit_batch": unit_batch,
            "rhythm_stage": "teacher_offline",
            "rhythm_teacher_as_main": 0.0,
            "disable_acoustic_train_path": 1.0,
            "rhythm_schedule_only_stage": 0.0,
            "rhythm_teacher_only_stage": 1.0,
            "rhythm_module_only_objective": 1.0,
            "rhythm_skip_acoustic_objective": 1.0,
            "rhythm_teacher_projector_force_full_commit": float(teacher_force_full_commit),
            **self._task_runtime_support().build_offline_confidence_outputs(confidence),
        }
        if teacher_scale is not None:
            output["rhythm_teacher_source_boundary_scale"] = unit_batch.dur_anchor_src.new_full(
                (unit_batch.dur_anchor_src.size(0), 1),
                float(teacher_scale),
            )
        if teacher_soft_pause_selection is not None:
            output["rhythm_teacher_projector_soft_pause_selection"] = unit_batch.dur_anchor_src.new_full(
                (unit_batch.dur_anchor_src.size(0), 1),
                1.0 if teacher_soft_pause_selection else 0.0,
            )
        output.update(collect_planner_runtime_outputs(execution))
        losses = {}
        self.add_rhythm_loss(output, sample, losses)
        self._task_runtime_support().route_conan_losses(losses, schedule_only_stage=False)
        return losses, output

    def run_model(self, sample, infer=False, test=False, **kwargs):
        # txt_tokens = sample["txt_tokens"]
        # mel2ph = sample["mel2ph"]
        # spk_id = sample["spk_ids"]
        content = sample["content"]
        spk_embed = sample.get("spk_embed")
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
        module_only_objective = runtime_state.module_only_objective
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
        if disable_acoustic_train_path and rhythm_ref_conditioning is not None and not infer:
            ref = None
        runtime_helper = self._task_runtime_support()
        runtime_offline_source_cache = runtime_helper.collect_runtime_offline_source_cache(sample, infer=infer)
        output = self.model(
            content,
            **runtime_helper.build_model_forward_kwargs(
                sample=sample,
                spk_embed=spk_embed,
                target=target,
                ref=ref,
                f0=f0,
                uv=uv,
                infer=infer,
                effective_global_step=effective_global_step,
                rhythm_apply_override=rhythm_apply_override,
                rhythm_ref_conditioning=rhythm_ref_conditioning,
                disable_source_pitch_supervision=disable_source_pitch_supervision,
                disable_acoustic_train_path=disable_acoustic_train_path,
                runtime_offline_source_cache=runtime_offline_source_cache,
                rhythm_state=kwargs.get("rhythm_state"),
            ),
        )
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
        acoustic_target, acoustic_weight = runtime_helper.attach_acoustic_target_bundle(
            output,
            acoustic_target=acoustic_target,
            acoustic_target_is_retimed=acoustic_target_is_retimed,
            acoustic_weight=acoustic_weight,
            acoustic_target_source=acoustic_target_source,
            disable_source_pitch_supervision=disable_source_pitch_supervision,
            disable_acoustic_train_path=disable_acoustic_train_path,
        )
        self._assert_pitch_supervision_ready(
            output,
            sample,
            infer=infer,
            test=test,
            retimed_stage_active=retimed_stage_active,
        )
        
        if not test:
            schedule_only_stage = stage == "legacy_schedule_only"
            skip_acoustic_objective = bool(schedule_only_stage or module_only_objective)
            output["rhythm_schedule_only_stage"] = float(schedule_only_stage)
            output["rhythm_stage"] = stage
            output["rhythm_module_only_objective"] = float(module_only_objective)
            output["rhythm_skip_acoustic_objective"] = float(skip_acoustic_objective)
            acoustic_loss_scale = self._resolve_stage3_acoustic_loss_scale(
                stage=stage,
                retimed_stage_active=bool(retimed_stage_active),
                acoustic_target_is_retimed=bool(output.get("acoustic_target_is_retimed", False)),
                infer=infer,
                test=test,
            )
            output["rhythm_stage3_acoustic_loss_scale"] = float(acoustic_loss_scale)
            output["rhythm_retimed_acoustic_loss_scale"] = float(acoustic_loss_scale)
            pitch_loss_scale = (
                acoustic_loss_scale
                if bool(hparams.get("rhythm_stage3_scale_pitch_loss", True))
                else 1.0
            )
            output["rhythm_stage3_pitch_loss_scale"] = float(pitch_loss_scale)
            output["rhythm_retimed_pitch_loss_scale"] = float(pitch_loss_scale)
            if not skip_acoustic_objective:
                self._add_acoustic_loss(
                    output['mel_out'],
                    acoustic_target,
                    losses,
                    frame_weight=acoustic_weight,
                    loss_scale=acoustic_loss_scale,
                )
                # self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
                self.add_pitch_loss(output, sample, losses, loss_scale=pitch_loss_scale)
            self.add_rhythm_loss(output, sample, losses)
            runtime_helper.add_style_losses(output, losses, schedule_only_stage=schedule_only_stage)
            runtime_helper.route_conan_losses(losses, schedule_only_stage=schedule_only_stage)
        
        return losses, output

    @staticmethod
    def _has_nonempty_pitch_payload(value) -> bool:
        if value is None:
            return False
        numel = getattr(value, "numel", None)
        if callable(numel):
            try:
                return int(numel()) > 0
            except Exception:
                pass
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                if len(shape) == 0:
                    return True
                return all(int(dim) > 0 for dim in shape)
            except Exception:
                pass
        try:
            return len(value) > 0
        except Exception:
            return True

    @staticmethod
    def _infer_time_steps(value) -> int | None:
        if not isinstance(value, torch.Tensor):
            return None
        if value.dim() >= 2:
            return int(value.size(1))
        if value.dim() == 1:
            return int(value.size(0))
        return None

    @classmethod
    def _assert_pitch_supervision_ready(
        cls,
        output,
        sample,
        *,
        infer: bool,
        test: bool,
        retimed_stage_active: bool,
    ) -> None:
        if infer or test:
            return
        if not bool(hparams.get("rhythm_fail_fast_missing_pitch_supervision", True)):
            return
        if not bool(hparams.get("use_pitch_embed", False)):
            return
        if bool(output.get("disable_acoustic_train_path", False)):
            return

        supervision_disabled = bool(output.get("rhythm_pitch_supervision_disabled", False))
        acoustic_target_is_retimed = bool(output.get("acoustic_target_is_retimed", False))
        retimed_f0 = output.get("retimed_f0_tgt")
        retimed_uv = output.get("retimed_uv_tgt")
        source_f0 = sample.get("f0")
        source_uv = sample.get("uv")

        has_retimed_pitch = cls._has_nonempty_pitch_payload(retimed_f0) and cls._has_nonempty_pitch_payload(retimed_uv)
        has_source_pitch = cls._has_nonempty_pitch_payload(source_f0) and cls._has_nonempty_pitch_payload(source_uv)
        expected_steps = cls._infer_time_steps(output.get("mel_out"))
        retimed_f0_steps = cls._infer_time_steps(retimed_f0)
        retimed_uv_steps = cls._infer_time_steps(retimed_uv)
        source_f0_steps = cls._infer_time_steps(source_f0)
        source_uv_steps = cls._infer_time_steps(source_uv)
        retimed_lengths_match = bool(
            expected_steps is None
            or (retimed_f0_steps == expected_steps and retimed_uv_steps == expected_steps)
        )
        source_lengths_match = bool(
            expected_steps is None
            or (source_f0_steps == expected_steps and source_uv_steps == expected_steps)
        )
        if acoustic_target_is_retimed:
            if has_retimed_pitch and retimed_lengths_match and not supervision_disabled:
                return
        else:
            if has_source_pitch and source_lengths_match and not supervision_disabled:
                return

        detail = (
            f"(acoustic_target_is_retimed={int(acoustic_target_is_retimed)}, "
            f"supervision_disabled={int(supervision_disabled)}, "
            f"expected_steps={expected_steps}, "
            f"has_retimed_f0={int(cls._has_nonempty_pitch_payload(retimed_f0))}, "
            f"has_retimed_uv={int(cls._has_nonempty_pitch_payload(retimed_uv))}, "
            f"retimed_f0_steps={retimed_f0_steps}, "
            f"retimed_uv_steps={retimed_uv_steps}, "
            f"has_source_f0={int(cls._has_nonempty_pitch_payload(source_f0))}, "
            f"has_source_uv={int(cls._has_nonempty_pitch_payload(source_uv))}, "
            f"source_f0_steps={source_f0_steps}, "
            f"source_uv_steps={source_uv_steps})"
        )
        if acoustic_target_is_retimed or retimed_stage_active:
            raise RuntimeError(
                "Rhythm retimed training is missing usable pitch supervision while use_pitch_embed=true. "
                "Retimed acoustic targets must be paired with matched retimed f0/uv targets and kept length-aligned with mel_out. "
                f"{detail} "
                "Retimed pitch supervision must not silently fall back to source-aligned supervision. "
                "Provide source f0/uv so retimed_f0_tgt/retimed_uv_tgt can be built, or explicitly set "
                "rhythm_fail_fast_missing_pitch_supervision=false only for debugging."
            )
        raise RuntimeError(
            "Pitch supervision is missing or length-misaligned while use_pitch_embed=true during training. "
            f"{detail} Provide f0/uv in the binary cache, keep pitch targets length-aligned with mel_out, disable use_pitch_embed, or explicitly set "
            "rhythm_fail_fast_missing_pitch_supervision=false only for debugging."
        )

    def add_pitch_loss(self, output, sample, losses, *, loss_scale: float = 1.0):
        # mel2ph = sample['mel2ph']  # [B, T_s]
        if bool(output.get("rhythm_pitch_supervision_disabled", False)):
            return
        loss_scale = float(max(0.0, loss_scale))
        content = output.get("content", sample['content'])
        acoustic_target_source = str(output.get("acoustic_target_source", "") or "").lower()
        use_retimed_pitch = acoustic_target_source == "online"
        f0 = output.get("retimed_f0_tgt") if use_retimed_pitch else sample.get('f0')
        uv = output.get("retimed_uv_tgt") if use_retimed_pitch else sample.get('uv')
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
            losses["fdiff"] = output["fdiff"] * loss_scale
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding_sum * hparams['lambda_uv'] * loss_scale
        elif hparams["f0_gen"] == "flow":
            losses["pflow"] = output["pflow"] * loss_scale
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding_sum * hparams['lambda_uv'] * loss_scale
        elif hparams["f0_gen"] == "gmdiff":
            losses["gdiff"] = output["gdiff"] * loss_scale
            losses["mdiff"] = output["mdiff"] * loss_scale
        elif hparams["f0_gen"] == "orig":
            losses["fdiff"] = output["fdiff"] * loss_scale
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding_sum * hparams['lambda_uv'] * loss_scale

    def _build_runtime_teacher_supervision_targets(self, output, sample):
        plan_local_weight, plan_cum_weight = self._resolve_rhythm_plan_weights()
        return build_runtime_teacher_supervision_targets_helper(
            output=output,
            sample=sample,
            plan_local_weight=plan_local_weight,
            plan_cum_weight=plan_cum_weight,
            pause_boundary_weight=self._resolve_rhythm_pause_boundary_weight(),
            budget_raw_weight=float(hparams.get("rhythm_budget_raw_weight", 1.0)),
            budget_exec_weight=float(hparams.get("rhythm_budget_exec_weight", 0.25)),
            feasible_debt_weight=float(hparams.get("rhythm_feasible_debt_weight", 0.05)),
        )

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
            build_prefix_carry_from_exec=build_prefix_state_from_exec_torch,
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
            build_prefix_carry_from_exec=build_prefix_state_from_exec_torch,
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
        rhythm_losses = build_rhythm_loss_dict(rhythm_execution, targets)
        scaled_rhythm_losses = scale_rhythm_loss_terms(
            rhythm_losses,
            hparams=hparams,
            cumplan_lambda=self._get_rhythm_prefix_state_lambda(),
        )
        scaled_rhythm_losses = self._maybe_balance_rhythm_loss_terms(scaled_rhythm_losses)
        losses.update(scaled_rhythm_losses)
        self._add_reference_descriptor_regularization(output, sample, losses)

        runtime_teacher = output.get("rhythm_offline_execution")
        lambda_teacher_aux = float(hparams.get("lambda_rhythm_teacher_aux", 0.0) or 0.0)
        if runtime_teacher is not None and lambda_teacher_aux > 0.0:
            teacher_bundle = self._build_runtime_teacher_supervision_targets(output, sample)
            if teacher_bundle is not None:
                teacher_execution, teacher_targets = teacher_bundle
                teacher_losses = build_rhythm_loss_dict(teacher_execution, teacher_targets)
                losses.update(
                    build_runtime_teacher_aux_loss_dict(
                        teacher_losses=teacher_losses,
                        hparams=hparams,
                        prefix_state_lambda=self._get_rhythm_prefix_state_lambda(),
                        lambda_teacher_aux=lambda_teacher_aux,
                    )
                )

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
            loss_output.update(self._collect_runtime_observability_outputs(model_out))
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
        outputs['losses'].update(outputs['rhythm_metrics'])
        outputs['total_loss'] = compute_reporting_total_loss(
            outputs['losses'],
            mel_loss_names=self._task_runtime_support().mel_loss_names(),
            hparams=hparams,
            schedule_only_stage=bool(model_out.get("rhythm_schedule_only_stage", False)),
        )
        outputs['nsamples'] = sample['nsamples']
        outputs = tensors_to_scalars(outputs)
        mel_out = model_out.get("mel_out")
        if batch_idx < hparams['num_valid_plots'] and mel_out is not None:
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
                wav_pred = self.vocoder.spec2wav(mel_out[0].cpu().numpy(), f0=pred_f0[0].cpu().numpy())
            else:
                wav_pred = self.vocoder.spec2wav(mel_out[0].cpu().numpy())
            self.logger.add_audio(f'wav_pred_{batch_idx}', wav_pred, self.global_step, sr)
            self.plot_mel(batch_idx, acoustic_target, mel_out[0], f'mel_{batch_idx}')
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

from __future__ import annotations

import torch
import torch.nn.functional as F

from tasks.Conan.rhythm.config_contract_rules.compat import (
    resolve_duplicate_primary_distill_dedupe_flag as _resolve_duplicate_primary_distill_dedupe_flag,
)
from tasks.Conan.rhythm.loss_routing import route_conan_optimizer_losses, update_public_loss_aliases
from tasks.Conan.rhythm.targets import RhythmTargetBuildConfig
from utils.commons.hparams import hparams


class RhythmTaskRuntimeSupport:
    _OFFLINE_CONFIDENCE_COMPONENTS = (
        ("rhythm_offline_confidence", "overall", None),
        ("rhythm_offline_confidence_exec", "exec", None),
        ("rhythm_offline_confidence_budget", "budget", None),
        ("rhythm_offline_confidence_prefix", "prefix", None),
        ("rhythm_offline_confidence_allocation", "allocation", None),
        ("rhythm_offline_confidence_shape", "shape", "exec"),
    )

    def __init__(self, owner) -> None:
        self.owner = owner

    @staticmethod
    def dedup_trainable_params(params):
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

    def mel_loss_names(self) -> tuple[str, ...]:
        return tuple(self.owner.mel_losses.keys())

    def build_rhythm_target_build_config(self) -> RhythmTargetBuildConfig:
        def _nonnegative_hparam(name: str, default: float) -> float:
            return max(0.0, float(hparams.get(name, default) or default))

        plan_local_weight, plan_cum_weight = self.owner._resolve_rhythm_plan_weights()
        return RhythmTargetBuildConfig(
            primary_target_surface=self.owner._resolve_rhythm_primary_target_surface(),
            distill_surface=self.owner._resolve_rhythm_distill_surface(),
            lambda_guidance=float(hparams.get("lambda_rhythm_guidance", 0.0) or 0.0),
            lambda_distill=float(hparams.get("lambda_rhythm_distill", 0.0) or 0.0),
            distill_exec_weight=float(hparams.get("rhythm_distill_exec_weight", 1.0)),
            distill_budget_weight=float(hparams.get("rhythm_distill_budget_weight", 0.5)),
            distill_allocation_weight=float(hparams.get("rhythm_distill_allocation_weight", 0.5)),
            distill_prefix_weight=float(hparams.get("rhythm_distill_prefix_weight", 0.25)),
            distill_speech_shape_weight=float(hparams.get("rhythm_distill_speech_shape_weight", 0.0)),
            distill_pause_shape_weight=float(hparams.get("rhythm_distill_pause_shape_weight", 0.0)),
            plan_local_weight=plan_local_weight,
            plan_cum_weight=plan_cum_weight,
            unit_logratio_weight=_nonnegative_hparam("rhythm_unit_logratio_weight", 0.0),
            pause_boundary_weight=self.owner._resolve_rhythm_pause_boundary_weight(),
            budget_raw_weight=float(hparams.get("rhythm_budget_raw_weight", 1.0)),
            budget_exec_weight=float(hparams.get("rhythm_budget_exec_weight", 0.25)),
            feasible_debt_weight=float(hparams.get("rhythm_feasible_debt_weight", 0.05)),
            plan_segment_shape_weight=_nonnegative_hparam("rhythm_plan_segment_shape_weight", 0.0),
            plan_pause_release_weight=_nonnegative_hparam("rhythm_plan_pause_release_weight", 0.0),
            pause_event_weight=_nonnegative_hparam("rhythm_pause_event_weight", 0.0),
            pause_support_weight=_nonnegative_hparam("rhythm_pause_support_weight", 0.0),
            pause_allocation_weight=_nonnegative_hparam("rhythm_pause_allocation_weight", 0.0),
            pause_event_threshold=_nonnegative_hparam("rhythm_pause_event_threshold", 0.5),
            pause_event_temperature=max(1e-4, _nonnegative_hparam("rhythm_pause_event_temperature", 0.25)),
            pause_event_pos_weight=max(1.0, _nonnegative_hparam("rhythm_pause_event_pos_weight", 2.0)),
            dedupe_primary_teacher_cache_distill=_resolve_duplicate_primary_distill_dedupe_flag(hparams),
            enable_distill_context_match=bool(
                hparams.get("rhythm_enable_distill_context_match", False)
            ),
            distill_context_floor=float(hparams.get("rhythm_distill_context_floor", 0.35)),
            distill_context_power=float(hparams.get("rhythm_distill_context_power", 1.0)),
            distill_context_open_run_penalty=float(
                hparams.get("rhythm_distill_context_open_run_penalty", 0.50)
            ),
        )

    def build_offline_confidence_outputs(self, confidence) -> dict:
        outputs = {}
        for output_key, confidence_key, fallback_key in self._OFFLINE_CONFIDENCE_COMPONENTS:
            value = None
            if isinstance(confidence, dict):
                value = confidence.get(confidence_key)
                if value is None and fallback_key is not None:
                    value = confidence.get(fallback_key)
            outputs[output_key] = value
        return outputs

    def route_conan_losses(self, losses, *, schedule_only_stage: bool) -> None:
        mel_loss_names = self.mel_loss_names()
        route_conan_optimizer_losses(
            losses,
            mel_loss_names=mel_loss_names,
            hparams=hparams,
            schedule_only_stage=schedule_only_stage,
        )
        update_public_loss_aliases(losses, mel_loss_names=mel_loss_names)

    def collect_runtime_offline_source_cache(self, sample, *, infer: bool):
        if self.owner._use_runtime_dual_mode_teacher() and not bool(infer):
            return self.owner._collect_rhythm_source_cache(sample, prefix="rhythm_offline_")
        return None

    def build_model_forward_kwargs(
        self,
        *,
        sample,
        spk_embed,
        target,
        ref,
        f0,
        uv,
        infer: bool,
        effective_global_step: int,
        rhythm_apply_override,
        rhythm_ref_conditioning,
        disable_source_pitch_supervision: bool,
        disable_acoustic_train_path: bool,
        runtime_offline_source_cache,
        rhythm_state,
    ) -> dict:
        return {
            "spk_embed": spk_embed,
            "target": target,
            "ref": ref,
            "f0": None if disable_source_pitch_supervision else f0,
            "uv": None if disable_source_pitch_supervision else uv,
            "infer": infer,
            "global_steps": effective_global_step,
            "content_lengths": sample.get("content_lengths", sample.get("mel_lengths")),
            "ref_lengths": sample.get("ref_mel_lengths"),
            "rhythm_apply_override": rhythm_apply_override,
            "rhythm_state": rhythm_state,
            "rhythm_ref_conditioning": rhythm_ref_conditioning,
            "rhythm_source_cache": self.owner._collect_rhythm_source_cache(sample),
            "rhythm_offline_source_cache": runtime_offline_source_cache,
            "disable_acoustic_train_path": disable_acoustic_train_path,
        }

    @staticmethod
    def _resize_rank2_sequence(value: torch.Tensor, target_len: int, *, mode: str) -> torch.Tensor:
        if value.dim() != 2:
            raise ValueError(f"Expected rank-2 tensor [B, T], got {tuple(value.shape)}")
        if value.size(1) == target_len:
            return value
        if target_len <= 0:
            return value[:, :0]
        if value.size(1) <= 0:
            return value.new_zeros((value.size(0), target_len), dtype=value.dtype)
        source_dtype = value.dtype
        work = value.float().unsqueeze(1)
        if mode == "linear":
            resized = F.interpolate(work, size=target_len, mode="linear", align_corners=False).squeeze(1)
        elif mode == "nearest":
            resized = F.interpolate(work, size=target_len, mode="nearest").squeeze(1)
        else:
            raise ValueError(f"Unsupported resize mode: {mode}")
        if source_dtype == torch.bool:
            return resized > 0.5
        if not value.is_floating_point():
            return resized.round().to(dtype=source_dtype)
        return resized.to(dtype=source_dtype)

    @classmethod
    def _align_rank2_sequence(
        cls,
        value: torch.Tensor,
        *,
        target_len: int,
        mode: str,
        prefer_resample: bool,
    ) -> tuple[torch.Tensor, bool, bool]:
        if value.size(1) == target_len:
            return value, False, False
        if prefer_resample or value.size(1) < target_len:
            return cls._resize_rank2_sequence(value, target_len, mode=mode), True, False
        return value[:, :target_len], False, True

    def _align_output_nonpadding(self, output) -> None:
        nonpadding = output.get("tgt_nonpadding")
        mel_out = output.get("mel_out")
        if not isinstance(nonpadding, torch.Tensor) or not isinstance(mel_out, torch.Tensor):
            return
        if nonpadding.dim() not in {2, 3} or mel_out.dim() < 2:
            return
        target_len = int(mel_out.size(1))
        squeezed = bool(nonpadding.dim() == 3)
        base = nonpadding.squeeze(-1) if squeezed else nonpadding
        if base.dim() != 2:
            return
        aligned, _, _ = self._align_rank2_sequence(
            base.float(),
            target_len=target_len,
            mode="nearest",
            prefer_resample=False,
        )
        output["tgt_nonpadding"] = aligned.unsqueeze(-1) if squeezed else aligned

    def _align_retimed_pitch_targets_to_output(self, output) -> None:
        mel_out = output.get("mel_out")
        if not isinstance(mel_out, torch.Tensor) or mel_out.dim() < 2:
            return
        target_len = int(mel_out.size(1))
        prefer_resample = bool(hparams.get("rhythm_resample_retimed_target_to_output", True))
        resampled = False
        trimmed = False
        before_len = None
        for key, mode in (("retimed_f0_tgt", "linear"), ("retimed_uv_tgt", "nearest")):
            value = output.get(key)
            if not isinstance(value, torch.Tensor) or value.dim() != 2:
                continue
            if before_len is None:
                before_len = int(value.size(1))
            aligned, key_resampled, key_trimmed = self._align_rank2_sequence(
                value,
                target_len=target_len,
                mode=mode,
                prefer_resample=prefer_resample,
            )
            if key == "retimed_uv_tgt" and isinstance(aligned, torch.Tensor) and aligned.is_floating_point():
                aligned = aligned.clamp(0.0, 1.0)
            output[key] = aligned
            resampled = bool(resampled or key_resampled)
            trimmed = bool(trimmed or key_trimmed)
        if before_len is not None:
            delta = float(before_len - target_len)
            output["retimed_pitch_target_length_frames_before_align"] = float(before_len)
            output["retimed_pitch_target_length_delta_before_align"] = delta
            output["retimed_pitch_target_length_mismatch_abs_before_align"] = abs(delta)
            output["retimed_pitch_target_resampled_to_output"] = float(resampled)
            output["retimed_pitch_target_trimmed_to_output"] = float(trimmed)
            after_f0 = output.get("retimed_f0_tgt")
            if isinstance(after_f0, torch.Tensor) and after_f0.dim() == 2:
                output["retimed_pitch_target_length_frames_after_align"] = float(after_f0.size(1))

    def attach_acoustic_target_bundle(
        self,
        output,
        *,
        acoustic_target,
        acoustic_target_is_retimed: bool,
        acoustic_weight,
        acoustic_target_source,
        disable_source_pitch_supervision: bool,
        disable_acoustic_train_path: bool,
    ):
        pitch_supervision_disabled = bool(disable_source_pitch_supervision)
        missing_retimed_pitch_target = False
        if (
            acoustic_target_is_retimed
            and not pitch_supervision_disabled
            and not bool(hparams.get("rhythm_allow_source_pitch_fallback_when_retimed", False))
        ):
            has_retimed_pitch = (
                output.get("retimed_f0_tgt") is not None
                and output.get("retimed_uv_tgt") is not None
            )
            if not has_retimed_pitch:
                pitch_supervision_disabled = True
                missing_retimed_pitch_target = True
        output["acoustic_target_mel"] = acoustic_target
        output["acoustic_target_is_retimed"] = bool(acoustic_target_is_retimed)
        output["acoustic_target_weight"] = acoustic_weight
        output["acoustic_target_source"] = acoustic_target_source
        output["rhythm_pitch_supervision_disabled"] = float(pitch_supervision_disabled)
        output["rhythm_missing_retimed_pitch_target"] = float(missing_retimed_pitch_target)
        output["disable_acoustic_train_path"] = float(disable_acoustic_train_path)
        target_len_before_align = None
        mel_len_before_align = None
        if hasattr(acoustic_target, "dim") and acoustic_target.dim() >= 2:
            target_len_before_align = int(acoustic_target.size(1))
            output["acoustic_target_length_frames_before_align"] = float(target_len_before_align)
        if hasattr(output.get("mel_out"), "dim") and output["mel_out"].dim() >= 2:
            mel_len_before_align = int(output["mel_out"].size(1))
            output["acoustic_output_length_frames_before_align"] = float(mel_len_before_align)
        if target_len_before_align is not None and mel_len_before_align is not None:
            delta = float(target_len_before_align - mel_len_before_align)
            output["acoustic_target_length_delta_before_align"] = delta
            mismatch_abs = abs(delta)
            output["acoustic_target_length_mismatch_abs_before_align"] = mismatch_abs
            output["acoustic_target_length_mismatch_present_before_align"] = float(mismatch_abs > 0.0)
            output["acoustic_target_length_mismatch_ratio_before_align"] = mismatch_abs / float(
                max(target_len_before_align, mel_len_before_align, 1)
            )
        output["acoustic_target_resampled_to_output"] = 0.0
        output["acoustic_target_trimmed_to_output"] = 0.0
        if acoustic_target_is_retimed:
            had_length_mismatch = (
                target_len_before_align is not None
                and mel_len_before_align is not None
                and target_len_before_align != mel_len_before_align
            )
            mel_out_aligned, acoustic_target, acoustic_weight = self.owner._align_acoustic_target_to_output(
                output["mel_out"],
                acoustic_target,
                acoustic_weight,
            )
            if had_length_mismatch:
                if bool(hparams.get("rhythm_resample_retimed_target_to_output", True)):
                    output["acoustic_target_resampled_to_output"] = 1.0
                else:
                    output["acoustic_target_trimmed_to_output"] = 1.0
            output["mel_out"] = mel_out_aligned
            output["acoustic_target_mel"] = acoustic_target
            output["acoustic_target_weight"] = acoustic_weight
        self._align_output_nonpadding(output)
        if acoustic_target_is_retimed:
            self._align_retimed_pitch_targets_to_output(output)
        if hasattr(output.get("acoustic_target_mel"), "dim") and output["acoustic_target_mel"].dim() >= 2:
            output["acoustic_target_length_frames_after_align"] = float(output["acoustic_target_mel"].size(1))
        if hasattr(output.get("mel_out"), "dim") and output["mel_out"].dim() >= 2:
            output["acoustic_output_length_frames_after_align"] = float(output["mel_out"].size(1))
        return output["acoustic_target_mel"], output["acoustic_target_weight"]

    def add_style_losses(self, output, losses, *, schedule_only_stage: bool) -> None:
        if (
            not hparams["style"]
            or schedule_only_stage
            or getattr(self.owner.model, "rhythm_minimal_style_only", False)
        ):
            return
        if (
            self.owner.global_step > hparams["forcing"]
            and self.owner.global_step < hparams["random_speaker_steps"]
            and "gloss" in output
        ):
            losses["gloss"] = output["gloss"]
        if self.owner.global_step > hparams["vq_start"] and "vq_loss" in output and "ppl" in output:
            losses["vq_loss"] = output["vq_loss"]
            losses["ppl"] = output["ppl"]


__all__ = ["RhythmTaskRuntimeSupport"]

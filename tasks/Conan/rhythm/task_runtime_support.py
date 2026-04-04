from __future__ import annotations

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
            pause_boundary_weight=self.owner._resolve_rhythm_pause_boundary_weight(),
            budget_raw_weight=float(hparams.get("rhythm_budget_raw_weight", 1.0)),
            budget_exec_weight=float(hparams.get("rhythm_budget_exec_weight", 0.25)),
            feasible_debt_weight=float(hparams.get("rhythm_feasible_debt_weight", 0.05)),
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
            "content_lengths": sample.get("mel_lengths"),
            "ref_lengths": sample.get("ref_mel_lengths"),
            "rhythm_apply_override": rhythm_apply_override,
            "rhythm_state": rhythm_state,
            "rhythm_ref_conditioning": rhythm_ref_conditioning,
            "rhythm_source_cache": self.owner._collect_rhythm_source_cache(sample),
            "rhythm_offline_source_cache": runtime_offline_source_cache,
            "disable_acoustic_train_path": disable_acoustic_train_path,
        }

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
        output["acoustic_target_mel"] = acoustic_target
        output["acoustic_target_is_retimed"] = bool(acoustic_target_is_retimed)
        output["acoustic_target_weight"] = acoustic_weight
        output["acoustic_target_source"] = acoustic_target_source
        output["rhythm_pitch_supervision_disabled"] = float(disable_source_pitch_supervision)
        output["disable_acoustic_train_path"] = float(disable_acoustic_train_path)
        if acoustic_target_is_retimed:
            mel_out_aligned, acoustic_target, acoustic_weight = self.owner._align_acoustic_target_to_output(
                output["mel_out"],
                acoustic_target,
                acoustic_weight,
            )
            output["mel_out"] = mel_out_aligned
            output["acoustic_target_mel"] = acoustic_target
            output["acoustic_target_weight"] = acoustic_weight
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

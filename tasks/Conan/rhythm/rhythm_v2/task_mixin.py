from __future__ import annotations

import torch

from modules.Conan.rhythm.prefix_state import build_prefix_state_from_exec_torch
from tasks.Conan.rhythm.distill_confidence import build_runtime_distill_confidence_bundle as build_runtime_distill_confidence_bundle_helper
from tasks.Conan.rhythm.reference_regularization import (
    build_predicted_compact_reference_descriptor,
    build_target_compact_reference_descriptor,
    compute_descriptor_consistency_loss,
    compute_group_reference_contrastive_loss,
)
from tasks.Conan.rhythm.runtime_teacher_supervision import (
    build_runtime_teacher_supervision_targets as build_runtime_teacher_supervision_targets_helper,
)
from tasks.Conan.rhythm.rhythm_v2.runtime_modes import build_legacy_v2_ref_conditioning, collect_legacy_planner_runtime_outputs
from tasks.Conan.rhythm.rhythm_v2.targets import (
    build_identity_rhythm_loss_targets,
    build_rhythm_loss_targets_from_sample,
)
from utils.commons.hparams import hparams


class RhythmV2TaskMixin:
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
        rhythm_ref_conditioning = build_legacy_v2_ref_conditioning(
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
        output.update(collect_legacy_planner_runtime_outputs(execution))
        losses = {}
        self.add_rhythm_loss(output, sample, losses)
        self._task_runtime_support().route_conan_losses(losses, schedule_only_stage=False)
        return losses, output

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

    def _build_rhythm_v2_loss_targets(self, output, sample):
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


__all__ = ["RhythmV2TaskMixin"]

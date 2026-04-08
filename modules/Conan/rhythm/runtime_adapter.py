from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from .bridge import attach_rhythm_outputs, run_rhythm_frontend
from .compat import resolve_bool_alias, resolve_float_alias, resolve_phase_decoupled_flag
from .factory import build_streaming_rhythm_module_from_hparams
from .stages import (
    detect_rhythm_stage,
    resolve_runtime_dual_mode_teacher_enable,
    resolve_runtime_offline_teacher_enable,
    resolve_teacher_as_main,
)
from .supervision import build_online_retimed_bundle
from .unit_frontend import RhythmUnitFrontend


def _resolve_phase_decoupled_override(runtime_overrides: dict) -> bool | None:
    return resolve_phase_decoupled_flag(
        default=None,
        phase_decoupled_timing=runtime_overrides.pop("phase_decoupled_timing", None),
        phase_free_timing=runtime_overrides.pop("phase_free_timing", None),
        where="ConanRhythmAdapter.runtime_overrides",
    )


class ConanRhythmAdapter(nn.Module):
    """Own the rhythm-specific runtime path used by Conan.

    This pulls the strong-rhythm stack out of modules/Conan/Conan.py so the
    acoustic model no longer has to directly manage all scheduler/projector/
    teacher runtime details.

    This adapter is the public runtime entrypoint. `bridge.run_rhythm_frontend`
    is the lower-level helper used underneath for tests and integration glue.
    """

    def __init__(self, hparams, hidden_size: int) -> None:
        super().__init__()
        self.hparams = hparams
        self.unit_frontend = RhythmUnitFrontend(
            silent_token=hparams.get("silent_token", 57),
            separator_aware=bool(hparams.get("rhythm_separator_aware", True)),
            tail_open_units=int(hparams.get("rhythm_tail_open_units", 1)),
        )
        self.module = build_streaming_rhythm_module_from_hparams(hparams)
        self.pause_state = nn.Parameter(torch.zeros(hidden_size))
        self.render_phase_mlp = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.render_phase_gain = nn.Parameter(
            torch.tensor(float(hparams.get("rhythm_renderer_phase_init_gain", 0.10)))
        )

    def _render_frame_state_post(
        self,
        frame_states: torch.Tensor,
        frame_phase_features: torch.Tensor,
        blank_mask: torch.Tensor,
        total_mask: torch.Tensor,
    ) -> torch.Tensor:
        edge_scale = (0.5 + 0.5 * frame_phase_features[..., 4:5]).clamp(0.0, 1.0)
        blank_feat = blank_mask.unsqueeze(-1).float()
        phase_input = torch.cat([frame_phase_features.float(), blank_feat], dim=-1)
        phase_residual = self.render_phase_mlp(phase_input)
        gain = torch.tanh(self.render_phase_gain).view(1, 1, 1)
        frame_states = frame_states + gain * edge_scale * phase_residual
        return frame_states * total_mask.unsqueeze(-1)

    def render_frame_state_post(
        self,
        frame_states: torch.Tensor,
        frame_phase_features: torch.Tensor,
        blank_mask: torch.Tensor,
        total_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._render_frame_state_post(
            frame_states=frame_states,
            frame_phase_features=frame_phase_features,
            blank_mask=blank_mask,
            total_mask=total_mask,
        )

    def _resolve_pause_topk_ratio(self, *, infer: bool, global_steps: int) -> float | None:
        final_ratio = float(self.hparams.get("rhythm_projector_pause_topk_ratio", 0.35))
        if infer:
            return float(max(0.0, min(1.0, final_ratio)))
        start_ratio = float(self.hparams.get("rhythm_projector_pause_topk_ratio_train_start", 1.0))
        end_ratio = float(self.hparams.get("rhythm_projector_pause_topk_ratio_train_end", final_ratio))
        warmup_steps = int(self.hparams.get("rhythm_projector_pause_topk_ratio_warmup_steps", 0) or 0)
        anneal_steps = int(self.hparams.get("rhythm_projector_pause_topk_ratio_anneal_steps", 20000) or 0)
        start_ratio = float(max(0.0, min(1.0, start_ratio)))
        end_ratio = float(max(0.0, min(1.0, end_ratio)))
        if anneal_steps <= 0:
            return end_ratio
        if global_steps <= warmup_steps:
            return start_ratio
        progress = min(max((int(global_steps) - warmup_steps) / float(anneal_steps), 0.0), 1.0)
        return float(start_ratio + (end_ratio - start_ratio) * progress)

    def _resolve_source_boundary_scale(
        self,
        *,
        infer: bool,
        global_steps: int,
        teacher: bool = False,
    ) -> float | None:
        if teacher:
            final_scale = float(
                self.hparams.get(
                    "rhythm_teacher_source_boundary_scale",
                    self.hparams.get("rhythm_source_boundary_scale", 1.0),
                )
            )
            return max(0.0, final_scale)
        final_scale = float(self.hparams.get("rhythm_source_boundary_scale", 1.0))
        if infer:
            return max(0.0, final_scale)
        start_scale = float(self.hparams.get("rhythm_source_boundary_scale_train_start", 1.0))
        end_scale = float(self.hparams.get("rhythm_source_boundary_scale_train_end", final_scale))
        warmup_steps = int(self.hparams.get("rhythm_source_boundary_scale_warmup_steps", 0) or 0)
        anneal_steps = int(self.hparams.get("rhythm_source_boundary_scale_anneal_steps", 20000) or 0)
        start_scale = max(0.0, start_scale)
        end_scale = max(0.0, end_scale)
        if anneal_steps <= 0:
            return end_scale
        if global_steps <= warmup_steps:
            return start_scale
        progress = min(max((int(global_steps) - warmup_steps) / float(anneal_steps), 0.0), 1.0)
        return float(start_scale + (end_scale - start_scale) * progress)

    def resolve_pause_topk_ratio(self, *, infer: bool, global_steps: int) -> float | None:
        return self._resolve_pause_topk_ratio(infer=infer, global_steps=global_steps)

    def resolve_source_boundary_scale(
        self,
        *,
        infer: bool,
        global_steps: int,
        teacher: bool = False,
    ) -> float | None:
        return self._resolve_source_boundary_scale(
            infer=infer,
            global_steps=global_steps,
            teacher=teacher,
        )

    def forward(
        self,
        *,
        ret: dict,
        content: torch.Tensor,
        ref: torch.Tensor | None,
        target: torch.Tensor | None,
        f0: torch.Tensor | None,
        uv: torch.Tensor | None,
        infer: bool,
        global_steps: int,
        content_embed: torch.Tensor,
        tgt_nonpadding: torch.Tensor,
        content_lengths: torch.Tensor | None = None,
        rhythm_state=None,
        rhythm_ref_conditioning=None,
        rhythm_apply_override=None,
        rhythm_runtime_overrides: dict | None = None,
        rhythm_source_cache: dict | None = None,
        rhythm_offline_source_cache: dict | None = None,
        speech_state_fn=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if ref is None and rhythm_ref_conditioning is None:
            return content_embed, tgt_nonpadding, f0, uv
        stage = detect_rhythm_stage(self.hparams)
        projector_pause_topk_ratio_override = self.resolve_pause_topk_ratio(
            infer=bool(infer),
            global_steps=int(global_steps),
        )
        source_boundary_scale_override = self.resolve_source_boundary_scale(
            infer=bool(infer),
            global_steps=int(global_steps),
            teacher=False,
        )
        runtime_overrides = dict(rhythm_runtime_overrides or {})
        projector_pause_topk_ratio_override = runtime_overrides.pop(
            "projector_pause_topk_ratio_override",
            projector_pause_topk_ratio_override,
        )
        source_boundary_scale_override = runtime_overrides.pop(
            "source_boundary_scale_override",
            source_boundary_scale_override,
        )
        configured_teacher_runtime_enabled = resolve_runtime_offline_teacher_enable(
            self.hparams,
            stage=stage,
        )
        module_supports_teacher_runtime = bool(
            getattr(
                self.module,
                "enable_learned_offline_teacher",
                configured_teacher_runtime_enabled,
            )
        )
        teacher_runtime_available = bool(
            configured_teacher_runtime_enabled and module_supports_teacher_runtime
        )
        teacher_as_main_requested = resolve_teacher_as_main(
            self.hparams,
            stage=stage,
            infer=bool(infer),
        )
        dual_mode_teacher_requested = resolve_runtime_dual_mode_teacher_enable(
            self.hparams,
            stage=stage,
            infer=bool(infer),
        )
        algorithmic_teacher_enabled = (
            bool(self.hparams.get("rhythm_enable_algorithmic_teacher", False))
            and not bool(infer)
            and stage in {"legacy_dual_mode_kd", "transitional"}
        )
        teacher_runtime_branch_requested = (
            bool(teacher_as_main_requested) or bool(dual_mode_teacher_requested)
        )
        teacher_source_boundary_scale_override = None
        if teacher_runtime_branch_requested or algorithmic_teacher_enabled:
            teacher_source_boundary_scale_override = self.resolve_source_boundary_scale(
                infer=bool(infer),
                global_steps=int(global_steps),
                teacher=True,
            )
        if teacher_runtime_branch_requested or algorithmic_teacher_enabled:
            teacher_source_boundary_scale_override = runtime_overrides.pop(
                "teacher_source_boundary_scale_override",
                teacher_source_boundary_scale_override,
            )
        teacher_projector_force_full_commit = None
        teacher_projector_soft_pause_selection = None
        if teacher_runtime_branch_requested:
            teacher_projector_force_full_commit = bool(
                runtime_overrides.pop(
                    "teacher_projector_force_full_commit",
                    self.hparams.get("rhythm_teacher_projector_force_full_commit", True),
                )
            )
            teacher_projector_soft_pause_selection = resolve_bool_alias(
                default=self.hparams.get("rhythm_teacher_projector_soft_pause_selection", None),
                canonical_value=runtime_overrides.pop("teacher_projector_soft_pause_selection", None),
                legacy_value=runtime_overrides.pop("teacher_projector_soft_pause_selection_override", None),
                canonical_name="teacher_projector_soft_pause_selection",
                legacy_name="teacher_projector_soft_pause_selection_override",
                where="ConanRhythmAdapter.runtime_overrides",
            )
        trace_active_tail_only = runtime_overrides.pop("trace_active_tail_only", None)
        if trace_active_tail_only is not None:
            trace_active_tail_only = bool(trace_active_tail_only)
        trace_offset_lookahead_units = runtime_overrides.pop("trace_offset_lookahead_units", None)
        if trace_offset_lookahead_units is not None:
            trace_offset_lookahead_units = int(trace_offset_lookahead_units)
        trace_cold_start_min_visible_units = runtime_overrides.pop(
            "trace_cold_start_min_visible_units",
            None,
        )
        if trace_cold_start_min_visible_units is not None:
            trace_cold_start_min_visible_units = int(trace_cold_start_min_visible_units)
        trace_cold_start_full_visible_units = runtime_overrides.pop(
            "trace_cold_start_full_visible_units",
            None,
        )
        if trace_cold_start_full_visible_units is not None:
            trace_cold_start_full_visible_units = int(trace_cold_start_full_visible_units)
        phase_decoupled_timing = _resolve_phase_decoupled_override(runtime_overrides)
        phase_decoupled_phrase_gate_boundary_threshold = resolve_float_alias(
            default=None,
            canonical_value=runtime_overrides.pop("phase_decoupled_phrase_gate_boundary_threshold", None),
            legacy_value=runtime_overrides.pop("phase_decoupled_phrase_boundary_threshold", None),
            canonical_name="phase_decoupled_phrase_gate_boundary_threshold",
            legacy_name="phase_decoupled_phrase_boundary_threshold",
            where="ConanRhythmAdapter.runtime_overrides",
        )
        phase_decoupled_phrase_gate_boundary_threshold = resolve_float_alias(
            default=phase_decoupled_phrase_gate_boundary_threshold,
            canonical_value=phase_decoupled_phrase_gate_boundary_threshold,
            legacy_value=runtime_overrides.pop("phase_free_phrase_boundary_threshold", None),
            canonical_name="phase_decoupled_phrase_gate_boundary_threshold",
            legacy_name="phase_free_phrase_boundary_threshold",
            where="ConanRhythmAdapter.runtime_overrides",
        )
        phase_decoupled_boundary_style_residual_scale = runtime_overrides.pop(
            "phase_decoupled_boundary_style_residual_scale",
            None,
        )
        if phase_decoupled_boundary_style_residual_scale is not None:
            phase_decoupled_boundary_style_residual_scale = float(
                phase_decoupled_boundary_style_residual_scale
            )
        debt_control_scale = runtime_overrides.pop("debt_control_scale", None)
        if debt_control_scale is not None:
            debt_control_scale = float(debt_control_scale)
        debt_pause_priority = runtime_overrides.pop("debt_pause_priority", None)
        if debt_pause_priority is not None:
            debt_pause_priority = float(debt_pause_priority)
        debt_speech_priority = runtime_overrides.pop("debt_speech_priority", None)
        if debt_speech_priority is not None:
            debt_speech_priority = float(debt_speech_priority)
        projector_debt_leak = runtime_overrides.pop("projector_debt_leak", None)
        if projector_debt_leak is not None:
            projector_debt_leak = float(projector_debt_leak)
        projector_debt_max_abs = runtime_overrides.pop("projector_debt_max_abs", None)
        if projector_debt_max_abs is not None:
            projector_debt_max_abs = float(projector_debt_max_abs)
        projector_debt_correction_horizon = runtime_overrides.pop(
            "projector_debt_correction_horizon",
            None,
        )
        if projector_debt_correction_horizon is not None:
            projector_debt_correction_horizon = float(projector_debt_correction_horizon)
        ret["rhythm_stage"] = stage
        ret["rhythm_teacher_runtime_enabled"] = float(teacher_runtime_available)
        ret["rhythm_teacher_as_main_requested"] = float(bool(teacher_as_main_requested))
        if projector_pause_topk_ratio_override is not None:
            ret["rhythm_projector_pause_topk_ratio"] = torch.full(
                (content.size(0), 1),
                float(projector_pause_topk_ratio_override),
                dtype=content_embed.dtype,
                device=content.device,
            )
        if source_boundary_scale_override is not None:
            ret["rhythm_source_boundary_scale"] = torch.full(
                (content.size(0), 1),
                float(source_boundary_scale_override),
                dtype=content_embed.dtype,
                device=content.device,
            )
        if teacher_source_boundary_scale_override is not None:
            ret["rhythm_teacher_source_boundary_scale"] = torch.full(
                (content.size(0), 1),
                float(teacher_source_boundary_scale_override),
                dtype=content_embed.dtype,
                device=content.device,
            )
        if teacher_projector_force_full_commit is not None:
            ret["rhythm_teacher_projector_force_full_commit"] = torch.full(
                (content.size(0), 1),
                1.0 if teacher_projector_force_full_commit else 0.0,
                dtype=content_embed.dtype,
                device=content.device,
            )
        if teacher_projector_soft_pause_selection is not None:
            ret["rhythm_teacher_projector_soft_pause_selection"] = torch.full(
                (content.size(0), 1),
                1.0 if teacher_projector_soft_pause_selection else 0.0,
                dtype=content_embed.dtype,
                device=content.device,
            )
        rhythm_bundle = run_rhythm_frontend(
            rhythm_enable_v2=True,
            rhythm_unit_frontend=self.unit_frontend,
            rhythm_module=self.module,
            content=content,
            ref=ref,
            infer=infer,
            content_lengths=content_lengths,
            rhythm_state=rhythm_state,
            rhythm_ref_conditioning=rhythm_ref_conditioning,
            rhythm_source_cache=rhythm_source_cache,
            rhythm_offline_source_cache=rhythm_offline_source_cache,
            enable_dual_mode_teacher=dual_mode_teacher_requested,
            enable_learned_offline_teacher=teacher_runtime_available,
            enable_algorithmic_teacher=algorithmic_teacher_enabled,
            teacher_as_main=teacher_as_main_requested,
            projector_pause_topk_ratio_override=projector_pause_topk_ratio_override,
            source_boundary_scale_override=source_boundary_scale_override,
            teacher_source_boundary_scale_override=teacher_source_boundary_scale_override,
            trace_horizon=runtime_overrides.pop("trace_horizon", None),
            trace_active_tail_only=trace_active_tail_only,
            trace_offset_lookahead_units=trace_offset_lookahead_units,
            trace_cold_start_min_visible_units=trace_cold_start_min_visible_units,
            trace_cold_start_full_visible_units=trace_cold_start_full_visible_units,
            phase_decoupled_timing=phase_decoupled_timing,
            phase_decoupled_phrase_gate_boundary_threshold=phase_decoupled_phrase_gate_boundary_threshold,
            phase_decoupled_boundary_style_residual_scale=phase_decoupled_boundary_style_residual_scale,
            debt_control_scale=debt_control_scale,
            debt_pause_priority=debt_pause_priority,
            debt_speech_priority=debt_speech_priority,
            projector_debt_leak=projector_debt_leak,
            projector_debt_max_abs=projector_debt_max_abs,
            projector_debt_correction_horizon=projector_debt_correction_horizon,
            projector_reuse_prefix=bool(runtime_overrides.pop("projector_reuse_prefix", True)),
            projector_force_full_commit=bool(runtime_overrides.pop("projector_force_full_commit", False)),
            teacher_projector_force_full_commit=bool(teacher_projector_force_full_commit)
            if teacher_projector_force_full_commit is not None
            else True,
            teacher_projector_soft_pause_selection=teacher_projector_soft_pause_selection,
            streaming_prefix_train=bool(self.hparams.get("rhythm_streaming_prefix_train", False)),
        )
        if runtime_overrides:
            warnings.warn(
                "Unused rhythm runtime overrides: "
                + ", ".join(sorted(str(key) for key in runtime_overrides.keys())),
                stacklevel=2,
            )
        content_embed, tgt_nonpadding = attach_rhythm_outputs(
            ret=ret,
            rhythm_bundle=rhythm_bundle,
            content_embed=content_embed,
            tgt_nonpadding=tgt_nonpadding,
            hparams=self.hparams,
            infer=infer,
            rhythm_apply_override=rhythm_apply_override,
            speech_state_fn=speech_state_fn,
            pause_state=self.pause_state,
            frame_state_post_fn=self._render_frame_state_post,
        )
        if bool(ret.get("rhythm_apply_render", 0.0)) and target is not None:
            frame_plan = ret.get("rhythm_frame_plan")
            if frame_plan is not None:
                online_bundle = build_online_retimed_bundle(
                    mel=target,
                    frame_plan=frame_plan,
                    f0=f0 if f0 is not None and uv is not None else None,
                    uv=uv if f0 is not None and uv is not None else None,
                    pause_frame_weight=float(self.hparams.get("rhythm_retimed_pause_frame_weight", 0.20)),
                    stretch_weight_min=float(self.hparams.get("rhythm_retimed_stretch_weight_min", 0.35)),
                )
                ret["rhythm_online_retimed_mel_tgt"] = online_bundle["mel_tgt"]
                ret["rhythm_online_retimed_frame_weight"] = online_bundle["frame_weight"]
                ret["rhythm_online_retimed_mel_len"] = online_bundle["mel_len"]
                if "f0_tgt" in online_bundle:
                    ret["retimed_f0_tgt"] = online_bundle["f0_tgt"]
                if "uv_tgt" in online_bundle:
                    ret["retimed_uv_tgt"] = online_bundle["uv_tgt"]
                if (
                    bool(self.hparams.get("rhythm_use_retimed_pitch_target", False))
                    and "retimed_f0_tgt" in ret
                    and "retimed_uv_tgt" in ret
                ):
                    f0 = ret["retimed_f0_tgt"]
                    uv = ret["retimed_uv_tgt"]
        return content_embed, tgt_nonpadding, f0, uv

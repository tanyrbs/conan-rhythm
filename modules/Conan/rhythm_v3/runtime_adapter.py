from __future__ import annotations

import torch
import torch.nn as nn

from modules.Conan.rhythm.bridge import resolve_rhythm_apply_mode
from modules.Conan.rhythm.renderer import render_rhythm_sequence

from .contracts import (
    ReferenceDurationMemory,
    move_duration_runtime_state,
    move_reference_duration_memory,
    move_source_unit_batch,
)
from .module import StreamingDurationModule
from .unit_frontend import DurationUnitFrontend


class ConanDurationAdapter(nn.Module):
    def __init__(self, hparams, hidden_size: int, *, vocab_size: int) -> None:
        super().__init__()
        self.hparams = hparams
        self.unit_frontend = DurationUnitFrontend(
            vocab_size=vocab_size,
            silent_token=hparams.get("silent_token", 57),
            separator_aware=bool(hparams.get("rhythm_separator_aware", True)),
            tail_open_units=int(hparams.get("rhythm_tail_open_units", 1)),
            anchor_hidden_size=int(hparams.get("rhythm_anchor_hidden_size", 128)),
            anchor_min_frames=float(hparams.get("rhythm_anchor_min_frames", 1.0)),
            anchor_max_frames=float(hparams.get("rhythm_anchor_max_frames", 12.0)),
        )
        self.module = StreamingDurationModule(
            vocab_size=vocab_size,
            hidden_size=int(hparams.get("rhythm_hidden_size", hidden_size)),
            role_dim=int(hparams.get("rhythm_role_dim", 64)),
            codebook_size=int(hparams.get("rhythm_role_codebook_size", 12)),
            role_window_left=int(hparams.get("rhythm_role_window_left", 4)),
            role_window_right=int(hparams.get("rhythm_role_window_right", 0)),
            trace_bins=int(hparams.get("rhythm_trace_bins", 24)),
            coverage_floor=float(hparams.get("rhythm_ref_coverage_floor", 0.05)),
            prefix_drift_gain=float(hparams.get("rhythm_prefix_drift_gain", 0.25)),
            prefix_drift_clip=float(hparams.get("rhythm_prefix_drift_clip", 0.05)),
            max_logstretch=float(hparams.get("rhythm_max_logstretch", 1.0)),
            anti_pos_bins=int(hparams.get("rhythm_anti_pos_bins", 8)),
            anti_pos_grl_scale=float(hparams.get("rhythm_anti_pos_grl_scale", 1.0)),
        )
        self.pause_state = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self.render_phase_mlp = None
        self.render_phase_gain = None

    def render_frame_state_post(
        self,
        frame_states: torch.Tensor,
        frame_phase_features: torch.Tensor,
        blank_mask: torch.Tensor,
        total_mask: torch.Tensor,
    ) -> torch.Tensor:
        del frame_phase_features, blank_mask
        return frame_states * total_mask.unsqueeze(-1)

    @staticmethod
    def resolve_pause_topk_ratio(*, infer: bool, global_steps: int):
        del infer, global_steps
        return None

    @staticmethod
    def resolve_source_boundary_scale(*, infer: bool, global_steps: int, teacher: bool = False):
        del infer, global_steps, teacher
        return None

    def _build_source_batch(
        self,
        *,
        content: torch.Tensor,
        content_lengths: torch.Tensor | None,
        rhythm_source_cache: dict | None,
        infer: bool,
    ):
        if rhythm_source_cache is not None:
            return self.unit_frontend.from_precomputed(
                content_units=rhythm_source_cache["content_units"],
                dur_anchor_src=rhythm_source_cache["dur_anchor_src"],
                unit_mask=rhythm_source_cache.get("unit_mask"),
                open_run_mask=rhythm_source_cache.get("open_run_mask"),
                sealed_mask=rhythm_source_cache.get("sealed_mask"),
                sep_hint=rhythm_source_cache.get("sep_hint"),
                boundary_confidence=rhythm_source_cache.get("boundary_confidence"),
                unit_anchor_base=rhythm_source_cache.get("unit_anchor_base"),
                edge_cue=rhythm_source_cache.get("edge_cue"),
            )
        return self.unit_frontend.from_content_tensor(
            content,
            content_lengths=content_lengths,
            mark_last_open=bool(infer) or bool(self.hparams.get("rhythm_streaming_prefix_train", False)),
        )

    def _attach_runtime_outputs(
        self,
        *,
        ret: dict,
        source_batch,
        ref_memory: ReferenceDurationMemory,
        execution,
    ) -> None:
        ret["rhythm_version"] = "v3"
        ret["rhythm_teacher_as_main"] = 0.0
        ret["rhythm_unit_batch"] = source_batch
        ret["rhythm_execution"] = execution
        ret["rhythm_state_next"] = execution.next_state
        ret["rhythm_ref_conditioning"] = ref_memory
        ret["ref_rhythm_stats"] = ref_memory.raw_stats
        ret["ref_rhythm_trace"] = ref_memory.raw_trace
        ret["global_rate"] = ref_memory.global_rate
        ret["role_value"] = ref_memory.role_value
        ret["role_coverage"] = ref_memory.role_coverage
        ret["role_attention"] = execution.role_attention
        ret["dur_logratio_unit"] = execution.unit_logstretch
        ret["speech_duration_exec"] = execution.speech_duration_exec
        ret["blank_duration_exec"] = execution.blank_duration_exec
        ret["pause_after_exec"] = execution.pause_after_exec
        ret["effective_duration_exec"] = execution.effective_duration_exec
        ret["commit_frontier"] = execution.commit_frontier
        ret["source_boundary_cue"] = source_batch.edge_cue
        ret["unit_anchor_base"] = source_batch.unit_anchor_base
        ret["source_runlen_src"] = source_batch.source_runlen_src
        ret["sealed_mask"] = source_batch.sealed_mask
        ret["boundary_confidence"] = source_batch.boundary_confidence
        if ref_memory.prompt_reconstruction is not None:
            ret["rhythm_prompt_reconstruction"] = ref_memory.prompt_reconstruction
        if ref_memory.prompt_rel_stretch is not None:
            ret["rhythm_prompt_rel_stretch"] = ref_memory.prompt_rel_stretch
        if ref_memory.prompt_mask is not None:
            ret["rhythm_prompt_mask"] = ref_memory.prompt_mask
        if execution.anti_pos_logits is not None:
            ret["rhythm_anti_pos_logits"] = execution.anti_pos_logits
        if execution.frame_plan is not None:
            ret["rhythm_frame_plan"] = execution.frame_plan

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
        ref_lengths: torch.Tensor | None = None,
        rhythm_state=None,
        rhythm_ref_conditioning=None,
        rhythm_apply_override=None,
        rhythm_runtime_overrides: dict | None = None,
        rhythm_source_cache: dict | None = None,
        rhythm_offline_source_cache: dict | None = None,
        speech_state_fn=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        del target, global_steps, rhythm_runtime_overrides, rhythm_offline_source_cache
        if ref is None and rhythm_ref_conditioning is None:
            return content_embed, tgt_nonpadding, f0, uv
        source_batch = self._build_source_batch(
            content=content,
            content_lengths=content_lengths,
            rhythm_source_cache=rhythm_source_cache,
            infer=infer,
        )
        source_batch = move_source_unit_batch(source_batch, device=content.device)
        rhythm_state = move_duration_runtime_state(rhythm_state, device=content.device)
        ref_memory = self.module.build_reference_conditioning(
            ref_conditioning=rhythm_ref_conditioning,
            ref_mel=ref,
            ref_lengths=ref_lengths,
        )
        ref_memory = move_reference_duration_memory(
            ref_memory,
            device=content.device,
            dtype=content_embed.dtype if content_embed.is_floating_point() else None,
        )
        execution = self.module(
            source_batch=source_batch,
            ref_memory=ref_memory,
            state=rhythm_state,
        )
        self._attach_runtime_outputs(
            ret=ret,
            source_batch=source_batch,
            ref_memory=ref_memory,
            execution=execution,
        )
        apply_rhythm_render = resolve_rhythm_apply_mode(
            self.hparams,
            infer=infer,
            override=rhythm_apply_override,
        )
        ret["rhythm_apply_render"] = float(apply_rhythm_render)
        if not apply_rhythm_render:
            return content_embed, tgt_nonpadding, f0, uv
        if speech_state_fn is None:
            raise ValueError("speech_state_fn is required when rhythm render is enabled.")
        rendered = render_rhythm_sequence(
            content_units=source_batch.content_units,
            silent_token=int(self.hparams.get("silent_token", 57)),
            speech_state_fn=speech_state_fn,
            pause_state=self.pause_state.to(device=content_embed.device, dtype=content_embed.dtype),
            frame_plan=execution.frame_plan,
            frame_state_post_fn=None,
        )
        ret["content"] = rendered.frame_tokens
        ret["content_rhythm_rendered"] = rendered.frame_tokens
        ret["content_embed_proj_rhythm"] = rendered.frame_states
        ret["rhythm_total_mask"] = rendered.total_mask
        ret["rhythm_speech_mask"] = rendered.speech_mask
        ret["rhythm_blank_mask"] = rendered.blank_mask
        ret["rhythm_render_slot_index"] = rendered.frame_slot_index
        ret["rhythm_render_unit_index"] = rendered.frame_unit_index
        ret["rhythm_render_phase_features"] = rendered.frame_phase_features
        ret["rhythm_frame_plan"] = rendered.frame_plan
        return rendered.frame_states, rendered.total_mask[:, :, None], f0, uv

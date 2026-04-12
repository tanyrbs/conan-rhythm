from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from .contracts import (
    ReferenceDurationMemory,
    collect_duration_v3_source_cache,
    move_duration_runtime_state,
    move_reference_duration_memory,
    move_source_unit_batch,
)
from .bridge import resolve_rhythm_apply_mode
from .frame_plan import RhythmFramePlan, build_frame_plan_from_execution
from .module import MixedEffectsDurationModule
from .renderer import RenderedRhythmSequence, render_rhythm_sequence
from .unit_frontend import DurationUnitFrontend
from tasks.Conan.rhythm.duration_v3.task_config import is_duration_v3_prompt_summary_backbone

_PROMPT_UNIT_REQUIRED_KEYS = (
    "prompt_content_units",
    "prompt_duration_obs",
)


class ConanDurationAdapter(nn.Module):
    def __init__(self, hparams, hidden_size: int, *, vocab_size: int) -> None:
        super().__init__()
        self.hparams = hparams
        self.baseline_train_mode = str(hparams.get("rhythm_v3_baseline_train_mode", "joint") or "joint").strip().lower()
        self.silent_token = int(hparams.get("silent_token", 57))
        self.emit_silence_runs = bool(hparams.get("rhythm_v3_emit_silence_runs", True))
        self.prompt_summary_backbone = is_duration_v3_prompt_summary_backbone(
            hparams.get("rhythm_v3_backbone", "global_only")
        )
        for removed_key, replacement in (
            ("rhythm_coarse_bins", "rhythm_progress_bins"),
            ("rhythm_coarse_support_tau", "rhythm_progress_support_tau"),
        ):
            if removed_key in hparams:
                raise ValueError(f"{removed_key} has been removed from rhythm_v3. Use {replacement} instead.")
        if self.prompt_summary_backbone and not self.emit_silence_runs:
            raise ValueError(
                "rhythm_v3 prompt_summary/unit_run backbone requires rhythm_v3_emit_silence_runs=true."
            )
        self.unit_frontend = DurationUnitFrontend(
            vocab_size=vocab_size,
            silent_token=self.silent_token,
            separator_aware=bool(hparams.get("rhythm_separator_aware", True)),
            tail_open_units=int(hparams.get("rhythm_tail_open_units", 1)),
            emit_silence_runs=self.emit_silence_runs,
            debounce_min_run_frames=int(hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
            anchor_hidden_size=int(hparams.get("rhythm_anchor_hidden_size", 128)),
            anchor_min_frames=float(hparams.get("rhythm_anchor_min_frames", 1.0)),
            anchor_max_frames=float(hparams.get("rhythm_anchor_max_frames", 12.0)),
            phrase_boundary_threshold=float(hparams.get("rhythm_source_phrase_threshold", 0.55)),
        )
        streaming_mode = str(hparams.get("rhythm_streaming_mode", "strict") or "strict")
        micro_lookahead_units = hparams.get("rhythm_micro_lookahead_units")
        self.module = MixedEffectsDurationModule(
            vocab_size=vocab_size,
            hidden_size=int(hparams.get("rhythm_hidden_size", hidden_size)),
            basis_rank=int(hparams.get("rhythm_response_rank", 12)),
            summary_dim=int(
                hparams.get(
                    "rhythm_summary_dim",
                    hparams.get("rhythm_role_dim", hparams.get("rhythm_hidden_size", hidden_size)),
                )
            ),
            num_summary_slots=int(hparams.get("rhythm_num_summary_slots", hparams.get("rhythm_num_role_slots", 12))),
            role_cov_floor=float(hparams.get("rhythm_prompt_cov_floor", 0.05)),
            summary_pool_speech_only=bool(hparams.get("rhythm_v3_summary_pool_speech_only", True)),
            summary_use_unit_embedding=bool(hparams.get("rhythm_v3_summary_use_unit_embedding", False)),
            max_logstretch=float(hparams.get("rhythm_max_logstretch", 1.2)),
            max_silence_logstretch=float(hparams.get("rhythm_v3_silence_max_logstretch", 0.35)),
            local_cold_start_runs=int(hparams.get("rhythm_v3_local_cold_start_runs", 2)),
            local_short_run_min_duration=float(hparams.get("rhythm_v3_local_short_run_min_duration", 2.0)),
            local_rate_decay=float(hparams.get("rhythm_v3_local_rate_decay", 0.95)),
            short_gap_silence_scale=float(hparams.get("rhythm_v3_short_gap_silence_scale", 0.35)),
            leading_silence_scale=float(hparams.get("rhythm_v3_leading_silence_scale", 0.0)),
            response_window_left=int(hparams.get("rhythm_response_window_left", 4)),
            response_window_right=int(hparams.get("rhythm_response_window_right", 0)),
            streaming_mode=streaming_mode,
            micro_lookahead_units=(None if micro_lookahead_units is None else int(micro_lookahead_units)),
            ridge_lambda=float(hparams.get("rhythm_operator_ridge_lambda", 1.0)),
            global_shrink_tau=float(hparams.get("rhythm_global_shrink_tau", 8.0)),
            progress_support_tau=float(hparams.get("rhythm_progress_support_tau", 8.0)),
            progress_bins=int(hparams.get("rhythm_progress_bins", 4)),
            ridge_support_tau=float(hparams.get("rhythm_operator_support_tau", 8.0)),
            operator_holdout_ratio=float(hparams.get("rhythm_operator_holdout_ratio", 0.30)),
            min_operator_support_factor=float(hparams.get("rhythm_operator_min_support_factor", 1.0)),
            backbone_mode=hparams.get("rhythm_v3_backbone"),
            warp_mode=hparams.get("rhythm_v3_warp_mode"),
            prefix_budget_pos=int(hparams.get("rhythm_v3_prefix_budget_pos", hparams.get("rhythm_v3_unit_budget_pos", 24))),
            prefix_budget_neg=int(hparams.get("rhythm_v3_prefix_budget_neg", hparams.get("rhythm_v3_unit_budget_neg", 24))),
            dynamic_budget_ratio=float(hparams.get("rhythm_v3_dynamic_budget_ratio", 0.0) or 0.0),
            min_prefix_budget=int(hparams.get("rhythm_v3_min_prefix_budget", 0) or 0),
            max_prefix_budget=int(hparams.get("rhythm_v3_max_prefix_budget", 0) or 0),
            boundary_carry_decay=float(hparams.get("rhythm_v3_boundary_carry_decay", 0.25) or 0.25),
            boundary_reset_thresh=float(hparams.get("rhythm_v3_boundary_reset_thresh", 0.5) or 0.5),
            allow_hybrid=(
                None
                if "rhythm_v3_allow_hybrid" not in hparams
                else bool(hparams.get("rhythm_v3_allow_hybrid", False))
            ),
            source_residual_gain=float(hparams.get("rhythm_v3_source_residual_gain", 0.0) or 0.0),
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
        precomputed_cache = collect_duration_v3_source_cache(rhythm_source_cache)
        if precomputed_cache is not None:
            if (
                self.prompt_summary_backbone
                and self.emit_silence_runs
                and precomputed_cache.get("source_silence_mask") is None
            ):
                raise ValueError(
                    "rhythm_v3 prompt_summary runtime with explicit silence runs requires "
                    "source_silence_mask in rhythm_source_cache / offline source cache."
                )
            return self.unit_frontend.from_precomputed(
                **precomputed_cache,
            )
        return self.unit_frontend.from_content_tensor(
            content,
            content_lengths=content_lengths,
            mark_last_open=bool(infer) or bool(self.hparams.get("rhythm_streaming_prefix_train", False)),
        )

    @staticmethod
    def _resolve_prompt_unit_mask(
        *,
        prompt_duration_obs: torch.Tensor,
        prompt_unit_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(prompt_unit_mask, torch.Tensor):
            return prompt_unit_mask.float().clamp(0.0, 1.0)
        return prompt_duration_obs.float().gt(0.0).float()

    def _resolve_prompt_speech_mask(
        self,
        *,
        prompt_content_units: torch.Tensor,
        prompt_valid_mask: torch.Tensor,
        prompt_speech_mask: torch.Tensor | None,
        prompt_silence_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(prompt_speech_mask, torch.Tensor):
            return prompt_speech_mask.float().clamp(0.0, 1.0) * prompt_valid_mask.float()
        if isinstance(prompt_silence_mask, torch.Tensor):
            return prompt_valid_mask.float() * (1.0 - prompt_silence_mask.float().clamp(0.0, 1.0))
        if self.emit_silence_runs:
            derived_silence = prompt_content_units.long().eq(int(self.silent_token)).float()
            return prompt_valid_mask.float() * (1.0 - derived_silence)
        if self.prompt_summary_backbone:
            raise ValueError(
                "rhythm_v3 prompt_summary/unit_run backbone requires prompt_speech_mask, "
                "prompt_silence_mask, or explicit silence-run prompt units."
            )
        return prompt_valid_mask.float()

    def _prepare_prompt_unit_conditioning(
        self,
        *,
        ref_conditioning,
        device: torch.device,
    ):
        if not isinstance(ref_conditioning, dict):
            return ref_conditioning
        if any(key not in ref_conditioning for key in _PROMPT_UNIT_REQUIRED_KEYS):
            return ref_conditioning
        prompt_content_units = ref_conditioning["prompt_content_units"].to(device=device).long()
        prompt_duration_obs = ref_conditioning["prompt_duration_obs"].to(device=device)
        prompt_valid_mask = self._resolve_prompt_unit_mask(
            prompt_duration_obs=prompt_duration_obs,
            prompt_unit_mask=ref_conditioning.get("prompt_unit_mask"),
        )
        prompt_silence_mask = (
            ref_conditioning["prompt_silence_mask"].to(device=device).float()
            if isinstance(ref_conditioning.get("prompt_silence_mask"), torch.Tensor)
            else None
        )
        prompt_speech_mask = self._resolve_prompt_speech_mask(
            prompt_content_units=prompt_content_units,
            prompt_valid_mask=prompt_valid_mask,
            prompt_speech_mask=ref_conditioning.get("prompt_speech_mask"),
            prompt_silence_mask=prompt_silence_mask,
        )
        if self.prompt_summary_backbone and not bool((prompt_speech_mask > 0.5).any().item()):
            raise ValueError(
                "rhythm_v3 prompt_summary/unit_run backbone requires at least one speech run in prompt conditioning."
            )
        enriched = {
            "prompt_content_units": prompt_content_units,
            "prompt_duration_obs": prompt_duration_obs,
            "prompt_unit_mask": prompt_valid_mask,
            "prompt_valid_mask": prompt_valid_mask,
            "prompt_speech_mask": prompt_speech_mask,
        }
        if isinstance(prompt_silence_mask, torch.Tensor):
            enriched["prompt_silence_mask"] = prompt_silence_mask * prompt_valid_mask
        if isinstance(ref_conditioning.get("prompt_spk_embed"), torch.Tensor):
            enriched["prompt_spk_embed"] = ref_conditioning["prompt_spk_embed"].to(device=device).float().detach()
        if isinstance(ref_conditioning.get("prompt_unit_anchor_base"), torch.Tensor):
            enriched["prompt_unit_anchor_base"] = ref_conditioning["prompt_unit_anchor_base"].to(device=device).detach()
        if isinstance(ref_conditioning.get("prompt_log_base"), torch.Tensor):
            enriched["prompt_log_base"] = ref_conditioning["prompt_log_base"].to(device=device).detach()
        for key in (
            "prompt_source_boundary_cue",
            "prompt_phrase_group_pos",
            "prompt_phrase_final_mask",
        ):
            if isinstance(ref_conditioning.get(key), torch.Tensor):
                enriched[key] = ref_conditioning[key].to(device=device).float()
        if enriched.get("prompt_log_base") is None and enriched.get("prompt_unit_anchor_base") is None:
            enriched["prompt_log_base"] = self.unit_frontend.compute_rate_log_base(
                prompt_content_units,
                prompt_valid_mask,
                stop_gradient=True,
            )
        return enriched

    def _validate_training_reference_semantics(
        self,
        *,
        infer: bool,
        ref_conditioning,
        ref,
    ) -> None:
        if infer:
            return
        if self.baseline_train_mode == "pretrain":
            return
        if isinstance(ref_conditioning, ReferenceDurationMemory):
            raise ValueError(
                "rhythm_v3 mainline training requires explicit prompt units "
                "(prompt_content_units / prompt_duration_obs / prompt_unit_mask)."
            )
        if isinstance(ref_conditioning, Mapping):
            nested = ref_conditioning.get("rhythm_ref_conditioning")
            if nested is not None and nested is not ref_conditioning:
                self._validate_training_reference_semantics(
                    infer=infer,
                    ref_conditioning=nested,
                    ref=ref,
                )
                return
            has_prompt_units = all(
                isinstance(ref_conditioning.get(key), torch.Tensor)
                for key in (*_PROMPT_UNIT_REQUIRED_KEYS, "prompt_unit_mask")
            )
            if has_prompt_units:
                return
            raise ValueError(
                "rhythm_v3 mainline training requires explicit prompt units "
                "(prompt_content_units / prompt_duration_obs / prompt_unit_mask)."
            )
        if ref_conditioning is None and ref is not None:
            raise ValueError(
                "rhythm_v3 mainline training requires explicit prompt units "
                "(prompt_content_units / prompt_duration_obs / prompt_unit_mask)."
            )

    def _attach_runtime_outputs(
        self,
        *,
        ret: dict,
        source_batch,
        ref_memory: ReferenceDurationMemory,
        execution,
        rhythm_state_prev,
    ) -> None:
        ret["rhythm_version"] = "v3"
        ret["rhythm_teacher_as_main"] = 0.0
        ret["rhythm_unit_batch"] = source_batch
        ret["rhythm_execution"] = execution
        ret["rhythm_state_prev"] = rhythm_state_prev
        ret["rhythm_state_next"] = execution.next_state
        ret["rhythm_ref_conditioning"] = ref_memory
        ret["speech_duration_exec"] = execution.speech_duration_exec
        ret["commit_frontier"] = execution.commit_frontier
        ret["rhythm_v3_runtime_mode"] = self.module.runtime_mode
        ret["rhythm_v3_backbone_mode"] = self.module.backbone_mode
        ret["rhythm_v3_warp_mode"] = self.module.warp_mode
        ret["rhythm_v3_allow_hybrid"] = float(bool(self.module.allow_hybrid))
        ret["rhythm_v3_baseline_train_mode"] = self.baseline_train_mode
        ret["rhythm_v3_source_residual_gain"] = float(self.module.source_residual_gain)
        if ref_memory is not None and isinstance(getattr(ref_memory, "summary_state", None), torch.Tensor):
            ret["rhythm_v3_summary_state_dim"] = float(ref_memory.summary_state.size(1))
        elif ref_memory is not None and isinstance(getattr(ref_memory, "operator_coeff", None), torch.Tensor):
            ret["rhythm_v3_summary_state_dim"] = float(ref_memory.operator_coeff.size(1))
        if ref_memory is not None and isinstance(getattr(ref_memory, "role_value", None), torch.Tensor):
            ret["rhythm_v3_summary_channels"] = float(ref_memory.role_value.size(1))
            ret["rhythm_v3_summary_slots"] = float(ref_memory.role_value.size(1))
        if self.module.duration_head is not None and hasattr(self.module.duration_head, "src_rate_init"):
            ret["rhythm_v3_source_rate_init"] = self.module.duration_head.src_rate_init.detach().reshape(1)
        if execution.frame_plan is not None:
            ret["rhythm_frame_plan"] = execution.frame_plan

    @staticmethod
    def _pad_tail_sequences(
        sequences: list[torch.Tensor],
        *,
        pad_value: int | float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        max_len = max((int(seq.numel()) for seq in sequences), default=0)
        out = torch.full((len(sequences), max_len), pad_value, dtype=dtype, device=device)
        for batch_idx, seq in enumerate(sequences):
            if int(seq.numel()) > 0:
                out[batch_idx, : seq.numel()] = seq.to(device=device, dtype=dtype)
        return out

    def _render_uncommitted_source_tail(
        self,
        *,
        source_batch,
        execution,
        content_embed: torch.Tensor,
        speech_state_fn,
    ) -> RenderedRhythmSequence | None:
        commit_mask = getattr(execution, "commit_mask", None)
        if not isinstance(commit_mask, torch.Tensor):
            return None
        unit_mask = source_batch.unit_mask.float()
        tail_unit_mask = (unit_mask > 0.5) & (commit_mask.float() <= 0.5)
        if not bool(tail_unit_mask.any().item()):
            return None
        device = content_embed.device
        batch_tokens: list[torch.Tensor] = []
        batch_durations: list[torch.Tensor] = []
        batch_masks: list[torch.Tensor] = []
        batch_unit_indices: list[torch.Tensor] = []
        for batch_idx in range(int(source_batch.content_units.size(0))):
            tail_indices = torch.nonzero(tail_unit_mask[batch_idx], as_tuple=False).reshape(-1)
            batch_tokens.append(source_batch.content_units[batch_idx].long().index_select(0, tail_indices))
            batch_durations.append(source_batch.source_duration_obs[batch_idx].float().index_select(0, tail_indices))
            batch_masks.append(torch.ones((tail_indices.numel(),), dtype=torch.float32, device=device))
            batch_unit_indices.append(tail_indices.long())
        tail_content_units = self._pad_tail_sequences(batch_tokens, pad_value=0, dtype=torch.long, device=device)
        if tail_content_units.size(1) <= 0:
            return None
        tail_duration_obs = self._pad_tail_sequences(batch_durations, pad_value=0.0, dtype=torch.float32, device=device)
        tail_mask = self._pad_tail_sequences(batch_masks, pad_value=0.0, dtype=torch.float32, device=device)
        tail_unit_indices = self._pad_tail_sequences(batch_unit_indices, pad_value=0, dtype=torch.long, device=device)
        tail_plan = build_frame_plan_from_execution(
            dur_anchor_src=tail_duration_obs,
            speech_exec=tail_duration_obs,
            pause_exec=torch.zeros_like(tail_duration_obs),
            unit_mask=tail_mask,
        )
        prefix_src_offset = torch.round(source_batch.source_duration_obs.float().clamp_min(0.0) * commit_mask.float()).long()
        prefix_src_offset = prefix_src_offset.sum(dim=1, keepdim=True)
        tail_frame_src_index = tail_plan.frame_src_index.clone()
        positive_src = tail_frame_src_index >= 0
        tail_frame_src_index = torch.where(
            positive_src,
            tail_frame_src_index + prefix_src_offset.expand_as(tail_frame_src_index),
            tail_frame_src_index,
        )
        safe_rel_index = tail_plan.frame_unit_index.clamp(min=0, max=max(0, tail_unit_indices.size(1) - 1))
        original_unit_index = tail_unit_indices.gather(1, safe_rel_index)
        total_mask = tail_plan.total_mask > 0.5
        original_unit_index = original_unit_index.masked_fill(~total_mask, -1)
        render_tail_plan = RhythmFramePlan(
            frame_src_index=tail_frame_src_index,
            frame_is_blank=tail_plan.frame_is_blank,
            frame_slot_index=tail_plan.frame_slot_index,
            frame_unit_index=tail_plan.frame_unit_index,
            total_mask=tail_plan.total_mask,
            speech_mask=tail_plan.speech_mask,
            blank_mask=tail_plan.blank_mask,
            frame_phase_features=tail_plan.frame_phase_features,
        )
        rendered = render_rhythm_sequence(
            content_units=tail_content_units,
            silent_token=int(self.hparams.get("silent_token", 57)),
            speech_state_fn=speech_state_fn,
            pause_state=self.pause_state.to(device=content_embed.device, dtype=content_embed.dtype),
            frame_plan=render_tail_plan,
            frame_state_post_fn=None,
        )
        frame_slot_index = torch.where(total_mask, 2 * original_unit_index.clamp_min(0), original_unit_index)
        frame_plan = RhythmFramePlan(
            frame_src_index=tail_frame_src_index,
            frame_is_blank=rendered.frame_plan.frame_is_blank,
            frame_slot_index=frame_slot_index,
            frame_unit_index=original_unit_index,
            total_mask=rendered.frame_plan.total_mask,
            speech_mask=rendered.frame_plan.speech_mask,
            blank_mask=rendered.frame_plan.blank_mask,
            frame_phase_features=rendered.frame_plan.frame_phase_features,
        )
        return RenderedRhythmSequence(
            frame_states=rendered.frame_states,
            frame_tokens=rendered.frame_tokens,
            speech_mask=rendered.speech_mask,
            blank_mask=rendered.blank_mask,
            total_mask=rendered.total_mask,
            frame_slot_index=frame_slot_index,
            frame_unit_index=original_unit_index,
            frame_phase_features=rendered.frame_phase_features,
            frame_plan=frame_plan,
        )

    @staticmethod
    def _concat_rendered_sequences(
        prefix: RenderedRhythmSequence,
        tail: RenderedRhythmSequence | None,
    ) -> RenderedRhythmSequence:
        if tail is None:
            return prefix
        frame_plan = RhythmFramePlan(
            frame_src_index=torch.cat([prefix.frame_plan.frame_src_index, tail.frame_plan.frame_src_index], dim=1),
            frame_is_blank=torch.cat([prefix.frame_plan.frame_is_blank, tail.frame_plan.frame_is_blank], dim=1),
            frame_slot_index=torch.cat([prefix.frame_plan.frame_slot_index, tail.frame_plan.frame_slot_index], dim=1),
            frame_unit_index=torch.cat([prefix.frame_plan.frame_unit_index, tail.frame_plan.frame_unit_index], dim=1),
            total_mask=torch.cat([prefix.frame_plan.total_mask, tail.frame_plan.total_mask], dim=1),
            speech_mask=torch.cat([prefix.frame_plan.speech_mask, tail.frame_plan.speech_mask], dim=1),
            blank_mask=torch.cat([prefix.frame_plan.blank_mask, tail.frame_plan.blank_mask], dim=1),
            frame_phase_features=torch.cat([prefix.frame_plan.frame_phase_features, tail.frame_plan.frame_phase_features], dim=1),
        )
        return RenderedRhythmSequence(
            frame_states=torch.cat([prefix.frame_states, tail.frame_states], dim=1),
            frame_tokens=torch.cat([prefix.frame_tokens, tail.frame_tokens], dim=1),
            speech_mask=torch.cat([prefix.speech_mask, tail.speech_mask], dim=1),
            blank_mask=torch.cat([prefix.blank_mask, tail.blank_mask], dim=1),
            total_mask=torch.cat([prefix.total_mask, tail.total_mask], dim=1),
            frame_slot_index=torch.cat([prefix.frame_slot_index, tail.frame_slot_index], dim=1),
            frame_unit_index=torch.cat([prefix.frame_unit_index, tail.frame_unit_index], dim=1),
            frame_phase_features=torch.cat([prefix.frame_phase_features, tail.frame_phase_features], dim=1),
            frame_plan=frame_plan,
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
        ref_lengths: torch.Tensor | None = None,
        spk_embed: torch.Tensor | None = None,
        rhythm_state=None,
        rhythm_ref_conditioning=None,
        rhythm_apply_override=None,
        rhythm_runtime_overrides: dict | None = None,
        rhythm_source_cache: dict | None = None,
        rhythm_offline_source_cache: dict | None = None,
        speech_state_fn=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        del target, global_steps, rhythm_runtime_overrides, rhythm_offline_source_cache
        if ref is None and rhythm_ref_conditioning is None and not (not infer and self.baseline_train_mode == "pretrain"):
            return content_embed, tgt_nonpadding, f0, uv
        if content_embed.device != content.device:
            raise ValueError(
                f"content/content_embed device mismatch: content={content.device}, content_embed={content_embed.device}"
            )
        if tgt_nonpadding.device != content.device:
            raise ValueError(
                f"content/tgt_nonpadding device mismatch: content={content.device}, tgt_nonpadding={tgt_nonpadding.device}"
            )
        source_batch = self._build_source_batch(
            content=content,
            content_lengths=content_lengths,
            rhythm_source_cache=rhythm_source_cache,
            infer=infer,
        )
        source_batch = move_source_unit_batch(source_batch, device=content.device)
        rhythm_state_prev = move_duration_runtime_state(rhythm_state, device=content.device)
        self._validate_training_reference_semantics(
            infer=bool(infer),
            ref_conditioning=rhythm_ref_conditioning,
            ref=ref,
        )
        if ref is None and rhythm_ref_conditioning is None and not infer and self.baseline_train_mode == "pretrain":
            rhythm_ref_conditioning = {
                "global_rate": torch.zeros((content.size(0), 1), device=content.device, dtype=torch.float32),
            }
        rhythm_ref_conditioning = self._prepare_prompt_unit_conditioning(
            ref_conditioning=rhythm_ref_conditioning,
            device=content.device,
        )
        if isinstance(rhythm_ref_conditioning, dict) and "prompt_spk_embed" not in rhythm_ref_conditioning and isinstance(spk_embed, torch.Tensor):
            prompt_spk = spk_embed
            if prompt_spk.dim() == 3 and prompt_spk.size(-1) == 1:
                prompt_spk = prompt_spk.squeeze(-1)
            elif prompt_spk.dim() == 3 and prompt_spk.size(1) == 1:
                prompt_spk = prompt_spk.squeeze(1)
            if prompt_spk.dim() == 2:
                rhythm_ref_conditioning["prompt_spk_embed"] = prompt_spk.detach()
        if ref is not None and rhythm_ref_conditioning is None:
            raise ValueError(
                "rhythm_v3 inference requires explicit prompt units. "
                "Set rhythm_ref_conditioning with prompt_content_units/prompt_duration_obs/prompt_unit_mask."
            )
        ref_memory = self.module.build_reference_conditioning(
            ref_conditioning=rhythm_ref_conditioning,
        )
        ref_memory = move_reference_duration_memory(
            ref_memory,
            device=content.device,
            dtype=content_embed.dtype if content_embed.is_floating_point() else None,
        )
        execution = self.module(
            source_batch=source_batch,
            ref_memory=ref_memory,
            state=rhythm_state_prev,
        )
        self._attach_runtime_outputs(
            ret=ret,
            source_batch=source_batch,
            ref_memory=ref_memory,
            execution=execution,
            rhythm_state_prev=rhythm_state_prev,
        )
        apply_rhythm_render = resolve_rhythm_apply_mode(
            self.hparams,
            infer=infer,
            override=rhythm_apply_override,
        )
        has_uncommitted_tail = False
        if isinstance(getattr(execution, "commit_mask", None), torch.Tensor):
            has_uncommitted_tail = bool(
                (((source_batch.unit_mask.float() > 0.5) & (execution.commit_mask.float() <= 0.5))).any().item()
            )
        if (
            bool(apply_rhythm_render)
            and (
                source_batch.content_units.size(1) <= 0
                or (
                    execution.frame_plan is None
                    and not has_uncommitted_tail
                )
                or (
                    execution.frame_plan is not None
                    and execution.frame_plan.total_mask.numel() <= 0
                    and not has_uncommitted_tail
                )
                or (
                    execution.frame_plan is not None
                    and float(execution.frame_plan.total_mask.sum().item()) <= 0.0
                    and not has_uncommitted_tail
                )
            )
        ):
            ret["rhythm_apply_render"] = 0.0
            ret["rhythm_render_skipped_empty"] = 1.0
            return content_embed, tgt_nonpadding, f0, uv
        ret["rhythm_apply_render"] = float(apply_rhythm_render)
        if not apply_rhythm_render:
            return content_embed, tgt_nonpadding, f0, uv
        if speech_state_fn is None:
            raise ValueError("speech_state_fn is required when rhythm render is enabled.")
        rendered_tail = self._render_uncommitted_source_tail(
            source_batch=source_batch,
            execution=execution,
            content_embed=content_embed,
            speech_state_fn=speech_state_fn,
        )
        has_prefix_render = bool(
            execution.frame_plan is not None
            and execution.frame_plan.total_mask.numel() > 0
            and float(execution.frame_plan.total_mask.sum().item()) > 0.0
        )
        if has_prefix_render:
            rendered = render_rhythm_sequence(
                content_units=source_batch.content_units,
                silent_token=int(self.hparams.get("silent_token", 57)),
                speech_state_fn=speech_state_fn,
                pause_state=self.pause_state.to(device=content_embed.device, dtype=content_embed.dtype),
                frame_plan=execution.frame_plan,
                frame_state_post_fn=None,
            )
            rendered = self._concat_rendered_sequences(rendered, rendered_tail)
        elif rendered_tail is not None:
            rendered = rendered_tail
        else:
            ret["rhythm_apply_render"] = 0.0
            ret["rhythm_render_skipped_empty"] = 1.0
            return content_embed, tgt_nonpadding, f0, uv
        ret["content"] = rendered.frame_tokens
        ret["content_rhythm_rendered"] = rendered.frame_tokens
        ret["content_embed_proj_rhythm"] = rendered.frame_states
        ret["rhythm_total_mask"] = rendered.total_mask
        ret["rhythm_speech_mask"] = rendered.speech_mask
        ret["rhythm_blank_mask"] = rendered.blank_mask
        ret["rhythm_render_unit_index"] = rendered.frame_unit_index
        ret["rhythm_frame_plan"] = rendered.frame_plan
        return rendered.frame_states, rendered.total_mask[:, :, None], f0, uv

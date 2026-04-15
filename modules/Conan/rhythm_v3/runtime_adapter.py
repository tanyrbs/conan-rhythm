from __future__ import annotations

from dataclasses import replace
from collections.abc import Mapping

import torch
import torch.nn as nn

from .contracts import (
    ReferenceDurationMemory,
    collect_duration_v3_source_cache,
    ensure_duration_runtime_state_batch,
    move_duration_runtime_state,
    move_reference_duration_memory,
    move_source_unit_batch,
)
from .bridge import resolve_rhythm_apply_mode
from .frame_plan import RhythmFramePlan, build_frame_plan_from_execution
from .g_stats import compute_duration_weighted_speech_ratio, summarize_global_rate_support
from .math_utils import src_prefix_stat_mode_requires_full_history
from .module import MixedEffectsDurationModule
from .renderer import RenderedRhythmSequence, render_rhythm_sequence
from .source_cache import (
    DURATION_V3_CACHE_META_KEY,
    assert_duration_v3_cache_meta_compatible,
    build_duration_v3_cache_meta,
)
from .unit_frontend import DurationUnitFrontend
from tasks.Conan.rhythm.duration_v3.task_config import (
    is_duration_v3_prompt_summary_backbone,
    resolve_duration_v3_rate_mode,
)
from tasks.Conan.rhythm.duration_v3.gate_contract import (
    build_runtime_contract_fingerprint_from_values,
    build_runtime_contract_fingerprint_json_from_values,
    build_runtime_contract_id_from_values,
)

_PROMPT_UNIT_REQUIRED_KEYS = (
    "prompt_content_units",
    "prompt_duration_obs",
)
_MINIMAL_REFERENCE_SUMMARY_KEYS = (
    "summary_state",
    "role_value",
    "role_var",
    "role_coverage",
)
_MISSING = object()


def _resolve_reference_summary_usage(
    *,
    minimal_v1_profile: bool,
    requested_use_reference_summary,
) -> bool:
    use_reference_summary = False if requested_use_reference_summary is _MISSING else bool(requested_use_reference_summary)
    if minimal_v1_profile and use_reference_summary:
        raise ValueError(
            "rhythm_v3_minimal_v1_profile selects the minimal_v1_global path directly; "
            "rhythm_v3_use_reference_summary only enables optional non-minimal reference-summary tensors and must be false."
        )
    return use_reference_summary


class ConanDurationAdapter(nn.Module):
    def __init__(self, hparams, hidden_size: int, *, vocab_size: int) -> None:
        super().__init__()
        self.hparams = hparams
        minimal_v1_profile = bool(hparams.get("rhythm_v3_minimal_v1_profile", False))
        self.minimal_v1_profile = minimal_v1_profile
        rate_mode = resolve_duration_v3_rate_mode(hparams)
        self.rate_mode = rate_mode
        simple_global_stats = rate_mode == "simple_global" or bool(
            hparams.get("rhythm_v3_simple_global_stats", minimal_v1_profile)
        )
        use_log_base_rate = bool(hparams.get("rhythm_v3_use_log_base_rate", False))
        if rate_mode == "simple_global" or simple_global_stats:
            use_log_base_rate = False
        disable_learned_gate = hparams.get("rhythm_v3_disable_learned_gate", _MISSING)
        requested_use_learned_residual_gate = hparams.get("rhythm_v3_use_learned_residual_gate", _MISSING)
        if requested_use_learned_residual_gate is not _MISSING:
            use_learned_residual_gate = bool(requested_use_learned_residual_gate)
        elif minimal_v1_profile:
            use_learned_residual_gate = False
        elif disable_learned_gate is not _MISSING:
            use_learned_residual_gate = not bool(disable_learned_gate)
        else:
            use_learned_residual_gate = True
        requested_use_reference_summary = (
            hparams.get("rhythm_v3_use_reference_summary", False)
            if "rhythm_v3_use_reference_summary" in hparams
            else _MISSING
        )
        use_reference_summary = _resolve_reference_summary_usage(
            minimal_v1_profile=minimal_v1_profile,
            requested_use_reference_summary=requested_use_reference_summary,
        )
        self.baseline_train_mode = str(hparams.get("rhythm_v3_baseline_train_mode", "joint") or "joint").strip().lower()
        self.silent_token = int(hparams.get("silent_token", 57))
        self.emit_silence_runs = bool(hparams.get("rhythm_v3_emit_silence_runs", True))
        self.separator_aware = bool(hparams.get("rhythm_separator_aware", True))
        self.tail_open_units = int(hparams.get("rhythm_tail_open_units", 1))
        self.debounce_min_run_frames = int(hparams.get("rhythm_v3_debounce_min_run_frames", 2))
        self.phrase_boundary_threshold = float(hparams.get("rhythm_source_phrase_threshold", 0.55))
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
            separator_aware=self.separator_aware,
            tail_open_units=self.tail_open_units,
            emit_silence_runs=self.emit_silence_runs,
            debounce_min_run_frames=self.debounce_min_run_frames,
            anchor_hidden_size=int(hparams.get("rhythm_anchor_hidden_size", 128)),
            anchor_min_frames=float(hparams.get("rhythm_anchor_min_frames", 1.0)),
            anchor_max_frames=float(hparams.get("rhythm_anchor_max_frames", 12.0)),
            phrase_boundary_threshold=self.phrase_boundary_threshold,
            rate_mode=rate_mode,
            simple_global_stats=simple_global_stats,
        )
        self.duration_v3_cache_meta = build_duration_v3_cache_meta(
            silent_token=self.silent_token,
            separator_aware=self.separator_aware,
            tail_open_units=self.tail_open_units,
            emit_silence_runs=self.emit_silence_runs,
            debounce_min_run_frames=self.debounce_min_run_frames,
            phrase_boundary_threshold=self.phrase_boundary_threshold,
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
            analytic_gap_clip=float(hparams.get("rhythm_v3_analytic_gap_clip", 0.35) or 0.0),
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
            budget_mode=str(hparams.get("rhythm_v3_budget_mode", "total") or "total"),
            boundary_carry_decay=float(hparams.get("rhythm_v3_boundary_carry_decay", 0.25) or 0.25),
            boundary_offset_decay=hparams.get("rhythm_v3_boundary_offset_decay", None),
            boundary_reset_thresh=float(hparams.get("rhythm_v3_boundary_reset_thresh", 0.5) or 0.5),
            projection_mode=str(hparams.get("rhythm_v3_projection_mode", "greedy") or "greedy"),
            integer_projection_mode=str(
                hparams.get(
                    "rhythm_v3_integer_projection_mode",
                    hparams.get("rhythm_v3_projection_mode", "greedy"),
                )
                or "greedy"
            ),
            integer_projection_anchor_mode=str(
                hparams.get("rhythm_v3_integer_projection_anchor_mode", "rounded") or "rounded"
            ),
            prefix_optimal_step_weight=float(hparams.get("rhythm_v3_prefix_optimal_step_weight", 0.10) or 0.10),
            prefix_optimal_prefix_weight=float(hparams.get("rhythm_v3_prefix_optimal_prefix_weight", 1.00) or 1.00),
            prefix_optimal_terminal_weight=float(hparams.get("rhythm_v3_prefix_optimal_terminal_weight", 1.00) or 1.00),
            prefix_optimal_boundary_weight=float(hparams.get("rhythm_v3_prefix_optimal_boundary_weight", 0.75) or 0.75),
            prefix_optimal_coarse_weight=float(hparams.get("rhythm_v3_prefix_optimal_coarse_weight", 0.50) or 0.50),
            prefix_optimal_phrase_final_boost=float(hparams.get("rhythm_v3_prefix_optimal_phrase_final_boost", 1.50) or 1.50),
            prefix_optimal_max_window=int(hparams.get("rhythm_v3_prefix_optimal_max_window", 96) or 96),
            prefix_optimal_max_states=int(hparams.get("rhythm_v3_prefix_optimal_max_states", 97) or 97),
            projection_repair_max_steps=int(hparams.get("rhythm_v3_projection_repair_max_steps", 0) or 0),
            projection_repair_speech_bonus=float(
                hparams.get("rhythm_v3_projection_repair_speech_bonus", 1.0) or 0.0
            ),
            projection_repair_boundary_penalty=float(
                hparams.get("rhythm_v3_projection_repair_boundary_penalty", 0.35) or 0.0
            ),
            rate_mode=rate_mode,
            simple_global_stats=simple_global_stats,
            minimal_v1_profile=minimal_v1_profile,
            strict_minimal_claim_profile=bool(hparams.get("rhythm_v3_strict_minimal_claim_profile", True)),
            use_log_base_rate=use_log_base_rate,
            disable_learned_gate=(
                None if disable_learned_gate is _MISSING else bool(disable_learned_gate)
            ),
            use_learned_residual_gate=use_learned_residual_gate,
            use_reference_summary=use_reference_summary,
            emit_prompt_diagnostics=bool(hparams.get("rhythm_v3_emit_prompt_diagnostics", True)),
            eval_mode=str(hparams.get("rhythm_v3_eval_mode", "learned")),
            g_variant=str(hparams.get("rhythm_v3_g_variant", "raw_median")),
            g_trim_ratio=float(hparams.get("rhythm_v3_g_trim_ratio", 0.2)),
            drop_edge_runs_for_g=int(hparams.get("rhythm_v3_drop_edge_runs_for_g", 0) or 0),
            min_boundary_confidence_for_g=hparams.get("rhythm_v3_min_boundary_confidence_for_g", None),
            prompt_domain_mode=str(hparams.get("rhythm_v3_prompt_domain_mode", "minimal_strict") or "minimal_strict"),
            prompt_require_clean_support=bool(hparams.get("rhythm_v3_prompt_require_clean_support", True)),
            require_prompt_ref_len_gate=bool(hparams.get("rhythm_v3_require_prompt_ref_len_gate", False)),
            min_prompt_support_runs=int(hparams.get("rhythm_v3_min_prompt_support_runs", 3) or 0),
            min_prompt_support_fraction=float(hparams.get("rhythm_v3_min_prompt_support_fraction", 0.20) or 0.0),
            min_prompt_support_weight=float(hparams.get("rhythm_v3_min_prompt_support_weight", 2.0) or 0.0),
            prompt_g_variant=str(
                hparams.get(
                    "rhythm_v3_prompt_g_variant",
                    hparams.get("rhythm_v3_g_variant", "raw_median"),
                )
                or "raw_median"
            ),
            prompt_g_trim_ratio=float(
                hparams.get(
                    "rhythm_v3_prompt_g_trim_ratio",
                    hparams.get("rhythm_v3_g_trim_ratio", 0.2),
                )
                or 0.2
            ),
            prompt_g_drop_edge_runs=int(
                hparams.get(
                    "rhythm_v3_prompt_g_drop_edge_runs",
                    hparams.get("rhythm_v3_drop_edge_runs_for_g", 0),
                )
                or 0
            ),
            prompt_min_boundary_confidence_for_g=hparams.get(
                "rhythm_v3_prompt_min_boundary_confidence_for_g",
                hparams.get("rhythm_v3_min_boundary_confidence_for_g", None),
            ),
            src_g_variant=str(
                hparams.get(
                    "rhythm_v3_src_g_variant",
                    hparams.get("rhythm_v3_g_variant", "raw_median"),
                )
                or "raw_median"
            ),
            src_g_trim_ratio=float(
                hparams.get(
                    "rhythm_v3_src_g_trim_ratio",
                    hparams.get("rhythm_v3_g_trim_ratio", 0.2),
                )
                or 0.2
            ),
            src_g_drop_edge_runs=int(
                hparams.get(
                    "rhythm_v3_src_g_drop_edge_runs",
                    hparams.get("rhythm_v3_drop_edge_runs_for_g", 0),
                )
                or 0
            ),
            src_min_boundary_confidence_for_g=hparams.get(
                "rhythm_v3_src_min_boundary_confidence_for_g",
                hparams.get("rhythm_v3_min_boundary_confidence_for_g", None),
            ),
            disable_local_residual=bool(hparams.get("rhythm_v3_disable_local_residual", False)),
            disable_coarse_bias=bool(hparams.get("rhythm_v3_disable_coarse_bias", False)),
            use_src_gap_in_coarse_head=bool(hparams.get("rhythm_v3_use_src_gap_in_coarse_head", False)),
            detach_global_term_in_local_head=bool(
                hparams.get(
                    "rhythm_v3_detach_global_term_in_local_head",
                    bool(hparams.get("rhythm_v3_minimal_v1_profile", False)),
                )
            ),
            coarse_delta_scale=float(hparams.get("rhythm_v3_coarse_delta_scale", 0.20) or 0.0),
            local_residual_scale=float(hparams.get("rhythm_v3_local_residual_scale", 0.35) or 0.0),
            src_rate_init_mode=str(hparams.get("rhythm_v3_src_rate_init_mode", "auto") or "auto"),
            src_prefix_stat_mode=str(hparams.get("rhythm_v3_src_prefix_stat_mode", "ema") or "ema"),
            src_prefix_min_support=int(hparams.get("rhythm_v3_src_prefix_min_support", 3) or 3),
            src_rate_init_value=float(hparams.get("rhythm_v3_src_rate_init_value", 0.0) or 0.0),
            freeze_src_rate_init=bool(
                hparams.get(
                    "rhythm_v3_freeze_src_rate_init",
                    bool(hparams.get("rhythm_v3_minimal_v1_profile", False)),
                )
            ),
            strict_eval_invalid_g=bool(
                hparams.get(
                    "rhythm_v3_strict_eval_invalid_g",
                    bool(hparams.get("rhythm_v3_minimal_v1_profile", False)),
                )
            ),
            debug_export=bool(hparams.get("rhythm_v3_debug_export", False)),
            export_projector_telemetry=bool(
                hparams.get(
                    "rhythm_v3_export_projector_telemetry",
                    hparams.get("rhythm_v3_debug_export", False),
                )
            ),
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

    def _assert_minimal_ref_conditioning_contract(self, ref_conditioning) -> None:
        if not self.minimal_v1_profile or ref_conditioning is None:
            return
        if isinstance(ref_conditioning, ReferenceDurationMemory):
            violations = [
                field_name
                for field_name in _MINIMAL_REFERENCE_SUMMARY_KEYS
                if getattr(ref_conditioning, field_name, None) is not None
            ]
            if violations:
                raise RuntimeError(
                    "rhythm_v3_minimal_v1_profile forbids non-minimal reference-summary memory at runtime: "
                    + ", ".join(violations)
                )
            return
        if not isinstance(ref_conditioning, Mapping):
            return
        nested = ref_conditioning.get("rhythm_ref_conditioning")
        if nested is not None and nested is not ref_conditioning:
            self._assert_minimal_ref_conditioning_contract(nested)
        violations = [
            field_name
            for field_name in _MINIMAL_REFERENCE_SUMMARY_KEYS
            if ref_conditioning.get(field_name) is not None
        ]
        if violations:
            raise ValueError(
                "rhythm_v3_minimal_v1_profile prompt conditioning must stay on the minimal path; "
                "remove non-minimal reference-summary fields: "
                + ", ".join(violations)
            )

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
        state=None,
        return_state: bool = False,
    ):
        resolved_prefix_mode = str(
            getattr(self.module, "src_prefix_stat_mode", "ema") or "ema"
        ).strip().lower()
        if state is not None and resolved_prefix_mode != "ema":
            raise ValueError(
                "strict online continuation currently only guarantees src_prefix_stat_mode=ema. "
                "dual_timescale requires a 2-state carry, family_hybrid requires a richer prefix summary state, "
                "and exact_global_family remains local/offline only."
            )
        precomputed_cache = collect_duration_v3_source_cache(rhythm_source_cache)
        if precomputed_cache is not None:
            raw_cache_meta = rhythm_source_cache.get(DURATION_V3_CACHE_META_KEY) if isinstance(rhythm_source_cache, Mapping) else None
            if raw_cache_meta is not None or self.minimal_v1_profile:
                assert_duration_v3_cache_meta_compatible(
                    raw_cache_meta,
                    silent_token=self.silent_token,
                    separator_aware=self.separator_aware,
                    tail_open_units=self.tail_open_units,
                    emit_silence_runs=self.emit_silence_runs,
                    debounce_min_run_frames=self.debounce_min_run_frames,
                    phrase_boundary_threshold=self.phrase_boundary_threshold,
                )
            if self.minimal_v1_profile:
                required_fields = (
                    "unit_mask",
                    "sealed_mask",
                    "source_silence_mask",
                    "source_run_stability",
                    "source_boundary_cue",
                )
                missing_fields = [key for key in required_fields if precomputed_cache.get(key) is None]
                if missing_fields:
                    raise ValueError(
                        "rhythm_v3_minimal_v1_profile requires fully materialized rhythm_source_cache fields: "
                        + ", ".join(missing_fields)
                    )
            if (
                self.prompt_summary_backbone
                and self.emit_silence_runs
                and precomputed_cache.get("source_silence_mask") is None
            ):
                raise ValueError(
                    "rhythm_v3 prompt_summary runtime with explicit silence runs requires "
                    "source_silence_mask in rhythm_source_cache / offline source cache."
                )
            batch = self.unit_frontend.from_precomputed(
                **precomputed_cache,
            )
            return (batch, state) if return_state else batch
        mark_last_open = bool(infer) or bool(self.hparams.get("rhythm_streaming_prefix_train", False))
        batch_size, total_steps = content.shape
        if content_lengths is None:
            resolved_lengths = torch.full(
                (batch_size,),
                int(total_steps),
                dtype=torch.long,
                device=content.device,
            )
        else:
            resolved_lengths = content_lengths.to(device=content.device).long().clamp(min=0, max=int(total_steps))
        if not infer:
            batch = self.unit_frontend.from_content_tensor(
                content,
                content_lengths=resolved_lengths,
                mark_last_open=mark_last_open,
            )
            return (batch, state) if return_state else batch
        if state is None:
            state = self.module.init_state(batch_size=batch_size, device=content.device)
        frontend_state = getattr(state, "frontend_state", None)
        consumed = getattr(state, "consumed_content_steps", None)
        if frontend_state is None or not isinstance(consumed, torch.Tensor):
            frontend_state = self.unit_frontend.init_stream_state(
                batch_size=batch_size,
                device=content.device,
            )
            batch, next_frontend_state = self.unit_frontend.step_content_tensor(
                content,
                frontend_state,
                content_lengths=resolved_lengths,
                mark_last_open=mark_last_open,
            )
            next_state = replace(
                state,
                frontend_state=next_frontend_state,
                consumed_content_steps=resolved_lengths.reshape(batch_size, 1).clone(),
            )
            return (batch, next_state) if return_state else batch
        consumed = consumed.to(device=content.device, dtype=torch.long).reshape(batch_size, 1)
        total_lengths = resolved_lengths.reshape(batch_size, 1)
        if bool((consumed > total_lengths).any().item()):
            raise ValueError(
                "Incremental rhythm frontend requires non-decreasing content lengths. "
                "Provide a precomputed rhythm_source_cache when content history is trimmed."
            )
        delta_lengths = (total_lengths - consumed).reshape(batch_size)
        max_delta = int(delta_lengths.max().item()) if batch_size > 0 else 0
        if max_delta <= 0:
            batch = self.unit_frontend.materialize_stream_state(
                frontend_state,
                mark_last_open=mark_last_open,
                device=content.device,
            )
            next_state = replace(
                state,
                consumed_content_steps=total_lengths.clone(),
            )
            return (batch, next_state) if return_state else batch
        positions = torch.arange(max_delta, device=content.device, dtype=torch.long).reshape(1, max_delta)
        start_positions = consumed.reshape(batch_size, 1)
        gather_idx = start_positions + positions
        delta_mask = positions < delta_lengths.reshape(batch_size, 1)
        safe_gather_idx = torch.minimum(
            gather_idx,
            torch.full_like(gather_idx, int(max(0, total_steps - 1))),
        )
        delta = torch.gather(content, 1, safe_gather_idx)
        delta = delta * delta_mask.to(dtype=content.dtype)
        batch, next_frontend_state = self.unit_frontend.step_content_tensor(
            delta,
            frontend_state,
            content_lengths=delta_lengths,
            mark_last_open=mark_last_open,
        )
        next_state = replace(
            state,
            frontend_state=next_frontend_state,
            consumed_content_steps=total_lengths.clone(),
        )
        return (batch, next_state) if return_state else batch

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
        prompt_cache_meta: Mapping[str, object] | None = None,
    ) -> torch.Tensor:
        if self.minimal_v1_profile and not isinstance(prompt_speech_mask, torch.Tensor):
            raise ValueError(
                "rhythm_v3 minimal_v1_profile requires explicit prompt_speech_mask. "
                "prompt_silence_mask is auxiliary-only and silence-token fallback is disabled."
            )
        if isinstance(prompt_speech_mask, torch.Tensor):
            return prompt_speech_mask.float().clamp(0.0, 1.0) * prompt_valid_mask.float()
        if isinstance(prompt_silence_mask, torch.Tensor):
            return prompt_valid_mask.float() * (1.0 - prompt_silence_mask.float().clamp(0.0, 1.0))
        if self.emit_silence_runs:
            if self.minimal_v1_profile:
                assert_duration_v3_cache_meta_compatible(
                    prompt_cache_meta,
                    silent_token=self.silent_token,
                    separator_aware=self.separator_aware,
                    tail_open_units=self.tail_open_units,
                    emit_silence_runs=self.emit_silence_runs,
                    debounce_min_run_frames=self.debounce_min_run_frames,
                    phrase_boundary_threshold=self.phrase_boundary_threshold,
                )
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
        prompt_speech_mask_raw = (
            ref_conditioning["prompt_speech_mask"].to(device=device).float()
            if isinstance(ref_conditioning.get("prompt_speech_mask"), torch.Tensor)
            else None
        )
        prompt_cache_meta = ref_conditioning.get(DURATION_V3_CACHE_META_KEY)
        expected_shape = tuple(prompt_duration_obs.shape)
        if tuple(prompt_content_units.shape) != expected_shape:
            raise ValueError(
                "prompt_content_units / prompt_duration_obs / prompt_unit_mask must share the same shape."
            )
        if tuple(prompt_valid_mask.shape) != expected_shape:
            raise ValueError(
                "prompt_content_units / prompt_duration_obs / prompt_unit_mask must share the same shape."
            )
        if isinstance(prompt_silence_mask, torch.Tensor) and tuple(prompt_silence_mask.shape) != expected_shape:
            raise ValueError("prompt_silence_mask must match prompt_duration_obs shape.")
        prompt_speech_mask = self._resolve_prompt_speech_mask(
            prompt_content_units=prompt_content_units,
            prompt_valid_mask=prompt_valid_mask,
            prompt_speech_mask=prompt_speech_mask_raw,
            prompt_silence_mask=prompt_silence_mask,
            prompt_cache_meta=prompt_cache_meta if isinstance(prompt_cache_meta, Mapping) else None,
        )
        if tuple(prompt_speech_mask.shape) != expected_shape:
            raise ValueError("prompt_speech_mask must match prompt_duration_obs shape.")
        if bool((prompt_speech_mask > (prompt_valid_mask + 1.0e-6)).any().item()):
            raise ValueError("prompt_speech_mask must be bounded by prompt_unit_mask / prompt_valid_mask.")
        if isinstance(prompt_silence_mask, torch.Tensor):
            combined = prompt_speech_mask + prompt_silence_mask
            if bool((combined > (prompt_valid_mask + 1.0e-6)).any().item()):
                raise ValueError("prompt_speech_mask + prompt_silence_mask must not exceed prompt_valid_mask.")
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
        if isinstance(prompt_cache_meta, Mapping):
            assert_duration_v3_cache_meta_compatible(
                prompt_cache_meta,
                silent_token=self.silent_token,
                separator_aware=self.separator_aware,
                tail_open_units=self.tail_open_units,
                emit_silence_runs=self.emit_silence_runs,
                debounce_min_run_frames=self.debounce_min_run_frames,
                phrase_boundary_threshold=self.phrase_boundary_threshold,
            )
            enriched[DURATION_V3_CACHE_META_KEY] = dict(prompt_cache_meta)
        if isinstance(ref_conditioning.get("prompt_spk_embed"), torch.Tensor):
            enriched["prompt_spk_embed"] = ref_conditioning["prompt_spk_embed"].to(device=device).float().detach()
        rate_mode = str(getattr(self.module, "rate_mode", "" ) or "").strip().lower()
        use_log_base_rate = (
            rate_mode != "simple_global"
            and bool(getattr(self.module, "use_log_base_rate", False))
        )
        if use_log_base_rate and isinstance(ref_conditioning.get("prompt_unit_anchor_base"), torch.Tensor):
            enriched["prompt_unit_anchor_base"] = ref_conditioning["prompt_unit_anchor_base"].to(device=device).detach()
        if use_log_base_rate and isinstance(ref_conditioning.get("prompt_log_base"), torch.Tensor):
            enriched["prompt_log_base"] = ref_conditioning["prompt_log_base"].to(device=device).detach()
        def _normalize_prompt_scalar(value, *, key: str) -> torch.Tensor | None:
            if value is None:
                return None
            tensor = value.to(device=device).float() if isinstance(value, torch.Tensor) else torch.as_tensor(
                value,
                device=device,
                dtype=torch.float32,
            )
            if tensor.ndim == 0:
                tensor = tensor.reshape(1, 1).expand(prompt_duration_obs.size(0), 1)
            elif tensor.ndim == 1:
                tensor = tensor.reshape(-1, 1)
                if int(tensor.size(0)) == 1 and int(prompt_duration_obs.size(0)) > 1:
                    tensor = tensor.expand(prompt_duration_obs.size(0), 1)
            else:
                tensor = tensor.reshape(int(tensor.size(0)), -1)
            if int(tensor.size(0)) != int(prompt_duration_obs.size(0)):
                raise ValueError(
                    f"{key} batch mismatch: expected batch_size={int(prompt_duration_obs.size(0))}, "
                    f"got {tuple(tensor.shape)}."
                )
            return tensor[:, :1]
        for key in (
            "prompt_closed_mask",
            "prompt_boundary_confidence",
            "prompt_global_weight",
            "prompt_global_weight_present",
            "prompt_unit_log_prior",
            "prompt_unit_log_prior_present",
            "prompt_unit_prior_vocab_size",
            "prompt_source_boundary_cue",
            "prompt_phrase_group_pos",
            "prompt_phrase_final_mask",
            "g_trim_ratio",
            "prompt_g_trim_ratio",
            "prompt_g_drop_edge_runs",
            "prompt_min_boundary_confidence_for_g",
        ):
            if isinstance(ref_conditioning.get(key), torch.Tensor):
                enriched[key] = ref_conditioning[key].to(device=device).float()
        prompt_ref_len_sec = _normalize_prompt_scalar(
            ref_conditioning.get("prompt_ref_len_sec"),
            key="prompt_ref_len_sec",
        )
        if isinstance(prompt_ref_len_sec, torch.Tensor):
            enriched["prompt_ref_len_sec"] = prompt_ref_len_sec
        prompt_speech_ratio_scalar = _normalize_prompt_scalar(
            ref_conditioning.get("prompt_speech_ratio_scalar"),
            key="prompt_speech_ratio_scalar",
        )
        if prompt_speech_ratio_scalar is None:
            prompt_speech_ratio_scalar = compute_duration_weighted_speech_ratio(
                duration_obs=prompt_duration_obs.float(),
                speech_mask=prompt_speech_mask.float(),
                valid_mask=prompt_valid_mask.float(),
            )
        enriched["prompt_speech_ratio_scalar"] = prompt_speech_ratio_scalar
        if isinstance(enriched.get("prompt_unit_prior_vocab_size"), torch.Tensor):
            enriched["prompt_unit_prior_vocab_size"] = enriched["prompt_unit_prior_vocab_size"].to(device=device).long()
        if (
            use_log_base_rate
            and enriched.get("prompt_log_base") is None
            and enriched.get("prompt_unit_anchor_base") is None
        ):
            enriched["prompt_log_base"] = self.unit_frontend.compute_rate_log_base(
                prompt_content_units,
                prompt_valid_mask,
                stop_gradient=True,
            )
        return enriched

    def _collect_contract_meta(self) -> dict[str, object]:
        prompt_domain_mode = str(getattr(self.module, "prompt_domain_mode", "minimal_strict"))
        prompt_ref_len_contract_active = prompt_domain_mode.strip().lower() == "minimal_strict"
        min_prompt_ref_len_sec = None
        max_prompt_ref_len_sec = None
        if prompt_ref_len_contract_active:
            min_prompt_ref_len_sec = float(getattr(self.module, "min_prompt_ref_len_sec", 0.0) or 0.0)
            max_prompt_ref_len_sec = float(getattr(self.module, "max_prompt_ref_len_sec", 0.0) or 0.0)
        values: dict[str, object] = {
            "rhythm_v3_g_variant": str(getattr(self.module, "g_variant", "raw_median")),
            "rhythm_v3_g_trim_ratio": float(getattr(self.module, "g_trim_ratio", 0.2)),
            "rhythm_v3_drop_edge_runs_for_g": int(getattr(self.module, "g_drop_edge_runs", 0)),
            "rhythm_v3_min_boundary_confidence_for_g": (
                None
                if getattr(self.module, "min_boundary_confidence_for_g", None) is None
                else float(self.module.min_boundary_confidence_for_g)
            ),
            "rhythm_v3_min_prompt_speech_ratio": float(
                getattr(self.module, "min_prompt_speech_ratio", self.hparams.get("rhythm_v3_min_prompt_speech_ratio", 0.6))
            ),
            "rhythm_v3_min_prompt_ref_len_sec": min_prompt_ref_len_sec,
            "rhythm_v3_max_prompt_ref_len_sec": max_prompt_ref_len_sec,
            "rhythm_v3_disallow_same_text_reference": bool(
                self.hparams.get("rhythm_v3_disallow_same_text_reference", True)
            ),
            "rhythm_v3_disallow_same_text_paired_target": bool(
                self.hparams.get("rhythm_v3_disallow_same_text_paired_target", False)
            ),
            "rhythm_v3_require_same_text_paired_target": bool(
                self.hparams.get("rhythm_v3_require_same_text_paired_target", True)
            ),
            "rhythm_v3_strict_eval_invalid_g": bool(
                self.hparams.get("rhythm_v3_strict_eval_invalid_g", True)
            ),
            "rhythm_v3_alignment_prefilter_bad_samples": bool(
                self.hparams.get("rhythm_v3_alignment_prefilter_bad_samples", False)
            ),
            "rhythm_v3_alignment_prefilter_max_attempts": int(
                self.hparams.get("rhythm_v3_alignment_prefilter_max_attempts", 4) or 0
            ),
            "rhythm_v3_alignment_unmatched_speech_ratio_max": float(
                self.hparams.get("rhythm_v3_alignment_unmatched_speech_ratio_max", 1.0) or 0.0
            ),
            "rhythm_v3_alignment_mean_local_confidence_speech_min": float(
                self.hparams.get("rhythm_v3_alignment_mean_local_confidence_speech_min", 0.0) or 0.0
            ),
            "rhythm_v3_alignment_mean_coarse_confidence_speech_min": float(
                self.hparams.get("rhythm_v3_alignment_mean_coarse_confidence_speech_min", 0.0) or 0.0
            ),
            "rhythm_v3_alignment_local_margin_p10_min": float(
                self.hparams.get("rhythm_v3_alignment_local_margin_p10_min", 0.0) or 0.0
            ),
            "rhythm_v3_prompt_domain_mode": prompt_domain_mode,
            "rhythm_v3_prompt_require_clean_support": bool(
                getattr(self.module, "prompt_require_clean_support", True)
            ),
            "rhythm_v3_prompt_g_variant": str(
                getattr(self.module, "prompt_g_variant", getattr(self.module, "g_variant", "raw_median"))
            ),
            "rhythm_v3_prompt_g_trim_ratio": float(
                getattr(self.module, "prompt_g_trim_ratio", getattr(self.module, "g_trim_ratio", 0.2))
            ),
            "rhythm_v3_prompt_g_drop_edge_runs": int(
                getattr(self.module, "prompt_g_drop_edge_runs", getattr(self.module, "g_drop_edge_runs", 0))
            ),
            "rhythm_v3_prompt_min_boundary_confidence_for_g": (
                None
                if getattr(self.module, "prompt_min_boundary_confidence_for_g", None) is None
                else float(self.module.prompt_min_boundary_confidence_for_g)
            ),
            "rhythm_v3_src_g_variant": str(
                getattr(self.module, "src_g_variant", getattr(self.module, "g_variant", "raw_median"))
            ),
            "rhythm_v3_src_g_trim_ratio": float(
                getattr(self.module, "src_g_trim_ratio", getattr(self.module, "g_trim_ratio", 0.2))
            ),
            "rhythm_v3_src_g_drop_edge_runs": int(
                getattr(self.module, "src_g_drop_edge_runs", getattr(self.module, "g_drop_edge_runs", 0))
            ),
            "rhythm_v3_src_min_boundary_confidence_for_g": (
                None
                if getattr(self.module, "src_min_boundary_confidence_for_g", None) is None
                else float(self.module.src_min_boundary_confidence_for_g)
            ),
            "rhythm_v3_prompt_ref_len_contract_active": prompt_ref_len_contract_active,
            "rhythm_v3_src_prefix_stat_mode": str(getattr(self.module, "src_prefix_stat_mode", "ema")),
            "rhythm_v3_src_prefix_min_support": int(getattr(self.module, "src_prefix_min_support", 3)),
            "rhythm_v3_src_rate_init_mode": str(getattr(self.module, "src_rate_init_mode", "first_speech")),
            "rhythm_v3_use_src_gap_in_coarse_head": bool(getattr(self.module, "use_src_gap_in_coarse_head", False)),
            "rhythm_v3_analytic_gap_clip": float(getattr(self.module, "analytic_gap_clip", 0.35)),
            "rhythm_v3_prefix_budget_pos": int(getattr(self.module.projector, "prefix_budget_pos", 24)),
            "rhythm_v3_prefix_budget_neg": int(getattr(self.module.projector, "prefix_budget_neg", 24)),
            "rhythm_v3_dynamic_budget_ratio": float(getattr(self.module.projector, "dynamic_budget_ratio", 0.0)),
            "rhythm_v3_min_prefix_budget": int(getattr(self.module.projector, "min_prefix_budget", 0)),
            "rhythm_v3_max_prefix_budget": int(getattr(self.module.projector, "max_prefix_budget", 0)),
            "rhythm_v3_budget_mode": str(getattr(self.module.projector, "budget_mode", "total")),
            "rhythm_v3_boundary_carry_decay": float(getattr(self.module.projector, "boundary_carry_decay", 0.25)),
            "rhythm_v3_boundary_offset_decay": float(getattr(self.module.projector, "boundary_offset_decay", 0.25)),
            "rhythm_v3_boundary_reset_thresh": float(getattr(self.module.projector, "boundary_reset_thresh", 0.5)),
            "rhythm_v3_projection_mode": str(getattr(self.module.projector, "projection_mode", "greedy")),
            "rhythm_v3_integer_projection_mode": str(
                getattr(
                    self.module.projector,
                    "integer_projection_mode",
                    getattr(self.module.projector, "projection_mode", "greedy"),
                )
            ),
            "rhythm_v3_integer_projection_anchor_mode": str(
                getattr(self.module.projector, "integer_projection_anchor_mode", "rounded")
            ),
            "rhythm_v3_prefix_optimal_step_weight": float(
                getattr(self.module.projector, "prefix_optimal_step_weight", 0.10)
            ),
            "rhythm_v3_prefix_optimal_prefix_weight": float(
                getattr(self.module.projector, "prefix_optimal_prefix_weight", 1.00)
            ),
            "rhythm_v3_prefix_optimal_terminal_weight": float(
                getattr(self.module.projector, "prefix_optimal_terminal_weight", 1.00)
            ),
            "rhythm_v3_prefix_optimal_boundary_weight": float(
                getattr(self.module.projector, "prefix_optimal_boundary_weight", 0.75)
            ),
            "rhythm_v3_prefix_optimal_coarse_weight": float(
                getattr(self.module.projector, "prefix_optimal_coarse_weight", 0.50)
            ),
            "rhythm_v3_prefix_optimal_phrase_final_boost": float(
                getattr(self.module.projector, "prefix_optimal_phrase_final_boost", 1.50)
            ),
            "rhythm_v3_prefix_optimal_max_window": int(
                getattr(self.module.projector, "prefix_optimal_max_window", 96)
            ),
            "rhythm_v3_prefix_optimal_max_states": int(
                getattr(self.module.projector, "prefix_optimal_max_states", 97)
            ),
            "rhythm_v3_projection_repair_max_steps": int(
                getattr(self.module.projector, "projection_repair_max_steps", 0)
            ),
            "rhythm_v3_projection_repair_speech_bonus": float(
                getattr(self.module.projector, "projection_repair_speech_bonus", 1.0)
            ),
            "rhythm_v3_projection_repair_boundary_penalty": float(
                getattr(self.module.projector, "projection_repair_boundary_penalty", 0.35)
            ),
            "rhythm_v3_use_continuous_alignment": bool(
                self.hparams.get("rhythm_v3_use_continuous_alignment", False)
            ),
            "rhythm_v3_alignment_mode": str(
                self.hparams.get("rhythm_v3_alignment_mode", "continuous_viterbi_v1")
                or "continuous_viterbi_v1"
            ),
            "rhythm_v3_minimal_v1_profile": bool(self.hparams.get("rhythm_v3_minimal_v1_profile", False)),
            "rhythm_v3_strict_minimal_claim_profile": bool(
                self.hparams.get("rhythm_v3_strict_minimal_claim_profile", True)
            ),
        }
        values["rhythm_v3_control_contract_fingerprint"] = build_runtime_contract_fingerprint_json_from_values(values)
        values["rhythm_v3_control_contract_id"] = build_runtime_contract_id_from_values(values)
        values["rhythm_v3_control_contract_values"] = build_runtime_contract_fingerprint_from_values(values)
        return values

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
                if bool(getattr(self.module, "is_minimal_v1", False)):
                    required_meta = (
                        "prompt_speech_mask",
                        "prompt_closed_mask",
                        "prompt_boundary_confidence",
                        "prompt_ref_len_sec",
                    )
                    missing_meta = [
                        key
                        for key in required_meta
                        if not isinstance(ref_conditioning.get(key), torch.Tensor)
                    ]
                    if missing_meta:
                        raise ValueError(
                            "rhythm_v3 minimal_v1 training requires explicit prompt-side domain metadata: "
                            + ", ".join(missing_meta)
                        )
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
        ref_conditioning_meta=None,
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
        ret["unit_duration_exec"] = execution.unit_duration_exec
        ret["speech_duration_exec"] = execution.speech_duration_exec
        ret["commit_frontier"] = execution.commit_frontier
        ret["rhythm_v3_runtime_mode"] = getattr(self.module, "public_runtime_mode", self.module.runtime_mode)
        ret["rhythm_v3_backbone_mode"] = getattr(self.module, "public_backbone_mode", self.module.backbone_mode)
        ret["rhythm_v3_warp_mode"] = self.module.warp_mode
        ret["rhythm_v3_allow_hybrid"] = float(bool(self.module.allow_hybrid))
        ret["rhythm_v3_baseline_train_mode"] = self.baseline_train_mode
        ret["rhythm_v3_source_residual_gain"] = float(self.module.source_residual_gain)
        ret["rhythm_v3_eval_mode"] = self.module.eval_mode
        contract_meta = self._collect_contract_meta()
        ret.update({k: v for k, v in contract_meta.items() if k != "rhythm_v3_control_contract_values"})
        src_prefix_stat_mode = str(contract_meta["rhythm_v3_src_prefix_stat_mode"])
        ret["rhythm_v3_src_prefix_stat_mode"] = src_prefix_stat_mode
        ret["rhythm_v3_src_prefix_requires_full_history"] = float(
            src_prefix_stat_mode_requires_full_history(src_prefix_stat_mode)
        )
        ret["rhythm_v3_src_prefix_contract_scope"] = (
            "local_offline_candidate_only"
            if src_prefix_stat_mode_requires_full_history(src_prefix_stat_mode)
            else "state_safe_runtime"
        )
        ret["rhythm_v3_src_rate_init_mode"] = str(ret["rhythm_v3_src_rate_init_mode"])
        ret["rhythm_v3_src_prefix_min_support"] = float(ret["rhythm_v3_src_prefix_min_support"])
        ret["rhythm_v3_detach_global_term_in_local_head"] = float(
            bool(getattr(self.module, "detach_global_term_in_local_head", False))
        )
        if isinstance(getattr(execution, "prompt_speech_ratio", None), torch.Tensor):
            ret["rhythm_prompt_speech_ratio"] = execution.prompt_speech_ratio.detach()
        if isinstance(getattr(execution, "prompt_valid_len", None), torch.Tensor):
            ret["rhythm_prompt_valid_len"] = execution.prompt_valid_len.detach()
        if isinstance(getattr(execution, "g_src_utt", None), torch.Tensor):
            ret["rhythm_g_src_utt"] = execution.g_src_utt.detach()
        if isinstance(getattr(execution, "g_src_prefix_mean", None), torch.Tensor):
            ret["rhythm_g_src_prefix_mean"] = execution.g_src_prefix_mean.detach()
        if isinstance(getattr(execution, "g_src_prefix_final", None), torch.Tensor):
            ret["rhythm_g_src_prefix_final"] = execution.g_src_prefix_final.detach()
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
        if isinstance(getattr(execution, "commit_closed_prefix_ok", None), torch.Tensor):
            ret["rhythm_v3_commit_closed_prefix_ok"] = execution.commit_closed_prefix_ok.detach()
        if isinstance(getattr(execution, "open_tail_commit_violation", None), torch.Tensor):
            ret["rhythm_v3_open_tail_commit_violation"] = execution.open_tail_commit_violation.detach()
        if isinstance(getattr(execution, "open_tail_commit_violation_count", None), torch.Tensor):
            ret["rhythm_v3_open_tail_commit_violation_count"] = (
                execution.open_tail_commit_violation_count.detach()
            )
        if isinstance(ref_conditioning_meta, Mapping):
            if isinstance(ref_conditioning_meta.get("prompt_global_weight_present"), torch.Tensor):
                ret["rhythm_prompt_global_weight_present"] = ref_conditioning_meta["prompt_global_weight_present"].detach()
            elif isinstance(ref_conditioning_meta.get("prompt_global_weight"), torch.Tensor):
                ret["rhythm_prompt_global_weight_present"] = torch.ones(
                    (int(ref_memory.global_rate.size(0)), 1),
                    device=ref_memory.global_rate.device,
                    dtype=ref_memory.global_rate.dtype,
                )
            if isinstance(ref_conditioning_meta.get("prompt_unit_log_prior_present"), torch.Tensor):
                ret["rhythm_prompt_unit_log_prior_present"] = ref_conditioning_meta["prompt_unit_log_prior_present"].detach()
            elif isinstance(ref_conditioning_meta.get("prompt_unit_log_prior"), torch.Tensor):
                ret["rhythm_prompt_unit_log_prior_present"] = torch.ones(
                    (int(ref_memory.global_rate.size(0)), 1),
                    device=ref_memory.global_rate.device,
                    dtype=ref_memory.global_rate.dtype,
                )
            if isinstance(ref_conditioning_meta.get("prompt_unit_prior_vocab_size"), torch.Tensor):
                ret["rhythm_prompt_unit_prior_vocab_size"] = ref_conditioning_meta["prompt_unit_prior_vocab_size"].detach()
            if isinstance(ref_conditioning_meta.get("prompt_ref_len_sec"), torch.Tensor):
                ret["rhythm_prompt_ref_len_sec"] = ref_conditioning_meta["prompt_ref_len_sec"].detach()
        if self.module.debug_export:
            g_debug_stats = {}
            prompt_ref_len_sec = None
            prompt_speech_mask = getattr(ref_memory, "prompt_speech_mask", None) if ref_memory is not None else None
            prompt_valid_mask = getattr(ref_memory, "prompt_valid_mask", None) if ref_memory is not None else None
            if isinstance(prompt_speech_mask, torch.Tensor):
                min_prompt_speech_ratio = float(
                    getattr(getattr(self.module, "prompt_memory_encoder", None), "min_speech_ratio", 0.0)
                )
                speech_count = prompt_speech_mask.float().sum(dim=1, keepdim=True)
                valid_count = (
                    prompt_valid_mask.float().sum(dim=1, keepdim=True)
                    if isinstance(prompt_valid_mask, torch.Tensor)
                    else prompt_speech_mask.new_full((prompt_speech_mask.size(0), 1), float(prompt_speech_mask.size(1)))
                )
                def _meta_tensor(key: str) -> torch.Tensor | None:
                    if isinstance(ref_conditioning_meta, Mapping):
                        value = ref_conditioning_meta.get(key)
                        if isinstance(value, torch.Tensor):
                            return value
                    return None
                prompt_duration_obs = _meta_tensor("prompt_duration_obs")
                prompt_evidence = getattr(ref_memory, "prompt", None) if ref_memory is not None else None
                if not isinstance(prompt_duration_obs, torch.Tensor) and prompt_evidence is not None:
                    prompt_log_duration = getattr(prompt_evidence, "prompt_log_duration", None)
                    if isinstance(prompt_log_duration, torch.Tensor):
                        prompt_duration_obs = torch.exp(prompt_log_duration.float()).clamp_min(0.0)
                speech_ratio = compute_duration_weighted_speech_ratio(
                    duration_obs=(
                        prompt_duration_obs.float()
                        if isinstance(prompt_duration_obs, torch.Tensor)
                        else None
                    ),
                    speech_mask=prompt_speech_mask.float(),
                    valid_mask=prompt_valid_mask.float() if isinstance(prompt_valid_mask, torch.Tensor) else None,
                )
                prompt_closed_mask = _meta_tensor("prompt_closed_mask")
                prompt_boundary_confidence = _meta_tensor("prompt_boundary_confidence")
                prompt_global_weight = _meta_tensor("prompt_global_weight")
                prompt_ref_len_sec = _meta_tensor("prompt_ref_len_sec")
                prompt_ref_len_valid = None
                support_stats = summarize_global_rate_support(
                    speech_mask=prompt_speech_mask.float().clamp(0.0, 1.0),
                    valid_mask=(
                        prompt_valid_mask.float().clamp(0.0, 1.0)
                        if isinstance(prompt_valid_mask, torch.Tensor)
                        else None
                    ),
                    duration_obs=(
                        prompt_duration_obs.float().clamp_min(0.0)
                        if isinstance(prompt_duration_obs, torch.Tensor)
                        else None
                    ),
                    drop_edge_runs=int(getattr(self.module, "g_drop_edge_runs", 0)),
                    closed_mask=(
                        prompt_closed_mask.float().clamp(0.0, 1.0)
                        if isinstance(prompt_closed_mask, torch.Tensor)
                        else None
                    ),
                    boundary_confidence=(
                        prompt_boundary_confidence.float().clamp(0.0, 1.0)
                        if isinstance(prompt_boundary_confidence, torch.Tensor)
                        else None
                    ),
                    min_boundary_confidence=getattr(self.module, "min_boundary_confidence_for_g", None),
                )
                support_mask = (
                    ref_memory.prompt_g_support_mask.float()
                    if isinstance(getattr(ref_memory, "prompt_g_support_mask", None), torch.Tensor)
                    else support_stats.support_mask.float()
                )
                clean_mask = (
                    ref_memory.prompt_g_clean_mask.float()
                    if isinstance(getattr(ref_memory, "prompt_g_clean_mask", None), torch.Tensor)
                    else (
                        support_stats.clean_mask.float()
                        if isinstance(support_stats.clean_mask, torch.Tensor)
                        else support_mask
                    )
                )
                support_count = (
                    ref_memory.prompt_g_support_count.float()
                    if isinstance(getattr(ref_memory, "prompt_g_support_count", None), torch.Tensor)
                    else support_stats.support_count.float()
                )
                clean_count = (
                    ref_memory.prompt_g_clean_count.float()
                    if isinstance(getattr(ref_memory, "prompt_g_clean_count", None), torch.Tensor)
                    else (
                        support_stats.clean_count.float()
                        if isinstance(support_stats.clean_count, torch.Tensor)
                        else support_count
                    )
                )
                support_weight = (
                    ref_memory.prompt_g_support_weight.float()
                    if isinstance(getattr(ref_memory, "prompt_g_support_weight", None), torch.Tensor)
                    else (
                        torch.where(
                            support_mask > 0.5,
                            prompt_global_weight.float().clamp_min(0.0),
                            torch.zeros_like(prompt_global_weight.float()),
                        ).sum(dim=1, keepdim=True)
                        if isinstance(prompt_global_weight, torch.Tensor)
                        else support_count
                    )
                )
                support_ratio_vs_speech = (
                    ref_memory.prompt_g_support_ratio_vs_speech.float()
                    if isinstance(getattr(ref_memory, "prompt_g_support_ratio_vs_speech", None), torch.Tensor)
                    else (support_count / speech_count.clamp_min(1.0))
                )
                support_ratio_vs_valid = (
                    ref_memory.prompt_g_support_ratio_vs_valid.float()
                    if isinstance(getattr(ref_memory, "prompt_g_support_ratio_vs_valid", None), torch.Tensor)
                    else (support_count / valid_count.clamp_min(1.0))
                )
                clean_ratio_vs_speech = (
                    ref_memory.prompt_g_clean_ratio_vs_speech.float()
                    if isinstance(getattr(ref_memory, "prompt_g_clean_ratio_vs_speech", None), torch.Tensor)
                    else (clean_count / speech_count.clamp_min(1.0))
                )
                clean_ratio_vs_valid = (
                    ref_memory.prompt_g_clean_ratio_vs_valid.float()
                    if isinstance(getattr(ref_memory, "prompt_g_clean_ratio_vs_valid", None), torch.Tensor)
                    else (clean_count / valid_count.clamp_min(1.0))
                )
                if bool(getattr(self.module, "is_minimal_v1", False)):
                    min_ref_len = float(getattr(self.module, "min_prompt_ref_len_sec", 0.0))
                    max_ref_len = float(getattr(self.module, "max_prompt_ref_len_sec", float("inf")))
                    if isinstance(prompt_ref_len_sec, torch.Tensor):
                        ref_len = prompt_ref_len_sec.float().reshape(int(prompt_speech_mask.size(0)), -1)[:, :1]
                        prompt_ref_len_valid = (
                            torch.isfinite(ref_len)
                            & (ref_len >= min_ref_len)
                            & (ref_len <= max_ref_len)
                        )
                    else:
                        prompt_ref_len_valid = prompt_speech_mask.new_zeros(
                            (prompt_speech_mask.size(0), 1),
                            dtype=torch.bool,
                        )
                g_valid_support = ((support_count > 0.5) & (support_weight > 1.0e-6)).float()
                g_domain_valid = (
                    ref_memory.prompt_g_domain_valid.float()
                    if isinstance(getattr(ref_memory, "prompt_g_domain_valid", None), torch.Tensor)
                    else (
                        (
                            (g_valid_support > 0.5)
                            & (clean_count > 0.5)
                            & (speech_ratio >= (min_prompt_speech_ratio - 1.0e-6))
                        ).float()
                    )
                )
                if (
                    not isinstance(getattr(ref_memory, "prompt_g_domain_valid", None), torch.Tensor)
                    and prompt_ref_len_valid is not None
                ):
                    g_domain_valid = g_domain_valid * prompt_ref_len_valid.float()
                g_debug_stats = {
                    "g_support_count": support_count.detach(),
                    "g_clean_count": clean_count.detach(),
                    "g_support_weight": support_weight.detach(),
                    "g_speech_count": speech_count.detach(),
                    "g_valid_count": valid_count.detach(),
                    "g_valid_support": g_valid_support.detach(),
                    "g_domain_valid": g_domain_valid.detach(),
                    "g_min_speech_ratio": support_count.new_full(
                        support_count.shape,
                        float(min_prompt_speech_ratio),
                    ),
                    "prompt_speech_ratio": speech_ratio.detach(),
                    "g_support_ratio_vs_speech": support_ratio_vs_speech.detach(),
                    "g_support_ratio_vs_valid": support_ratio_vs_valid.detach(),
                    "g_clean_ratio_vs_speech": clean_ratio_vs_speech.detach(),
                    "g_clean_ratio_vs_valid": clean_ratio_vs_valid.detach(),
                    "g_valid": g_domain_valid.float().detach(),
                    "g_drop_edge_runs": support_count.new_full(
                        support_count.shape,
                        float(int(getattr(self.module, "g_drop_edge_runs", 0))),
                    ),
                    "g_edge_runs_dropped": support_stats.edge_runs_dropped.detach(),
                    "g_ref_len_valid": (
                        prompt_ref_len_valid.float().detach()
                        if prompt_ref_len_valid is not None
                        else None
                    ),
                    "g_strict_speech_only": support_count.new_ones(support_count.shape),
                    "prompt_g_support_mask": support_mask.detach(),
                    "prompt_g_clean_mask": clean_mask.detach(),
                    "prompt_g_speech_ratio_weighted": (
                        ref_memory.prompt_g_speech_ratio_weighted.detach()
                        if isinstance(getattr(ref_memory, "prompt_g_speech_ratio_weighted", None), torch.Tensor)
                        else speech_ratio.detach()
                    ),
                    "prompt_g_speech_ratio_count": (
                        ref_memory.prompt_g_speech_ratio_count.detach()
                        if isinstance(getattr(ref_memory, "prompt_g_speech_ratio_count", None), torch.Tensor)
                        else (speech_count / valid_count.clamp_min(1.0)).detach()
                    ),
                    "prompt_g_invalid_no_speech": (
                        ref_memory.prompt_g_invalid_no_speech.detach()
                        if isinstance(getattr(ref_memory, "prompt_g_invalid_no_speech", None), torch.Tensor)
                        else (speech_count <= 0.0).float().detach()
                    ),
                    "prompt_g_invalid_low_speech_ratio": (
                        ref_memory.prompt_g_invalid_low_speech_ratio.detach()
                        if isinstance(getattr(ref_memory, "prompt_g_invalid_low_speech_ratio", None), torch.Tensor)
                        else (speech_ratio < (min_prompt_speech_ratio - 1.0e-6)).float().detach()
                    ),
                    "prompt_g_invalid_ref_len": (
                        ref_memory.prompt_g_invalid_ref_len.detach()
                        if isinstance(getattr(ref_memory, "prompt_g_invalid_ref_len", None), torch.Tensor)
                        else (
                            None
                            if prompt_ref_len_valid is None
                            else (1.0 - prompt_ref_len_valid.float()).detach()
                        )
                    ),
                    "prompt_g_invalid_support": (
                        ref_memory.prompt_g_invalid_support.detach()
                        if isinstance(getattr(ref_memory, "prompt_g_invalid_support", None), torch.Tensor)
                        else (1.0 - g_valid_support).detach()
                    ),
                    "prompt_g_invalid_clean": (
                        ref_memory.prompt_g_invalid_clean.detach()
                        if isinstance(getattr(ref_memory, "prompt_g_invalid_clean", None), torch.Tensor)
                        else (clean_count <= 0.5).float().detach()
                    ),
                    "prompt_g_invalid_missing_closed": (
                        ref_memory.prompt_g_invalid_missing_closed.detach()
                        if isinstance(getattr(ref_memory, "prompt_g_invalid_missing_closed", None), torch.Tensor)
                        else None
                    ),
                    "prompt_g_invalid_missing_boundary": (
                        ref_memory.prompt_g_invalid_missing_boundary.detach()
                        if isinstance(getattr(ref_memory, "prompt_g_invalid_missing_boundary", None), torch.Tensor)
                        else None
                    ),
                }
            coarse_correction_used = (
                execution.coarse_correction.detach()
                if isinstance(getattr(execution, "coarse_correction", None), torch.Tensor)
                else None
            )
            coarse_correction_pred = (
                execution.coarse_correction_pred.detach()
                if isinstance(getattr(execution, "coarse_correction_pred", None), torch.Tensor)
                else coarse_correction_used
            )
            residual_logstretch_used = (
                execution.local_residual.detach()
                if isinstance(getattr(execution, "local_residual", None), torch.Tensor)
                else None
            )
            residual_logstretch_pred = (
                execution.local_residual_pred.detach()
                if isinstance(getattr(execution, "local_residual_pred", None), torch.Tensor)
                else residual_logstretch_used
            )
            debug_bundle = {
                "g_variant": self.module.g_variant,
                "g_trim_ratio": float(getattr(self.module, "g_trim_ratio", 0.2)),
                "prompt_g_variant": str(getattr(self.module, "prompt_g_variant", self.module.g_variant)),
                "prompt_g_trim_ratio": float(
                    getattr(self.module, "prompt_g_trim_ratio", getattr(self.module, "g_trim_ratio", 0.2))
                ),
                "prompt_g_drop_edge_runs": float(
                    getattr(self.module, "prompt_g_drop_edge_runs", getattr(self.module, "g_drop_edge_runs", 0))
                ),
                "prompt_min_boundary_confidence_for_g": (
                    None
                    if getattr(self.module, "prompt_min_boundary_confidence_for_g", None) is None
                    else float(self.module.prompt_min_boundary_confidence_for_g)
                ),
                "src_g_variant": str(getattr(self.module, "src_g_variant", self.module.g_variant)),
                "src_g_trim_ratio": float(
                    getattr(self.module, "src_g_trim_ratio", getattr(self.module, "g_trim_ratio", 0.2))
                ),
                "src_g_drop_edge_runs": float(
                    getattr(self.module, "src_g_drop_edge_runs", getattr(self.module, "g_drop_edge_runs", 0))
                ),
                "src_min_boundary_confidence_for_g": (
                    None
                    if getattr(self.module, "src_min_boundary_confidence_for_g", None) is None
                    else float(self.module.src_min_boundary_confidence_for_g)
                ),
                "src_prefix_stat_mode": str(getattr(self.module, "src_prefix_stat_mode", "ema")),
                "src_prefix_min_support": float(getattr(self.module, "src_prefix_min_support", 3)),
                "prompt_g_support_ratio_vs_speech": support_ratio_vs_speech.detach(),
                "prompt_g_support_ratio_vs_valid": support_ratio_vs_valid.detach(),
                "prompt_g_clean_ratio_vs_speech": clean_ratio_vs_speech.detach(),
                "prompt_g_clean_ratio_vs_valid": clean_ratio_vs_valid.detach(),
                "g_ref": (
                    execution.g_ref.detach()
                    if isinstance(getattr(execution, "g_ref", None), torch.Tensor)
                    else ref_memory.global_rate.detach()
                ),
                "g_ref_scalar": (
                    execution.g_ref.detach()
                    if isinstance(getattr(execution, "g_ref", None), torch.Tensor)
                    else ref_memory.global_rate.detach()
                ),
                "g_src_prefix": (
                    execution.g_src_prefix.detach()
                    if isinstance(getattr(execution, "g_src_prefix", None), torch.Tensor)
                    else None
                ),
                "g_src_prefix_seq": (
                    execution.g_src_prefix.detach()
                    if isinstance(getattr(execution, "g_src_prefix", None), torch.Tensor)
                    else None
                ),
                "g_src_utt": (
                    execution.g_src_utt.detach()
                    if isinstance(getattr(execution, "g_src_utt", None), torch.Tensor)
                    else None
                ),
                "g_src_prefix_mean": (
                    execution.g_src_prefix_mean.detach()
                    if isinstance(getattr(execution, "g_src_prefix_mean", None), torch.Tensor)
                    else None
                ),
                "g_src_prefix_final": (
                    execution.g_src_prefix_final.detach()
                    if isinstance(getattr(execution, "g_src_prefix_final", None), torch.Tensor)
                    else None
                ),
                "global_shift_analytic": (
                    execution.global_shift_analytic.detach()
                    if isinstance(getattr(execution, "global_shift_analytic", None), torch.Tensor)
                    else None
                ),
                "analytic_gap_raw": (
                    execution.analytic_gap_raw.detach()
                    if isinstance(getattr(execution, "analytic_gap_raw", None), torch.Tensor)
                    else None
                ),
                "analytic_gap_clipped": (
                    execution.analytic_gap_clipped.detach()
                    if isinstance(getattr(execution, "analytic_gap_clipped", None), torch.Tensor)
                    else None
                ),
                "analytic_clip_hit": (
                    execution.analytic_clip_hit.detach()
                    if isinstance(getattr(execution, "analytic_clip_hit", None), torch.Tensor)
                    else None
                ),
                "analytic_clip_hit_rate": (
                    execution.analytic_clip_hit_rate.detach()
                    if isinstance(getattr(execution, "analytic_clip_hit_rate", None), torch.Tensor)
                    else None
                ),
                "analytic_logstretch": (
                    execution.global_shift_analytic.detach()
                    if isinstance(getattr(execution, "global_shift_analytic", None), torch.Tensor)
                    else None
                ),
                "coarse_correction": (
                    coarse_correction_used
                ),
                "coarse_correction_used": (
                    coarse_correction_used
                ),
                "coarse_correction_pred": (
                    coarse_correction_pred
                ),
                "coarse_delta": (
                    coarse_correction_used
                ),
                "coarse_path_logstretch": (
                    execution.coarse_path_logstretch.detach()
                    if isinstance(getattr(execution, "coarse_path_logstretch", None), torch.Tensor)
                    else None
                ),
                "local_residual": (
                    residual_logstretch_used
                ),
                "residual_logstretch": (
                    residual_logstretch_used
                ),
                "residual_logstretch_used": (
                    residual_logstretch_used
                ),
                "residual_logstretch_pred": (
                    residual_logstretch_pred
                ),
                "speech_pred": (
                    execution.speech_pred.detach()
                    if isinstance(getattr(execution, "speech_pred", None), torch.Tensor)
                    else None
                ),
                "silence_pred": (
                    execution.silence_pred.detach()
                    if isinstance(getattr(execution, "silence_pred", None), torch.Tensor)
                    else None
                ),
                "source_rate_seq": (
                    execution.source_rate_seq.detach()
                    if isinstance(getattr(execution, "source_rate_seq", None), torch.Tensor)
                    else None
                ),
                "coarse_scalar_raw": (
                    execution.coarse_scalar_raw.detach()
                    if isinstance(getattr(execution, "coarse_scalar_raw", None), torch.Tensor)
                    else None
                ),
                "global_term_before_local": (
                    execution.global_term_before_local.detach()
                    if isinstance(getattr(execution, "global_term_before_local", None), torch.Tensor)
                    else None
                ),
                "unit_residual_gate": (
                    execution.unit_residual_gate.detach()
                    if isinstance(getattr(execution, "unit_residual_gate", None), torch.Tensor)
                    else None
                ),
                "unit_residual_cold_gate": (
                    execution.unit_residual_cold_gate.detach()
                    if isinstance(getattr(execution, "unit_residual_cold_gate", None), torch.Tensor)
                    else None
                ),
                "unit_residual_short_gate": (
                    execution.unit_residual_short_gate.detach()
                    if isinstance(getattr(execution, "unit_residual_short_gate", None), torch.Tensor)
                    else None
                ),
                "unit_residual_gate_stability": (
                    execution.unit_residual_gate_stability.detach()
                    if isinstance(getattr(execution, "unit_residual_gate_stability", None), torch.Tensor)
                    else None
                ),
                "residual_gate_cold": (
                    execution.unit_residual_cold_gate.detach()
                    if isinstance(getattr(execution, "unit_residual_cold_gate", None), torch.Tensor)
                    else None
                ),
                "residual_gate_short": (
                    execution.unit_residual_short_gate.detach()
                    if isinstance(getattr(execution, "unit_residual_short_gate", None), torch.Tensor)
                    else None
                ),
                "residual_gate_stability": (
                    execution.unit_residual_gate_stability.detach()
                    if isinstance(getattr(execution, "unit_residual_gate_stability", None), torch.Tensor)
                    else None
                ),
                "unit_runtime_stability": (
                    execution.unit_runtime_stability.detach()
                    if isinstance(getattr(execution, "unit_runtime_stability", None), torch.Tensor)
                    else None
                ),
                "residual_gate_mean": (
                    execution.residual_gate_mean.detach()
                    if isinstance(getattr(execution, "residual_gate_mean", None), torch.Tensor)
                    else None
                ),
                "detach_global_term_in_local_head": (
                    execution.detach_global_term_in_local_head.detach()
                    if isinstance(getattr(execution, "detach_global_term_in_local_head", None), torch.Tensor)
                    else None
                ),
                "prompt_speech_ratio": (
                    execution.prompt_speech_ratio.detach()
                    if isinstance(getattr(execution, "prompt_speech_ratio", None), torch.Tensor)
                    else None
                ),
                "prompt_valid_len": (
                    execution.prompt_valid_len.detach()
                    if isinstance(getattr(execution, "prompt_valid_len", None), torch.Tensor)
                    else None
                ),
                "prompt_ref_len_sec": (
                    prompt_ref_len_sec.detach()
                    if isinstance(prompt_ref_len_sec, torch.Tensor)
                    else None
                ),
                "prompt_global_weight_present": (
                    ret.get("rhythm_prompt_global_weight_present").detach()
                    if isinstance(ret.get("rhythm_prompt_global_weight_present"), torch.Tensor)
                    else None
                ),
                "prompt_unit_log_prior_present": (
                    ret.get("rhythm_prompt_unit_log_prior_present").detach()
                    if isinstance(ret.get("rhythm_prompt_unit_log_prior_present"), torch.Tensor)
                    else None
                ),
                "prompt_unit_prior_vocab_size": (
                    ret.get("rhythm_prompt_unit_prior_vocab_size").detach()
                    if isinstance(ret.get("rhythm_prompt_unit_prior_vocab_size"), torch.Tensor)
                    else None
                ),
                **g_debug_stats,
                "projector_prefix_offset": (
                    execution.prefix_unit_offset.detach()
                    if isinstance(getattr(execution, "prefix_unit_offset", None), torch.Tensor)
                    else None
                ),
                "projector_rounding_residual": (
                    execution.projector_rounding_residual.detach()
                    if isinstance(getattr(execution, "projector_rounding_residual", None), torch.Tensor)
                    else (
                        execution.next_state.rounding_residual.detach()
                        if isinstance(getattr(execution.next_state, "rounding_residual", None), torch.Tensor)
                        else None
                    )
                ),
                "projector_boundary_decay_applied": (
                    execution.projector_boundary_decay_applied.detach()
                    if isinstance(getattr(execution, "projector_boundary_decay_applied", None), torch.Tensor)
                    else None
                ),
                "projector_boundary_hit": (
                    execution.projector_boundary_hit.detach()
                    if isinstance(getattr(execution, "projector_boundary_hit", None), torch.Tensor)
                    else None
                ),
                "projector_budget_pos_used": (
                    execution.projector_budget_pos_used.detach()
                    if isinstance(getattr(execution, "projector_budget_pos_used", None), torch.Tensor)
                    else None
                ),
                "projector_budget_neg_used": (
                    execution.projector_budget_neg_used.detach()
                    if isinstance(getattr(execution, "projector_budget_neg_used", None), torch.Tensor)
                    else None
                ),
                "projector_budget_hit_mask": (
                    execution.projector_budget_hit_mask.detach()
                    if isinstance(getattr(execution, "projector_budget_hit_mask", None), torch.Tensor)
                    else (
                        (
                            execution.projector_budget_hit_pos.detach()
                            | execution.projector_budget_hit_neg.detach()
                        )
                        if isinstance(getattr(execution, "projector_budget_hit_pos", None), torch.Tensor)
                        and isinstance(getattr(execution, "projector_budget_hit_neg", None), torch.Tensor)
                        else None
                    )
                ),
                "projector_since_last_boundary": (
                    execution.projector_since_last_boundary.detach()
                    if isinstance(getattr(execution, "projector_since_last_boundary", None), torch.Tensor)
                    else (
                        execution.next_state.since_last_boundary.detach()
                        if isinstance(getattr(execution.next_state, "since_last_boundary", None), torch.Tensor)
                        else None
                    )
                ),
                "projected_prefix_cumsum": (
                    execution.projected_prefix_cumsum.detach()
                    if isinstance(getattr(execution, "projected_prefix_cumsum", None), torch.Tensor)
                    else None
                ),
                "projector_prefix_drift": (
                    execution.projector_prefix_drift.detach()
                    if isinstance(getattr(execution, "projector_prefix_drift", None), torch.Tensor)
                    else (
                        execution.prefix_unit_offset.detach()
                        if isinstance(getattr(execution, "prefix_unit_offset", None), torch.Tensor)
                        else None
                    )
                ),
                "projector_preclamp_exec": (
                    execution.projector_preclamp_exec.detach()
                    if isinstance(getattr(execution, "projector_preclamp_exec", None), torch.Tensor)
                    else None
                ),
                "projector_preclamp_duration_exec": (
                    execution.projector_preclamp_duration_exec.detach()
                    if isinstance(getattr(execution, "projector_preclamp_duration_exec", None), torch.Tensor)
                    else None
                ),
                "projector_prefreeze_exec": (
                    execution.projector_prefreeze_exec.detach()
                    if isinstance(getattr(execution, "projector_prefreeze_exec", None), torch.Tensor)
                    else None
                ),
                "projector_prefreeze_duration_exec": (
                    execution.projector_prefreeze_duration_exec.detach()
                    if isinstance(getattr(execution, "projector_prefreeze_duration_exec", None), torch.Tensor)
                    else (
                        execution.projector_prefreeze_exec.detach()
                        if isinstance(getattr(execution, "projector_prefreeze_exec", None), torch.Tensor)
                        else None
                    )
                ),
                "projector_repair_candidate_delta": (
                    execution.projector_repair_candidate_delta.detach()
                    if isinstance(getattr(execution, "projector_repair_candidate_delta", None), torch.Tensor)
                    else None
                ),
                "projector_repair_candidate_steps": (
                    execution.projector_repair_candidate_steps.detach()
                    if isinstance(getattr(execution, "projector_repair_candidate_steps", None), torch.Tensor)
                    else None
                ),
                "projector_repair_delta": (
                    execution.projector_repair_delta.detach()
                    if isinstance(getattr(execution, "projector_repair_delta", None), torch.Tensor)
                    else None
                ),
                "projector_repair_steps": (
                    execution.projector_repair_steps.detach()
                    if isinstance(getattr(execution, "projector_repair_steps", None), torch.Tensor)
                    else None
                ),
                "projector_clamp_delta": (
                    execution.projector_clamp_delta.detach()
                    if isinstance(getattr(execution, "projector_clamp_delta", None), torch.Tensor)
                    else None
                ),
                "projector_clamp_mass": (
                    execution.projector_clamp_mass.detach()
                    if isinstance(getattr(execution, "projector_clamp_mass", None), torch.Tensor)
                    else None
                ),
                "projector_rounding_only_regret": (
                    execution.projector_rounding_only_regret.detach()
                    if isinstance(getattr(execution, "projector_rounding_only_regret", None), torch.Tensor)
                    else None
                ),
                "projector_rounding_regret": (
                    execution.projector_rounding_regret.detach()
                    if isinstance(getattr(execution, "projector_rounding_regret", None), torch.Tensor)
                    else None
                ),
                "projector_projection_regret": (
                    execution.projector_projection_regret.detach()
                    if isinstance(getattr(execution, "projector_projection_regret", None), torch.Tensor)
                    else None
                ),
                "projector_preclamp_prefix_cumsum": (
                    execution.projector_preclamp_prefix_cumsum.detach()
                    if isinstance(getattr(execution, "projector_preclamp_prefix_cumsum", None), torch.Tensor)
                    else None
                ),
                "projector_prefreeze_prefix_cumsum": (
                    execution.projector_prefreeze_prefix_cumsum.detach()
                    if isinstance(getattr(execution, "projector_prefreeze_prefix_cumsum", None), torch.Tensor)
                    else None
                ),
                "source_prefix_cumsum": (
                    execution.source_prefix_cumsum.detach()
                    if isinstance(getattr(execution, "source_prefix_cumsum", None), torch.Tensor)
                    else None
                ),
                "commit_closed_prefix_ok": (
                    execution.commit_closed_prefix_ok.detach()
                    if isinstance(getattr(execution, "commit_closed_prefix_ok", None), torch.Tensor)
                    else None
                ),
                "open_tail_commit_violation": (
                    execution.open_tail_commit_violation.detach()
                    if isinstance(getattr(execution, "open_tail_commit_violation", None), torch.Tensor)
                    else None
                ),
                "open_tail_commit_violation_count": (
                    execution.open_tail_commit_violation_count.detach()
                    if isinstance(getattr(execution, "open_tail_commit_violation_count", None), torch.Tensor)
                    else None
                ),
                "projector_budget_mode": getattr(execution, "projector_budget_mode", None),
                "eval_mode": getattr(execution, "eval_mode", self.module.eval_mode),
            }
            ret["rhythm_v3_debug"] = debug_bundle
            ret["rhythm_debug_g_ref"] = debug_bundle["g_ref"]
            ret["rhythm_debug_g_ref_scalar"] = debug_bundle["g_ref_scalar"]
            ret["rhythm_debug_g_src_prefix"] = debug_bundle["g_src_prefix"]
            ret["rhythm_debug_g_src_prefix_seq"] = debug_bundle["g_src_prefix_seq"]
            ret["rhythm_debug_g_src_utt"] = debug_bundle["g_src_utt"]
            ret["rhythm_debug_g_src_prefix_mean"] = debug_bundle["g_src_prefix_mean"]
            ret["rhythm_debug_g_src_prefix_final"] = debug_bundle["g_src_prefix_final"]
            ret["rhythm_debug_analytic_gap"] = debug_bundle["global_shift_analytic"]
            ret["rhythm_debug_analytic_logstretch"] = debug_bundle["analytic_logstretch"]
            ret["rhythm_debug_analytic_gap_raw"] = debug_bundle["analytic_gap_raw"]
            ret["rhythm_debug_analytic_gap_clipped"] = debug_bundle["analytic_gap_clipped"]
            ret["rhythm_debug_analytic_clip_hit"] = debug_bundle["analytic_clip_hit"]
            ret["rhythm_debug_analytic_clip_hit_rate"] = debug_bundle["analytic_clip_hit_rate"]
            ret["rhythm_debug_coarse_bias"] = debug_bundle["coarse_correction"]
            ret["rhythm_debug_coarse_used"] = debug_bundle["coarse_correction_used"]
            ret["rhythm_debug_coarse_pred"] = debug_bundle["coarse_correction_pred"]
            ret["rhythm_debug_coarse_delta"] = debug_bundle["coarse_delta"]
            ret["rhythm_debug_coarse_path"] = debug_bundle["coarse_path_logstretch"]
            ret["rhythm_debug_coarse_scalar_raw"] = debug_bundle["coarse_scalar_raw"]
            ret["rhythm_debug_global_term_before_local"] = debug_bundle["global_term_before_local"]
            ret["rhythm_debug_local_residual"] = debug_bundle["local_residual"]
            ret["rhythm_debug_residual_logstretch"] = debug_bundle["residual_logstretch"]
            ret["rhythm_debug_residual_used"] = debug_bundle["residual_logstretch_used"]
            ret["rhythm_debug_residual_pred"] = debug_bundle["residual_logstretch_pred"]
            ret["rhythm_debug_unit_residual_gate"] = debug_bundle["unit_residual_gate"]
            ret["rhythm_debug_unit_residual_cold_gate"] = debug_bundle["unit_residual_cold_gate"]
            ret["rhythm_debug_unit_residual_short_gate"] = debug_bundle["unit_residual_short_gate"]
            ret["rhythm_debug_unit_residual_gate_stability"] = debug_bundle["unit_residual_gate_stability"]
            ret["rhythm_debug_residual_gate_cold"] = debug_bundle["residual_gate_cold"]
            ret["rhythm_debug_residual_gate_short"] = debug_bundle["residual_gate_short"]
            ret["rhythm_debug_residual_gate_stability"] = debug_bundle["residual_gate_stability"]
            ret["rhythm_debug_unit_runtime_stability"] = debug_bundle["unit_runtime_stability"]
            ret["rhythm_debug_residual_gate_mean"] = debug_bundle["residual_gate_mean"]
            ret["rhythm_debug_detach_global_term_in_local_head"] = debug_bundle["detach_global_term_in_local_head"]
            ret["rhythm_debug_speech_pred"] = debug_bundle["speech_pred"]
            ret["rhythm_debug_silence_pred"] = debug_bundle["silence_pred"]
            ret["rhythm_debug_projector_prefix_offset"] = debug_bundle["projector_prefix_offset"]
            ret["rhythm_debug_projector_rounding_residual"] = debug_bundle["projector_rounding_residual"]
            ret["rhythm_debug_projector_boundary_hit"] = debug_bundle["projector_boundary_hit"]
            ret["rhythm_debug_projector_boundary_decay"] = debug_bundle["projector_boundary_decay_applied"]
            ret["rhythm_debug_projector_budget_pos_used"] = debug_bundle["projector_budget_pos_used"]
            ret["rhythm_debug_projector_budget_neg_used"] = debug_bundle["projector_budget_neg_used"]
            ret["rhythm_debug_projector_budget_hit_mask"] = debug_bundle["projector_budget_hit_mask"]
            ret["rhythm_debug_projector_since_last_boundary"] = debug_bundle["projector_since_last_boundary"]
            ret["rhythm_debug_projector_prefix_drift"] = debug_bundle["projector_prefix_drift"]
            ret["rhythm_debug_projected_prefix_cumsum"] = debug_bundle["projected_prefix_cumsum"]
            ret["rhythm_debug_source_prefix_cumsum"] = debug_bundle["source_prefix_cumsum"]
            ret["rhythm_debug_commit_closed_prefix_ok"] = debug_bundle["commit_closed_prefix_ok"]
            ret["rhythm_debug_open_tail_commit_violation"] = debug_bundle["open_tail_commit_violation"]
            ret["rhythm_debug_open_tail_commit_violation_count"] = debug_bundle[
                "open_tail_commit_violation_count"
            ]
            ret["rhythm_debug_projector_preclamp_exec"] = debug_bundle["projector_preclamp_exec"]
            ret["rhythm_debug_projector_preclamp_duration_exec"] = debug_bundle["projector_preclamp_duration_exec"]
            ret["rhythm_debug_projector_prefreeze_exec"] = debug_bundle["projector_prefreeze_exec"]
            ret["rhythm_debug_projector_prefreeze_duration_exec"] = debug_bundle["projector_prefreeze_duration_exec"]
            ret["rhythm_debug_projector_repair_candidate_delta"] = debug_bundle["projector_repair_candidate_delta"]
            ret["rhythm_debug_projector_repair_candidate_steps"] = debug_bundle["projector_repair_candidate_steps"]
            ret["rhythm_debug_projector_repair_delta"] = debug_bundle["projector_repair_delta"]
            ret["rhythm_debug_projector_repair_steps"] = debug_bundle["projector_repair_steps"]
            ret["rhythm_debug_projector_clamp_delta"] = debug_bundle["projector_clamp_delta"]
            ret["rhythm_debug_projector_rounding_only_regret"] = debug_bundle["projector_rounding_only_regret"]
            ret["rhythm_debug_projector_projection_regret"] = debug_bundle["projector_projection_regret"]
            ret["rhythm_debug_projector_preclamp_prefix_cumsum"] = debug_bundle["projector_preclamp_prefix_cumsum"]
            ret["rhythm_debug_projector_prefreeze_prefix_cumsum"] = debug_bundle["projector_prefreeze_prefix_cumsum"]
            for key, value in g_debug_stats.items():
                ret[f"rhythm_debug_{key}"] = value
            if isinstance(ret.get("rhythm_prompt_global_weight_present"), torch.Tensor):
                ret["rhythm_debug_prompt_global_weight_present"] = ret["rhythm_prompt_global_weight_present"]
            if isinstance(ret.get("rhythm_prompt_unit_log_prior_present"), torch.Tensor):
                ret["rhythm_debug_prompt_unit_log_prior_present"] = ret["rhythm_prompt_unit_log_prior_present"]
            if isinstance(ret.get("rhythm_prompt_unit_prior_vocab_size"), torch.Tensor):
                ret["rhythm_debug_prompt_unit_prior_vocab_size"] = ret["rhythm_prompt_unit_prior_vocab_size"]
            if isinstance(debug_bundle.get("prompt_ref_len_sec"), torch.Tensor):
                ret["rhythm_debug_prompt_ref_len_sec"] = debug_bundle["prompt_ref_len_sec"]
            if isinstance(getattr(source_batch, "source_silence_mask", None), torch.Tensor):
                debug_bundle["is_speech"] = (
                    source_batch.unit_mask.float() * (1.0 - source_batch.source_silence_mask.float().clamp(0.0, 1.0))
                ).detach()
                ret["rhythm_debug_is_speech"] = debug_bundle["is_speech"]
            if isinstance(getattr(execution, "projector_budget_hit_pos", None), torch.Tensor):
                debug_bundle["budget_hit_pos"] = execution.projector_budget_hit_pos.detach()
                ret["rhythm_debug_budget_hit_pos"] = debug_bundle["budget_hit_pos"]
            if isinstance(getattr(execution, "projector_budget_hit_neg", None), torch.Tensor):
                debug_bundle["budget_hit_neg"] = execution.projector_budget_hit_neg.detach()
                ret["rhythm_debug_budget_hit_neg"] = debug_bundle["budget_hit_neg"]
            if isinstance(debug_bundle.get("projector_budget_hit_mask"), torch.Tensor):
                ret["rhythm_debug_budget_hit_mask"] = debug_bundle["projector_budget_hit_mask"]
            ret["rhythm_debug_prompt_g_support_ratio_vs_speech"] = debug_bundle["prompt_g_support_ratio_vs_speech"]
            ret["rhythm_debug_prompt_g_support_ratio_vs_valid"] = debug_bundle["prompt_g_support_ratio_vs_valid"]
            ret["rhythm_debug_prompt_g_clean_ratio_vs_speech"] = debug_bundle["prompt_g_clean_ratio_vs_speech"]
            ret["rhythm_debug_prompt_g_clean_ratio_vs_valid"] = debug_bundle["prompt_g_clean_ratio_vs_valid"]
            if isinstance(g_debug_stats.get("prompt_g_support_mask"), torch.Tensor):
                ret["rhythm_prompt_g_support_mask"] = g_debug_stats["prompt_g_support_mask"]
                ret["rhythm_debug_prompt_g_support_mask"] = g_debug_stats["prompt_g_support_mask"]
            if isinstance(g_debug_stats.get("prompt_g_clean_mask"), torch.Tensor):
                ret["rhythm_prompt_g_clean_mask"] = g_debug_stats["prompt_g_clean_mask"]
                ret["rhythm_debug_prompt_g_clean_mask"] = g_debug_stats["prompt_g_clean_mask"]

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
        del target, global_steps, rhythm_offline_source_cache
        if ref is None and rhythm_ref_conditioning is None and not (not infer and self.baseline_train_mode == "pretrain"):
            return content_embed, tgt_nonpadding, f0, uv
        if self.minimal_v1_profile and isinstance(rhythm_runtime_overrides, dict):
            legacy_override_keys = [
                key for key in rhythm_runtime_overrides.keys()
                if any(token in str(key).lower() for token in ("phase", "pause", "debt"))
            ]
            if legacy_override_keys:
                raise ValueError(
                    "rhythm_v3_minimal_v1_profile forbids legacy phase/pause/debt runtime overrides: "
                    + ", ".join(sorted(str(key) for key in legacy_override_keys)),
                )
        if content_embed.device != content.device:
            raise ValueError(
                f"content/content_embed device mismatch: content={content.device}, content_embed={content_embed.device}"
            )
        if tgt_nonpadding.device != content.device:
            raise ValueError(
                f"content/tgt_nonpadding device mismatch: content={content.device}, tgt_nonpadding={tgt_nonpadding.device}"
            )
        rhythm_state_prev = move_duration_runtime_state(rhythm_state, device=content.device)
        if rhythm_state_prev is not None:
            rhythm_state_prev = ensure_duration_runtime_state_batch(
                rhythm_state_prev,
                batch_size=int(content.size(0)),
            )
        source_batch, rhythm_state_prev = self._build_source_batch(
            content=content,
            content_lengths=content_lengths,
            rhythm_source_cache=rhythm_source_cache,
            infer=infer,
            state=rhythm_state_prev,
            return_state=True,
        )
        source_batch = move_source_unit_batch(source_batch, device=content.device)
        self._validate_training_reference_semantics(
            infer=bool(infer),
            ref_conditioning=rhythm_ref_conditioning,
            ref=ref,
        )
        self._assert_minimal_ref_conditioning_contract(rhythm_ref_conditioning)
        if ref is None and rhythm_ref_conditioning is None and not infer and self.baseline_train_mode == "pretrain":
            rhythm_ref_conditioning = {
                "global_rate": torch.zeros((content.size(0), 1), device=content.device, dtype=torch.float32),
            }
        rhythm_ref_conditioning = self._prepare_prompt_unit_conditioning(
            ref_conditioning=rhythm_ref_conditioning,
            device=content.device,
        )
        self._assert_minimal_ref_conditioning_contract(rhythm_ref_conditioning)
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
        if rhythm_state_prev is not None:
            execution.next_state = replace(
                execution.next_state,
                frontend_state=getattr(rhythm_state_prev, "frontend_state", None),
                consumed_content_steps=(
                    None
                    if getattr(rhythm_state_prev, "consumed_content_steps", None) is None
                    else rhythm_state_prev.consumed_content_steps.detach().clone()
                ),
            )
        self._attach_runtime_outputs(
            ret=ret,
            source_batch=source_batch,
            ref_memory=ref_memory,
            ref_conditioning_meta=rhythm_ref_conditioning,
            execution=execution,
            rhythm_state_prev=rhythm_state_prev,
        )
        prompt_domain_valid_tensor = None
        domain_invalid_mask = None
        domain_invalid_any = False
        prompt_domain_valid = getattr(ref_memory, "prompt_g_domain_valid", None)
        if isinstance(prompt_domain_valid, torch.Tensor):
            prompt_domain_valid_tensor = prompt_domain_valid.float().reshape(
                int(prompt_domain_valid.size(0)),
                -1,
            )[:, :1]
            domain_invalid_mask = prompt_domain_valid_tensor <= 0.5
            domain_invalid_any = bool(domain_invalid_mask.any().item())
            ret["rhythm_prompt_domain_valid"] = prompt_domain_valid_tensor.detach()
            ret["rhythm_domain_invalid"] = domain_invalid_mask.float().detach()
        ret["rhythm_domain_invalid_any"] = float(domain_invalid_any)
        apply_rhythm_render = resolve_rhythm_apply_mode(
            self.hparams,
            infer=infer,
            override=rhythm_apply_override,
        )
        if bool(apply_rhythm_render) and domain_invalid_any:
            ret["rhythm_apply_render"] = 0.0
            ret["rhythm_render_skipped_invalid_prompt"] = 1.0
            return content_embed, tgt_nonpadding, f0, uv
        has_uncommitted_tail = False
        if isinstance(getattr(execution, "commit_mask", None), torch.Tensor):
            has_uncommitted_tail = bool(
                (((source_batch.unit_mask.float() > 0.5) & (execution.commit_mask.float() <= 0.5))).any().item()
            )
        ret["rhythm_v3_has_uncommitted_tail"] = float(has_uncommitted_tail)
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
        rendered_tail_frames = 0
        if rendered_tail is not None:
            rendered_tail_frames = int(rendered_tail.total_mask.float().sum().item())
        ret["rhythm_v3_render_open_tail_frame_count"] = float(rendered_tail_frames)
        ret["rhythm_v3_render_frame_plan_contains_uncommitted_tail"] = float(rendered_tail_frames > 0)
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

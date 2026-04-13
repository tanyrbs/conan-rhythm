from __future__ import annotations

import torch
import torch.nn as nn

from .g_stats import normalize_falsification_eval_mode
from .math_utils import apply_analytic_gap_clip, build_causal_local_rate_seq
from .silence_surface import build_silence_tau_surface_meta
from .summary_memory import CausalUnitRunEncoder


class MinimalStreamingDurationHeadV1G(nn.Module):
    """Hard-bounded V1 head: scalar coarse + speech-only local residual."""

    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        num_slots: int,
        spk_dim: int | None = None,
        simple_global_stats: bool = True,
        use_log_base_rate: bool = False,
        use_learned_residual_gate: bool = False,
        max_logstretch: float = 1.2,
        max_silence_logstretch: float = 0.35,
        local_cold_start_runs: int = 2,
        local_short_run_min_duration: float = 2.0,
        local_rate_decay: float = 0.95,
        analytic_gap_clip: float = 0.35,
        short_gap_silence_scale: float = 0.35,
        leading_silence_scale: float = 0.0,
        eval_mode: str = "learned",
        disable_local_residual: bool = False,
        disable_coarse_bias: bool = False,
        detach_global_term_in_local_head: bool = False,
        codebook=None,
    ) -> None:
        super().__init__()
        del num_slots, codebook
        if not bool(simple_global_stats):
            raise ValueError("MinimalStreamingDurationHeadV1G requires simple_global_stats=true.")
        if bool(use_log_base_rate):
            raise ValueError("MinimalStreamingDurationHeadV1G requires use_log_base_rate=false.")
        if bool(use_learned_residual_gate):
            raise ValueError("MinimalStreamingDurationHeadV1G forbids learned residual gates.")
        if abs(float(short_gap_silence_scale) - 0.35) > 1.0e-6:
            raise ValueError(
                "MinimalStreamingDurationHeadV1G uses a constant silence clip; "
                "short_gap_silence_scale is not part of the minimal runtime surface."
            )
        if abs(float(leading_silence_scale) - 0.0) > 1.0e-6:
            raise ValueError(
                "MinimalStreamingDurationHeadV1G uses a constant silence clip; "
                "leading_silence_scale is not part of the minimal runtime surface."
            )
        self.rate_mode = "simple_global"
        self.simple_global_stats = True
        self.use_log_base_rate = False
        self.use_learned_residual_gate = False
        self.max_logstretch = float(max(0.1, max_logstretch))
        self.max_silence_logstretch = float(max(0.01, min(self.max_logstretch, max_silence_logstretch)))
        self.local_cold_start_runs = int(max(0, local_cold_start_runs))
        self.local_short_run_min_duration = float(max(1.0, local_short_run_min_duration))
        self.local_rate_decay = float(max(0.0, min(0.999, local_rate_decay)))
        self.analytic_gap_clip = float(max(0.0, analytic_gap_clip))
        self.eval_mode = normalize_falsification_eval_mode(eval_mode)
        self.disable_local_residual = bool(disable_local_residual)
        self.disable_coarse_bias = bool(disable_coarse_bias)
        self.detach_global_term_in_local_head = bool(detach_global_term_in_local_head)
        self.query_dim = int(max(8, dim))
        self.query_encoder = CausalUnitRunEncoder(vocab_size=vocab_size, dim=self.query_dim)
        self.spk_dim = int(max(8, spk_dim if spk_dim is not None else dim))
        self.spk_proj = nn.Linear(self.spk_dim, self.query_dim)
        self.coarse_head = nn.Sequential(
            nn.Linear(self.query_dim + 1, self.query_dim),
            nn.GELU(),
            nn.Linear(self.query_dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear((self.query_dim * 2) + 1, self.query_dim),
            nn.GELU(),
            nn.Linear(self.query_dim, 1),
        )
        self.coarse_delta_scale = 0.20
        self.local_residual_scale = 0.35
        self.src_rate_init = nn.Parameter(torch.zeros((1,)))

    @staticmethod
    def _resolve_scalar_column(value: torch.Tensor, *, batch_size: int) -> torch.Tensor:
        scalar = value.float()
        if scalar.dim() == 0:
            scalar = scalar.reshape(1, 1).expand(batch_size, 1)
        elif scalar.dim() == 1:
            scalar = scalar.unsqueeze(-1)
        elif scalar.dim() > 2:
            scalar = scalar.reshape(batch_size, -1)
        if scalar.size(0) != batch_size:
            raise ValueError(
                f"MinimalStreamingDurationHeadV1G.global_rate batch mismatch: got {tuple(scalar.shape)} "
                f"for batch_size={batch_size}."
            )
        if scalar.size(1) <= 0:
            raise ValueError("MinimalStreamingDurationHeadV1G.global_rate must contain at least one scalar.")
        return scalar[:, :1]

    def forward(
        self,
        *,
        content_units: torch.Tensor,
        log_anchor: torch.Tensor,
        log_base: torch.Tensor | None = None,
        unit_mask: torch.Tensor,
        sealed_mask: torch.Tensor,
        sep_hint: torch.Tensor,
        edge_cue: torch.Tensor,
        phrase_final_mask: torch.Tensor | None = None,
        global_rate: torch.Tensor,
        summary_state: torch.Tensor | None = None,
        spk_embed: torch.Tensor | None = None,
        role_value: torch.Tensor | None = None,
        role_var: torch.Tensor | None = None,
        role_coverage: torch.Tensor | None = None,
        local_rate_ema: torch.Tensor,
        silence_mask: torch.Tensor | None = None,
        run_stability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | str]:
        if log_base is not None:
            raise ValueError("MinimalStreamingDurationHeadV1G requires log_base=None.")
        if any(isinstance(value, torch.Tensor) for value in (summary_state, role_value, role_var, role_coverage)):
            raise ValueError("MinimalStreamingDurationHeadV1G forbids summary/role conditioning inputs.")
        if not isinstance(silence_mask, torch.Tensor):
            raise ValueError("MinimalStreamingDurationHeadV1G requires explicit silence_mask.")
        mask = unit_mask.float().clamp(0.0, 1.0)
        sealed = sealed_mask.float().clamp(0.0, 1.0)
        silence = silence_mask.float().clamp(0.0, 1.0) * mask
        runtime_stability = (
            run_stability.float().clamp(0.0, 1.0) * mask
            if isinstance(run_stability, torch.Tensor)
            else torch.ones_like(mask)
        )
        commit_valid_mask = mask * sealed
        silence_commit_mask = commit_valid_mask * silence
        speech_mask = commit_valid_mask * (1.0 - silence)

        init_local_rate = None
        if isinstance(local_rate_ema, torch.Tensor):
            init_local_rate = local_rate_ema.float()
            if init_local_rate.dim() != 2 or init_local_rate.size(1) != 1:
                raise ValueError(
                    f"MinimalStreamingDurationHeadV1G.local_rate_ema must have shape [B, 1], got {tuple(init_local_rate.shape)}"
                )

        local_rate_seq, local_rate_final = build_causal_local_rate_seq(
            observed_log=log_anchor.float(),
            speech_mask=speech_mask,
            init_rate=init_local_rate,
            default_init_rate=self.src_rate_init,
            decay=self.local_rate_decay,
        )
        query = self.query_encoder(
            unit_ids=content_units.long(),
            log_anchor=log_anchor.float(),
            log_base=None,
            use_log_base_rate=False,
            source_rate=local_rate_seq.float(),
            silence_mask=silence,
            sep_hint=sep_hint.float(),
            edge_cue=edge_cue.float(),
            phrase_final_mask=phrase_final_mask,
            unit_mask=mask,
        ) * mask.unsqueeze(-1)

        batch_size = int(query.size(0))
        g_ref_col = self._resolve_scalar_column(global_rate, batch_size=batch_size)
        if isinstance(spk_embed, torch.Tensor):
            spk = spk_embed.float()
            if spk.dim() == 3 and spk.size(-1) == 1:
                spk = spk.squeeze(-1)
            elif spk.dim() == 3 and spk.size(1) == 1:
                spk = spk.squeeze(1)
            if spk.dim() != 2:
                raise ValueError(
                    f"MinimalStreamingDurationHeadV1G.spk_embed must have shape [B, H], got {tuple(spk.shape)}"
                )
            if spk.size(-1) != self.spk_dim:
                if spk.size(-1) > self.spk_dim:
                    spk = spk[:, : self.spk_dim]
                else:
                    pad = spk.new_zeros((spk.size(0), self.spk_dim - spk.size(-1)))
                    spk = torch.cat([spk, pad], dim=-1)
            spk_ctx = torch.tanh(self.spk_proj(spk))
        else:
            spk_ctx = query.new_zeros((batch_size, self.query_dim))

        analytic_gap = apply_analytic_gap_clip(
            g_ref_col - local_rate_seq.float(),
            self.analytic_gap_clip,
        )
        global_shift_analytic = analytic_gap * commit_valid_mask
        coarse_context = torch.cat([spk_ctx, g_ref_col], dim=-1)
        coarse_scalar = self.coarse_delta_scale * torch.tanh(self.coarse_head(coarse_context).squeeze(-1))
        predicted_coarse = coarse_scalar.unsqueeze(1).expand_as(global_shift_analytic) * commit_valid_mask
        coarse_correction = predicted_coarse
        if self.eval_mode == "analytic" or self.disable_coarse_bias:
            coarse_correction = torch.zeros_like(predicted_coarse)
        global_term = (global_shift_analytic + coarse_correction) * commit_valid_mask

        residual_global_term = (
            global_term.detach() if self.detach_global_term_in_local_head else global_term
        )
        residual_input = torch.cat(
            [
                query,
                spk_ctx.unsqueeze(1).expand(-1, query.size(1), -1),
                residual_global_term.unsqueeze(-1),
            ],
            dim=-1,
        )
        residual = self.local_residual_scale * torch.tanh(self.residual_head(residual_input).squeeze(-1))
        prefix_speech_prev = (torch.cumsum(speech_mask, dim=1) - speech_mask).clamp_min(0.0)
        if self.local_cold_start_runs > 0:
            cold_gate = (prefix_speech_prev / float(self.local_cold_start_runs)).clamp(0.0, 1.0)
        else:
            cold_gate = torch.ones_like(prefix_speech_prev)
        min_duration = max(1.0, float(self.local_short_run_min_duration))
        short_gate = (
            (torch.exp(log_anchor.float()).clamp_min(1.0) - 1.0)
            / max(1.0, min_duration - 1.0)
        ).clamp(0.0, 1.0)
        residual_gate = cold_gate * short_gate * runtime_stability * speech_mask
        residual = residual * residual_gate
        predicted_residual = residual
        if self.eval_mode in {"analytic", "coarse_only"} or self.disable_local_residual:
            residual = torch.zeros_like(predicted_residual)

        pred_speech = (global_term + residual).clamp(
            min=-self.max_logstretch,
            max=self.max_logstretch,
        ) * speech_mask
        silence_surface = build_silence_tau_surface_meta(
            prediction_anchor=torch.exp(log_anchor.float()),
            committed_silence_mask=silence_commit_mask,
            sep_hint=sep_hint,
            boundary_cue=edge_cue,
            max_silence_logstretch=self.max_silence_logstretch,
            short_gap_scale=0.35,
            minimal_v1_profile=True,
        )
        silence_tau = silence_surface["silence_tau"]
        pred_silence = torch.clamp(global_term, min=-silence_tau, max=silence_tau)
        pred_silence = pred_silence * silence_commit_mask
        pred = pred_speech + pred_silence
        global_bias_scalar = coarse_scalar.reshape(-1, 1)
        analytic_term = global_shift_analytic * mask
        coarse_delta = coarse_correction * mask
        coarse_delta_pred = predicted_coarse * mask
        coarse_path = global_term * mask
        global_term_before_local = global_term * mask
        residual_used = residual * mask
        residual_pred = predicted_residual * mask
        residual_gate_mean = (
            (residual_gate * speech_mask).sum(dim=1, keepdim=True)
            / speech_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        leading_gate = torch.ones_like(mask) * commit_valid_mask

        return {
            "unit_logstretch": pred,
            "unit_global_shift": coarse_path,
            "unit_global_shift_analytic": analytic_term,
            "unit_analytic_gap": analytic_term,
            "unit_analytic_logstretch": analytic_term,
            "global_bias_scalar": global_bias_scalar,
            "unit_coarse_logstretch": coarse_path,
            "unit_coarse_path_logstretch": coarse_path,
            "unit_coarse_correction": coarse_delta,
            "unit_coarse_correction_used": coarse_delta,
            "unit_coarse_delta": coarse_delta,
            "unit_coarse_correction_pred": coarse_delta_pred,
            "unit_coarse_correction_predicted": coarse_delta_pred,
            "coarse_scalar_raw": coarse_scalar.reshape(-1, 1),
            "unit_global_term_before_local": global_term_before_local,
            "unit_residual_logstretch": residual_used,
            "unit_local_residual_used": residual_used,
            "unit_residual_logstretch_pred": residual_pred,
            "unit_residual_gate": residual_gate * mask,
            "residual_gate_mean": residual_gate_mean,
            "unit_residual_cold_gate": cold_gate * speech_mask,
            "unit_residual_short_gate": short_gate * speech_mask,
            "unit_residual_gate_stability": runtime_stability * speech_mask,
            "residual_gate_cold": cold_gate * speech_mask,
            "residual_gate_short": short_gate * speech_mask,
            "residual_gate_stability": runtime_stability * speech_mask,
            "unit_runtime_stability": runtime_stability * mask,
            "unit_silence_tau": silence_tau,
            "unit_silence_tau_surface_kind": silence_surface["silence_surface_kind"],
            "unit_boundary_shaping": silence_surface["silence_boundary_shaping"],
            "unit_leading_gate": leading_gate,
            "unit_leading_gate_mode": "disabled_constant_one",
            "unit_speech_pred": pred_speech,
            "unit_silence_pred": pred_silence,
            "runtime_surface_kind": "minimal_v1_constant_clip",
            "runtime_silence_tau_mode": "constant",
            "runtime_local_residual_mode": "speech_only",
            "role_attn_unit": mask.unsqueeze(-1),
            "role_value_unit": mask.new_zeros(mask.shape),
            "role_var_unit": mask.new_zeros(mask.shape),
            "role_conf_unit": mask.new_zeros(mask.shape),
            "role_query_unit": query,
            "local_response": residual_used,
            "local_response_pred": residual_pred,
            "local_rate_seq": local_rate_seq * mask,
            "local_rate_final": local_rate_final,
            "source_rate_seq": local_rate_seq * mask,
            "g_ref": g_ref_col.squeeze(-1),
            "g_src_prefix": local_rate_seq * mask,
            "detach_global_term_in_local_head": mask.new_full(
                (mask.size(0), 1),
                1.0 if self.detach_global_term_in_local_head else 0.0,
            ),
            "eval_mode": self.eval_mode,
            "falsification_eval_mode": mask.new_full((mask.size(0), 1), {"analytic": 0.0, "coarse_only": 1.0, "learned": 2.0}[self.eval_mode]),
        }


__all__ = ["MinimalStreamingDurationHeadV1G"]

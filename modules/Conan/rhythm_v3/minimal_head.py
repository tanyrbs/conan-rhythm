from __future__ import annotations

import torch
import torch.nn as nn

from .math_utils import build_causal_local_rate_seq
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
        short_gap_silence_scale: float = 0.35,
        leading_silence_scale: float = 0.0,
        codebook=None,
    ) -> None:
        super().__init__()
        del num_slots, short_gap_silence_scale, leading_silence_scale, codebook
        if not bool(simple_global_stats):
            raise ValueError("MinimalStreamingDurationHeadV1G requires simple_global_stats=true.")
        if bool(use_log_base_rate):
            raise ValueError("MinimalStreamingDurationHeadV1G requires use_log_base_rate=false.")
        if bool(use_learned_residual_gate):
            raise ValueError("MinimalStreamingDurationHeadV1G forbids learned residual gates.")
        self.rate_mode = "simple_global"
        self.simple_global_stats = True
        self.use_log_base_rate = False
        self.use_learned_residual_gate = False
        self.max_logstretch = float(max(0.1, max_logstretch))
        self.max_silence_logstretch = float(max(0.01, min(self.max_logstretch, max_silence_logstretch)))
        self.local_cold_start_runs = int(max(0, local_cold_start_runs))
        self.local_short_run_min_duration = float(max(1.0, local_short_run_min_duration))
        self.local_rate_decay = float(max(0.0, min(0.999, local_rate_decay)))
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
    ) -> dict[str, torch.Tensor]:
        del log_base
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
        global_rate_col = self._resolve_scalar_column(global_rate, batch_size=batch_size)
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

        global_shift_analytic = (global_rate_col - local_rate_seq.float()) * commit_valid_mask
        coarse_context = torch.cat([spk_ctx, global_rate_col], dim=-1)
        coarse_scalar = self.coarse_delta_scale * torch.tanh(self.coarse_head(coarse_context).squeeze(-1))
        coarse_correction = coarse_scalar.unsqueeze(1).expand_as(global_shift_analytic) * commit_valid_mask
        global_term = (global_shift_analytic + coarse_correction) * commit_valid_mask

        residual_input = torch.cat(
            [
                query,
                spk_ctx.unsqueeze(1).expand(-1, query.size(1), -1),
                global_term.unsqueeze(-1),
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

        pred_speech = (global_term + residual).clamp(
            min=-self.max_logstretch,
            max=self.max_logstretch,
        ) * speech_mask
        silence_tau = torch.full_like(global_term, self.max_silence_logstretch)
        pred_silence = torch.clamp(global_term, min=-self.max_silence_logstretch, max=self.max_silence_logstretch)
        pred_silence = pred_silence * silence_commit_mask
        pred = pred_speech + pred_silence
        global_bias_scalar = coarse_scalar.reshape(-1, 1)

        return {
            "unit_logstretch": pred,
            "unit_global_shift": global_term * mask,
            "unit_global_shift_analytic": global_shift_analytic * mask,
            "global_bias_scalar": global_bias_scalar,
            "unit_coarse_logstretch": global_term * mask,
            "unit_coarse_correction": coarse_correction * mask,
            "unit_residual_logstretch": residual * mask,
            "unit_residual_gate": residual_gate * mask,
            "unit_runtime_stability": runtime_stability * mask,
            "unit_silence_tau": silence_tau * silence_commit_mask,
            "role_attn_unit": mask.unsqueeze(-1),
            "role_value_unit": mask.new_zeros(mask.shape),
            "role_var_unit": mask.new_zeros(mask.shape),
            "role_conf_unit": mask.new_zeros(mask.shape),
            "role_query_unit": query,
            "local_response": residual * mask,
            "local_rate_seq": local_rate_seq * mask,
            "local_rate_final": local_rate_final,
            "source_rate_seq": local_rate_seq * mask,
        }


__all__ = ["MinimalStreamingDurationHeadV1G"]

from __future__ import annotations

from dataclasses import dataclass

import torch


_GROUP_LOSS_KEYS = {
    "exec": ("rhythm_exec_speech", "rhythm_exec_pause"),
    "state": ("rhythm_budget", "rhythm_prefix_state"),
    "plan": ("rhythm_plan",),
    "guidance": ("rhythm_guidance",),
    "distill": ("rhythm_distill",),
}

_GROUP_APPLY_KEYS = {
    "exec": ("rhythm_exec_speech", "rhythm_exec_pause"),
    "state": (
        "rhythm_budget",
        "rhythm_budget_raw_surface",
        "rhythm_budget_exec_surface",
        "rhythm_budget_total_surface",
        "rhythm_budget_pause_share_surface",
        "rhythm_feasible_debt",
        "rhythm_prefix_clock",
        "rhythm_prefix_backlog",
        "rhythm_prefix_state",
        "rhythm_cumplan",
        "rhythm_carry",
    ),
    "plan": ("rhythm_plan", "rhythm_plan_local", "rhythm_plan_cum"),
    "guidance": ("rhythm_guidance",),
    "distill": (
        "rhythm_distill",
        "rhythm_distill_student",
        "rhythm_distill_exec",
        "rhythm_distill_budget",
        "rhythm_distill_budget_raw_surface",
        "rhythm_distill_budget_exec_surface",
        "rhythm_distill_budget_total_surface",
        "rhythm_distill_budget_pause_share_surface",
        "rhythm_distill_prefix",
        "rhythm_distill_speech_shape",
        "rhythm_distill_pause_shape",
        "rhythm_distill_allocation",
    ),
}


@dataclass(frozen=True)
class AdaptiveRhythmLossBalancerConfig:
    mode: str = "none"
    beta: float = 0.98
    alpha: float = 0.50
    warmup_steps: int = 0
    min_scale: float = 0.50
    max_scale: float = 2.00
    eps: float = 1.0e-6


class AdaptiveRhythmLossBalancer:
    def __init__(self, config: AdaptiveRhythmLossBalancerConfig | None = None) -> None:
        self.config = config or AdaptiveRhythmLossBalancerConfig()
        self._ema_by_group: dict[str, float] = {}

    @classmethod
    def from_hparams(cls, hparams) -> "AdaptiveRhythmLossBalancer":
        return cls(
            AdaptiveRhythmLossBalancerConfig(
                mode=str(hparams.get("rhythm_loss_balance_mode", "none") or "none").strip().lower(),
                beta=float(hparams.get("rhythm_loss_balance_beta", 0.98)),
                alpha=float(hparams.get("rhythm_loss_balance_alpha", 0.50)),
                warmup_steps=int(hparams.get("rhythm_loss_balance_warmup_steps", 0) or 0),
                min_scale=float(hparams.get("rhythm_loss_balance_min_scale", 0.50)),
                max_scale=float(hparams.get("rhythm_loss_balance_max_scale", 2.00)),
                eps=float(hparams.get("rhythm_loss_balance_eps", 1.0e-6)),
            )
        )

    def _resolve_group_magnitude(self, losses: dict[str, torch.Tensor], group: str) -> tuple[torch.Tensor | None, float]:
        total = None
        for key in _GROUP_LOSS_KEYS[group]:
            value = losses.get(key)
            if not isinstance(value, torch.Tensor) or not value.requires_grad:
                continue
            total = value if total is None else total + value
        if total is None:
            return None, 0.0
        magnitude = float(total.detach().abs().item())
        return total, magnitude

    def apply(
        self,
        losses: dict[str, torch.Tensor],
        *,
        global_step: int,
        training: bool,
    ) -> dict[str, torch.Tensor]:
        if self.config.mode in {"", "none", "off", "disabled"}:
            return losses
        if self.config.mode != "ema_group":
            raise ValueError(f"Unsupported rhythm_loss_balance_mode: {self.config.mode}")
        if not training or int(global_step) < int(self.config.warmup_steps):
            return losses

        active_groups: dict[str, tuple[torch.Tensor, float]] = {}
        for group in _GROUP_LOSS_KEYS:
            total, magnitude = self._resolve_group_magnitude(losses, group)
            if total is None or magnitude <= float(self.config.eps):
                continue
            previous = self._ema_by_group.get(group, magnitude)
            ema = float(self.config.beta) * float(previous) + (1.0 - float(self.config.beta)) * magnitude
            self._ema_by_group[group] = ema
            active_groups[group] = (total, ema)

        if len(active_groups) <= 1:
            return losses

        target_magnitude = sum(ema for _, ema in active_groups.values()) / float(len(active_groups))
        raw_scales = {
            group: max(
                float(self.config.min_scale),
                min(
                    float(self.config.max_scale),
                    (target_magnitude / max(ema, float(self.config.eps))) ** float(self.config.alpha),
                ),
            )
            for group, (_, ema) in active_groups.items()
        }
        mean_scale = sum(raw_scales.values()) / float(len(raw_scales))
        scales = {
            group: max(
                float(self.config.min_scale),
                min(float(self.config.max_scale), scale / max(mean_scale, float(self.config.eps))),
            )
            for group, scale in raw_scales.items()
        }

        balanced = dict(losses)
        device = None
        for value in losses.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        for group, scale in scales.items():
            for key in _GROUP_APPLY_KEYS[group]:
                value = balanced.get(key)
                if isinstance(value, torch.Tensor):
                    balanced[key] = value * float(scale)
            if device is not None:
                balanced[f"rhythm_loss_balance_{group}_scale"] = torch.tensor(float(scale), device=device)
        if device is not None:
            balanced["rhythm_loss_balance_active_groups"] = torch.tensor(float(len(scales)), device=device)
        return balanced


__all__ = [
    "AdaptiveRhythmLossBalancer",
    "AdaptiveRhythmLossBalancerConfig",
]

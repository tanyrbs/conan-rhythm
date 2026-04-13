from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalUnitRunEncoder(nn.Module):
    """Shared causal run encoder used by both minimal and richer duration writers."""

    def __init__(
        self,
        *,
        vocab_size: int = 2048,
        dim: int = 64,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.dim = int(max(8, dim))
        self.kernel_size = int(max(2, kernel_size))
        self.dilations = tuple(max(1, int(value)) for value in dilations)
        self.unit_emb = nn.Embedding(int(vocab_size), self.dim)
        self.in_proj = nn.Linear(self.dim + 6, self.dim)
        self.dw = nn.ModuleList(
            [
                nn.Conv1d(
                    self.dim,
                    self.dim,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    groups=self.dim,
                )
                for dilation in self.dilations
            ]
        )
        self.pw = nn.ModuleList([nn.Conv1d(self.dim, self.dim, kernel_size=1) for _ in self.dilations])
        self.norm = nn.LayerNorm(self.dim)

    def forward(
        self,
        *,
        unit_ids: torch.Tensor,
        log_anchor: torch.Tensor,
        log_base: torch.Tensor | None,
        use_log_base_rate: bool = True,
        source_rate: torch.Tensor,
        silence_mask: torch.Tensor,
        sep_hint: torch.Tensor,
        edge_cue: torch.Tensor,
        phrase_final_mask: torch.Tensor | None,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = unit_mask.float().clamp(0.0, 1.0)
        base = (
            log_base.float()
            if (bool(use_log_base_rate) and isinstance(log_base, torch.Tensor))
            else torch.zeros_like(log_anchor.float())
        )
        normalized_anchor = (log_anchor.float() - base) * mask
        centered = (normalized_anchor - source_rate.float()) * mask
        silence = silence_mask.float().clamp(0.0, 1.0) * mask
        sep = sep_hint.float().clamp(0.0, 1.0) * mask
        edge = edge_cue.float().clamp(0.0, 1.0) * mask
        phrase_final = (
            phrase_final_mask.float().clamp(0.0, 1.0) * mask
            if isinstance(phrase_final_mask, torch.Tensor)
            else torch.zeros_like(mask)
        )
        hidden = torch.cat(
            [
                self.unit_emb(unit_ids.long()),
                log_anchor.float().unsqueeze(-1),
                centered.unsqueeze(-1),
                silence.unsqueeze(-1),
                sep.unsqueeze(-1),
                edge.unsqueeze(-1),
                phrase_final.unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden = self.in_proj(hidden) * mask.unsqueeze(-1)
        hidden_t = hidden.transpose(1, 2)
        for depthwise, pointwise, dilation in zip(self.dw, self.pw, self.dilations):
            padded = F.pad(hidden_t, (dilation * (self.kernel_size - 1), 0))
            update = pointwise(F.gelu(depthwise(padded)))
            hidden_t = hidden_t + update
        hidden = self.norm(hidden_t.transpose(1, 2))
        return hidden * mask.unsqueeze(-1)


__all__ = ["CausalUnitRunEncoder"]

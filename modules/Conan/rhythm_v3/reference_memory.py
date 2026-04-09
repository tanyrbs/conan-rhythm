from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.Conan.rhythm.reference_encoder import ReferenceRhythmEncoder

from .contracts import ReferenceDurationMemory


class PromptDurationMemoryBuilder(nn.Module):
    def __init__(
        self,
        *,
        trace_bins: int = 24,
        trace_dim: int = 5,
        stats_dim: int = 6,
        role_dim: int = 64,
        codebook_size: int = 12,
        coverage_floor: float = 0.05,
    ) -> None:
        super().__init__()
        self.reference_encoder = ReferenceRhythmEncoder(trace_bins=trace_bins)
        self.role_dim = int(role_dim)
        self.codebook_size = int(codebook_size)
        self.coverage_floor = float(max(0.0, min(0.5, coverage_floor)))
        self.role_proj = nn.Sequential(
            nn.Linear(trace_dim + stats_dim, role_dim),
            nn.SiLU(),
            nn.Linear(role_dim, role_dim),
        )
        self.role_codebook = nn.Parameter(torch.randn(self.codebook_size, self.role_dim) * 0.02)

    @staticmethod
    def _derive_prompt_rel_stretch(trace: torch.Tensor) -> torch.Tensor:
        local_rate = trace[..., 1]
        boundary = trace[..., 2]
        segment_bias = trace[..., 3]
        voiced = trace[..., 4]
        pause = trace[..., 0]
        stretch = 0.55 * segment_bias + 0.35 * boundary - 0.30 * local_rate + 0.10 * voiced - 0.15 * pause
        return stretch - stretch.mean(dim=1, keepdim=True)

    def _build_from_stats_trace(
        self,
        *,
        ref_rhythm_stats: torch.Tensor,
        ref_rhythm_trace: torch.Tensor,
    ) -> ReferenceDurationMemory:
        trace = ref_rhythm_trace.float()
        stats = ref_rhythm_stats.float()
        prompt_mask = torch.ones(trace.shape[:2], device=trace.device, dtype=trace.dtype)
        stats_feat = stats.unsqueeze(1).expand(-1, trace.size(1), -1)
        prompt_role_feat = self.role_proj(torch.cat([trace, stats_feat], dim=-1)) * prompt_mask.unsqueeze(-1)
        prompt_rel_stretch = self._derive_prompt_rel_stretch(trace) * prompt_mask
        role_keys = F.normalize(self.role_codebook, dim=-1)
        prompt_keys = F.normalize(prompt_role_feat, dim=-1)
        prompt_role_attention = F.softmax(torch.einsum("bnd,md->bnm", prompt_keys, role_keys), dim=-1)
        role_mass = prompt_role_attention.sum(dim=1)
        role_value = (prompt_role_attention * prompt_rel_stretch.unsqueeze(-1)).sum(dim=1) / role_mass.clamp_min(1.0e-6)
        role_coverage = role_mass / prompt_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        role_coverage = role_coverage.clamp_min(self.coverage_floor)
        prompt_reconstruction = (prompt_role_attention * role_value.unsqueeze(1)).sum(dim=-1) * prompt_mask
        global_rate = prompt_rel_stretch.sum(dim=1, keepdim=True) / prompt_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        role_summary = (role_coverage * role_value).sum(dim=1, keepdim=True)
        return ReferenceDurationMemory(
            global_rate=global_rate,
            role_keys=role_keys,
            role_value=role_value,
            role_coverage=role_coverage,
            prompt_role_feat=prompt_role_feat,
            prompt_rel_stretch=prompt_rel_stretch,
            prompt_mask=prompt_mask,
            prompt_role_attention=prompt_role_attention,
            prompt_reconstruction=prompt_reconstruction,
            role_summary=role_summary,
            raw_stats=stats,
            raw_trace=trace,
        )

    def encode_reference(
        self,
        ref_mel: torch.Tensor,
        *,
        ref_lengths: torch.Tensor | None = None,
    ) -> ReferenceDurationMemory:
        encoded = self.reference_encoder(ref_mel, ref_lengths=ref_lengths)
        return self._build_from_stats_trace(
            ref_rhythm_stats=encoded["ref_rhythm_stats"],
            ref_rhythm_trace=encoded["ref_rhythm_trace"],
        )

    def from_conditioning(self, ref_conditioning) -> ReferenceDurationMemory:
        if isinstance(ref_conditioning, ReferenceDurationMemory):
            return ref_conditioning
        if ref_conditioning is None:
            raise ValueError("Reference conditioning is required when no reference mel is provided.")
        if not isinstance(ref_conditioning, dict):
            raise TypeError(f"Unsupported reference conditioning type: {type(ref_conditioning)!r}")
        if "role_value" in ref_conditioning and "role_coverage" in ref_conditioning and "global_rate" in ref_conditioning:
            role_keys = F.normalize(ref_conditioning.get("role_keys", self.role_codebook).float(), dim=-1)
            return ReferenceDurationMemory(
                global_rate=ref_conditioning["global_rate"],
                role_keys=role_keys,
                role_value=ref_conditioning["role_value"],
                role_coverage=ref_conditioning["role_coverage"],
                prompt_role_feat=ref_conditioning.get("prompt_role_feat"),
                prompt_rel_stretch=ref_conditioning.get("prompt_rel_stretch"),
                prompt_mask=ref_conditioning.get("prompt_mask"),
                prompt_role_attention=ref_conditioning.get("prompt_role_attention"),
                prompt_reconstruction=ref_conditioning.get("prompt_reconstruction"),
                role_summary=ref_conditioning.get("role_summary"),
                raw_stats=ref_conditioning.get("ref_rhythm_stats"),
                raw_trace=ref_conditioning.get("ref_rhythm_trace"),
            )
        ref_rhythm_stats = ref_conditioning.get("ref_rhythm_stats")
        ref_rhythm_trace = ref_conditioning.get("ref_rhythm_trace")
        if ref_rhythm_stats is None or ref_rhythm_trace is None:
            raise ValueError("Reference conditioning must provide ref_rhythm_stats/ref_rhythm_trace or v3 role memory.")
        return self._build_from_stats_trace(
            ref_rhythm_stats=ref_rhythm_stats,
            ref_rhythm_trace=ref_rhythm_trace,
        )

    def forward(
        self,
        *,
        ref_conditioning=None,
        ref_mel: torch.Tensor | None = None,
        ref_lengths: torch.Tensor | None = None,
    ) -> ReferenceDurationMemory:
        if ref_conditioning is not None:
            return self.from_conditioning(ref_conditioning)
        if ref_mel is None:
            raise ValueError("Either ref_conditioning or ref_mel must be provided.")
        return self.encode_reference(ref_mel, ref_lengths=ref_lengths)

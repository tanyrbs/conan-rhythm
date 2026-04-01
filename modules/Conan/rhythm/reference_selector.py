from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ReferenceSelection:
    slow_rhythm_memory: torch.Tensor
    slow_rhythm_indices: torch.Tensor
    slow_rhythm_scores: torch.Tensor
    slow_rhythm_starts: torch.Tensor
    slow_rhythm_ends: torch.Tensor


class ReferenceSelector:
    """Select a small slow-rhythm memory bank from the low-rate trace.

    This is intentionally lightweight:
    - no heavy learned module yet
    - select a few prosodic cells from the explicit trace
    - expose scores / indices for debugging and future cache export
    """

    def __init__(
        self,
        *,
        slow_topk: int = 6,
        cell_size: int = 3,
        include_phrase_final: bool = True,
    ) -> None:
        self.slow_topk = max(1, int(slow_topk))
        self.cell_size = max(1, int(cell_size))
        self.include_phrase_final = bool(include_phrase_final)

    def _select_anchor_indices(self, score: torch.Tensor) -> torch.Tensor:
        trace_bins = int(score.size(0))
        topk = min(self.slow_topk, trace_bins)
        if topk <= 0:
            return score.new_zeros((0,), dtype=torch.long)
        anchors: list[int] = []
        if self.include_phrase_final and trace_bins > 0:
            anchors.append(trace_bins - 1)
        min_spacing = max(1, self.cell_size // 2)
        ranked = torch.argsort(score, descending=True)
        for idx in ranked.tolist():
            idx = int(idx)
            if any(abs(idx - kept) < min_spacing for kept in anchors):
                continue
            anchors.append(idx)
            if len(anchors) >= topk:
                break
        if len(anchors) < topk:
            for idx in range(trace_bins):
                if idx not in anchors:
                    anchors.append(idx)
                if len(anchors) >= topk:
                    break
        anchors = sorted(anchors[:topk])
        return torch.tensor(anchors, dtype=torch.long, device=score.device)

    def _build_cell_memory(
        self,
        trace: torch.Tensor,
        score: torch.Tensor,
        anchors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trace_bins, trace_dim = trace.shape
        if anchors.numel() <= 0:
            empty_long = torch.zeros((0,), dtype=torch.long, device=trace.device)
            return trace.new_zeros((0, trace_dim)), empty_long, empty_long
        radius = self.cell_size // 2
        memory = []
        starts = []
        ends = []
        for anchor in anchors.tolist():
            center = int(anchor)
            start = max(0, center - radius)
            end = min(trace_bins, start + self.cell_size)
            start = max(0, end - self.cell_size)
            cell = trace[start:end]
            cell_score = score[start:end].clamp_min(0.0)
            if cell.size(0) <= 0:
                pooled = trace.new_zeros((trace_dim,))
            else:
                weight = (1.0 + cell_score).unsqueeze(-1)
                pooled = (cell * weight).sum(dim=0) / weight.sum(dim=0).clamp_min(1e-6)
            memory.append(pooled)
            starts.append(start)
            ends.append(max(start, end - 1))
        return (
            torch.stack(memory, dim=0),
            torch.tensor(starts, dtype=torch.long, device=trace.device),
            torch.tensor(ends, dtype=torch.long, device=trace.device),
        )

    def __call__(self, ref_rhythm_trace: torch.Tensor) -> ReferenceSelection:
        if ref_rhythm_trace.dim() != 3:
            raise ValueError(f"ref_rhythm_trace must be [B,bins,dim], got {tuple(ref_rhythm_trace.shape)}")
        batch_size, trace_bins, trace_dim = ref_rhythm_trace.shape
        local_rate = ref_rhythm_trace[:, :, 1].abs()
        boundary = ref_rhythm_trace[:, :, 2].clamp_min(0.0)
        pause = ref_rhythm_trace[:, :, 0].clamp_min(0.0)
        voiced = ref_rhythm_trace[:, :, 4].clamp_min(0.0) if trace_dim > 4 else torch.ones_like(boundary)
        final_bias = torch.zeros_like(boundary)
        if trace_bins > 0 and self.include_phrase_final:
            final_bias[:, -1] = 1.0
        score = 0.28 * local_rate + 0.28 * boundary + 0.18 * pause + 0.16 * voiced + 0.10 * final_bias
        memory_rows = []
        index_rows = []
        score_rows = []
        start_rows = []
        end_rows = []
        topk = min(self.slow_topk, trace_bins)
        for batch_idx in range(batch_size):
            anchors = self._select_anchor_indices(score[batch_idx])
            pooled, starts, ends = self._build_cell_memory(
                ref_rhythm_trace[batch_idx],
                score[batch_idx],
                anchors,
            )
            values = score[batch_idx].gather(0, anchors)
            if anchors.numel() < topk:
                pad_count = topk - int(anchors.numel())
                pad_memory = pooled.new_zeros((pad_count, trace_dim))
                pad_index = anchors.new_full((pad_count,), max(trace_bins - 1, 0))
                pad_value = values.new_zeros((pad_count,))
                pad_start = starts.new_zeros((pad_count,))
                pad_end = ends.new_zeros((pad_count,))
                pooled = torch.cat([pooled, pad_memory], dim=0)
                anchors = torch.cat([anchors, pad_index], dim=0)
                values = torch.cat([values, pad_value], dim=0)
                starts = torch.cat([starts, pad_start], dim=0)
                ends = torch.cat([ends, pad_end], dim=0)
            memory_rows.append(pooled)
            index_rows.append(anchors)
            score_rows.append(values)
            start_rows.append(starts)
            end_rows.append(ends)
        memory = torch.stack(memory_rows, dim=0)
        indices = torch.stack(index_rows, dim=0)
        values = torch.stack(score_rows, dim=0)
        starts = torch.stack(start_rows, dim=0)
        ends = torch.stack(end_rows, dim=0)
        return ReferenceSelection(
            slow_rhythm_memory=memory,
            slow_rhythm_indices=indices,
            slow_rhythm_scores=values,
            slow_rhythm_starts=starts,
            slow_rhythm_ends=ends,
        )

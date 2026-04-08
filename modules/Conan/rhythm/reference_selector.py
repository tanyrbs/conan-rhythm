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
        phrase_select_window: int = 3,
        boundary_weight: float = 0.28,
        local_rate_weight: float = 0.28,
        pause_weight: float = 0.18,
        voiced_weight: float = 0.16,
        final_bias_weight: float = 0.10,
        monotonic_bias: float = 0.0,
        phrase_length_bias: float = 0.0,
    ) -> None:
        self.slow_topk = max(1, int(slow_topk))
        self.cell_size = max(1, int(cell_size))
        self.include_phrase_final = bool(include_phrase_final)
        self.phrase_select_window = max(1, int(phrase_select_window))
        self.boundary_weight = float(boundary_weight)
        self.local_rate_weight = float(local_rate_weight)
        self.pause_weight = float(pause_weight)
        self.voiced_weight = float(voiced_weight)
        self.final_bias_weight = float(final_bias_weight)
        self.monotonic_bias = float(monotonic_bias)
        self.phrase_length_bias = float(phrase_length_bias)

    @staticmethod
    def _gather_phrase_bank_rows(value: torch.Tensor | None, index: torch.Tensor) -> torch.Tensor | None:
        if value is None:
            return None
        if value.dim() < 2:
            raise ValueError(f"phrase-bank value must have rank >= 2, got {tuple(value.shape)}")
        gather_index = index.long().clamp_min(0).view(value.size(0), 1, *([1] * (value.dim() - 2)))
        expand_shape = [value.size(0), 1, *list(value.shape[2:])]
        gather_index = gather_index.expand(*expand_shape)
        gathered = value.gather(1, gather_index)
        return gathered.squeeze(1)

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
        length_axis = torch.linspace(0.0, 1.0, trace_bins, device=ref_rhythm_trace.device).unsqueeze(0)
        monotonic_axis = (
            torch.arange(trace_bins, dtype=torch.float32, device=ref_rhythm_trace.device)
            / float(max(trace_bins - 1, 1))
        ).unsqueeze(0)
        score = (
            self.local_rate_weight * local_rate
            + self.boundary_weight * boundary
            + self.pause_weight * pause
            + self.voiced_weight * voiced
            + self.final_bias_weight * final_bias
            + self.phrase_length_bias * length_axis
            - self.monotonic_bias * monotonic_axis
        )
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

    @staticmethod
    def select_monotonic_phrase(
        *,
        ref_phrase_trace: torch.Tensor,
        planner_ref_phrase_trace: torch.Tensor,
        ref_phrase_valid: torch.Tensor,
        ref_phrase_lengths: torch.Tensor,
        ref_phrase_starts: torch.Tensor,
        ref_phrase_ends: torch.Tensor,
        ref_phrase_boundary_strength: torch.Tensor,
        ref_phrase_stats: torch.Tensor | None,
        ref_phrase_ptr: torch.Tensor,
        query_chunk_summary: torch.Tensor | None = None,
        query_commit_confidence: torch.Tensor | None = None,
        query_phrase_close_prob: torch.Tensor | None = None,
        monotonic_window: int = 3,
        strict_pointer_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        valid_counts = ref_phrase_valid.long().sum(dim=1).clamp_min(1)
        base_index = ref_phrase_ptr.long().clamp_min(0)
        base_index = torch.minimum(base_index, valid_counts - 1)
        selected_index = base_index
        if not strict_pointer_only:
            window = max(1, int(monotonic_window))
            steps = torch.arange(ref_phrase_valid.size(1), device=ref_phrase_valid.device)[None, :]
            candidate_mask = (ref_phrase_valid.float() > 0.5) & (steps >= base_index[:, None])
            candidate_mask = candidate_mask & (steps < (base_index[:, None] + window))
            if query_phrase_close_prob is None:
                if query_commit_confidence is not None:
                    commit_conf = query_commit_confidence.float().reshape(base_index.size(0), 1)
                else:
                    commit_conf = torch.ones((base_index.size(0), 1), device=ref_phrase_valid.device) * 0.5
            else:
                commit_conf = query_phrase_close_prob.float().reshape(base_index.size(0), 1)
            structure_progress = (
                query_chunk_summary.float()[:, 0:1]
                if query_chunk_summary is not None
                else torch.full_like(commit_conf, 0.5)
            )
            max_len = ref_phrase_lengths.float().amax(dim=1, keepdim=True).clamp_min(1.0)
            norm_len = ref_phrase_lengths.float() / max_len
            boundary_score = ref_phrase_boundary_strength.float()
            distance = (steps - base_index[:, None]).float()
            monotonic_bias = -0.15 * distance
            score = (
                0.40 * boundary_score
                + 0.25 * norm_len
                + 0.15 * commit_conf
                + 0.10 * structure_progress
                + monotonic_bias
            )
            score = score.masked_fill(~candidate_mask, float("-inf"))
            winner = torch.argmax(score, dim=1)
            has_candidate = candidate_mask.any(dim=1)
            selected_index = torch.where(has_candidate, winner.long(), base_index)
        selection = {
            "selected_ref_phrase_index": selected_index,
            "selected_ref_phrase_trace": ReferenceSelector._gather_phrase_bank_rows(ref_phrase_trace, selected_index),
            "selected_planner_ref_phrase_trace": ReferenceSelector._gather_phrase_bank_rows(
                planner_ref_phrase_trace,
                selected_index,
            ),
            "selected_ref_phrase_valid": ReferenceSelector._gather_phrase_bank_rows(
                ref_phrase_valid.float(),
                selected_index,
            ),
            "selected_ref_phrase_length": ReferenceSelector._gather_phrase_bank_rows(
                ref_phrase_lengths.float(),
                selected_index,
            ),
            "selected_ref_phrase_start": ReferenceSelector._gather_phrase_bank_rows(
                ref_phrase_starts.float(),
                selected_index,
            ),
            "selected_ref_phrase_end": ReferenceSelector._gather_phrase_bank_rows(
                ref_phrase_ends.float(),
                selected_index,
            ),
            "selected_ref_phrase_boundary_strength": ReferenceSelector._gather_phrase_bank_rows(
                ref_phrase_boundary_strength.float(),
                selected_index,
            ),
            "selected_ref_phrase_stats": ReferenceSelector._gather_phrase_bank_rows(
                ref_phrase_stats,
                selected_index,
            ),
        }
        selection.update(ReferenceSelector.build_phrase_prototype(selection))
        return selection

    @staticmethod
    def build_phrase_prototype(selection: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        trace = selection.get("selected_planner_ref_phrase_trace")
        if trace is None:
            trace = selection.get("selected_ref_phrase_trace")
        trace_summary = None
        if trace is not None:
            if trace.dim() == 3:
                trace_summary = trace.float().mean(dim=1)
            elif trace.dim() == 2:
                trace_summary = trace.float()
            else:
                raise ValueError(
                    "selected phrase trace must have rank 2 or 3, "
                    f"got {tuple(trace.shape)}"
                )
        stats = selection.get("selected_ref_phrase_stats")
        valid = selection.get("selected_ref_phrase_valid")
        if valid is not None:
            valid = valid.float().reshape(valid.size(0), -1)[:, :1]
        boundary_strength = selection.get("selected_ref_phrase_boundary_strength")
        if boundary_strength is not None:
            boundary_strength = boundary_strength.float().reshape(boundary_strength.size(0), -1)[:, :1]
        prototype = {}
        if trace_summary is not None:
            prototype["selected_phrase_prototype_summary"] = trace_summary
        if stats is not None:
            stats = stats.float()
            if stats.dim() == 2 and stats.size(-1) > 2 and trace_summary is not None and trace_summary.size(-1) >= 2:
                prototype["selected_phrase_prototype_stats"] = trace_summary[:, :2]
            else:
                prototype["selected_phrase_prototype_stats"] = stats
        if valid is not None:
            prototype["selected_phrase_prototype_valid"] = valid
        if boundary_strength is not None:
            prototype["selected_phrase_prototype_boundary_strength"] = boundary_strength
        return prototype

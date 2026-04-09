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

    @staticmethod
    def _mix_phrase_bank_rows(
        value: torch.Tensor | None,
        *,
        center_index: torch.Tensor,
        valid_mask: torch.Tensor,
        mix_alpha: float,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        alpha = float(max(0.0, min(0.49, mix_alpha)))
        center = ReferenceSelector._gather_phrase_bank_rows(value, center_index)
        if center is None or alpha <= 0.0:
            return center
        prev_index = (center_index.long() - 1).clamp_min(0)
        next_index = torch.minimum(center_index.long() + 1, valid_mask.long().sum(dim=1).clamp_min(1) - 1)
        prev_valid = (center_index.long() > 0).float().reshape(center_index.size(0), *([1] * (center.dim() - 1)))
        next_valid = (
            center_index.long() < (valid_mask.long().sum(dim=1).clamp_min(1) - 1)
        ).float().reshape(center_index.size(0), *([1] * (center.dim() - 1)))
        prev_value = ReferenceSelector._gather_phrase_bank_rows(value, prev_index)
        next_value = ReferenceSelector._gather_phrase_bank_rows(value, next_index)
        prev_weight = alpha * prev_valid
        next_weight = alpha * next_valid
        center_weight = 1.0 - (prev_weight + next_weight)
        mixed = center.float() * center_weight
        if prev_value is not None:
            mixed = mixed + prev_value.float() * prev_weight
        if next_value is not None:
            mixed = mixed + next_value.float() * next_weight
        return mixed

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
    def _compose_runtime_phrase_score(
        *,
        ref_phrase_trace: torch.Tensor,
        ref_phrase_stats: torch.Tensor | None,
        ref_phrase_lengths: torch.Tensor,
        ref_phrase_valid: torch.Tensor,
        ref_phrase_boundary_strength: torch.Tensor,
        base_index: torch.Tensor,
        query_chunk_summary: torch.Tensor | None,
        query_commit_confidence: torch.Tensor | None,
        query_phrase_close_prob: torch.Tensor | None,
        monotonic_window: int,
        boundary_weight: float,
        local_rate_weight: float,
        pause_weight: float,
        voiced_weight: float,
        final_bias_weight: float,
        monotonic_bias: float,
        phrase_length_bias: float,
    ) -> torch.Tensor:
        steps = torch.arange(ref_phrase_valid.size(1), device=ref_phrase_valid.device)[None, :]
        candidate_mask = (ref_phrase_valid.float() > 0.5) & (steps >= base_index[:, None])
        candidate_mask = candidate_mask & (steps < (base_index[:, None] + max(1, int(monotonic_window))))
        if ref_phrase_stats is None:
            if ref_phrase_trace.dim() == 4:
                phrase_stats = ref_phrase_trace.float().mean(dim=2)
            elif ref_phrase_trace.dim() == 3:
                phrase_stats = ref_phrase_trace.float()
            else:
                raise ValueError(
                    "ref_phrase_trace must be rank-3 [B,P,D] or rank-4 [B,P,T,D], "
                    f"got {tuple(ref_phrase_trace.shape)}"
                )
        else:
            phrase_stats = ref_phrase_stats.float()
        zeros = torch.zeros_like(ref_phrase_boundary_strength.float())
        pause_score = phrase_stats[:, :, 0].clamp(0.0, 1.0) if phrase_stats.size(-1) > 0 else zeros
        local_rate_score = phrase_stats[:, :, 1].abs() if phrase_stats.size(-1) > 1 else zeros
        voiced_score = phrase_stats[:, :, 4].clamp(0.0, 1.0) if phrase_stats.size(-1) > 4 else torch.ones_like(zeros)
        boundary_score = ref_phrase_boundary_strength.float().clamp(0.0, 1.0)
        max_len = ref_phrase_lengths.float().amax(dim=1, keepdim=True).clamp_min(1.0)
        norm_len = ref_phrase_lengths.float() / max_len
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
        valid_counts = ref_phrase_valid.long().sum(dim=1).clamp_min(1)
        final_index = (valid_counts - 1)[:, None]
        final_bias = steps.eq(final_index).float()
        distance = (steps - base_index[:, None]).float().clamp_min(0.0)
        distance_penalty = (0.15 + max(float(monotonic_bias), 0.0)) * distance
        score = (
            float(boundary_weight) * boundary_score
            + float(local_rate_weight) * local_rate_score
            + float(pause_weight) * pause_score
            + float(voiced_weight) * voiced_score
            + float(final_bias_weight) * final_bias
            + float(phrase_length_bias) * norm_len
            + 0.15 * commit_conf
            + 0.10 * structure_progress
            - distance_penalty
        )
        return score.masked_fill(~candidate_mask, float("-inf"))

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
        neighbor_mix_alpha: float = 0.0,
        boundary_weight: float = 0.40,
        local_rate_weight: float = 0.25,
        pause_weight: float = 0.10,
        voiced_weight: float = 0.05,
        final_bias_weight: float = 0.05,
        monotonic_bias: float = 0.0,
        phrase_length_bias: float = 0.25,
    ) -> dict[str, torch.Tensor]:
        valid_counts = ref_phrase_valid.long().sum(dim=1).clamp_min(1)
        base_index = ref_phrase_ptr.long().clamp_min(0)
        base_index = torch.minimum(base_index, valid_counts - 1)
        selected_index = base_index
        if not strict_pointer_only:
            score = ReferenceSelector._compose_runtime_phrase_score(
                ref_phrase_trace=ref_phrase_trace,
                ref_phrase_stats=ref_phrase_stats,
                ref_phrase_lengths=ref_phrase_lengths,
                ref_phrase_valid=ref_phrase_valid,
                ref_phrase_boundary_strength=ref_phrase_boundary_strength,
                base_index=base_index,
                query_chunk_summary=query_chunk_summary,
                query_commit_confidence=query_commit_confidence,
                query_phrase_close_prob=query_phrase_close_prob,
                monotonic_window=monotonic_window,
                boundary_weight=boundary_weight,
                local_rate_weight=local_rate_weight,
                pause_weight=pause_weight,
                voiced_weight=voiced_weight,
                final_bias_weight=final_bias_weight,
                monotonic_bias=monotonic_bias,
                phrase_length_bias=phrase_length_bias,
            )
            winner = torch.argmax(score, dim=1)
            has_candidate = torch.isfinite(score).any(dim=1)
            selected_index = torch.where(has_candidate, winner.long(), base_index)
        selected_ref_phrase_trace = ReferenceSelector._mix_phrase_bank_rows(
            ref_phrase_trace,
            center_index=selected_index,
            valid_mask=ref_phrase_valid,
            mix_alpha=neighbor_mix_alpha,
        )
        selected_planner_ref_phrase_trace = ReferenceSelector._mix_phrase_bank_rows(
            planner_ref_phrase_trace,
            center_index=selected_index,
            valid_mask=ref_phrase_valid,
            mix_alpha=neighbor_mix_alpha,
        )
        selected_ref_phrase_valid = ReferenceSelector._gather_phrase_bank_rows(
            ref_phrase_valid.float(),
            selected_index,
        )
        selected_ref_phrase_length = ReferenceSelector._mix_phrase_bank_rows(
            ref_phrase_lengths.float(),
            center_index=selected_index,
            valid_mask=ref_phrase_valid,
            mix_alpha=neighbor_mix_alpha,
        )
        selected_ref_phrase_start = ReferenceSelector._mix_phrase_bank_rows(
            ref_phrase_starts.float(),
            center_index=selected_index,
            valid_mask=ref_phrase_valid,
            mix_alpha=neighbor_mix_alpha,
        )
        selected_ref_phrase_end = ReferenceSelector._mix_phrase_bank_rows(
            ref_phrase_ends.float(),
            center_index=selected_index,
            valid_mask=ref_phrase_valid,
            mix_alpha=neighbor_mix_alpha,
        )
        selected_ref_phrase_boundary_strength = ReferenceSelector._mix_phrase_bank_rows(
            ref_phrase_boundary_strength.float(),
            center_index=selected_index,
            valid_mask=ref_phrase_valid,
            mix_alpha=neighbor_mix_alpha,
        )
        selected_ref_phrase_stats = ReferenceSelector._mix_phrase_bank_rows(
            ref_phrase_stats,
            center_index=selected_index,
            valid_mask=ref_phrase_valid,
            mix_alpha=neighbor_mix_alpha,
        )
        next_index = torch.minimum(selected_index.long() + 1, valid_counts - 1)
        next_ref_phrase_trace = ReferenceSelector._gather_phrase_bank_rows(
            ref_phrase_trace,
            next_index,
        )
        next_planner_ref_phrase_trace = ReferenceSelector._gather_phrase_bank_rows(
            planner_ref_phrase_trace,
            next_index,
        )
        next_ref_phrase_stats = ReferenceSelector._gather_phrase_bank_rows(
            ref_phrase_stats,
            next_index,
        )
        selection = {
            "selected_ref_phrase_index": selected_index,
            "selected_ref_phrase_trace": selected_ref_phrase_trace,
            "selected_planner_ref_phrase_trace": selected_planner_ref_phrase_trace,
            "selected_ref_phrase_valid": selected_ref_phrase_valid,
            "selected_ref_phrase_length": selected_ref_phrase_length,
            "selected_ref_phrase_start": selected_ref_phrase_start,
            "selected_ref_phrase_end": selected_ref_phrase_end,
            "selected_ref_phrase_boundary_strength": selected_ref_phrase_boundary_strength,
            "selected_ref_phrase_stats": selected_ref_phrase_stats,
            "next_ref_phrase_index": next_index,
            "next_ref_phrase_trace": next_ref_phrase_trace,
            "next_planner_ref_phrase_trace": next_planner_ref_phrase_trace,
            "next_ref_phrase_stats": next_ref_phrase_stats,
        }
        selection.update(ReferenceSelector.build_phrase_prototype(selection))
        return selection

    def select_monotonic_phrase_bank(
        self,
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
        monotonic_window: int | None = None,
        strict_pointer_only: bool = False,
        neighbor_mix_alpha: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        return self.select_monotonic_phrase(
            ref_phrase_trace=ref_phrase_trace,
            planner_ref_phrase_trace=planner_ref_phrase_trace,
            ref_phrase_valid=ref_phrase_valid,
            ref_phrase_lengths=ref_phrase_lengths,
            ref_phrase_starts=ref_phrase_starts,
            ref_phrase_ends=ref_phrase_ends,
            ref_phrase_boundary_strength=ref_phrase_boundary_strength,
            ref_phrase_stats=ref_phrase_stats,
            ref_phrase_ptr=ref_phrase_ptr,
            query_chunk_summary=query_chunk_summary,
            query_commit_confidence=query_commit_confidence,
            query_phrase_close_prob=query_phrase_close_prob,
            monotonic_window=self.phrase_select_window if monotonic_window is None else monotonic_window,
            strict_pointer_only=strict_pointer_only,
            neighbor_mix_alpha=neighbor_mix_alpha,
            boundary_weight=self.boundary_weight,
            local_rate_weight=self.local_rate_weight,
            pause_weight=self.pause_weight,
            voiced_weight=self.voiced_weight,
            final_bias_weight=self.final_bias_weight,
            monotonic_bias=self.monotonic_bias,
            phrase_length_bias=self.phrase_length_bias,
        )

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

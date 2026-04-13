from __future__ import annotations

import numpy as np


CONTINUOUS_ALIGNER_VERSION = "2026-04-13"
_ALIGNMENT_MODE_DISCRETE = "discrete"
_ALIGNMENT_MODE_CONTINUOUS_PRECOMPUTED = "continuous_precomputed"
_ALIGNMENT_MODE_CONTINUOUS_VITERBI_V1 = "continuous_viterbi_v1"


def as_int64_1d(value) -> np.ndarray:
    return np.asarray(value, dtype=np.int64).reshape(-1)


def as_float32_1d(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def as_float32_2d(value, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape={arr.shape}")
    return arr


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1.0e-6, None)


def _build_progress_from_anchor(durations: np.ndarray) -> np.ndarray:
    dur = as_float32_1d(durations)
    if dur.size <= 1:
        return np.zeros((dur.size,), dtype=np.float32)
    dur = np.clip(dur, 1.0e-4, None)
    cdf = np.cumsum(dur, dtype=np.float64)
    total = max(float(cdf[-1]), 1.0e-6)
    progress = (cdf / total).astype(np.float32, copy=False)
    progress[-1] = np.float32(1.0)
    return progress


def _build_uniform_progress(size: int) -> np.ndarray:
    if int(size) <= 0:
        return np.zeros((0,), dtype=np.float32)
    if int(size) == 1:
        return np.zeros((1,), dtype=np.float32)
    return np.linspace(0.0, 1.0, int(size), dtype=np.float32)


def _build_progress_from_weights(weights: np.ndarray) -> np.ndarray:
    w = as_float32_1d(weights)
    if w.size <= 1:
        return np.zeros((w.size,), dtype=np.float32)
    w = np.clip(w, 1.0e-4, None)
    cdf = np.cumsum(w, dtype=np.float64)
    total = max(float(cdf[-1]), 1.0e-6)
    progress = (cdf / total).astype(np.float32, copy=False)
    progress[-1] = np.float32(1.0)
    return progress


def _nearest_progress_index(progress: np.ndarray, value: float) -> int:
    if progress.size <= 1:
        return 0
    idx = int(np.searchsorted(progress, np.float32(value), side="left"))
    if idx <= 0:
        return 0
    if idx >= progress.size:
        return int(progress.size - 1)
    prev_value = float(progress[idx - 1])
    next_value = float(progress[idx])
    if abs(value - prev_value) <= abs(next_value - value):
        return int(idx - 1)
    return int(idx)


def _resolve_optional_run_signal(
    value,
    *,
    num_source_runs: int,
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    arr = as_float32_1d(value)
    if arr.shape[0] != int(num_source_runs):
        raise ValueError(f"{name} length mismatch: expected {int(num_source_runs)}, got {int(arr.shape[0])}")
    return arr.astype(np.float32, copy=False)


def _normalize_continuous_alignment_mode(
    value,
    *,
    precomputed_alignment=None,
) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "auto"}:
        return (
            _ALIGNMENT_MODE_CONTINUOUS_PRECOMPUTED
            if precomputed_alignment is not None
            else _ALIGNMENT_MODE_CONTINUOUS_VITERBI_V1
        )
    aliases = {
        "precomputed": _ALIGNMENT_MODE_CONTINUOUS_PRECOMPUTED,
        "continuous": _ALIGNMENT_MODE_CONTINUOUS_VITERBI_V1,
        "viterbi": _ALIGNMENT_MODE_CONTINUOUS_VITERBI_V1,
        "run_state_viterbi": _ALIGNMENT_MODE_CONTINUOUS_VITERBI_V1,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {
        _ALIGNMENT_MODE_DISCRETE,
        _ALIGNMENT_MODE_CONTINUOUS_PRECOMPUTED,
        _ALIGNMENT_MODE_CONTINUOUS_VITERBI_V1,
    }:
        raise ValueError(
            "Unsupported continuous alignment mode. "
            f"Expected one of: {_ALIGNMENT_MODE_DISCRETE}, "
            f"{_ALIGNMENT_MODE_CONTINUOUS_PRECOMPUTED}, "
            f"{_ALIGNMENT_MODE_CONTINUOUS_VITERBI_V1}; got {value!r}"
        )
    return normalized


def resolve_precomputed_alignment(
    *,
    precomputed_alignment=None,
    num_target: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if precomputed_alignment is None:
        return None
    if isinstance(precomputed_alignment, tuple) and len(precomputed_alignment) == 2:
        assigned_source, assigned_cost = precomputed_alignment
    elif isinstance(precomputed_alignment, dict):
        assigned_source = precomputed_alignment.get("assigned_source")
        assigned_cost = precomputed_alignment.get("assigned_cost")
    else:
        raise TypeError(
            "precomputed_alignment must be a dict with assigned_source/assigned_cost "
            "or a 2-tuple of arrays."
        )
    if assigned_source is None:
        return None
    assigned_source = as_int64_1d(assigned_source)
    if assigned_source.shape[0] != int(num_target):
        raise ValueError(
            "precomputed alignment assigned_source length mismatch: "
            f"expected {int(num_target)}, got {int(assigned_source.shape[0])}"
        )
    if assigned_cost is None:
        assigned_cost = np.zeros((int(num_target),), dtype=np.float32)
    else:
        assigned_cost = as_float32_1d(assigned_cost)
        if assigned_cost.shape[0] != int(num_target):
            raise ValueError(
                "precomputed alignment assigned_cost length mismatch: "
                f"expected {int(num_target)}, got {int(assigned_cost.shape[0])}"
            )
    return assigned_source, assigned_cost


def resolve_run_silence_mask(*, size: int, silence_mask=None) -> np.ndarray:
    if silence_mask is None:
        return np.zeros((size,), dtype=np.float32)
    silence = as_float32_1d(silence_mask)
    if silence.shape[0] == size:
        return silence
    out = np.zeros((size,), dtype=np.float32)
    limit = min(size, silence.shape[0])
    out[:limit] = silence[:limit]
    return out


def filter_valid_runs(
    *,
    units: np.ndarray,
    durations: np.ndarray,
    silence_mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keep = np.asarray(durations, dtype=np.float32).reshape(-1) > 0.0
    if valid_mask is not None:
        keep = keep & (as_float32_1d(valid_mask) > 0.5)
    indices = np.nonzero(keep)[0].astype(np.int64)
    return (
        as_int64_1d(units)[indices],
        as_float32_1d(durations)[indices],
        as_float32_1d(silence_mask)[indices],
        indices,
    )


class ContinuousRunAligner:
    def __init__(
        self,
        *,
        lambda_emb: float = 1.0,
        lambda_type: float = 0.5,
        lambda_band: float = 0.2,
        lambda_unit: float = 0.0,
        band_width: int | None = None,
        band_ratio: float = 0.08,
        bad_cost_threshold: float = 1.2,
        allow_source_skip: bool = False,
        skip_penalty: float = 1.0,
    ) -> None:
        self.lambda_emb = float(lambda_emb)
        self.lambda_type = float(lambda_type)
        self.lambda_band = float(lambda_band)
        self.lambda_unit = float(lambda_unit)
        self.band_width = None if band_width is None else int(max(1, band_width))
        self.band_ratio = float(max(0.0, band_ratio))
        self.bad_cost_threshold = float(max(0.0, bad_cost_threshold))
        self.allow_source_skip = bool(allow_source_skip)
        self.skip_penalty = float(max(0.0, skip_penalty))

    def _resolve_band_width(self, *, num_source: int) -> int:
        if self.band_width is not None:
            return int(self.band_width)
        resolved = int(max(4, round(self.band_ratio * float(max(1, num_source)))))
        return int(min(resolved, max(4, int(num_source) // 2)))

    def _resolve_band_slack(self, *, num_source: int, num_target: int) -> np.float32:
        if self.band_width is not None:
            slack = float(self.band_width) / float(max(1, num_source - 1))
        else:
            slack = float(self.band_ratio)
        slack = max(slack, 1.0 / float(max(1, num_target)))
        return np.float32(min(0.30, max(0.02, slack)))

    @staticmethod
    def _build_source_run_prototypes(
        *,
        source_frame_states: np.ndarray,
        source_frame_to_run: np.ndarray,
        num_source_runs: int,
        source_valid_run_index: np.ndarray | None = None,
        source_run_stability: np.ndarray | None = None,
        source_boundary_cue: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frame_states = as_float32_2d(source_frame_states, name="source_frame_states")
        frame_to_run = as_int64_1d(source_frame_to_run)
        if frame_states.shape[0] != frame_to_run.shape[0]:
            raise ValueError(
                "source_frame_states/source_frame_to_run length mismatch: "
                f"{frame_states.shape[0]} vs {frame_to_run.shape[0]}"
            )
        if frame_to_run.size <= 0:
            raise RuntimeError("continuous aligner requires non-empty source frame states.")

        if frame_to_run.min(initial=0) < 0 or frame_to_run.max(initial=-1) >= int(num_source_runs):
            if source_valid_run_index is None:
                raise RuntimeError(
                    "source_frame_to_run indexes exceed current source run lattice, "
                    "but no source_valid_run_index remapping was provided."
                )
            source_valid_run_index = as_int64_1d(source_valid_run_index)
            if source_valid_run_index.size != int(num_source_runs):
                raise RuntimeError(
                    "source_valid_run_index must match the filtered source run lattice: "
                    f"expected {int(num_source_runs)}, got {int(source_valid_run_index.size)}"
                )
            max_index = int(
                max(
                    int(frame_to_run.max(initial=-1)),
                    int(source_valid_run_index.max(initial=-1)),
                )
            )
            orig_to_valid = np.full((max_index + 1,), -1, dtype=np.int64)
            orig_to_valid[source_valid_run_index] = np.arange(num_source_runs, dtype=np.int64)
            remapped = np.full_like(frame_to_run, -1)
            valid = (frame_to_run >= 0) & (frame_to_run < orig_to_valid.shape[0])
            remapped[valid] = orig_to_valid[frame_to_run[valid]]
            frame_to_run = remapped

        keep = (frame_to_run >= 0) & (frame_to_run < int(num_source_runs))
        if not bool(np.any(keep)):
            raise RuntimeError("continuous aligner could not map any source frames onto valid source runs.")
        frame_states = frame_states[keep]
        frame_to_run = frame_to_run[keep]

        run_stability = _resolve_optional_run_signal(
            source_run_stability,
            num_source_runs=num_source_runs,
            name="source_run_stability",
        )
        if run_stability is None:
            run_stability = np.ones((int(num_source_runs),), dtype=np.float32)
        else:
            run_stability = np.clip(run_stability, 0.0, 1.0)
        boundary_cue = _resolve_optional_run_signal(
            source_boundary_cue,
            num_source_runs=num_source_runs,
            name="source_boundary_cue",
        )
        if boundary_cue is None:
            boundary_cue = np.zeros((int(num_source_runs),), dtype=np.float32)
        else:
            boundary_cue = np.clip(boundary_cue, 0.0, 1.0)

        frame_count = np.bincount(frame_to_run, minlength=int(num_source_runs)).astype(np.float32, copy=False)
        frame_pos = np.zeros((frame_to_run.shape[0],), dtype=np.int64)
        seen = np.zeros((int(num_source_runs),), dtype=np.int64)
        for frame_idx, run_idx in enumerate(frame_to_run.tolist()):
            frame_pos[frame_idx] = seen[run_idx]
            seen[run_idx] += 1

        run_len = np.clip(frame_count[frame_to_run].astype(np.float32, copy=False), 1.0, None)
        max_pos = np.maximum(run_len - 1.0, 1.0)
        dist_to_edge = np.minimum(frame_pos.astype(np.float32, copy=False), max_pos - frame_pos.astype(np.float32, copy=False))
        center_emphasis = np.clip((2.0 * dist_to_edge) / max_pos, 0.0, 1.0)
        stability_frame = run_stability[frame_to_run]
        boundary_frame = boundary_cue[frame_to_run]
        edge_strength = np.clip(boundary_frame * (1.25 - 0.75 * stability_frame), 0.0, 1.0)
        edge_floor = 1.0 - (0.75 * edge_strength)
        edge_profile = edge_floor + ((1.0 - edge_floor) * center_emphasis)
        frame_weight = np.clip((0.35 + 0.65 * stability_frame) * edge_profile, 1.0e-4, None).astype(np.float32, copy=False)

        proto = np.zeros((int(num_source_runs), int(frame_states.shape[1])), dtype=np.float32)
        proto_sq = np.zeros_like(proto)
        weight_sum = np.zeros((int(num_source_runs),), dtype=np.float32)
        np.add.at(proto, frame_to_run, frame_states * frame_weight[:, None])
        np.add.at(proto_sq, frame_to_run, np.square(frame_states) * frame_weight[:, None])
        np.add.at(weight_sum, frame_to_run, frame_weight)
        missing = np.nonzero(weight_sum <= 0.0)[0]
        if missing.size > 0:
            raise RuntimeError(
                "continuous aligner requires source frame support for every valid run; "
                f"missing run ids={missing.tolist()}"
            )
        proto /= weight_sum[:, None]
        proto_var = np.maximum((proto_sq / weight_sum[:, None]) - np.square(proto), 0.0).astype(np.float32, copy=False)
        return (
            proto.astype(np.float32),
            frame_count.astype(np.float32),
            weight_sum.astype(np.float32),
            proto_var.astype(np.float32),
        )

    def build_local_cost(
        self,
        source_run_proto: np.ndarray,
        source_run_types: np.ndarray,
        target_frame_state: np.ndarray,
        target_frame_speech_prob: np.ndarray,
        target_frame_weight: np.ndarray | None = None,
        target_frame_unit_hint: np.ndarray | None = None,
        source_run_units: np.ndarray | None = None,
        source_durations: np.ndarray | None = None,
    ) -> np.ndarray:
        source_run_proto = as_float32_2d(source_run_proto, name="source_run_proto")
        target_frame_state = as_float32_2d(target_frame_state, name="target_frame_state")
        source_run_types = as_float32_1d(source_run_types)
        target_frame_speech_prob = as_float32_1d(target_frame_speech_prob).clip(0.0, 1.0)
        if source_run_proto.shape[0] != source_run_types.shape[0]:
            raise ValueError(
                "source_run_proto/source_run_types length mismatch: "
                f"{source_run_proto.shape[0]} vs {source_run_types.shape[0]}"
            )
        if target_frame_state.shape[0] != target_frame_speech_prob.shape[0]:
            raise ValueError(
                "target_frame_state/target_frame_speech_prob length mismatch: "
                f"{target_frame_state.shape[0]} vs {target_frame_speech_prob.shape[0]}"
            )
        if source_run_proto.shape[1] != target_frame_state.shape[1]:
            raise ValueError(
                "source/target frame state dim mismatch: "
                f"{source_run_proto.shape[1]} vs {target_frame_state.shape[1]}"
            )

        source_norm = _normalize_rows(source_run_proto)
        target_norm = _normalize_rows(target_frame_state)
        emb_cost = 1.0 - np.clip(np.matmul(target_norm, source_norm.T), -1.0, 1.0)

        source_is_speech = ~(source_run_types > 0.5)
        type_cost = np.where(
            source_is_speech[None, :],
            1.0 - target_frame_speech_prob[:, None],
            target_frame_speech_prob[:, None],
        ).astype(np.float32, copy=False)

        num_target = int(target_frame_state.shape[0])
        num_source = int(source_run_proto.shape[0])
        if target_frame_weight is None:
            target_progress = _build_uniform_progress(num_target).reshape(num_target, 1)
        else:
            target_progress = _build_progress_from_weights(target_frame_weight).reshape(num_target, 1)
        if source_durations is None:
            source_progress = _build_uniform_progress(num_source).reshape(1, num_source)
        else:
            source_progress = _build_progress_from_anchor(source_durations).reshape(1, num_source)
        band_slack = self._resolve_band_slack(num_source=num_source, num_target=num_target)
        band_cost = np.square(np.maximum(0.0, np.abs(target_progress - source_progress) - band_slack)).astype(np.float32)

        unit_cost = np.zeros_like(emb_cost, dtype=np.float32)
        if target_frame_unit_hint is not None and source_run_units is not None:
            target_frame_unit_hint = as_int64_1d(target_frame_unit_hint)
            source_run_units = as_int64_1d(source_run_units)
            if target_frame_unit_hint.shape[0] != num_target:
                raise ValueError(
                    "target_frame_unit_hint length mismatch: "
                    f"{target_frame_unit_hint.shape[0]} vs {num_target}"
                )
            if source_run_units.shape[0] != num_source:
                raise ValueError(
                    "source_run_units length mismatch: "
                    f"{source_run_units.shape[0]} vs {num_source}"
                )
            valid_hint = target_frame_unit_hint >= 0
            if bool(np.any(valid_hint)):
                unit_cost[valid_hint] = (
                    source_run_units[None, :] != target_frame_unit_hint[valid_hint][:, None]
                ).astype(np.float32)

        total = (
            (self.lambda_emb * emb_cost)
            + (self.lambda_type * type_cost)
            + (self.lambda_band * band_cost)
            + (self.lambda_unit * unit_cost)
        )
        return total.astype(np.float32)

    def viterbi_align(
        self,
        local_cost: np.ndarray,
        *,
        source_progress: np.ndarray | None = None,
        target_progress: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.float32, np.ndarray]:
        local_cost = as_float32_2d(local_cost, name="local_cost")
        num_target, num_source = local_cost.shape
        if num_target <= 0 or num_source <= 0:
            raise RuntimeError("continuous aligner requires non-empty target observations and source runs.")
        if num_target < num_source and not self.allow_source_skip:
            raise RuntimeError(
                "continuous aligner requires at least one target frame per source run under strict monotonicity: "
                f"target_frames={num_target}, source_runs={num_source}"
            )

        if source_progress is None:
            source_progress = _build_uniform_progress(num_source)
        else:
            source_progress = as_float32_1d(source_progress)
        if target_progress is None:
            target_progress = _build_uniform_progress(num_target)
        else:
            target_progress = as_float32_1d(target_progress)
        if source_progress.shape[0] != num_source:
            raise ValueError("source_progress/local_cost width mismatch")
        if target_progress.shape[0] != num_target:
            raise ValueError("target_progress/local_cost height mismatch")

        band_width = self._resolve_band_width(num_source=num_source)
        inf = np.float32(1.0e12)
        if self.allow_source_skip:
            dp_prev = np.full((num_source,), inf, dtype=np.float32)
            dp_curr = np.full((num_source,), inf, dtype=np.float32)
            back = np.full((num_target, num_source), -1, dtype=np.int32)
            margin = np.zeros((num_target, num_source), dtype=np.float32)

            center0 = _nearest_progress_index(source_progress, float(target_progress[0]))
            init_left = max(0, center0 - band_width)
            init_right = min(num_source - 1, center0 + band_width)
            if init_left > init_right:
                init_left, init_right = 0, num_source - 1
            for src_idx in range(init_left, init_right + 1):
                cost = local_cost[0, src_idx]
                if not np.isfinite(cost):
                    continue
                dp_prev[src_idx] = cost + (np.float32(self.skip_penalty) * np.float32(max(0, src_idx)))
                back[0, src_idx] = src_idx

            for tgt_idx in range(1, num_target):
                dp_curr.fill(inf)
                center = _nearest_progress_index(source_progress, float(target_progress[tgt_idx]))
                left = max(0, center - band_width)
                right = min(num_source - 1, center + band_width)
                if left > right:
                    left, right = 0, num_source - 1
                for src_idx in range(left, right + 1):
                    cost = local_cost[tgt_idx, src_idx]
                    if not np.isfinite(cost):
                        continue
                    best_prev_cost = inf
                    best_prev_idx = -1
                    second_prev_cost = inf
                    for prev_idx in range(0, src_idx + 1):
                        prev_cost = dp_prev[prev_idx]
                        if not np.isfinite(prev_cost):
                            continue
                        candidate = prev_cost
                        if prev_idx < src_idx:
                            candidate = candidate + (np.float32(self.skip_penalty) * np.float32(src_idx - prev_idx - 1))
                        if candidate < best_prev_cost:
                            second_prev_cost = best_prev_cost
                            best_prev_cost = candidate
                            best_prev_idx = prev_idx
                        elif candidate < second_prev_cost:
                            second_prev_cost = candidate
                    if best_prev_idx < 0:
                        continue
                    dp_curr[src_idx] = cost + best_prev_cost
                    back[tgt_idx, src_idx] = best_prev_idx
                    margin[tgt_idx, src_idx] = (
                        np.float32(abs(second_prev_cost - best_prev_cost))
                        if np.isfinite(second_prev_cost)
                        else np.float32(1.0e3)
                    )
                dp_prev, dp_curr = dp_curr, dp_prev

            terminal_total = np.full((num_source,), inf, dtype=np.float32)
            for src_idx in range(num_source):
                if not np.isfinite(dp_prev[src_idx]):
                    continue
                terminal_total[src_idx] = dp_prev[src_idx] + (
                    np.float32(self.skip_penalty) * np.float32(max(0, (num_source - 1) - src_idx))
                )
            terminal_src = int(np.argmin(terminal_total))
            terminal_cost = terminal_total[terminal_src]
            if not np.isfinite(terminal_cost):
                raise RuntimeError("continuous aligner failed to reach a terminal source run under the current band.")

            assigned_run = np.zeros((num_target,), dtype=np.int64)
            path_margin = np.zeros((num_target,), dtype=np.float32)
            src_idx = terminal_src
            for tgt_idx in range(num_target - 1, -1, -1):
                assigned_run[tgt_idx] = np.int64(src_idx)
                path_margin[tgt_idx] = margin[tgt_idx, src_idx]
                if tgt_idx <= 0:
                    continue
                prev_idx = int(back[tgt_idx, src_idx])
                if prev_idx < 0 or prev_idx > src_idx:
                    raise RuntimeError(
                        "continuous aligner backtrace encountered an invalid skip predecessor marker "
                        f"at target={tgt_idx}, source={src_idx}: prev={prev_idx}"
                    )
                src_idx = prev_idx
            assigned_local_cost = local_cost[np.arange(num_target, dtype=np.int64), assigned_run].astype(np.float32)
            return assigned_run, assigned_local_cost, np.float32(terminal_cost), path_margin

        dp_prev = np.full((num_source,), inf, dtype=np.float32)
        dp_curr = np.full((num_source,), inf, dtype=np.float32)
        back = np.full((num_target, num_source), -1, dtype=np.int64)
        margin = np.zeros((num_target, num_source), dtype=np.float32)
        center = _nearest_progress_index(source_progress, float(target_progress[0]))
        left = max(0, center - band_width)
        right = min(num_source - 1, center + band_width)
        if left > right:
            left, right = 0, min(num_source - 1, max(0, center))
        for src_idx in range(left, right + 1):
            if not self.allow_source_skip and src_idx != 0:
                continue
            cost = local_cost[0, src_idx]
            if not np.isfinite(cost):
                continue
            skip_cost = np.float32(self.skip_penalty * float(src_idx)) if self.allow_source_skip else np.float32(0.0)
            dp_prev[src_idx] = cost + skip_cost
            back[0, src_idx] = -1
        if not np.isfinite(dp_prev).any():
            raise RuntimeError("continuous aligner could not initialize any first-frame state.")

        for tgt_idx in range(1, num_target):
            dp_curr.fill(inf)
            hard_left = 0 if self.allow_source_skip else max(0, num_source - num_target + tgt_idx)
            hard_right = num_source - 1 if self.allow_source_skip else min(num_source - 1, tgt_idx)
            center = _nearest_progress_index(source_progress, float(target_progress[tgt_idx]))
            left = max(hard_left, center - band_width)
            right = min(hard_right, center + band_width)
            if left > right:
                left, right = hard_left, hard_right
            for src_idx in range(left, right + 1):
                if self.allow_source_skip:
                    prev_idx = np.arange(src_idx + 1, dtype=np.int64)
                    prev_cost = dp_prev[: src_idx + 1]
                    if prev_cost.size <= 0:
                        continue
                    cand = prev_cost + (self.skip_penalty * (src_idx - prev_idx).astype(np.float32))
                    best_rank = np.argsort(cand, kind="stable")
                    best_prev_idx = int(prev_idx[best_rank[0]])
                    best_prev = cand[best_rank[0]]
                    alt_prev = cand[best_rank[1]] if cand.size > 1 else inf
                else:
                    stay = dp_prev[src_idx]
                    advance = dp_prev[src_idx - 1] if src_idx > 0 else inf
                    if src_idx <= 0:
                        best_prev = stay
                        alt_prev = inf
                        best_prev_idx = src_idx
                    elif advance <= stay:
                        best_prev = advance
                        alt_prev = stay
                        best_prev_idx = src_idx - 1
                    else:
                        best_prev = stay
                        alt_prev = advance
                        best_prev_idx = src_idx
                if not np.isfinite(best_prev):
                    continue
                cost = local_cost[tgt_idx, src_idx]
                if not np.isfinite(cost):
                    continue
                dp_curr[src_idx] = cost + best_prev
                back[tgt_idx, src_idx] = best_prev_idx
                margin[tgt_idx, src_idx] = (
                    np.float32(abs(float(best_prev) - float(alt_prev)))
                    if np.isfinite(alt_prev)
                    else np.float32(1.0e3)
                )
            dp_prev, dp_curr = dp_curr, dp_prev

        terminal_cost = dp_prev[num_source - 1]
        if not np.isfinite(terminal_cost):
            raise RuntimeError("continuous aligner failed to reach the final source run under the current band.")

        assigned_run = np.zeros((num_target,), dtype=np.int64)
        path_margin = np.zeros((num_target,), dtype=np.float32)
        src_idx = num_source - 1
        for tgt_idx in range(num_target - 1, -1, -1):
            assigned_run[tgt_idx] = np.int64(src_idx)
            path_margin[tgt_idx] = margin[tgt_idx, src_idx]
            if tgt_idx <= 0:
                continue
            prev_src_idx = int(back[tgt_idx, src_idx])
            if prev_src_idx < 0:
                raise RuntimeError(
                    "continuous aligner backtrace encountered an invalid predecessor marker "
                    f"at target={tgt_idx}, source={src_idx}: move={prev_src_idx}"
                )
            src_idx = prev_src_idx
        if not self.allow_source_skip and src_idx != 0:
            raise RuntimeError(
                "continuous aligner backtrace did not return to the first source run: "
                f"final_state={src_idx}"
            )

        assigned_local_cost = local_cost[np.arange(num_target, dtype=np.int64), assigned_run].astype(np.float32)
        return assigned_run, assigned_local_cost, np.float32(terminal_cost), path_margin

    def summarize_alignment(
        self,
        assigned_run: np.ndarray,
        assigned_local_cost: np.ndarray,
        source_run_types: np.ndarray,
        target_frame_speech_prob: np.ndarray,
        *,
        target_frame_weight: np.ndarray | None = None,
        target_frame_unit_hint: np.ndarray | None = None,
        source_run_units: np.ndarray | None = None,
        path_margin: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | np.float32]:
        assigned_run = as_int64_1d(assigned_run)
        assigned_local_cost = as_float32_1d(assigned_local_cost)
        source_run_types = as_float32_1d(source_run_types)
        target_frame_speech_prob = as_float32_1d(target_frame_speech_prob).clip(0.0, 1.0)
        if assigned_run.shape[0] != assigned_local_cost.shape[0]:
            raise ValueError("assigned_run/assigned_local_cost length mismatch")
        if assigned_run.shape[0] != target_frame_speech_prob.shape[0]:
            raise ValueError("assigned_run/target_frame_speech_prob length mismatch")
        num_source = int(source_run_types.shape[0])
        if target_frame_weight is None:
            target_frame_weight = np.ones((assigned_run.shape[0],), dtype=np.float32)
        else:
            target_frame_weight = as_float32_1d(target_frame_weight).clip(1.0e-4, None)
        if target_frame_weight.shape[0] != assigned_run.shape[0]:
            raise ValueError("target_frame_weight length mismatch")
        if path_margin is None:
            path_margin = np.zeros((assigned_run.shape[0],), dtype=np.float32)
        else:
            path_margin = as_float32_1d(path_margin)
        if path_margin.shape[0] != assigned_run.shape[0]:
            raise ValueError("path_margin length mismatch")

        run_occ_viterbi = np.bincount(assigned_run, minlength=num_source).astype(np.float32, copy=False)
        run_occ_weighted = np.bincount(
            assigned_run,
            weights=target_frame_weight.astype(np.float32, copy=False),
            minlength=num_source,
        ).astype(np.float32, copy=False)
        run_cost_sum = np.bincount(
            assigned_run,
            weights=assigned_local_cost.astype(np.float32, copy=False),
            minlength=num_source,
        ).astype(np.float32, copy=False)
        run_mean_cost = np.divide(
            run_cost_sum,
            np.clip(run_occ_viterbi, 1.0, None),
            out=np.zeros((num_source,), dtype=np.float32),
            where=run_occ_viterbi > 0.0,
        )

        source_is_speech = ~(source_run_types > 0.5)
        frame_type_agree = np.where(
            source_is_speech[assigned_run],
            target_frame_speech_prob,
            1.0 - target_frame_speech_prob,
        ).astype(np.float32, copy=False)
        run_type_agree = np.divide(
            np.bincount(
                assigned_run,
                weights=frame_type_agree,
                minlength=num_source,
            ).astype(np.float32, copy=False),
            np.clip(run_occ_viterbi, 1.0, None),
            out=np.zeros((num_source,), dtype=np.float32),
            where=run_occ_viterbi > 0.0,
        )

        if target_frame_unit_hint is not None and source_run_units is not None:
            target_frame_unit_hint = as_int64_1d(target_frame_unit_hint)
            source_run_units = as_int64_1d(source_run_units)
            if target_frame_unit_hint.shape[0] != assigned_run.shape[0]:
                raise ValueError("target_frame_unit_hint length mismatch")
            if source_run_units.shape[0] != num_source:
                raise ValueError("source_run_units length mismatch")
            valid_hint = target_frame_unit_hint >= 0
            frame_match = frame_type_agree.copy()
            if bool(np.any(valid_hint)):
                frame_match[valid_hint] = (
                    (target_frame_unit_hint[valid_hint] == source_run_units[assigned_run[valid_hint]]).astype(np.float32)
                    * frame_type_agree[valid_hint]
                )
        else:
            frame_match = frame_type_agree
        run_match_rate = np.divide(
            np.bincount(
                assigned_run,
                weights=frame_match,
                minlength=num_source,
            ).astype(np.float32, copy=False),
            np.clip(run_occ_viterbi, 1.0, None),
            out=np.zeros((num_source,), dtype=np.float32),
            where=run_occ_viterbi > 0.0,
        )

        run_margin = np.zeros((num_source,), dtype=np.float32)
        for src_idx in range(num_source):
            selected = path_margin[assigned_run == src_idx]
            finite = selected[np.isfinite(selected)]
            if finite.size > 0:
                run_margin[src_idx] = np.float32(np.median(finite))

        margin_term = np.tanh(np.clip(run_margin, 0.0, 5.0))
        cost_term_local = _sigmoid(-1.6 * run_mean_cost)
        cost_term_coarse = _sigmoid(-1.1 * run_mean_cost)
        margin_term_local = _sigmoid(0.9 * margin_term)
        margin_term_coarse = _sigmoid(0.5 * margin_term)
        type_term_local = _sigmoid(1.2 * (run_type_agree - 0.5))
        type_term_coarse = _sigmoid(0.8 * (run_type_agree - 0.5))
        match_term_local = np.clip(run_match_rate, 0.0, 1.0).astype(np.float32)
        match_term_coarse = np.sqrt(np.clip(run_match_rate, 0.0, 1.0)).astype(np.float32)
        confidence_local = _sigmoid(
            (-1.6 * run_mean_cost) + (0.9 * margin_term) + (1.2 * (run_type_agree - 0.5))
        ).astype(np.float32)
        confidence_coarse = _sigmoid(
            (-1.1 * run_mean_cost) + (0.5 * margin_term) + (0.8 * (run_type_agree - 0.5))
        ).astype(np.float32)
        confidence_local = np.where(run_occ_viterbi > 0.0, confidence_local, 0.0).astype(np.float32, copy=False)
        confidence_coarse = np.where(run_occ_viterbi > 0.0, confidence_coarse, 0.0).astype(np.float32, copy=False)
        confidence_local[~source_is_speech] = 0.0

        speech_frame = target_frame_speech_prob > 0.5
        unmatched = speech_frame & (
            (assigned_local_cost > np.float32(self.bad_cost_threshold))
            | (~source_is_speech[assigned_run])
        )
        unmatched_speech_ratio = (
            np.float32(float(np.count_nonzero(unmatched)) / float(max(1, np.count_nonzero(speech_frame))))
            if bool(np.any(speech_frame))
            else np.float32(0.0)
        )
        mean_local_confidence_speech = (
            np.float32(np.mean(confidence_local[source_is_speech], dtype=np.float32))
            if bool(np.any(source_is_speech))
            else np.float32(0.0)
        )
        mean_coarse_confidence_speech = (
            np.float32(np.mean(confidence_coarse[source_is_speech], dtype=np.float32))
            if bool(np.any(source_is_speech))
            else np.float32(0.0)
        )

        return {
            "run_occ_viterbi": run_occ_viterbi.astype(np.float32),
            "run_occ_weighted": run_occ_weighted.astype(np.float32),
            "run_margin": run_margin.astype(np.float32),
            "run_mean_cost": run_mean_cost.astype(np.float32),
            "run_type_agree": run_type_agree.astype(np.float32),
            "run_match_rate": run_match_rate.astype(np.float32),
            "run_confidence_cost_term": cost_term_local.astype(np.float32),
            "run_confidence_margin_term": margin_term_local.astype(np.float32),
            "run_confidence_type_term": type_term_local.astype(np.float32),
            "run_confidence_match_term": match_term_local.astype(np.float32),
            "run_confidence_local_final": confidence_local.astype(np.float32),
            "run_confidence_coarse_cost_term": cost_term_coarse.astype(np.float32),
            "run_confidence_coarse_margin_term": margin_term_coarse.astype(np.float32),
            "run_confidence_coarse_type_term": type_term_coarse.astype(np.float32),
            "run_confidence_coarse_match_term": match_term_coarse.astype(np.float32),
            "run_confidence_coarse_final": confidence_coarse.astype(np.float32),
            "run_confidence_local": confidence_local.astype(np.float32),
            "run_confidence_coarse": confidence_coarse.astype(np.float32),
            "unmatched_speech_ratio": unmatched_speech_ratio,
            "mean_local_confidence_speech": mean_local_confidence_speech,
            "mean_coarse_confidence_speech": mean_coarse_confidence_speech,
        }

    def align(
        self,
        *,
        source_run_units: np.ndarray,
        source_run_types: np.ndarray,
        source_frame_states: np.ndarray,
        source_frame_to_run: np.ndarray,
        target_frame_states: np.ndarray,
        source_valid_run_index: np.ndarray | None = None,
        source_durations: np.ndarray | None = None,
        source_run_stability: np.ndarray | None = None,
        source_boundary_cue: np.ndarray | None = None,
        target_frame_speech_prob: np.ndarray | None = None,
        target_frame_weight: np.ndarray | None = None,
        target_frame_valid: np.ndarray | None = None,
        target_frame_unit_hint: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | np.float32 | str]:
        source_run_units = as_int64_1d(source_run_units)
        source_run_types = as_float32_1d(source_run_types)
        if source_run_units.shape[0] != source_run_types.shape[0]:
            raise ValueError(
                "source_run_units/source_run_types length mismatch: "
                f"{source_run_units.shape[0]} vs {source_run_types.shape[0]}"
            )

        target_frame_states = as_float32_2d(target_frame_states, name="target_frame_states")
        num_target_frames = int(target_frame_states.shape[0])
        if target_frame_valid is None:
            target_frame_valid = np.ones((num_target_frames,), dtype=np.float32)
        else:
            target_frame_valid = as_float32_1d(target_frame_valid)
        if target_frame_valid.shape[0] != num_target_frames:
            raise ValueError(
                "target_frame_valid/target_frame_states length mismatch: "
                f"{target_frame_valid.shape[0]} vs {num_target_frames}"
            )
        valid_frame = target_frame_valid > 0.5
        target_valid_index = np.nonzero(valid_frame)[0].astype(np.int64)
        if target_valid_index.size <= 0:
            raise RuntimeError("continuous aligner requires at least one valid target frame.")

        target_frame_states_valid = target_frame_states[valid_frame]
        if target_frame_speech_prob is None:
            target_frame_speech_prob_valid = np.ones((target_valid_index.size,), dtype=np.float32)
        else:
            target_frame_speech_prob = as_float32_1d(target_frame_speech_prob)
            if target_frame_speech_prob.shape[0] != num_target_frames:
                raise ValueError(
                    "target_frame_speech_prob/target_frame_states length mismatch: "
                    f"{target_frame_speech_prob.shape[0]} vs {num_target_frames}"
                )
            target_frame_speech_prob_valid = target_frame_speech_prob[valid_frame].astype(np.float32, copy=False)
        has_target_frame_weight = target_frame_weight is not None
        if target_frame_weight is None:
            target_frame_weight_valid = np.ones((target_valid_index.size,), dtype=np.float32)
        else:
            target_frame_weight = as_float32_1d(target_frame_weight)
            if target_frame_weight.shape[0] != num_target_frames:
                raise ValueError(
                    "target_frame_weight/target_frame_states length mismatch: "
                    f"{target_frame_weight.shape[0]} vs {num_target_frames}"
                )
            target_frame_weight_valid = target_frame_weight[valid_frame].astype(np.float32, copy=False)
        if target_frame_unit_hint is None:
            target_frame_unit_hint_valid = None
        else:
            target_frame_unit_hint = as_int64_1d(target_frame_unit_hint)
            if target_frame_unit_hint.shape[0] != num_target_frames:
                raise ValueError(
                    "target_frame_unit_hint/target_frame_states length mismatch: "
                    f"{target_frame_unit_hint.shape[0]} vs {num_target_frames}"
                )
            target_frame_unit_hint_valid = target_frame_unit_hint[valid_frame].astype(np.int64, copy=False)

        source_run_proto, source_frame_count, source_frame_weight_sum, source_run_proto_var = self._build_source_run_prototypes(
            source_frame_states=source_frame_states,
            source_frame_to_run=source_frame_to_run,
            num_source_runs=int(source_run_units.shape[0]),
            source_valid_run_index=source_valid_run_index,
            source_run_stability=source_run_stability,
            source_boundary_cue=source_boundary_cue,
        )
        local_cost = self.build_local_cost(
            source_run_proto=source_run_proto,
            source_run_types=source_run_types,
            target_frame_state=target_frame_states_valid,
            target_frame_speech_prob=target_frame_speech_prob_valid,
            target_frame_weight=target_frame_weight_valid if has_target_frame_weight else None,
            target_frame_unit_hint=target_frame_unit_hint_valid,
            source_run_units=source_run_units,
            source_durations=source_durations,
        )
        source_progress = (
            _build_progress_from_anchor(source_durations)
            if source_durations is not None
            else _build_uniform_progress(source_run_units.shape[0])
        )
        target_progress = (
            _build_progress_from_weights(target_frame_weight_valid)
            if has_target_frame_weight
            else _build_uniform_progress(target_frame_states_valid.shape[0])
        )
        assigned_run, assigned_local_cost, global_path_cost, path_margin = self.viterbi_align(
            local_cost,
            source_progress=source_progress,
            target_progress=target_progress,
        )
        summary = self.summarize_alignment(
            assigned_run=assigned_run,
            assigned_local_cost=assigned_local_cost,
            source_run_types=source_run_types,
            target_frame_speech_prob=target_frame_speech_prob_valid,
            target_frame_weight=target_frame_weight_valid,
            target_frame_unit_hint=target_frame_unit_hint_valid,
            source_run_units=source_run_units,
            path_margin=path_margin,
        )
        source_progress_kind = "anchor_duration_cdf" if source_durations is not None else "uniform_index"
        target_progress_kind = "weighted_cdf" if has_target_frame_weight else "uniform_index"
        summary.update(
            {
                "alignment_kind": _ALIGNMENT_MODE_CONTINUOUS_VITERBI_V1,
                "alignment_mode": _ALIGNMENT_MODE_CONTINUOUS_VITERBI_V1,
                "alignment_source": "run_state_viterbi",
                "alignment_version": CONTINUOUS_ALIGNER_VERSION,
                "posterior_kind": "none",
                "confidence_kind": "heuristic_v1",
                "confidence_formula_version": "heuristic_margin_cost_type_v1",
                "assigned_source": assigned_run.astype(np.int64),
                "assigned_cost": assigned_local_cost.astype(np.float32),
                "target_valid_observation_index": target_valid_index.astype(np.int64),
                "source_frame_count": source_frame_count.astype(np.float32),
                "source_frame_weight_sum": source_frame_weight_sum.astype(np.float32),
                "source_run_proto": source_run_proto.astype(np.float32),
                "source_run_proto_var": source_run_proto_var.astype(np.float32),
                "source_run_proto_kind": "stability_boundary_weighted_mean_v1",
                "source_run_proto_weighting_uses_stability": np.asarray([1], dtype=np.int64),
                "source_run_proto_weighting_uses_boundary": np.asarray([1], dtype=np.int64),
                "source_progress_kind": source_progress_kind,
                "target_progress_kind": target_progress_kind,
                "alignment_lambda_emb": np.asarray([self.lambda_emb], dtype=np.float32),
                "alignment_lambda_type": np.asarray([self.lambda_type], dtype=np.float32),
                "alignment_lambda_band": np.asarray([self.lambda_band], dtype=np.float32),
                "alignment_lambda_unit": np.asarray([self.lambda_unit], dtype=np.float32),
                "alignment_band_ratio": np.asarray([self.band_ratio], dtype=np.float32),
                "alignment_bad_cost_threshold": np.asarray([self.bad_cost_threshold], dtype=np.float32),
                "alignment_band_width": np.asarray(
                    [-1 if self.band_width is None else int(self.band_width)],
                    dtype=np.int64,
                ),
                "alignment_allow_source_skip": np.asarray([1 if self.allow_source_skip else 0], dtype=np.int64),
                "alignment_skip_penalty": np.asarray([self.skip_penalty], dtype=np.float32),
                "run_occ_hard": summary["run_occ_viterbi"],
                "run_occ_expected": summary["run_occ_viterbi"],
                "run_occ_expected_is_hard_proxy": np.asarray([1], dtype=np.int64),
                "run_occ_expected_semantics": "hard_viterbi_proxy",
                "run_entropy": np.zeros_like(summary["run_occ_viterbi"], dtype=np.float32),
                "run_posterior_mass_on_path": np.where(
                    np.asarray(summary["run_occ_viterbi"], dtype=np.float32) > 0.0,
                    1.0,
                    0.0,
                ).astype(np.float32),
                "run_posterior_mass_on_path_is_hard_proxy": np.asarray([1], dtype=np.int64),
                "global_path_cost": np.asarray([global_path_cost], dtype=np.float32),
            }
        )
        return summary


def align_target_runs_to_source_discrete(
    *,
    source_units: np.ndarray,
    source_durations: np.ndarray,
    source_silence: np.ndarray,
    target_units: np.ndarray,
    target_durations: np.ndarray,
    target_silence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_source = int(source_units.shape[0])
    num_target = int(target_units.shape[0])
    if num_source <= 0 or num_target <= 0:
        raise RuntimeError("paired projection requires non-empty source and target run lattices.")

    gap_penalty = np.float32(0.40)
    src_pos = (np.arange(num_source, dtype=np.float32) + 1.0) / float(max(1, num_source))
    tgt_pos = (np.arange(num_target, dtype=np.float32) + 1.0) / float(max(1, num_target))
    token_match = source_units[:, None] == target_units[None, :]
    sil_match = (source_silence[:, None] > 0.5) == (target_silence[None, :] > 0.5)
    log_duration_delta = np.abs(
        np.log(source_durations[:, None].clip(min=1.0e-4))
        - np.log(target_durations[None, :].clip(min=1.0e-4))
    ).astype(np.float32)
    local_cost = np.where(
        token_match & sil_match,
        0.0,
        np.where(
            ~sil_match,
            1.60,
            np.where(token_match, 0.15, 0.90),
        ),
    ).astype(np.float32)
    local_cost += (0.20 * log_duration_delta).astype(np.float32)
    local_cost += (0.15 * np.abs(src_pos[:, None] - tgt_pos[None, :])).astype(np.float32)

    back = np.zeros((num_source + 1, num_target + 1), dtype=np.uint8)
    dp_prev = np.full((num_target + 1,), np.inf, dtype=np.float32)
    dp_curr = np.full((num_target + 1,), np.inf, dtype=np.float32)
    dp_prev[0] = 0.0
    for tgt_idx in range(1, num_target + 1):
        dp_prev[tgt_idx] = dp_prev[tgt_idx - 1] + gap_penalty
        back[0, tgt_idx] = 2

    band = int(max(8, round(0.15 * max(num_source, num_target))))
    for src_idx in range(1, num_source + 1):
        dp_curr.fill(np.inf)
        dp_curr[0] = dp_prev[0] + gap_penalty
        back[src_idx, 0] = 1
        center = int(round((src_idx - 1) * num_target / float(max(1, num_source))))
        left = max(1, center - band)
        right = min(num_target + 1, center + band + 1)
        for tgt_idx in range(left, right):
            cost = local_cost[src_idx - 1, tgt_idx - 1]
            diag = dp_prev[tgt_idx - 1] + cost
            up = dp_prev[tgt_idx] + gap_penalty
            left_cost = dp_curr[tgt_idx - 1] + gap_penalty
            if diag <= up and diag <= left_cost:
                dp_curr[tgt_idx] = diag
                back[src_idx, tgt_idx] = 0
            elif up <= left_cost:
                dp_curr[tgt_idx] = up
                back[src_idx, tgt_idx] = 1
            else:
                dp_curr[tgt_idx] = left_cost
                back[src_idx, tgt_idx] = 2
        dp_prev, dp_curr = dp_curr, dp_prev

    assigned_source = np.full((num_target,), -1, dtype=np.int64)
    assigned_cost = np.full((num_target,), gap_penalty, dtype=np.float32)
    src_idx = num_source
    tgt_idx = num_target
    while src_idx > 0 or tgt_idx > 0:
        move = int(back[src_idx, tgt_idx])
        if move == 0:
            assigned_source[tgt_idx - 1] = max(0, src_idx - 1)
            assigned_cost[tgt_idx - 1] = local_cost[src_idx - 1, tgt_idx - 1]
            src_idx -= 1
            tgt_idx -= 1
        elif move == 1:
            src_idx -= 1
        else:
            tgt_idx -= 1
    return assigned_source, assigned_cost


def align_target_frames_to_source_runs(
    *,
    source_units: np.ndarray,
    source_durations: np.ndarray,
    source_silence: np.ndarray,
    target_units: np.ndarray,
    target_durations: np.ndarray,
    target_silence: np.ndarray,
    source_frame_states: np.ndarray | None = None,
    target_frame_states: np.ndarray | None = None,
    source_frame_to_run: np.ndarray | None = None,
    source_valid_run_index: np.ndarray | None = None,
    source_run_stability: np.ndarray | None = None,
    source_boundary_cue: np.ndarray | None = None,
    target_frame_speech_prob: np.ndarray | None = None,
    target_frame_weight: np.ndarray | None = None,
    target_frame_valid: np.ndarray | None = None,
    target_frame_unit_hint: np.ndarray | None = None,
    continuous_alignment_mode: str | None = None,
    continuous_aligner_kwargs: dict | None = None,
    precomputed_alignment=None,
) -> dict[str, np.ndarray | np.float32 | str] | None:
    resolved_mode = _normalize_continuous_alignment_mode(
        continuous_alignment_mode,
        precomputed_alignment=precomputed_alignment,
    )
    resolved = resolve_precomputed_alignment(
        precomputed_alignment=precomputed_alignment,
        num_target=int(target_units.shape[0]),
    )
    if resolved_mode == _ALIGNMENT_MODE_CONTINUOUS_PRECOMPUTED:
        if resolved is None:
            return None
        alignment_kind = _ALIGNMENT_MODE_CONTINUOUS_PRECOMPUTED
        alignment_source = ""
        alignment_version = ""
        if isinstance(precomputed_alignment, dict):
            alignment_kind = str(precomputed_alignment.get("alignment_kind", alignment_kind) or alignment_kind)
            alignment_source = str(precomputed_alignment.get("alignment_source", "") or "")
            alignment_version = str(precomputed_alignment.get("alignment_version", "") or "")
        return {
            "assigned_source": resolved[0].astype(np.int64),
            "assigned_cost": resolved[1].astype(np.float32),
            "alignment_kind": alignment_kind,
            "alignment_mode": _ALIGNMENT_MODE_CONTINUOUS_PRECOMPUTED,
            "alignment_source": alignment_source,
            "alignment_version": alignment_version,
        }
    if resolved_mode != _ALIGNMENT_MODE_CONTINUOUS_VITERBI_V1:
        raise ValueError(f"Unsupported continuous alignment mode for frame/run aligner: {resolved_mode!r}")
    if source_frame_states is None or target_frame_states is None or source_frame_to_run is None:
        return None
    del target_durations, target_silence
    aligner = ContinuousRunAligner(**(continuous_aligner_kwargs or {}))
    return aligner.align(
        source_run_units=source_units,
        source_run_types=source_silence,
        source_frame_states=source_frame_states,
        source_frame_to_run=source_frame_to_run,
        target_frame_states=target_frame_states,
        source_valid_run_index=source_valid_run_index,
        source_durations=source_durations,
        source_run_stability=source_run_stability,
        source_boundary_cue=source_boundary_cue,
        target_frame_speech_prob=target_frame_speech_prob,
        target_frame_weight=target_frame_weight,
        target_frame_valid=target_frame_valid,
        target_frame_unit_hint=target_frame_unit_hint,
    )


def align_target_to_source(
    *,
    source_units: np.ndarray,
    source_durations: np.ndarray,
    source_silence: np.ndarray,
    target_units: np.ndarray,
    target_durations: np.ndarray,
    target_silence: np.ndarray,
    use_continuous_alignment: bool = False,
    source_frame_states: np.ndarray | None = None,
    target_frame_states: np.ndarray | None = None,
    source_frame_to_run: np.ndarray | None = None,
    source_valid_run_index: np.ndarray | None = None,
    source_run_stability: np.ndarray | None = None,
    source_boundary_cue: np.ndarray | None = None,
    target_frame_speech_prob: np.ndarray | None = None,
    target_frame_weight: np.ndarray | None = None,
    target_frame_valid: np.ndarray | None = None,
    target_frame_unit_hint: np.ndarray | None = None,
    continuous_alignment_mode: str | None = None,
    continuous_aligner_kwargs: dict | None = None,
    precomputed_alignment=None,
) -> dict[str, np.ndarray | np.float32 | str]:
    if use_continuous_alignment:
        continuous = align_target_frames_to_source_runs(
            source_units=source_units,
            source_durations=source_durations,
            source_silence=source_silence,
            target_units=target_units,
            target_durations=target_durations,
            target_silence=target_silence,
            source_frame_states=source_frame_states,
            target_frame_states=target_frame_states,
            source_frame_to_run=source_frame_to_run,
            source_valid_run_index=source_valid_run_index,
            source_run_stability=source_run_stability,
            source_boundary_cue=source_boundary_cue,
            target_frame_speech_prob=target_frame_speech_prob,
            target_frame_weight=target_frame_weight,
            target_frame_valid=target_frame_valid,
            target_frame_unit_hint=target_frame_unit_hint,
            continuous_alignment_mode=continuous_alignment_mode,
            continuous_aligner_kwargs=continuous_aligner_kwargs,
            precomputed_alignment=precomputed_alignment,
        )
        if continuous is not None:
            return continuous
        resolved_mode = _normalize_continuous_alignment_mode(
            continuous_alignment_mode,
            precomputed_alignment=precomputed_alignment,
        )
        if resolved_mode == _ALIGNMENT_MODE_CONTINUOUS_PRECOMPUTED:
            raise RuntimeError(
                "rhythm_v3_alignment_mode=continuous_precomputed requires explicit continuous_precomputed metadata."
            )
        raise RuntimeError(
            "rhythm_v3_use_continuous_alignment=true requires explicit source_frame_states/source_frame_to_run/"
            "target_frame_states sidecars for continuous_viterbi_v1."
        )
    assigned_source, assigned_cost = align_target_runs_to_source_discrete(
        source_units=source_units,
        source_durations=source_durations,
        source_silence=source_silence,
        target_units=target_units,
        target_durations=target_durations,
        target_silence=target_silence,
    )
    return {
        "assigned_source": assigned_source.astype(np.int64),
        "assigned_cost": assigned_cost.astype(np.float32),
        "alignment_kind": "discrete",
        "alignment_source": "",
        "alignment_version": "",
    }


def project_target_runs_onto_source(
    *,
    source_units: np.ndarray,
    source_durations: np.ndarray,
    source_silence_mask: np.ndarray,
    target_units: np.ndarray,
    target_durations: np.ndarray,
    target_valid_mask: np.ndarray,
    target_speech_mask: np.ndarray,
    use_continuous_alignment: bool = False,
    source_frame_states: np.ndarray | None = None,
    target_frame_states: np.ndarray | None = None,
    source_frame_to_run: np.ndarray | None = None,
    source_run_stability: np.ndarray | None = None,
    source_boundary_cue: np.ndarray | None = None,
    target_frame_speech_prob: np.ndarray | None = None,
    target_frame_weight: np.ndarray | None = None,
    target_frame_valid: np.ndarray | None = None,
    target_frame_unit_hint: np.ndarray | None = None,
    continuous_alignment_mode: str | None = None,
    continuous_aligner_kwargs: dict | None = None,
    precomputed_alignment=None,
    allow_source_self_target_fallback: bool = False,
    alignment_soft_repair: bool = False,
) -> dict[str, np.ndarray | np.float32 | str]:
    source_silence_mask = resolve_run_silence_mask(size=len(source_units), silence_mask=source_silence_mask)
    target_silence_mask = ((target_valid_mask > 0.5) & ~(target_speech_mask > 0.5)).astype(np.float32)
    (
        src_units_valid,
        src_durations_valid,
        src_silence_valid,
        src_run_index,
    ) = filter_valid_runs(
        units=source_units,
        durations=source_durations,
        silence_mask=source_silence_mask,
    )
    (
        tgt_units_valid,
        tgt_durations_valid,
        tgt_silence_valid,
        tgt_run_index,
    ) = filter_valid_runs(
        units=target_units,
        durations=target_durations,
        silence_mask=target_silence_mask,
        valid_mask=target_valid_mask,
    )
    if src_units_valid.size <= 0:
        raise RuntimeError("paired projection requires a non-empty source run lattice.")
    if tgt_units_valid.size <= 0:
        raise RuntimeError("paired projection requires a non-empty paired target prompt lattice.")

    alignment = align_target_to_source(
        source_units=src_units_valid,
        source_durations=src_durations_valid,
        source_silence=src_silence_valid,
        target_units=tgt_units_valid,
        target_durations=tgt_durations_valid,
        target_silence=tgt_silence_valid,
        use_continuous_alignment=use_continuous_alignment,
        source_frame_states=source_frame_states,
        target_frame_states=target_frame_states,
        source_frame_to_run=source_frame_to_run,
        source_run_stability=(
            None
            if source_run_stability is None
            else as_float32_1d(source_run_stability)[src_run_index]
        ),
        source_boundary_cue=(
            None
            if source_boundary_cue is None
            else as_float32_1d(source_boundary_cue)[src_run_index]
        ),
        source_valid_run_index=src_run_index,
        target_frame_speech_prob=target_frame_speech_prob,
        target_frame_weight=target_frame_weight,
        target_frame_valid=target_frame_valid,
        target_frame_unit_hint=target_frame_unit_hint,
        continuous_alignment_mode=continuous_alignment_mode,
        continuous_aligner_kwargs=continuous_aligner_kwargs,
        precomputed_alignment=precomputed_alignment,
    )

    assigned_source = np.asarray(alignment["assigned_source"], dtype=np.int64)
    assigned_cost = np.asarray(alignment["assigned_cost"], dtype=np.float32)
    alignment_kind = str(alignment.get("alignment_kind", "discrete") or "discrete").strip().lower()

    num_source_runs = int(source_units.shape[0])
    projected = np.zeros((num_source_runs,), dtype=np.float32)
    projected_weighted = np.zeros((num_source_runs,), dtype=np.float32)
    aligned_target = np.zeros((num_source_runs,), dtype=np.float32)
    exact_match = np.zeros((num_source_runs,), dtype=np.float32)
    cost_mass = np.zeros((num_source_runs,), dtype=np.float32)
    source_support = np.zeros((num_source_runs,), dtype=np.float32)
    source_support[src_run_index] = 1.0
    confidence_local = np.zeros((num_source_runs,), dtype=np.float32)
    confidence_coarse = np.zeros((num_source_runs,), dtype=np.float32)
    coverage = np.zeros((num_source_runs,), dtype=np.float32)
    coverage_binary = np.zeros((num_source_runs,), dtype=np.float32)
    coverage_fraction = np.zeros((num_source_runs,), dtype=np.float32)
    expected_frame_support = np.zeros((num_source_runs,), dtype=np.float32)
    match_rate = np.zeros((num_source_runs,), dtype=np.float32)
    mean_cost = np.zeros((num_source_runs,), dtype=np.float32)
    confidence_cost_term = np.zeros((num_source_runs,), dtype=np.float32)
    confidence_margin_term = np.zeros((num_source_runs,), dtype=np.float32)
    confidence_type_term = np.zeros((num_source_runs,), dtype=np.float32)
    confidence_match_term = np.zeros((num_source_runs,), dtype=np.float32)

    if "run_occ_viterbi" in alignment:
        projected[src_run_index] = np.asarray(alignment["run_occ_viterbi"], dtype=np.float32)
        projected_weighted[src_run_index] = np.asarray(alignment["run_occ_weighted"], dtype=np.float32)
        aligned_target[src_run_index] = np.asarray(alignment["run_occ_viterbi"], dtype=np.float32)
        coverage[src_run_index] = (
            np.asarray(alignment["run_occ_viterbi"], dtype=np.float32) > 0.0
        ).astype(np.float32, copy=False)
        coverage_binary[src_run_index] = coverage[src_run_index]
        expected_frame_support[src_run_index] = np.clip(src_durations_valid.astype(np.float32), 1.0, None)
        coverage_fraction[src_run_index] = np.maximum(
            np.asarray(alignment["run_occ_viterbi"], dtype=np.float32)
            / np.clip(expected_frame_support[src_run_index], 1.0, None),
            0.0,
        ).astype(np.float32, copy=False)
        match_rate[src_run_index] = np.asarray(alignment["run_match_rate"], dtype=np.float32)
        mean_cost[src_run_index] = np.asarray(alignment["run_mean_cost"], dtype=np.float32)
        confidence_local[src_run_index] = np.asarray(alignment["run_confidence_local"], dtype=np.float32)
        confidence_coarse[src_run_index] = np.asarray(alignment["run_confidence_coarse"], dtype=np.float32)
        confidence_cost_term[src_run_index] = np.asarray(
            alignment.get("run_confidence_cost_term", np.zeros((src_run_index.size,), dtype=np.float32)),
            dtype=np.float32,
        )
        confidence_margin_term[src_run_index] = np.asarray(
            alignment.get("run_confidence_margin_term", np.zeros((src_run_index.size,), dtype=np.float32)),
            dtype=np.float32,
        )
        confidence_type_term[src_run_index] = np.asarray(
            alignment.get("run_confidence_type_term", np.zeros((src_run_index.size,), dtype=np.float32)),
            dtype=np.float32,
        )
        confidence_match_term[src_run_index] = np.asarray(
            alignment.get("run_confidence_match_term", np.zeros((src_run_index.size,), dtype=np.float32)),
            dtype=np.float32,
        )
        unmatched_speech_ratio = float(alignment.get("unmatched_speech_ratio", 0.0))
        mean_local_confidence_speech = float(alignment.get("mean_local_confidence_speech", 0.0))
        mean_coarse_confidence_speech = float(alignment.get("mean_coarse_confidence_speech", 0.0))
    else:
        valid = (assigned_source >= 0) & (assigned_source < int(src_run_index.shape[0]))
        if bool(valid.any()):
            src_token_idx = assigned_source[valid].astype(np.int64, copy=False)
            run_idx = src_run_index[src_token_idx]
            projected = np.bincount(
                run_idx,
                weights=tgt_durations_valid[valid].astype(np.float32, copy=False),
                minlength=num_source_runs,
            ).astype(np.float32, copy=False)
            projected_weighted = projected.copy()
            aligned_target = np.bincount(
                run_idx,
                minlength=num_source_runs,
            ).astype(np.float32, copy=False)
            exact_mask = (
                (src_units_valid[src_token_idx].astype(np.int64, copy=False) == tgt_units_valid[valid].astype(np.int64, copy=False))
                & ((src_silence_valid[src_token_idx] > 0.5) == (tgt_silence_valid[valid] > 0.5))
            ).astype(np.float32, copy=False)
            exact_match = np.bincount(
                run_idx,
                weights=exact_mask,
                minlength=num_source_runs,
            ).astype(np.float32, copy=False)
            cost_mass = np.bincount(
                run_idx,
                weights=assigned_cost[valid].astype(np.float32, copy=False),
                minlength=num_source_runs,
            ).astype(np.float32, copy=False)

        coverage = np.divide(
            aligned_target,
            np.clip(source_support, 1.0, None),
            out=np.zeros_like(aligned_target),
            where=source_support > 0.0,
        )
        coverage = np.clip(coverage, 0.0, 1.0).astype(np.float32)
        coverage_binary = (aligned_target > 0.0).astype(np.float32, copy=False)
        expected_frame_support = np.clip(source_durations.astype(np.float32), 1.0, None)
        coverage_fraction = np.maximum(
            aligned_target / expected_frame_support,
            0.0,
        ).astype(np.float32, copy=False)
        match_rate = np.divide(
            exact_match,
            np.clip(aligned_target, 1.0, None),
            out=np.zeros_like(exact_match),
            where=aligned_target > 0.0,
        )
        mean_cost = np.divide(
            cost_mass,
            np.clip(aligned_target, 1.0, None),
            out=np.zeros_like(cost_mass),
            where=aligned_target > 0.0,
        )
        mass_agree = np.divide(
            np.minimum(projected, source_durations.astype(np.float32)),
            np.maximum(np.maximum(projected, source_durations.astype(np.float32)), 1.0),
            out=np.zeros_like(projected),
            where=(projected > 0.0) | (source_durations.astype(np.float32) > 0.0),
        )
        confidence_coarse = np.clip(
            (0.70 * np.sqrt(np.clip(mass_agree, 0.0, 1.0))) + (0.15 * coverage) + (0.15 * np.exp(-0.5 * mean_cost)),
            0.0,
            1.0,
        ).astype(np.float32)
        confidence_local = np.clip(
            (0.55 * mass_agree) + (0.30 * match_rate) + (0.15 * np.exp(-mean_cost)),
            0.0,
            1.0,
        ).astype(np.float32)
        confidence_cost_term = np.exp(-mean_cost).astype(np.float32, copy=False)
        confidence_margin_term = coverage_fraction.astype(np.float32, copy=False)
        confidence_type_term = mass_agree.astype(np.float32, copy=False)
        confidence_match_term = match_rate.astype(np.float32, copy=False)

        source_is_silence = source_silence_mask > 0.5
        projected[source_is_silence] = np.maximum(projected[source_is_silence], 1.0).astype(np.float32)
        projected_weighted[source_is_silence] = np.maximum(projected_weighted[source_is_silence], 1.0).astype(np.float32)
        confidence_local[source_is_silence] = 0.0
        sil_floor = np.float32(0.15)
        sil_shape = np.clip(source_durations.astype(np.float32) / 3.0, 0.0, 1.0)
        confidence_coarse[source_is_silence] = np.clip(
            np.maximum(
                sil_floor,
                0.5 * confidence_coarse[source_is_silence] * sil_shape[source_is_silence],
            ),
            sil_floor,
            np.float32(0.35),
        ).astype(np.float32)

        speech_zero = (~source_is_silence) & (projected <= 0.0)
        if bool(alignment_soft_repair or allow_source_self_target_fallback):
            projected[speech_zero] = source_durations[speech_zero].astype(np.float32)
            projected_weighted[speech_zero] = source_durations[speech_zero].astype(np.float32)
        confidence_local[speech_zero] = 0.0
        confidence_coarse[speech_zero] = 0.0
        speech_run_count = int(np.count_nonzero(~source_is_silence))
        unmatched_speech_ratio = (
            float(np.count_nonzero(speech_zero)) / float(max(1, speech_run_count))
            if speech_run_count > 0
            else 0.0
        )
        mean_local_confidence_speech = (
            float(np.mean(confidence_local[~source_is_silence], dtype=np.float32))
            if speech_run_count > 0
            else 0.0
        )
        mean_coarse_confidence_speech = (
            float(np.mean(confidence_coarse[~source_is_silence], dtype=np.float32))
            if speech_run_count > 0
            else 0.0
        )

    source_is_silence = source_silence_mask > 0.5
    confidence_local[source_is_silence] = 0.0
    sil_floor = np.float32(0.15)
    sil_shape = np.clip(source_durations.astype(np.float32) / 3.0, 0.0, 1.0)
    confidence_coarse[source_is_silence] = np.clip(
        np.maximum(
            sil_floor,
            0.5 * confidence_coarse[source_is_silence] * sil_shape[source_is_silence],
        ),
        sil_floor,
        np.float32(0.35),
    ).astype(np.float32)

    return {
        "projected": projected.astype(np.float32),
        "projected_weighted": projected_weighted.astype(np.float32),
        "confidence_local": confidence_local.astype(np.float32),
        "confidence_coarse": confidence_coarse.astype(np.float32),
        "coverage_binary": coverage_binary.astype(np.float32),
        "coverage_fraction": coverage_fraction.astype(np.float32),
        "expected_frame_support": expected_frame_support.astype(np.float32),
        "confidence_cost_term": confidence_cost_term.astype(np.float32),
        "confidence_margin_term": confidence_margin_term.astype(np.float32),
        "confidence_type_term": confidence_type_term.astype(np.float32),
        "confidence_match_term": confidence_match_term.astype(np.float32),
        "coverage": coverage.astype(np.float32),
        "match_rate": match_rate.astype(np.float32),
        "mean_cost": mean_cost.astype(np.float32),
        "unmatched_speech_ratio": np.float32(unmatched_speech_ratio),
        "mean_local_confidence_speech": np.float32(mean_local_confidence_speech),
        "mean_coarse_confidence_speech": np.float32(mean_coarse_confidence_speech),
        "alignment_kind": alignment_kind,
        "alignment_mode": str(
            alignment.get("alignment_mode", alignment_kind or _ALIGNMENT_MODE_DISCRETE)
            or (alignment_kind or _ALIGNMENT_MODE_DISCRETE)
        ),
        "alignment_source": str(alignment.get("alignment_source", "") or ""),
        "alignment_version": str(alignment.get("alignment_version", "") or ""),
        "confidence_kind": str(alignment.get("confidence_kind", "heuristic_v1") or "heuristic_v1"),
        "confidence_formula_version": str(
            alignment.get("confidence_formula_version", "heuristic_margin_cost_type_v1")
            or "heuristic_margin_cost_type_v1"
        ),
        "assigned_source": assigned_source.astype(np.int64),
        "assigned_cost": assigned_cost.astype(np.float32),
        "source_valid_run_index": src_run_index.astype(np.int64),
        "target_valid_run_index": tgt_run_index.astype(np.int64),
        "target_valid_observation_index": np.asarray(
            alignment.get("target_valid_observation_index", tgt_run_index),
            dtype=np.int64,
        ),
        "run_occ_hard": np.asarray(
            alignment.get("run_occ_hard", alignment.get("run_occ_viterbi", projected[src_run_index] if src_run_index.size > 0 else np.zeros((0,), dtype=np.float32))),
            dtype=np.float32,
        ),
        "run_occ_weighted": np.asarray(
            alignment.get("run_occ_weighted", projected_weighted[src_run_index] if src_run_index.size > 0 else np.zeros((0,), dtype=np.float32)),
            dtype=np.float32,
        ),
        "posterior_kind": str(alignment.get("posterior_kind", "none") or "none"),
        "run_occ_expected": np.asarray(
            alignment.get(
                "run_occ_expected",
                alignment.get(
                    "run_occ_viterbi",
                    projected[src_run_index] if src_run_index.size > 0 else np.zeros((0,), dtype=np.float32),
                ),
            ),
            dtype=np.float32,
        ),
        "run_occ_expected_is_hard_proxy": np.asarray(
            alignment.get(
                "run_occ_expected_is_hard_proxy",
                np.asarray([1], dtype=np.int64) if str(alignment.get("posterior_kind", "none") or "none") == "none" else np.asarray([0], dtype=np.int64),
            ),
            dtype=np.int64,
        ),
        "run_occ_expected_semantics": str(
            alignment.get(
                "run_occ_expected_semantics",
                "hard_viterbi_proxy" if str(alignment.get("posterior_kind", "none") or "none") == "none" else "",
            )
            or ("hard_viterbi_proxy" if str(alignment.get("posterior_kind", "none") or "none") == "none" else "")
        ),
        "run_entropy": np.asarray(
            alignment.get("run_entropy", np.zeros((src_run_index.size,), dtype=np.float32)),
            dtype=np.float32,
        ),
        "run_posterior_mass_on_path": np.asarray(
            alignment.get("run_posterior_mass_on_path", np.zeros((src_run_index.size,), dtype=np.float32)),
            dtype=np.float32,
        ),
        "run_posterior_mass_on_path_is_hard_proxy": np.asarray(
            alignment.get("run_posterior_mass_on_path_is_hard_proxy", np.asarray([0], dtype=np.int64)),
            dtype=np.int64,
        ),
        "source_frame_count": np.asarray(
            alignment.get("source_frame_count", np.zeros((src_run_index.size,), dtype=np.float32)),
            dtype=np.float32,
        ),
        "source_frame_weight_sum": np.asarray(
            alignment.get("source_frame_weight_sum", np.zeros((src_run_index.size,), dtype=np.float32)),
            dtype=np.float32,
        ),
        "source_run_proto": np.asarray(
            alignment.get("source_run_proto", np.zeros((src_run_index.size, 0), dtype=np.float32)),
            dtype=np.float32,
        ),
        "source_run_proto_var": np.asarray(
            alignment.get("source_run_proto_var", np.zeros((src_run_index.size, 0), dtype=np.float32)),
            dtype=np.float32,
        ),
        "source_run_proto_kind": str(
            alignment.get("source_run_proto_kind", "stability_boundary_weighted_mean_v1")
            or "stability_boundary_weighted_mean_v1"
        ),
        "source_run_proto_weighting_uses_stability": np.asarray(
            alignment.get("source_run_proto_weighting_uses_stability", np.asarray([0], dtype=np.int64)),
            dtype=np.int64,
        ),
        "source_run_proto_weighting_uses_boundary": np.asarray(
            alignment.get("source_run_proto_weighting_uses_boundary", np.asarray([0], dtype=np.int64)),
            dtype=np.int64,
        ),
        "source_progress_kind": str(alignment.get("source_progress_kind", "") or ""),
        "target_progress_kind": str(alignment.get("target_progress_kind", "") or ""),
        "alignment_lambda_emb": np.asarray(
            alignment.get("alignment_lambda_emb", np.asarray([np.nan], dtype=np.float32)),
            dtype=np.float32,
        ),
        "alignment_lambda_type": np.asarray(
            alignment.get("alignment_lambda_type", np.asarray([np.nan], dtype=np.float32)),
            dtype=np.float32,
        ),
        "alignment_lambda_band": np.asarray(
            alignment.get("alignment_lambda_band", np.asarray([np.nan], dtype=np.float32)),
            dtype=np.float32,
        ),
        "alignment_lambda_unit": np.asarray(
            alignment.get("alignment_lambda_unit", np.asarray([np.nan], dtype=np.float32)),
            dtype=np.float32,
        ),
        "alignment_band_ratio": np.asarray(
            alignment.get("alignment_band_ratio", np.asarray([np.nan], dtype=np.float32)),
            dtype=np.float32,
        ),
        "alignment_bad_cost_threshold": np.asarray(
            alignment.get("alignment_bad_cost_threshold", np.asarray([np.nan], dtype=np.float32)),
            dtype=np.float32,
        ),
        "alignment_band_width": np.asarray(
            alignment.get("alignment_band_width", np.asarray([-1], dtype=np.int64)),
            dtype=np.int64,
        ),
        "global_path_cost": np.asarray(
            alignment.get("global_path_cost", np.asarray([0.0], dtype=np.float32)),
            dtype=np.float32,
        ),
    }


__all__ = [
    "ContinuousRunAligner",
    "CONTINUOUS_ALIGNER_VERSION",
    "align_target_frames_to_source_runs",
    "align_target_runs_to_source_discrete",
    "align_target_to_source",
    "as_float32_1d",
    "as_float32_2d",
    "as_int64_1d",
    "filter_valid_runs",
    "project_target_runs_onto_source",
    "resolve_precomputed_alignment",
    "resolve_run_silence_mask",
]

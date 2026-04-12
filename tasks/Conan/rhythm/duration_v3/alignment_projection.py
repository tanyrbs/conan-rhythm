from __future__ import annotations

import numpy as np


CONTINUOUS_ALIGNER_VERSION = "2026-04-13"


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
    ) -> None:
        self.lambda_emb = float(lambda_emb)
        self.lambda_type = float(lambda_type)
        self.lambda_band = float(lambda_band)
        self.lambda_unit = float(lambda_unit)
        self.band_width = None if band_width is None else int(max(1, band_width))
        self.band_ratio = float(max(0.0, band_ratio))
        self.bad_cost_threshold = float(max(0.0, bad_cost_threshold))

    def _resolve_band_width(self, *, num_source: int) -> int:
        if self.band_width is not None:
            return int(self.band_width)
        return int(max(16, round(self.band_ratio * float(max(1, num_source)))))

    @staticmethod
    def _build_source_run_prototypes(
        *,
        source_frame_states: np.ndarray,
        source_frame_to_run: np.ndarray,
        num_source_runs: int,
        source_valid_run_index: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
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

        proto = np.zeros((int(num_source_runs), int(frame_states.shape[1])), dtype=np.float32)
        count = np.zeros((int(num_source_runs),), dtype=np.float32)
        np.add.at(proto, frame_to_run, frame_states)
        np.add.at(count, frame_to_run, 1.0)
        missing = np.nonzero(count <= 0.0)[0]
        if missing.size > 0:
            raise RuntimeError(
                "continuous aligner requires source frame support for every valid run; "
                f"missing run ids={missing.tolist()}"
            )
        proto /= count[:, None]
        return proto.astype(np.float32), count.astype(np.float32)

    def build_local_cost(
        self,
        source_run_proto: np.ndarray,
        source_run_types: np.ndarray,
        target_frame_state: np.ndarray,
        target_frame_speech_prob: np.ndarray,
        target_frame_unit_hint: np.ndarray | None = None,
        source_run_units: np.ndarray | None = None,
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
        target_progress = np.zeros((num_target, 1), dtype=np.float32)
        source_progress = np.zeros((1, num_source), dtype=np.float32)
        if num_target > 1:
            target_progress[:, 0] = np.linspace(0.0, 1.0, num_target, dtype=np.float32)
        if num_source > 1:
            source_progress[0, :] = np.linspace(0.0, 1.0, num_source, dtype=np.float32)
        band_slack = min(
            0.45,
            max(
                1.0 / float(max(1, num_source - 1)),
                float(self._resolve_band_width(num_source=num_source)) / float(max(1, num_source - 1)),
            ),
        )
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
    ) -> tuple[np.ndarray, np.ndarray, np.float32, np.ndarray]:
        local_cost = as_float32_2d(local_cost, name="local_cost")
        num_target, num_source = local_cost.shape
        if num_target <= 0 or num_source <= 0:
            raise RuntimeError("continuous aligner requires non-empty target observations and source runs.")
        if num_target < num_source:
            raise RuntimeError(
                "continuous aligner requires at least one target frame per source run under strict monotonicity: "
                f"target_frames={num_target}, source_runs={num_source}"
            )

        band_width = self._resolve_band_width(num_source=num_source)
        inf = np.float32(1.0e12)
        dp_prev = np.full((num_source,), inf, dtype=np.float32)
        dp_curr = np.full((num_source,), inf, dtype=np.float32)
        back = np.full((num_target, num_source), 255, dtype=np.uint8)
        margin = np.zeros((num_target, num_source), dtype=np.float32)

        if not np.isfinite(local_cost[0, 0]):
            raise RuntimeError("continuous aligner local_cost[0, 0] is not finite.")
        dp_prev[0] = local_cost[0, 0]
        back[0, 0] = 0

        for tgt_idx in range(1, num_target):
            dp_curr.fill(inf)
            hard_left = max(0, num_source - num_target + tgt_idx)
            hard_right = min(num_source - 1, tgt_idx)
            center = int(round((tgt_idx * max(0, num_source - 1)) / float(max(1, num_target - 1))))
            left = max(hard_left, center - band_width)
            right = min(hard_right, center + band_width)
            if left > right:
                left, right = hard_left, hard_right
            for src_idx in range(left, right + 1):
                stay = dp_prev[src_idx]
                advance = dp_prev[src_idx - 1] if src_idx > 0 else inf
                if src_idx <= 0:
                    best_prev = stay
                    alt_prev = inf
                    move = 0
                elif advance <= stay:
                    best_prev = advance
                    alt_prev = stay
                    move = 1
                else:
                    best_prev = stay
                    alt_prev = advance
                    move = 0
                if not np.isfinite(best_prev):
                    continue
                cost = local_cost[tgt_idx, src_idx]
                if not np.isfinite(cost):
                    continue
                dp_curr[src_idx] = cost + best_prev
                back[tgt_idx, src_idx] = move
                margin[tgt_idx, src_idx] = (
                    np.float32(abs(stay - advance))
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
            move = int(back[tgt_idx, src_idx])
            if move == 1:
                src_idx -= 1
            elif move != 0:
                raise RuntimeError(
                    "continuous aligner backtrace encountered an invalid predecessor marker "
                    f"at target={tgt_idx}, source={src_idx}: move={move}"
                )
        if src_idx != 0:
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

        source_run_proto, source_frame_count = self._build_source_run_prototypes(
            source_frame_states=source_frame_states,
            source_frame_to_run=source_frame_to_run,
            num_source_runs=int(source_run_units.shape[0]),
            source_valid_run_index=source_valid_run_index,
        )
        local_cost = self.build_local_cost(
            source_run_proto=source_run_proto,
            source_run_types=source_run_types,
            target_frame_state=target_frame_states_valid,
            target_frame_speech_prob=target_frame_speech_prob_valid,
            target_frame_unit_hint=target_frame_unit_hint_valid,
            source_run_units=source_run_units,
        )
        assigned_run, assigned_local_cost, global_path_cost, path_margin = self.viterbi_align(local_cost)
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
        summary.update(
            {
                "alignment_kind": "continuous_viterbi_v1",
                "alignment_source": "run_state_viterbi",
                "alignment_version": CONTINUOUS_ALIGNER_VERSION,
                "posterior_kind": "none",
                "assigned_source": assigned_run.astype(np.int64),
                "assigned_cost": assigned_local_cost.astype(np.float32),
                "target_valid_observation_index": target_valid_index.astype(np.int64),
                "source_frame_count": source_frame_count.astype(np.float32),
                "source_run_proto": source_run_proto.astype(np.float32),
                "run_occ_expected": summary["run_occ_weighted"],
                "run_entropy": np.zeros_like(summary["run_occ_viterbi"], dtype=np.float32),
                "run_posterior_mass_on_path": np.where(
                    np.asarray(summary["run_occ_viterbi"], dtype=np.float32) > 0.0,
                    1.0,
                    0.0,
                ).astype(np.float32),
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
    target_frame_speech_prob: np.ndarray | None = None,
    target_frame_weight: np.ndarray | None = None,
    target_frame_valid: np.ndarray | None = None,
    target_frame_unit_hint: np.ndarray | None = None,
    precomputed_alignment=None,
) -> dict[str, np.ndarray | np.float32 | str] | None:
    resolved = resolve_precomputed_alignment(
        precomputed_alignment=precomputed_alignment,
        num_target=int(target_units.shape[0]),
    )
    if resolved is not None:
        alignment_kind = "continuous_precomputed"
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
            "alignment_source": alignment_source,
            "alignment_version": alignment_version,
        }
    if source_frame_states is None or target_frame_states is None or source_frame_to_run is None:
        return None
    del source_durations, target_durations, target_silence
    aligner = ContinuousRunAligner()
    return aligner.align(
        source_run_units=source_units,
        source_run_types=source_silence,
        source_frame_states=source_frame_states,
        source_frame_to_run=source_frame_to_run,
        target_frame_states=target_frame_states,
        source_valid_run_index=source_valid_run_index,
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
    target_frame_speech_prob: np.ndarray | None = None,
    target_frame_weight: np.ndarray | None = None,
    target_frame_valid: np.ndarray | None = None,
    target_frame_unit_hint: np.ndarray | None = None,
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
            target_frame_speech_prob=target_frame_speech_prob,
            target_frame_weight=target_frame_weight,
            target_frame_valid=target_frame_valid,
            target_frame_unit_hint=target_frame_unit_hint,
            precomputed_alignment=precomputed_alignment,
        )
        if continuous is not None:
            return continuous
        raise RuntimeError(
            "rhythm_v3_use_continuous_alignment=true requires either continuous_precomputed metadata "
            "or explicit source_frame_states/source_frame_to_run/target_frame_states sidecars."
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
    target_frame_speech_prob: np.ndarray | None = None,
    target_frame_weight: np.ndarray | None = None,
    target_frame_valid: np.ndarray | None = None,
    target_frame_unit_hint: np.ndarray | None = None,
    precomputed_alignment=None,
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
        source_valid_run_index=src_run_index,
        target_frame_speech_prob=target_frame_speech_prob,
        target_frame_weight=target_frame_weight,
        target_frame_valid=target_frame_valid,
        target_frame_unit_hint=target_frame_unit_hint,
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
    match_rate = np.zeros((num_source_runs,), dtype=np.float32)
    mean_cost = np.zeros((num_source_runs,), dtype=np.float32)

    if "run_occ_viterbi" in alignment:
        projected[src_run_index] = np.asarray(alignment["run_occ_viterbi"], dtype=np.float32)
        projected_weighted[src_run_index] = np.asarray(alignment["run_occ_weighted"], dtype=np.float32)
        aligned_target[src_run_index] = np.asarray(alignment["run_occ_viterbi"], dtype=np.float32)
        coverage[src_run_index] = (
            np.asarray(alignment["run_occ_viterbi"], dtype=np.float32) > 0.0
        ).astype(np.float32, copy=False)
        match_rate[src_run_index] = np.asarray(alignment["run_match_rate"], dtype=np.float32)
        mean_cost[src_run_index] = np.asarray(alignment["run_mean_cost"], dtype=np.float32)
        confidence_local[src_run_index] = np.asarray(alignment["run_confidence_local"], dtype=np.float32)
        confidence_coarse[src_run_index] = np.asarray(alignment["run_confidence_coarse"], dtype=np.float32)
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
        "coverage": coverage.astype(np.float32),
        "match_rate": match_rate.astype(np.float32),
        "mean_cost": mean_cost.astype(np.float32),
        "unmatched_speech_ratio": np.float32(unmatched_speech_ratio),
        "mean_local_confidence_speech": np.float32(mean_local_confidence_speech),
        "mean_coarse_confidence_speech": np.float32(mean_coarse_confidence_speech),
        "alignment_kind": alignment_kind,
        "alignment_source": str(alignment.get("alignment_source", "") or ""),
        "alignment_version": str(alignment.get("alignment_version", "") or ""),
        "assigned_source": assigned_source.astype(np.int64),
        "assigned_cost": assigned_cost.astype(np.float32),
        "source_valid_run_index": src_run_index.astype(np.int64),
        "target_valid_run_index": tgt_run_index.astype(np.int64),
        "target_valid_observation_index": np.asarray(
            alignment.get("target_valid_observation_index", tgt_run_index),
            dtype=np.int64,
        ),
        "posterior_kind": str(alignment.get("posterior_kind", "none") or "none"),
        "run_occ_expected": np.asarray(
            alignment.get("run_occ_expected", projected[src_run_index] if src_run_index.size > 0 else np.zeros((0,), dtype=np.float32)),
            dtype=np.float32,
        ),
        "run_entropy": np.asarray(
            alignment.get("run_entropy", np.zeros((src_run_index.size,), dtype=np.float32)),
            dtype=np.float32,
        ),
        "run_posterior_mass_on_path": np.asarray(
            alignment.get("run_posterior_mass_on_path", np.zeros((src_run_index.size,), dtype=np.float32)),
            dtype=np.float32,
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

from __future__ import annotations

import numpy as np


def as_int64_1d(value) -> np.ndarray:
    return np.asarray(value, dtype=np.int64).reshape(-1)


def as_float32_1d(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


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
) -> tuple[np.ndarray, np.ndarray] | None:
    del (
        source_units,
        source_durations,
        source_silence,
        target_units,
        target_durations,
        target_silence,
    )
    return None


def align_target_to_source(
    *,
    source_units: np.ndarray,
    source_durations: np.ndarray,
    source_silence: np.ndarray,
    target_units: np.ndarray,
    target_durations: np.ndarray,
    target_silence: np.ndarray,
    use_continuous_alignment: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if use_continuous_alignment:
        continuous = align_target_frames_to_source_runs(
            source_units=source_units,
            source_durations=source_durations,
            source_silence=source_silence,
            target_units=target_units,
            target_durations=target_durations,
            target_silence=target_silence,
        )
        if continuous is not None:
            return continuous
    return align_target_runs_to_source_discrete(
        source_units=source_units,
        source_durations=source_durations,
        source_silence=source_silence,
        target_units=target_units,
        target_durations=target_durations,
        target_silence=target_silence,
    )


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
) -> dict[str, np.ndarray]:
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
        _,
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

    assigned_source, assigned_cost = align_target_to_source(
        source_units=src_units_valid,
        source_durations=src_durations_valid,
        source_silence=src_silence_valid,
        target_units=tgt_units_valid,
        target_durations=tgt_durations_valid,
        target_silence=tgt_silence_valid,
        use_continuous_alignment=use_continuous_alignment,
    )

    num_source_runs = int(source_units.shape[0])
    projected = np.zeros((num_source_runs,), dtype=np.float32)
    aligned_target = np.zeros((num_source_runs,), dtype=np.float32)
    exact_match = np.zeros((num_source_runs,), dtype=np.float32)
    cost_mass = np.zeros((num_source_runs,), dtype=np.float32)
    source_support = np.zeros((num_source_runs,), dtype=np.float32)
    source_support[src_run_index] = 1.0

    for tgt_idx, src_token_idx in enumerate(assigned_source.tolist()):
        if src_token_idx < 0 or src_token_idx >= int(src_run_index.shape[0]):
            continue
        safe_src_token_idx = int(src_token_idx)
        run_idx = int(src_run_index[safe_src_token_idx])
        projected[run_idx] += float(tgt_durations_valid[tgt_idx])
        aligned_target[run_idx] += 1.0
        if (
            int(src_units_valid[safe_src_token_idx]) == int(tgt_units_valid[tgt_idx])
            and int(src_silence_valid[safe_src_token_idx] > 0.5) == int(tgt_silence_valid[tgt_idx] > 0.5)
        ):
            exact_match[run_idx] += 1.0
        cost_mass[run_idx] += float(assigned_cost[tgt_idx])

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
    confidence_local[speech_zero] = 0.0
    confidence_coarse[speech_zero] = np.minimum(confidence_coarse[speech_zero], 0.20).astype(np.float32)
    return {
        "projected": projected.astype(np.float32),
        "confidence_local": confidence_local.astype(np.float32),
        "confidence_coarse": confidence_coarse.astype(np.float32),
        "coverage": coverage.astype(np.float32),
        "match_rate": match_rate.astype(np.float32),
        "mean_cost": mean_cost.astype(np.float32),
        "assigned_source": assigned_source.astype(np.int64),
        "assigned_cost": assigned_cost.astype(np.float32),
        "source_valid_run_index": src_run_index.astype(np.int64),
        "target_valid_run_index": np.nonzero(as_float32_1d(target_valid_mask) > 0.5)[0].astype(np.int64),
    }


__all__ = [
    "align_target_frames_to_source_runs",
    "align_target_runs_to_source_discrete",
    "align_target_to_source",
    "as_float32_1d",
    "as_int64_1d",
    "filter_valid_runs",
    "project_target_runs_onto_source",
    "resolve_run_silence_mask",
]

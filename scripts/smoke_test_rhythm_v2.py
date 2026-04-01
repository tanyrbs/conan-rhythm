import numpy as np
import torch
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from modules.Conan.rhythm.factory import build_streaming_rhythm_module_from_hparams
from modules.Conan.rhythm.supervision import (
    build_item_rhythm_bundle,
    build_reference_guided_targets,
    build_reference_teacher_targets,
    build_retimed_mel_target,
)
from modules.Conan.rhythm.unit_frontend import RhythmUnitFrontend
from modules.Conan.rhythm.unitizer import StreamingRunLengthUnitizer
from tasks.Conan.rhythm.metrics import build_rhythm_metric_dict


if __name__ == '__main__':
    hparams = {
        'content_embedding_dim': 128,
        'rhythm_hidden_size': 32,
        'rhythm_trace_bins': 12,
        'rhythm_stats_dim': 6,
        'rhythm_trace_dim': 5,
        'rhythm_trace_horizon': 0.35,
        'rhythm_trace_smooth_kernel': 5,
        'rhythm_max_total_logratio': 0.8,
        'rhythm_max_unit_logratio': 0.6,
        'rhythm_pause_share_max': 0.45,
        'rhythm_projector_min_speech_frames': 1.0,
        'rhythm_projector_max_speech_expand': 3.0,
        'rhythm_projector_tail_hold_units': 2,
        'rhythm_projector_boundary_commit_threshold': 0.45,
    }
    frontend = RhythmUnitFrontend(silent_token=57, separator_aware=True)
    batch = frontend.from_token_lists([
        [1, 1, 1, 2, 2, 57, 3, 3, 4, 4],
        [5, 5, 6, 6, 6, 7, 57, 8, 8, 8],
    ], device=torch.device('cpu'))
    model = build_streaming_rhythm_module_from_hparams(hparams)

    ref_mel = torch.randn(2, 80, 64)
    ref_conditioning = model.encode_reference(ref_mel)
    print('ref descriptor keys:', sorted(ref_conditioning.keys()))
    assert 'global_rate' in ref_conditioning and 'boundary_trace' in ref_conditioning
    assert 'slow_rhythm_memory' in ref_conditioning and 'selector_meta_indices' in ref_conditioning
    assert 'selector_meta_starts' in ref_conditioning and 'selector_meta_ends' in ref_conditioning
    assert batch.sealed_mask.shape == batch.open_run_mask.shape
    assert batch.boundary_confidence.shape == batch.open_run_mask.shape

    dual = model.forward_dual(
        content_units=batch.content_units,
        dur_anchor_src=batch.dur_anchor_src,
        unit_mask=batch.unit_mask,
        open_run_mask=batch.open_run_mask,
        sealed_mask=batch.sealed_mask,
        sep_hint=batch.sep_hint,
        boundary_confidence=batch.boundary_confidence,
        ref_rhythm_stats=ref_conditioning['ref_rhythm_stats'],
        ref_rhythm_trace=ref_conditioning['ref_rhythm_trace'],
        state=model.init_state(batch_size=2, device=torch.device('cpu')),
    )
    offline_exec = dual["offline_execution"]
    algo_teacher = dual["algorithmic_teacher"]
    assert offline_exec.commit_frontier.tolist() == batch.unit_mask.sum(dim=1).long().tolist()
    assert algo_teacher.allocation_tgt.shape == batch.unit_mask.shape
    assert algo_teacher.prefix_clock_tgt.shape == batch.unit_mask.shape

    stream_unitizer = StreamingRunLengthUnitizer(silent_token=57, separator_aware=True)
    unitizer_state = stream_unitizer.init_state(batch_size=1)
    _, unitizer_state = stream_unitizer.step_token_lists([[1, 1, 2, 57]], unitizer_state)
    unitized_step2, unitizer_state = stream_unitizer.step_token_lists([[2, 2, 3, 3]], unitizer_state)
    assert len(unitizer_state.raw_tokens[0]) == 8
    assert unitized_step2[0].units == [1, 2, 2, 3]

    state = model.init_state(batch_size=2, device=torch.device('cpu'))
    out1 = model(
        content_units=batch.content_units,
        dur_anchor_src=batch.dur_anchor_src,
        unit_mask=batch.unit_mask,
        open_run_mask=batch.open_run_mask,
        ref_rhythm_stats=ref_conditioning['ref_rhythm_stats'],
        ref_rhythm_trace=ref_conditioning['ref_rhythm_trace'],
        state=state,
    )
    out2 = model(
        content_units=batch.content_units,
        dur_anchor_src=batch.dur_anchor_src,
        unit_mask=batch.unit_mask,
        open_run_mask=batch.open_run_mask,
        ref_rhythm_stats=ref_conditioning['ref_rhythm_stats'],
        ref_rhythm_trace=ref_conditioning['ref_rhythm_trace'],
        state=out1.next_state,
    )
    assert out1.slot_duration_exec.shape[1] == batch.content_units.shape[1] * 2
    assert out1.slot_is_blank[:, 1::2].sum().item() > 0

    guidance = build_reference_guided_targets(
        dur_anchor_src=batch.dur_anchor_src[0].cpu().numpy(),
        unit_mask=batch.unit_mask[0].cpu().numpy(),
        ref_rhythm_stats=np.array([0.15, 2.0, 3.0, 0.0, 0.2, 0.8], dtype=np.float32),
        ref_rhythm_trace=np.random.randn(12, 5).astype(np.float32),
    )
    teacher = build_reference_teacher_targets(
        dur_anchor_src=batch.dur_anchor_src[0].cpu().numpy(),
        unit_mask=batch.unit_mask[0].cpu().numpy(),
        ref_rhythm_stats=np.array([0.15, 2.0, 3.0, 0.0, 0.2, 0.8], dtype=np.float32),
        ref_rhythm_trace=np.random.randn(12, 5).astype(np.float32),
    )
    retimed = build_retimed_mel_target(
        mel=np.random.randn(10, 80).astype(np.float32),
        dur_anchor_src=batch.dur_anchor_src[0].cpu().numpy(),
        speech_exec_tgt=guidance["rhythm_speech_exec_tgt"],
        pause_exec_tgt=guidance["rhythm_pause_exec_tgt"],
        unit_mask=batch.unit_mask[0].cpu().numpy(),
    )
    cached_bundle = build_item_rhythm_bundle(
        content_tokens=[1, 1, 1, 2, 2, 57, 3, 3],
        mel=np.random.randn(16, 80).astype(np.float32),
        trace_horizon=0.40,
        include_self_targets=True,
        include_teacher_targets=True,
        include_retimed_mel_target=True,
        retimed_mel_target_source="teacher",
    )
    metrics = build_rhythm_metric_dict(
        {
            "rhythm_execution": out1,
            "rhythm_offline_execution": offline_exec,
            "rhythm_algorithmic_teacher": algo_teacher,
            "rhythm_unit_batch": batch,
            "rhythm_state_next": out1.next_state,
            "rhythm_apply_render": 1.0,
            "acoustic_target_is_retimed": False,
        },
        {
            "rhythm_speech_exec_tgt": out1.speech_duration_exec.detach(),
            "rhythm_pause_exec_tgt": out1.pause_after_exec.detach(),
            "rhythm_speech_budget_tgt": out1.planner.speech_budget_win.detach(),
            "rhythm_pause_budget_tgt": out1.planner.pause_budget_win.detach(),
        },
    )
    print('speech_exec shape:', tuple(out1.speech_duration_exec.shape))
    print('pause_exec shape:', tuple(out1.pause_after_exec.shape))
    print('commit_frontier step1:', out1.commit_frontier.tolist())
    print('commit_frontier step2:', out2.commit_frontier.tolist())
    print('phase_ptr step2:', out2.next_state.phase_ptr.tolist())
    print('source_boundary_cue max:', float(out1.planner.source_boundary_cue.max().item()))
    print('offline total corr metric:', float(metrics['rhythm_metric_offline_online_total_corr']))
    print('algorithmic teacher alloc kl:', float(metrics['rhythm_metric_algorithmic_teacher_alloc_kl']))
    print('slow memory shape:', tuple(ref_conditioning['slow_rhythm_memory'].shape))
    print('blank slot ratio metric:', float(metrics['rhythm_metric_blank_slot_ratio_mean']))
    print('retimed mel len:', int(retimed['rhythm_retimed_mel_len'][0]))
    print('retimed frame weight mean:', float(retimed['rhythm_retimed_frame_weight'].mean()))
    print('guidance keys:', sorted(guidance.keys()))
    print('teacher keys:', sorted(teacher.keys()))
    teacher_gap = float(np.abs(teacher['rhythm_teacher_pause_exec_tgt'] - guidance['rhythm_pause_exec_tgt']).sum())
    print('teacher/guidance pause gap:', teacher_gap)
    print('cache version:', int(cached_bundle['rhythm_cache_version'][0]))
    print('trace bins:', int(cached_bundle['rhythm_trace_bins'][0]))
    print('trace horizon:', float(cached_bundle['rhythm_trace_horizon'][0]))
    print('target confidence:', float(cached_bundle['rhythm_target_confidence'][0]))
    print('teacher confidence:', float(cached_bundle['rhythm_teacher_confidence'][0]))
    print('selector starts:', cached_bundle['selector_meta_starts'].tolist())
    print('selector ends:', cached_bundle['selector_meta_ends'].tolist())
    print('phrase groups:', cached_bundle['phrase_group_index'].tolist())
    print('retimed source id:', int(cached_bundle['rhythm_retimed_target_source_id'][0]))
    print('retimed confidence:', float(cached_bundle['rhythm_retimed_target_confidence'][0]))
    print('metric exec total corr:', float(metrics['rhythm_metric_exec_total_corr']))
    print('metric prefix drift l1:', float(metrics['rhythm_metric_prefix_drift_l1']))
    print('metric prefix backlog mean:', float(metrics['rhythm_metric_prefix_backlog_mean']))
    assert teacher_gap >= 0.0
    assert float(metrics['rhythm_metric_exec_total_corr']) > 0.99
    assert float(metrics['rhythm_metric_prefix_drift_l1']) < 1e-6
    offline_corr = float(metrics['rhythm_metric_offline_online_total_corr'])
    assert np.isfinite(offline_corr)
    assert abs(offline_corr) <= 1.0 + 1e-6

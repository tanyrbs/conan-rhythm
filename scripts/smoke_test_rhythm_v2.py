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
        'content_vocab_size': 128,
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
        'rhythm_source_boundary_scale': 0.60,
        'rhythm_source_boundary_scale_train_start': 1.0,
        'rhythm_source_boundary_scale_train_end': 0.60,
        'rhythm_teacher_source_boundary_scale': 0.45,
    }
    frontend = RhythmUnitFrontend(silent_token=57, separator_aware=True)
    batch = frontend.from_token_lists([
        [1, 1, 1, 2, 2, 57, 3, 3, 4, 4, 5, 5],
        [5, 5, 6, 6, 6, 7, 57, 8, 8, 8, 9, 9],
    ], device=torch.device('cpu'))
    prefix_batch = frontend.from_token_lists([
        [1, 1, 1, 2, 2, 57, 3, 3],
        [5, 5, 6, 6, 6, 7, 57, 8],
    ], device=torch.device('cpu'))
    prefix_last = prefix_batch.unit_mask.sum(dim=1).long() - 1
    for b in range(prefix_batch.unit_mask.size(0)):
        li = int(prefix_last[b].item())
        assert int(prefix_batch.open_run_mask[b, li].item()) == 1
        assert float(prefix_batch.sealed_mask[b, li].item()) == 0.0
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
        content_units=prefix_batch.content_units,
        dur_anchor_src=prefix_batch.dur_anchor_src,
        unit_mask=prefix_batch.unit_mask,
        open_run_mask=prefix_batch.open_run_mask,
        sealed_mask=prefix_batch.sealed_mask,
        sep_hint=prefix_batch.sep_hint,
        boundary_confidence=prefix_batch.boundary_confidence,
        ref_rhythm_stats=ref_conditioning['ref_rhythm_stats'],
        ref_rhythm_trace=ref_conditioning['ref_rhythm_trace'],
        state=model.init_state(batch_size=2, device=torch.device('cpu')),
        offline_content_units=batch.content_units,
        offline_dur_anchor_src=batch.dur_anchor_src,
        offline_unit_mask=batch.unit_mask,
        offline_open_run_mask=batch.open_run_mask,
        offline_sealed_mask=batch.sealed_mask,
        offline_sep_hint=batch.sep_hint,
        offline_boundary_confidence=batch.boundary_confidence,
    )
    offline_exec = dual["offline_execution"]
    offline_conf = dual["offline_confidence"]
    algo_teacher = dual["algorithmic_teacher"]
    assert offline_exec.commit_frontier.tolist() == batch.unit_mask.sum(dim=1).long().tolist()
    assert dual["streaming_execution"].speech_duration_exec.size(1) < offline_exec.speech_duration_exec.size(1)
    assert isinstance(offline_conf, dict) and {"overall", "exec", "budget", "prefix", "allocation"} <= set(offline_conf.keys())
    assert algo_teacher.allocation_tgt.shape == batch.unit_mask.shape
    assert algo_teacher.prefix_clock_tgt.shape == batch.unit_mask.shape

    stream_unitizer = StreamingRunLengthUnitizer(silent_token=57, separator_aware=True)
    unitizer_state = stream_unitizer.init_state(batch_size=1)
    _, unitizer_state = stream_unitizer.step_token_lists([[1, 1, 2, 57]], unitizer_state)
    unitized_step2, unitizer_state = stream_unitizer.step_token_lists([[2, 2, 3, 3]], unitizer_state)
    assert unitizer_state.rows[0].units.tolist() == [1, 2, 2, 3]
    assert unitizer_state.rows[0].durations.tolist() == [2, 1, 2, 2]
    assert unitized_step2[0].units == [1, 2, 2, 3]
    frontend_state = frontend.init_stream_state(batch_size=1, device=torch.device("cpu"))
    frontend_step1, frontend_state = frontend.step_content_tensor(
        torch.tensor([[1, 1, 2, 57]], dtype=torch.long),
        frontend_state,
        content_lengths=torch.tensor([4], dtype=torch.long),
    )
    frontend_step2, frontend_state = frontend.step_content_tensor(
        torch.tensor([[2, 2, 3, 3]], dtype=torch.long),
        frontend_state,
        content_lengths=torch.tensor([4], dtype=torch.long),
    )
    assert frontend_step1.content_units[0, :2].tolist() == [1, 2]
    assert frontend_step2.content_units[0, :4].tolist() == [1, 2, 2, 3]

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
    out_hold = model(
        content_units=batch.content_units,
        dur_anchor_src=batch.dur_anchor_src,
        unit_mask=batch.unit_mask,
        open_run_mask=torch.ones_like(batch.open_run_mask),
        ref_rhythm_stats=ref_conditioning['ref_rhythm_stats'],
        ref_rhythm_trace=ref_conditioning['ref_rhythm_trace'],
        state=out2.next_state,
    )
    out_no_source_prior = model(
        content_units=batch.content_units,
        dur_anchor_src=batch.dur_anchor_src,
        unit_mask=batch.unit_mask,
        open_run_mask=batch.open_run_mask,
        ref_rhythm_stats=ref_conditioning['ref_rhythm_stats'],
        ref_rhythm_trace=ref_conditioning['ref_rhythm_trace'],
        state=model.init_state(batch_size=2, device=torch.device('cpu')),
        source_boundary_scale_override=0.0,
    )
    assert out1.slot_duration_exec.shape[1] == batch.content_units.shape[1] * 2
    assert out1.slot_is_blank[:, 1::2].sum().item() > 0
    assert torch.equal(out1.blank_slot_duration_exec, out1.slot_duration_exec)
    assert torch.equal(out1.blank_slot_is_blank, out1.slot_is_blank)
    assert float(out_no_source_prior.planner.source_boundary_cue.abs().max().item()) == 0.0
    assert torch.all(out2.next_state.phase_ptr + 1e-6 >= out1.next_state.phase_ptr)
    assert torch.all(out_hold.next_state.phase_ptr + 1e-6 >= out2.next_state.phase_ptr)
    assert torch.equal(out_hold.next_state.commit_frontier, out2.next_state.commit_frontier)
    assert out1.pause_after_exec.requires_grad
    frame_plan = out1.frame_plan
    assert frame_plan is not None
    assert frame_plan.frame_src_index.shape == frame_plan.total_mask.shape
    assert frame_plan.blank_mask.shape == frame_plan.total_mask.shape
    assert frame_plan.speech_mask.shape == frame_plan.total_mask.shape
    assert frame_plan.frame_phase_features.size(-1) == 5
    rounded_slot_frames = (torch.round(out1.slot_duration_exec.float()).clamp_min(0.0) * out1.slot_mask.float()).sum(dim=1)
    assert torch.equal(frame_plan.total_mask.sum(dim=1).long(), rounded_slot_frames.long())
    rounded_pause_frames = (torch.round(out1.pause_after_exec.float()).clamp_min(0.0) * batch.unit_mask.float()).sum(dim=1)
    assert torch.equal(frame_plan.blank_mask.sum(dim=1).long(), rounded_pause_frames.long())
    valid_blank = (frame_plan.blank_mask > 0.5) & (frame_plan.total_mask > 0.5)
    valid_speech = (frame_plan.blank_mask <= 0.5) & (frame_plan.total_mask > 0.5)
    assert torch.all(frame_plan.frame_src_index[valid_blank] < 0)
    assert torch.all(frame_plan.frame_src_index[valid_speech] >= 0)

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
            "rhythm_state_prev": state,
            "rhythm_state_next": out1.next_state,
            "rhythm_apply_render": 1.0,
            "acoustic_target_is_retimed": False,
            "rhythm_source_boundary_scale": torch.full((2, 1), 0.6),
            "rhythm_teacher_source_boundary_scale": torch.full((2, 1), 0.45),
            "rhythm_offline_confidence": offline_conf["overall"],
            "rhythm_offline_confidence_exec": offline_conf["exec"],
            "rhythm_offline_confidence_budget": offline_conf["budget"],
            "rhythm_offline_confidence_prefix": offline_conf["prefix"],
            "rhythm_offline_confidence_allocation": offline_conf["allocation"],
            "rhythm_frame_plan": out1.frame_plan,
            "acoustic_target_source": "online",
            "rhythm_exec": torch.tensor(0.6),
            "rhythm_stream_state": torch.tensor(0.2),
            "base": torch.tensor(1.4),
            "pitch": torch.tensor(0.3),
            "L_exec_speech": torch.tensor(0.11),
            "L_exec_pause": torch.tensor(0.09),
            "L_budget": torch.tensor(0.03),
            "L_cumplan": torch.tensor(0.05),
            "L_prefix_state": torch.tensor(0.05),
            "L_rhythm_exec": torch.tensor(0.2),
            "L_stream_state": torch.tensor(0.1),
            "L_base": torch.tensor(1.4),
            "L_pitch": torch.tensor(0.3),
        },
        {
            "rhythm_speech_exec_tgt": out1.speech_duration_exec.detach(),
            "rhythm_pause_exec_tgt": out1.pause_after_exec.detach(),
            "rhythm_speech_budget_tgt": out1.planner.speech_budget_win.detach(),
            "rhythm_pause_budget_tgt": out1.planner.pause_budget_win.detach(),
        },
    )
    minimal_metrics = build_rhythm_metric_dict(
        {
            "rhythm_execution": out1,
            "rhythm_unit_batch": batch,
            "rhythm_frame_plan": out1.frame_plan,
            "acoustic_target_source": "cached",
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
    print('source boundary scale metric:', float(metrics['rhythm_metric_source_boundary_scale_mean']))
    print('offline total corr metric:', float(metrics['rhythm_metric_offline_online_total_corr']))
    print('offline stream prefix ratio:', float(prefix_batch.unit_mask.sum().item() / batch.unit_mask.sum().item()))
    print('algorithmic teacher alloc kl:', float(metrics['rhythm_metric_algorithmic_teacher_alloc_kl']))
    print('offline confidence exec:', float(metrics['rhythm_metric_offline_confidence_exec_mean']))
    print('offline confidence coverage:', float(metrics['rhythm_metric_offline_confidence_component_coverage']))
    print('slow memory shape:', tuple(ref_conditioning['slow_rhythm_memory'].shape))
    print('blank slot ratio metric:', float(metrics['rhythm_metric_blank_slot_ratio_mean']))
    print('frame plan present:', float(metrics['rhythm_metric_frame_plan_present']))
    print('frame plan blank/src consistency:', float(metrics['rhythm_metric_frame_plan_blank_src_consistency']))
    print('frame plan speech/src consistency:', float(metrics['rhythm_metric_frame_plan_speech_src_consistency']))
    print('acoustic target source id:', float(metrics['rhythm_metric_acoustic_target_source_id']))
    print('phase nonretro rate:', float(metrics['rhythm_metric_phase_nonretro_rate']))
    print('alias L_rhythm_exec:', float(metrics['rhythm_metric_alias_L_rhythm_exec']))
    print('compact rhythm_exec:', float(metrics['rhythm_metric_compact_rhythm_exec']))
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
    print('minimal metric source id:', float(minimal_metrics['rhythm_metric_acoustic_target_source_id']))
    print('minimal metric frame plan present:', float(minimal_metrics['rhythm_metric_frame_plan_present']))
    assert "rhythm_blank_exec_tgt" in guidance and "rhythm_teacher_blank_exec_tgt" in teacher
    assert "rhythm_pause_exec_tgt" in cached_bundle and "rhythm_pause_budget_tgt" in cached_bundle
    assert "rhythm_blank_exec_tgt" in cached_bundle and "rhythm_blank_budget_tgt" in cached_bundle
    assert teacher_gap >= 0.0
    assert float(metrics['rhythm_metric_exec_total_corr']) > 0.99
    assert float(metrics['rhythm_metric_prefix_drift_l1']) < 1e-6
    assert float(metrics['rhythm_metric_frame_plan_present']) == 1.0
    assert float(metrics['rhythm_metric_frame_plan_blank_src_consistency']) > 0.99
    assert float(metrics['rhythm_metric_frame_plan_speech_src_consistency']) > 0.99
    assert float(metrics['rhythm_metric_acoustic_target_source_is_online']) == 1.0
    assert float(metrics['rhythm_metric_acoustic_target_source_unknown']) == 0.0
    assert float(metrics['rhythm_metric_phase_nonretro_rate']) == 1.0
    assert float(metrics['rhythm_metric_phase_delta_min']) >= -1e-6
    assert float(metrics['rhythm_metric_offline_confidence_component_coverage']) == 1.0
    assert "rhythm_metric_offline_confidence_component_std" in metrics
    assert "rhythm_metric_alias_L_rhythm_exec" in metrics
    assert "rhythm_metric_alias_L_stream_state" in metrics
    assert "rhythm_metric_alias_L_base" in metrics
    assert "rhythm_metric_alias_L_pitch" in metrics
    assert "rhythm_metric_compact_rhythm_exec" in metrics
    assert "rhythm_metric_compact_stream_state" in metrics
    assert "rhythm_metric_compact_base" in metrics
    assert "rhythm_metric_compact_pitch" in metrics
    assert abs(float(metrics["rhythm_metric_alias_L_rhythm_exec"]) - 0.2) < 1e-6
    assert abs(float(metrics["rhythm_metric_compact_rhythm_exec"]) - 0.6) < 1e-6
    assert float(minimal_metrics['rhythm_metric_acoustic_target_source_is_cached']) == 1.0
    assert float(minimal_metrics['rhythm_metric_frame_plan_present']) == 1.0
    offline_corr = float(metrics['rhythm_metric_offline_online_total_corr'])
    assert np.isfinite(offline_corr)
    assert abs(offline_corr) <= 1.0 + 1e-6

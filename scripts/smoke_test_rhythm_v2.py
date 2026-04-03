import argparse
import numpy as np
import torch
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from modules.Conan.rhythm.contracts import RhythmPlannerOutputs
from modules.Conan.rhythm.factory import build_streaming_rhythm_module_from_hparams
from modules.Conan.pitch_utils import (
    apply_silent_content_to_uv,
    f0_minmax_denorm,
    f0_minmax_norm,
    infer_uv_from_logits,
    pack_flow_f0_target,
)
from modules.Conan.rhythm.projector import _project_pause_impl, _project_pause_simple_impl
from modules.Conan.rhythm.stages import (
    detect_rhythm_stage,
    resolve_runtime_dual_mode_teacher_enable,
    resolve_runtime_offline_teacher_enable,
    resolve_teacher_as_main,
)
from modules.Conan.rhythm.supervision import (
    RHYTHM_CACHE_VERSION,
    RHYTHM_GUIDANCE_SURFACE_NAME,
    RHYTHM_RETIMED_SOURCE_TEACHER,
    RHYTHM_TEACHER_SURFACE_ALGORITHMIC_NAME,
    RHYTHM_TEACHER_SURFACE_LEARNED_OFFLINE_NAME,
    RHYTHM_TEACHER_TARGET_SOURCE_ALGORITHMIC,
    RHYTHM_TEACHER_TARGET_SOURCE_LEARNED_OFFLINE,
    build_item_rhythm_bundle,
    build_learned_offline_teacher_bundle,
    build_reference_guided_targets,
    build_reference_teacher_targets,
    build_retimed_mel_target,
    compatible_rhythm_cache_versions,
    infer_teacher_target_source_id_from_surface_name,
    is_rhythm_cache_version_compatible,
    materialize_rhythm_cache_compat_fields,
)
from modules.Conan.rhythm.unit_frontend import RhythmUnitFrontend
from modules.Conan.rhythm.unitizer import StreamingRunLengthUnitizer
from tasks.Conan.rhythm.losses import RhythmLossTargets, build_rhythm_loss_dict
from tasks.Conan.rhythm.metrics import build_rhythm_metric_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Structural smoke test for Rhythm V2.")
    parser.parse_args()

    def _scalar_str(x):
        arr = np.asarray(x).reshape(-1)
        if arr.size <= 0:
            return ""
        value = arr[0]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

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
        'rhythm_enable_dual_mode_teacher': True,
        'rhythm_enable_learned_offline_teacher': True,
        'rhythm_runtime_enable_learned_offline_teacher': True,
    }
    base_f0 = torch.tensor([[6.5, 7.0, 8.0]], dtype=torch.float32)
    base_uv = torch.tensor([[False, True, False]])
    norm_f0 = f0_minmax_norm(base_f0, base_uv)
    denorm_f0 = f0_minmax_denorm(norm_f0, base_uv)
    assert torch.allclose(denorm_f0[base_uv == 0], base_f0[base_uv == 0], atol=1e-5)
    assert float(denorm_f0[0, 1].item()) == 0.0
    packed_flow = pack_flow_f0_target(norm_f0)
    assert packed_flow.shape == (1, 1, 1, 3)
    uv_logits = torch.tensor([[[1.0, 0.0], [-0.5, 0.0], [0.7, 0.0]]], dtype=torch.float32)
    content = torch.tensor([[1, 57, 3]], dtype=torch.long)
    inferred_uv = infer_uv_from_logits(uv_logits, content=content, silent_token=57)
    assert inferred_uv.dtype == torch.bool
    assert bool(inferred_uv[0, 0].item()) is True
    assert bool(inferred_uv[0, 1].item()) is True
    assert bool(apply_silent_content_to_uv(inferred_uv, content=content, silent_token=57)[0, 1].item()) is True

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
    assert bool(getattr(model, "enable_learned_offline_teacher", False)) is True
    print('runtime offline teacher enabled:', bool(getattr(model, "enable_learned_offline_teacher", False)))

    schedule_hparams = dict(hparams)
    schedule_hparams.update({
        'rhythm_enable_dual_mode_teacher': False,
        'rhythm_enable_learned_offline_teacher': False,
        'rhythm_runtime_enable_learned_offline_teacher': False,
        'rhythm_schedule_only_stage': True,
        'lambda_rhythm_distill': 0.0,
        'rhythm_distill_surface': 'none',
    })
    schedule_model = build_streaming_rhythm_module_from_hparams(schedule_hparams)
    assert bool(getattr(schedule_model, "enable_learned_offline_teacher", True)) is False
    print('schedule-only runtime offline teacher enabled:', bool(getattr(schedule_model, "enable_learned_offline_teacher", True)))

    strict_hparams = dict(schedule_hparams)
    strict_hparams.update({
        'rhythm_strict_mainline': True,
        'rhythm_enable_learned_offline_teacher': False,
    })
    strict_model = build_streaming_rhythm_module_from_hparams(strict_hparams)
    assert strict_model.projector.config.pause_selection_mode == 'simple'
    assert strict_model.projector.config.use_boundary_commit_guard is False
    assert strict_model.projector.config.build_render_plan is False
    simple_pause = _project_pause_simple_impl(
        pause_weight_unit=torch.full((1, 4), 0.25),
        boundary_score_unit=torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
        unit_mask=torch.ones(1, 4),
        pause_budget_win=torch.tensor([[4.0]], dtype=torch.float32),
        previous_pause_exec=None,
        commit_frontier=torch.zeros(1, dtype=torch.long),
        reuse_prefix=False,
        pause_min_boundary_weight=0.10,
        pause_boundary_bias_weight=0.15,
    )
    assert float(simple_pause[0, 2].item()) > float(simple_pause[0, 0].item())
    compat_item = materialize_rhythm_cache_compat_fields({
        "rhythm_cache_version": np.asarray([4], dtype=np.int64),
        "rhythm_teacher_surface_name": np.asarray([RHYTHM_TEACHER_SURFACE_ALGORITHMIC_NAME], dtype=np.str_),
        "rhythm_retimed_target_surface_name": np.asarray([RHYTHM_GUIDANCE_SURFACE_NAME], dtype=np.str_),
    })
    assert infer_teacher_target_source_id_from_surface_name(RHYTHM_TEACHER_SURFACE_ALGORITHMIC_NAME) == RHYTHM_TEACHER_TARGET_SOURCE_ALGORITHMIC
    assert int(np.asarray(compat_item["rhythm_teacher_target_source_id"]).reshape(-1)[0]) == RHYTHM_TEACHER_TARGET_SOURCE_ALGORITHMIC
    assert int(np.asarray(compat_item["rhythm_retimed_target_source_id"]).reshape(-1)[0]) == 0
    assert is_rhythm_cache_version_compatible(4, RHYTHM_CACHE_VERSION) is True
    assert 4 in compatible_rhythm_cache_versions(RHYTHM_CACHE_VERSION)

    cache_kd_hparams = dict(strict_hparams)
    cache_kd_hparams.update({
        'rhythm_stage': 'student_kd',
        'lambda_rhythm_distill': 0.35,
        'rhythm_distill_surface': 'cache',
        'rhythm_require_cached_teacher': True,
        'rhythm_teacher_target_source': 'learned_offline',
    })
    cache_kd_model = build_streaming_rhythm_module_from_hparams(cache_kd_hparams)
    assert bool(getattr(cache_kd_model, 'enable_learned_offline_teacher', True)) is False

    teacher_offline_hparams = dict(hparams)
    teacher_offline_hparams.update({
        'rhythm_stage': 'teacher_offline',
        'rhythm_enable_dual_mode_teacher': False,
        'rhythm_teacher_as_main': True,
        'rhythm_schedule_only_stage': False,
    })
    assert detect_rhythm_stage(teacher_offline_hparams) == 'teacher_offline'
    assert resolve_runtime_offline_teacher_enable(teacher_offline_hparams, stage='teacher_offline') is True
    assert resolve_runtime_dual_mode_teacher_enable(teacher_offline_hparams, stage='teacher_offline', infer=False) is False
    assert resolve_teacher_as_main(teacher_offline_hparams, stage='teacher_offline', infer=False) is True

    student_kd_hparams = dict(cache_kd_hparams)
    assert detect_rhythm_stage(student_kd_hparams) == 'student_kd'
    assert resolve_runtime_offline_teacher_enable(student_kd_hparams, stage='student_kd') is False
    assert resolve_runtime_dual_mode_teacher_enable(student_kd_hparams, stage='student_kd', infer=False) is False
    assert resolve_teacher_as_main(student_kd_hparams, stage='student_kd', infer=False) is False

    student_retimed_hparams = dict(student_kd_hparams)
    student_retimed_hparams.update({
        'rhythm_stage': 'student_retimed',
        'rhythm_use_retimed_target_if_available': True,
        'rhythm_apply_train_override': True,
        'rhythm_apply_valid_override': True,
        'rhythm_require_retimed_cache': True,
    })
    assert detect_rhythm_stage(student_retimed_hparams) == 'student_retimed'
    assert resolve_runtime_offline_teacher_enable(student_retimed_hparams, stage='student_retimed') is False
    assert resolve_runtime_dual_mode_teacher_enable(student_retimed_hparams, stage='student_retimed', infer=False) is False
    assert resolve_teacher_as_main(student_retimed_hparams, stage='student_retimed', infer=False) is False

    legacy_dual_hparams = dict(hparams)
    legacy_dual_hparams.update({'rhythm_stage': 'legacy_dual_mode_kd'})
    assert detect_rhythm_stage(legacy_dual_hparams) == 'legacy_dual_mode_kd'
    assert resolve_runtime_offline_teacher_enable(legacy_dual_hparams, stage='legacy_dual_mode_kd') is True
    assert resolve_runtime_dual_mode_teacher_enable(legacy_dual_hparams, stage='legacy_dual_mode_kd', infer=False) is True
    assert resolve_teacher_as_main(legacy_dual_hparams, stage='legacy_dual_mode_kd', infer=False) is False

    teacher_flag_only_hparams = dict(schedule_hparams)
    teacher_flag_only_hparams.pop('rhythm_runtime_enable_learned_offline_teacher', None)
    teacher_flag_only_hparams.update({
        'rhythm_enable_learned_offline_teacher': True,
        'rhythm_enable_dual_mode_teacher': False,
    })
    teacher_flag_only_model = build_streaming_rhythm_module_from_hparams(teacher_flag_only_hparams)
    assert bool(getattr(teacher_flag_only_model, 'enable_learned_offline_teacher', True)) is False

    ref_mel = torch.randn(2, 80, 64)
    ref_conditioning = model.encode_reference(ref_mel)
    print('ref descriptor keys:', sorted(ref_conditioning.keys()))
    assert 'global_rate' in ref_conditioning and 'boundary_trace' in ref_conditioning
    assert 'planner_ref_stats' in ref_conditioning and ref_conditioning['planner_ref_stats'].size(-1) == 2
    assert 'planner_ref_trace' in ref_conditioning and ref_conditioning['planner_ref_trace'].size(-1) == 2
    # Maintained runtime contract only requires stats/trace; slow-memory/selector are optional sidecars.
    if 'slow_rhythm_memory' in ref_conditioning:
        print('slow memory shape:', tuple(ref_conditioning['slow_rhythm_memory'].shape))
    else:
        print('slow memory shape:', None)
    if 'selector_meta_starts' in ref_conditioning and 'selector_meta_ends' in ref_conditioning:
        print('selector spans present in runtime ref conditioning')
    else:
        print('selector spans absent in runtime ref conditioning (allowed in maintained path)')
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
    strict_out = strict_model(
        content_units=batch.content_units,
        dur_anchor_src=batch.dur_anchor_src,
        unit_mask=batch.unit_mask,
        open_run_mask=batch.open_run_mask,
        ref_rhythm_stats=ref_conditioning['ref_rhythm_stats'],
        ref_rhythm_trace=ref_conditioning['ref_rhythm_trace'],
        state=strict_model.init_state(batch_size=2, device=torch.device('cpu')),
    )
    assert strict_out.slot_duration_exec is None
    assert strict_out.slot_mask is None
    assert strict_out.slot_is_blank is None
    assert strict_out.slot_unit_index is None
    assert strict_out.frame_plan is None

    forced_planner = RhythmPlannerOutputs(
        speech_budget_win=torch.zeros((1, 1), dtype=torch.float32, requires_grad=True),
        pause_budget_win=torch.zeros((1, 1), dtype=torch.float32, requires_grad=True),
        dur_logratio_unit=torch.zeros((1, 4), dtype=torch.float32, requires_grad=True),
        pause_weight_unit=torch.full((1, 4), 0.25, dtype=torch.float32, requires_grad=True),
        boundary_score_unit=torch.zeros((1, 4), dtype=torch.float32),
        trace_context=torch.zeros((1, 4, 2), dtype=torch.float32),
        source_boundary_cue=torch.zeros((1, 4), dtype=torch.float32),
    )
    forced_exec = strict_model.projector(
        dur_anchor_src=torch.full((1, 4), 2.0, dtype=torch.float32),
        unit_mask=torch.ones((1, 4), dtype=torch.float32),
        speech_budget_win=forced_planner.speech_budget_win,
        pause_budget_win=forced_planner.pause_budget_win,
        dur_logratio_unit=forced_planner.dur_logratio_unit,
        pause_weight_unit=forced_planner.pause_weight_unit,
        boundary_score_unit=forced_planner.boundary_score_unit,
        state=strict_model.init_state(batch_size=1, device=torch.device('cpu')),
        planner=forced_planner,
        reuse_prefix=False,
        force_full_commit=True,
    )
    assert torch.allclose(forced_exec.planner.raw_speech_budget_win, forced_planner.speech_budget_win)
    assert torch.allclose(forced_exec.planner.raw_pause_budget_win, forced_planner.pause_budget_win)
    assert float(forced_exec.planner.speech_budget_win.item()) >= 4.0
    forced_losses = build_rhythm_loss_dict(
        forced_exec,
        RhythmLossTargets(
            speech_exec_tgt=forced_exec.speech_duration_exec.detach(),
            pause_exec_tgt=forced_exec.pause_after_exec.detach(),
            speech_budget_tgt=forced_exec.planner.speech_budget_win.detach(),
            pause_budget_tgt=forced_exec.planner.pause_budget_win.detach(),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            dur_anchor_src=torch.full((1, 4), 2.0, dtype=torch.float32),
        ),
    )
    assert forced_losses["rhythm_feasible_debt"].requires_grad
    assert float(forced_losses["rhythm_budget"].detach().item()) > 0.0

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
    scheduler_state = model.init_state(batch_size=2, device=torch.device('cpu'))
    unit_states = model.unit_embedding(batch.content_units)
    visible_sizes = batch.unit_mask.float().sum(dim=1).long().clamp_min(1)
    planner_trace_context = model.sample_planner_trace_window(
        ref_conditioning=ref_conditioning,
        phase_ptr=scheduler_state.phase_ptr,
        window_size=batch.content_units.size(1),
        visible_sizes=visible_sizes,
    )
    scheduler_inputs = dict(
        unit_states=unit_states,
        dur_anchor_src=batch.dur_anchor_src,
        unit_mask=batch.unit_mask,
        ref_conditioning={
            'planner_ref_stats': ref_conditioning['planner_ref_stats'],
            'planner_slow_rhythm_summary': ref_conditioning.get('planner_slow_rhythm_summary'),
        },
        trace_context=out1.planner.trace_context.detach(),
        planner_trace_context=planner_trace_context.detach(),
        state=scheduler_state,
        source_boundary_cue=out1.planner.source_boundary_cue.detach(),
    )
    scheduler_base = model.scheduler(**scheduler_inputs)
    budget_probe_weight = model.scheduler.window_budget.pool_mlp[0].weight.detach().clone()
    with torch.no_grad():
        model.scheduler.window_budget.pool_mlp[0].weight.add_(0.5)
    scheduler_shifted = model.scheduler(**scheduler_inputs)
    with torch.no_grad():
        model.scheduler.window_budget.pool_mlp[0].weight.copy_(budget_probe_weight)
    assert torch.allclose(
        scheduler_base.dur_logratio_unit,
        scheduler_shifted.dur_logratio_unit,
        atol=1e-6,
    )
    assert torch.allclose(
        scheduler_base.pause_weight_unit,
        scheduler_shifted.pause_weight_unit,
        atol=1e-6,
    )
    redistribution_inputs = dict(
        unit_states=unit_states,
        dur_anchor_src=batch.dur_anchor_src,
        planner_trace_context=planner_trace_context.detach(),
        unit_mask=batch.unit_mask,
        slow_rhythm_summary=ref_conditioning.get('planner_slow_rhythm_summary'),
        boundary_score_unit=out1.planner.boundary_score_unit.detach(),
    )
    redistribution_base = model.scheduler.unit_redistribution(**redistribution_inputs)
    redistribution_shifted = model.scheduler.unit_redistribution(**redistribution_inputs)
    assert torch.allclose(
        redistribution_base['dur_logratio_unit'],
        redistribution_shifted['dur_logratio_unit'],
        atol=1e-6,
    )
    assert torch.allclose(
        redistribution_base['pause_weight_unit'],
        redistribution_shifted['pause_weight_unit'],
        atol=1e-6,
    )
    assert not hasattr(model.scheduler.unit_redistribution, 'boundary_head')
    assert torch.allclose(out1.planner.boundary_score_unit, out1.planner.boundary_latent, atol=1e-6)
    assert out1.slot_duration_exec.shape[1] == batch.content_units.shape[1] * 2
    assert out1.slot_is_blank[:, 1::2].sum().item() > 0
    assert torch.equal(out1.blank_slot_duration_exec, out1.slot_duration_exec)
    assert torch.equal(out1.blank_slot_is_blank, out1.slot_is_blank)
    assert float(out_no_source_prior.planner.source_boundary_cue.abs().max().item()) == 0.0
    assert torch.all(out2.next_state.phase_ptr + 1e-6 >= out1.next_state.phase_ptr)
    assert torch.all(out_hold.next_state.phase_ptr + 1e-6 >= out2.next_state.phase_ptr)
    assert torch.equal(out_hold.next_state.commit_frontier, out2.next_state.commit_frontier)
    assert out1.pause_after_exec.requires_grad
    zero_pause = _project_pause_impl(
        pause_weight_unit=torch.randn(1, 4, requires_grad=True),
        boundary_score_unit=torch.randn(1, 4, requires_grad=True),
        unit_mask=torch.ones(1, 4),
        pause_budget_win=torch.zeros(1, 1, requires_grad=True),
        previous_pause_exec=None,
        commit_frontier=torch.zeros(1, dtype=torch.long),
        reuse_prefix=False,
        soft_pause_selection=True,
        topk_ratio=0.5,
        pause_min_boundary_weight=0.10,
        pause_boundary_bias_weight=0.15,
        temperature=0.12,
    )
    assert zero_pause.requires_grad
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
    distill_speech_tgt = torch.flip(out1.speech_duration_exec.detach(), dims=[1]) * batch.unit_mask.float()
    distill_pause_tgt = out1.pause_after_exec.detach() * batch.unit_mask.float()
    loss_dict = build_rhythm_loss_dict(
        out1,
        RhythmLossTargets(
            speech_exec_tgt=out1.speech_duration_exec.detach(),
            pause_exec_tgt=out1.pause_after_exec.detach(),
            speech_budget_tgt=out1.planner.speech_budget_win.detach(),
            pause_budget_tgt=out1.planner.pause_budget_win.detach(),
            unit_mask=batch.unit_mask.detach(),
            dur_anchor_src=batch.dur_anchor_src.detach(),
            distill_speech_tgt=distill_speech_tgt,
            distill_pause_tgt=distill_pause_tgt,
            distill_allocation_weight=0.0,
            distill_speech_shape_weight=1.0,
            distill_pause_shape_weight=1.0,
        ),
    )
    assert "rhythm_distill_speech_shape" in loss_dict
    assert "rhythm_distill_pause_shape" in loss_dict
    assert torch.isfinite(loss_dict["rhythm_plan_local"])
    assert torch.isfinite(loss_dict["rhythm_plan"])
    assert torch.isfinite(loss_dict["rhythm_distill_speech_shape"])
    assert torch.isfinite(loss_dict["rhythm_distill_pause_shape"])
    assert torch.isfinite(loss_dict["rhythm_distill_allocation"])
    assert torch.isfinite(loss_dict["rhythm_total"])
    assert abs(float(loss_dict["rhythm_distill_pause_shape"].detach())) < 1e-6

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
    learned_teacher_override = {
        "rhythm_teacher_speech_exec_tgt": np.asarray(cached_bundle["rhythm_teacher_speech_exec_tgt"], dtype=np.float32) * 1.05,
        "rhythm_teacher_pause_exec_tgt": np.asarray(cached_bundle["rhythm_teacher_pause_exec_tgt"], dtype=np.float32) * 0.95,
        "rhythm_teacher_confidence": np.asarray([0.83], dtype=np.float32),
    }
    learned_bundle = build_item_rhythm_bundle(
        content_tokens=[1, 1, 1, 2, 2, 57, 3, 3],
        mel=np.random.randn(16, 80).astype(np.float32),
        trace_horizon=0.40,
        include_self_targets=True,
        include_teacher_targets=False,
        include_retimed_mel_target=True,
        retimed_mel_target_source="teacher",
        teacher_target_source="learned_offline",
        teacher_bundle_override=learned_teacher_override,
    )
    exported_learned_bundle = build_learned_offline_teacher_bundle(
        speech_exec_tgt=learned_bundle["rhythm_teacher_speech_exec_tgt"],
        pause_exec_tgt=learned_bundle["rhythm_teacher_pause_exec_tgt"],
        dur_anchor_src=learned_bundle["dur_anchor_src"],
        confidence=learned_bundle["rhythm_teacher_confidence"],
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
            "rhythm_offline_confidence_shape": offline_conf.get("shape", offline_conf["exec"]),
            "rhythm_frame_plan": out1.frame_plan,
            "acoustic_target_source": "online",
            "rhythm_exec": torch.tensor(0.6),
            "rhythm_prefix_state": torch.tensor(0.12),
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
    print('source boundary scale metric:', float(metrics['rhythm_metric_source_boundary_scale_mean'].detach()))
    print('offline total corr metric:', float(metrics['rhythm_metric_offline_online_total_corr'].detach()))
    print('offline stream prefix ratio:', float(prefix_batch.unit_mask.sum().item() / batch.unit_mask.sum().item()))
    print('algorithmic teacher alloc kl:', float(metrics['rhythm_metric_algorithmic_teacher_alloc_kl'].detach()))
    print('offline confidence exec:', float(metrics['rhythm_metric_offline_confidence_exec_mean'].detach()))
    print('offline confidence coverage:', float(metrics['rhythm_metric_offline_confidence_component_coverage'].detach()))
    print('blank slot ratio metric:', float(metrics['rhythm_metric_blank_slot_ratio_mean'].detach()))
    print('frame plan present:', float(metrics['rhythm_metric_frame_plan_present'].detach()))
    print('frame plan blank/src consistency:', float(metrics['rhythm_metric_frame_plan_blank_src_consistency'].detach()))
    print('frame plan speech/src consistency:', float(metrics['rhythm_metric_frame_plan_speech_src_consistency'].detach()))
    print('acoustic target source id:', float(metrics['rhythm_metric_acoustic_target_source_id'].detach()))
    print('phase nonretro rate:', float(metrics['rhythm_metric_phase_nonretro_rate'].detach()))
    print('alias L_rhythm_exec:', float(metrics['rhythm_metric_alias_L_rhythm_exec'].detach()))
    print('compact rhythm_exec:', float(metrics['rhythm_metric_compact_rhythm_exec'].detach()))
    print('retimed mel len:', int(retimed['rhythm_retimed_mel_len'][0]))
    print('retimed frame weight mean:', float(retimed['rhythm_retimed_frame_weight'].mean()))
    print('guidance keys:', sorted(guidance.keys()))
    print('teacher keys:', sorted(teacher.keys()))
    teacher_gap = float(np.abs(teacher['rhythm_teacher_pause_exec_tgt'] - guidance['rhythm_pause_exec_tgt']).sum())
    print('teacher/guidance pause gap:', teacher_gap)
    print('cache version:', int(cached_bundle['rhythm_cache_version'][0]))
    print('trace bins:', int(cached_bundle['rhythm_trace_bins'][0]))
    print('trace horizon:', float(cached_bundle['rhythm_trace_horizon'][0]))
    print('guidance surface name:', _scalar_str(cached_bundle['rhythm_guidance_surface_name']))
    print('teacher surface name:', _scalar_str(cached_bundle['rhythm_teacher_surface_name']))
    print('teacher target source id:', int(cached_bundle['rhythm_teacher_target_source_id'][0]))
    print('retimed target surface:', _scalar_str(cached_bundle['rhythm_retimed_target_surface_name']))
    print('target confidence:', float(cached_bundle['rhythm_target_confidence'][0]))
    print('teacher confidence:', float(cached_bundle['rhythm_teacher_confidence'][0]))
    if 'selector_meta_starts' in cached_bundle and 'selector_meta_ends' in cached_bundle:
        print('selector starts:', cached_bundle['selector_meta_starts'].tolist())
        print('selector ends:', cached_bundle['selector_meta_ends'].tolist())
    else:
        print('selector starts:', None)
        print('selector ends:', None)
    print('phrase groups:', cached_bundle['phrase_group_index'].tolist())
    print('retimed source id:', int(cached_bundle['rhythm_retimed_target_source_id'][0]))
    print('retimed confidence:', float(cached_bundle['rhythm_retimed_target_confidence'][0]))
    print('learned teacher surface name:', _scalar_str(learned_bundle['rhythm_teacher_surface_name']))
    print('learned teacher target source id:', int(learned_bundle['rhythm_teacher_target_source_id'][0]))
    print('learned retimed target surface:', _scalar_str(learned_bundle['rhythm_retimed_target_surface_name']))
    print('metric exec total corr:', float(metrics['rhythm_metric_exec_total_corr'].detach()))
    print('metric prefix drift l1:', float(metrics['rhythm_metric_prefix_drift_l1'].detach()))
    print('metric prefix backlog mean:', float(metrics['rhythm_metric_prefix_backlog_mean'].detach()))
    print('minimal metric source id:', float(minimal_metrics['rhythm_metric_acoustic_target_source_id']))
    print('minimal metric frame plan present:', float(minimal_metrics['rhythm_metric_frame_plan_present']))
    assert "rhythm_blank_exec_tgt" in guidance and "rhythm_teacher_blank_exec_tgt" in teacher
    assert "rhythm_pause_exec_tgt" in cached_bundle and "rhythm_pause_budget_tgt" in cached_bundle
    assert "rhythm_blank_exec_tgt" in cached_bundle and "rhythm_blank_budget_tgt" in cached_bundle
    assert int(cached_bundle["rhythm_cache_version"][0]) == int(RHYTHM_CACHE_VERSION)
    assert _scalar_str(cached_bundle["rhythm_guidance_surface_name"]) == RHYTHM_GUIDANCE_SURFACE_NAME
    assert _scalar_str(cached_bundle["rhythm_teacher_surface_name"]) == RHYTHM_TEACHER_SURFACE_ALGORITHMIC_NAME
    assert int(cached_bundle["rhythm_teacher_target_source_id"][0]) == int(RHYTHM_TEACHER_TARGET_SOURCE_ALGORITHMIC)
    assert int(cached_bundle["rhythm_retimed_target_source_id"][0]) == int(RHYTHM_RETIMED_SOURCE_TEACHER)
    assert _scalar_str(cached_bundle["rhythm_retimed_target_surface_name"]) == RHYTHM_TEACHER_SURFACE_ALGORITHMIC_NAME
    assert _scalar_str(learned_bundle["rhythm_teacher_surface_name"]) == RHYTHM_TEACHER_SURFACE_LEARNED_OFFLINE_NAME
    assert int(learned_bundle["rhythm_teacher_target_source_id"][0]) == int(RHYTHM_TEACHER_TARGET_SOURCE_LEARNED_OFFLINE)
    assert _scalar_str(exported_learned_bundle["rhythm_teacher_surface_name"]) == RHYTHM_TEACHER_SURFACE_LEARNED_OFFLINE_NAME
    assert int(exported_learned_bundle["rhythm_teacher_target_source_id"][0]) == int(RHYTHM_TEACHER_TARGET_SOURCE_LEARNED_OFFLINE)
    assert int(learned_bundle["rhythm_retimed_target_source_id"][0]) == int(RHYTHM_RETIMED_SOURCE_TEACHER)
    assert _scalar_str(learned_bundle["rhythm_retimed_target_surface_name"]) == RHYTHM_TEACHER_SURFACE_LEARNED_OFFLINE_NAME
    assert "rhythm_teacher_prefix_clock_tgt" in learned_bundle
    assert "rhythm_teacher_prefix_backlog_tgt" in learned_bundle
    assert learned_bundle["rhythm_teacher_prefix_clock_tgt"].shape == learned_bundle["dur_anchor_src"].shape
    assert learned_bundle["rhythm_teacher_prefix_backlog_tgt"].shape == learned_bundle["dur_anchor_src"].shape
    assert teacher_gap >= 0.0
    assert float(metrics['rhythm_metric_exec_total_corr'].detach()) > 0.99
    assert float(metrics['rhythm_metric_prefix_drift_l1'].detach()) < 1e-6
    assert float(metrics['rhythm_metric_frame_plan_present'].detach()) == 1.0
    blank_src_consistency = float(metrics['rhythm_metric_frame_plan_blank_src_consistency'].detach())
    assert np.isfinite(blank_src_consistency)
    assert 0.0 <= blank_src_consistency <= 1.0
    assert float(metrics['rhythm_metric_frame_plan_speech_src_consistency'].detach()) > 0.99
    assert float(metrics['rhythm_metric_acoustic_target_source_is_online'].detach()) == 1.0
    assert float(metrics['rhythm_metric_acoustic_target_source_unknown'].detach()) == 0.0
    assert float(metrics['rhythm_metric_phase_nonretro_rate'].detach()) == 1.0
    assert float(metrics['rhythm_metric_phase_delta_min'].detach()) >= -1e-6
    assert float(metrics['rhythm_metric_offline_confidence_component_coverage'].detach()) == 1.0
    assert "rhythm_metric_offline_confidence_component_std" in metrics
    assert "rhythm_metric_alias_L_rhythm_exec" in metrics
    assert "rhythm_metric_alias_L_stream_state" in metrics
    assert "rhythm_metric_alias_L_base" in metrics
    assert "rhythm_metric_alias_L_pitch" in metrics
    assert "rhythm_metric_compact_rhythm_exec" in metrics
    assert "rhythm_metric_compact_stream_state" in metrics
    assert "rhythm_metric_compact_base" in metrics
    assert "rhythm_metric_compact_pitch" in metrics
    assert abs(float(metrics["rhythm_metric_alias_L_rhythm_exec"].detach()) - 0.2) < 1e-6
    assert abs(float(metrics["rhythm_metric_compact_rhythm_exec"].detach()) - 0.6) < 1e-6
    assert float(minimal_metrics['rhythm_metric_acoustic_target_source_is_cached']) == 1.0
    assert float(minimal_metrics['rhythm_metric_frame_plan_present']) == 1.0
    offline_corr = float(metrics['rhythm_metric_offline_online_total_corr'].detach())
    assert np.isfinite(offline_corr)
    assert abs(offline_corr) <= 1.0 + 1e-6

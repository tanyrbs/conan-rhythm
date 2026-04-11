from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.runtime_modes import merge_retimed_weight, resolve_acoustic_target_post_model
from tasks.Conan.rhythm.task_runtime_support import RhythmTaskRuntimeSupport


class RhythmTaskRuntimeSupportTests(unittest.TestCase):
    @staticmethod
    def _runtime_owner():
        class DummyOwner:
            mel_losses = {"l1": 1.0}

            @staticmethod
            def _align_acoustic_target_to_output(mel_out, acoustic_target, acoustic_weight):
                return mel_out, acoustic_target, acoustic_weight

        return DummyOwner()

    def test_dedup_trainable_params_filters_duplicates_and_frozen(self) -> None:
        p1 = torch.nn.Parameter(torch.tensor([1.0]))
        p2 = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
        dedup = RhythmTaskRuntimeSupport.dedup_trainable_params([p1, p1, None, p2])
        self.assertEqual(dedup, [p1])

    def test_offline_confidence_outputs_use_shape_fallback(self) -> None:
        owner = SimpleNamespace(mel_losses={"l1": 1.0})
        support = RhythmTaskRuntimeSupport(owner)
        outputs = support.build_offline_confidence_outputs(
            {
                "overall": torch.tensor([0.9]),
                "exec": torch.tensor([0.7]),
                "budget": torch.tensor([0.6]),
            }
        )
        self.assertTrue(torch.allclose(outputs["rhythm_offline_confidence"], torch.tensor([0.9])))
        self.assertTrue(torch.allclose(outputs["rhythm_offline_confidence_exec"], torch.tensor([0.7])))
        self.assertTrue(torch.allclose(outputs["rhythm_offline_confidence_budget"], torch.tensor([0.6])))
        self.assertIsNone(outputs["rhythm_offline_confidence_prefix"])
        self.assertTrue(torch.allclose(outputs["rhythm_offline_confidence_shape"], torch.tensor([0.7])))

    def test_build_model_forward_kwargs_carries_runtime_caches(self) -> None:
        class DummyOwner:
            mel_losses = {"l1": 1.0}

            @staticmethod
            def _collect_rhythm_source_cache(sample, *, prefix: str = ""):
                if prefix:
                    return {"dur_anchor_src": sample[f"{prefix}dur_anchor_src"]}
                return {"dur_anchor_src": sample["dur_anchor_src"]}

        support = RhythmTaskRuntimeSupport(DummyOwner())
        sample = {
            "content_lengths": torch.tensor([7]),
            "mel_lengths": torch.tensor([5]),
            "ref_mel_lengths": torch.tensor([4]),
            "dur_anchor_src": torch.tensor([[1, 2]]),
            "rhythm_offline_dur_anchor_src": torch.tensor([[1, 2, 3]]),
        }
        kwargs = support.build_model_forward_kwargs(
            sample=sample,
            spk_embed=None,
            target=torch.zeros((1, 5, 80)),
            ref=torch.zeros((1, 4, 80)),
            f0=torch.ones((1, 5)),
            uv=torch.zeros((1, 5)),
            infer=False,
            effective_global_step=10,
            rhythm_apply_override=False,
            rhythm_ref_conditioning={"ref_rhythm_stats": torch.zeros((1, 6))},
            disable_source_pitch_supervision=True,
            disable_acoustic_train_path=False,
            runtime_offline_source_cache={"dur_anchor_src": sample["rhythm_offline_dur_anchor_src"]},
            rhythm_state="state",
        )
        self.assertIsNone(kwargs["f0"])
        self.assertIsNone(kwargs["uv"])
        self.assertTrue(torch.equal(kwargs["content_lengths"], sample["content_lengths"]))
        self.assertEqual(kwargs["global_steps"], 10)
        self.assertEqual(kwargs["rhythm_state"], "state")
        self.assertEqual(kwargs["rhythm_source_cache"]["dur_anchor_src"].shape[1], 2)
        self.assertEqual(kwargs["rhythm_offline_source_cache"]["dur_anchor_src"].shape[1], 3)

    def test_target_build_config_accepts_duplicate_distill_alias(self) -> None:
        owner = SimpleNamespace(
            mel_losses={"l1": 1.0},
            _resolve_rhythm_plan_weights=lambda: (0.5, 1.0),
            _resolve_rhythm_primary_target_surface=lambda: "teacher",
            _resolve_rhythm_distill_surface=lambda: "cache",
            _resolve_rhythm_pause_boundary_weight=lambda: 0.35,
            _rhythm_policy=lambda: SimpleNamespace(strict_mainline=True),
        )
        support = RhythmTaskRuntimeSupport(owner)
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_runtime_support.hparams",
            {
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_distill": 0.35,
                "rhythm_distill_exec_weight": 0.0,
                "rhythm_distill_budget_weight": 0.0,
                "rhythm_distill_allocation_weight": 0.0,
                "rhythm_distill_prefix_weight": 0.0,
            "rhythm_distill_speech_shape_weight": 0.25,
            "rhythm_distill_pause_shape_weight": 0.25,
            "rhythm_budget_raw_weight": 1.0,
            "rhythm_budget_exec_weight": 0.25,
            "rhythm_feasible_debt_weight": 0.05,
            "rhythm_pause_event_weight": 0.20,
            "rhythm_pause_support_weight": 0.08,
            "rhythm_pause_allocation_weight": 0.04,
            "rhythm_plan_segment_shape_weight": 0.20,
            "rhythm_plan_pause_release_weight": 0.10,
            "rhythm_pause_event_threshold": 0.60,
            "rhythm_pause_event_temperature": 0.15,
            "rhythm_pause_event_pos_weight": 2.75,
            "rhythm_unit_logratio_weight": 0.35,
            "rhythm_srmdp_role_consistency_weight": 0.40,
            "rhythm_srmdp_notimeline_weight": 0.20,
            "rhythm_srmdp_memory_role_weight": 0.30,
            "rhythm_suppress_duplicate_primary_distill": True,
        },
        clear=True,
    ):
            config = support.build_rhythm_target_build_config()
        self.assertTrue(config.dedupe_primary_teacher_cache_distill)
        self.assertEqual(config.distill_exec_weight, 0.0)
        self.assertEqual(config.pause_event_weight, 0.20)
        self.assertEqual(config.pause_support_weight, 0.08)
        self.assertEqual(config.pause_allocation_weight, 0.04)
        self.assertEqual(config.plan_segment_shape_weight, 0.20)
        self.assertEqual(config.plan_pause_release_weight, 0.10)
        self.assertEqual(config.pause_event_threshold, 0.60)
        self.assertEqual(config.pause_event_temperature, 0.15)
        self.assertEqual(config.pause_event_pos_weight, 2.75)
        self.assertEqual(config.unit_logratio_weight, 0.35)
        self.assertEqual(config.srmdp_role_consistency_weight, 0.40)
        self.assertEqual(config.srmdp_notimeline_weight, 0.20)
        self.assertEqual(config.srmdp_memory_role_weight, 0.30)

    def test_build_duration_v3_target_build_config_defaults_consistency_off(self) -> None:
        support = RhythmTaskRuntimeSupport(SimpleNamespace(mel_losses={"l1": 1.0}))
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_runtime_support.hparams",
            {
                "lambda_rhythm_dur": 1.0,
                "lambda_rhythm_op": 0.25,
                "lambda_rhythm_pref": 0.20,
                "lambda_rhythm_zero": 0.05,
            },
            clear=True,
        ):
            config = support.build_duration_v3_target_build_config()
        self.assertEqual(config.lambda_dur, 1.0)
        self.assertEqual(config.lambda_op, 0.25)
        self.assertEqual(config.lambda_pref, 0.20)
        self.assertEqual(config.lambda_bias, 0.0)
        self.assertEqual(config.lambda_cons, 0.0)
        self.assertEqual(config.lambda_zero, 0.05)
        self.assertEqual(config.lambda_ortho, 0.0)
        self.assertTrue(config.strict_target_alignment)

    def test_build_duration_v3_target_build_config_respects_consistency_override(self) -> None:
        support = RhythmTaskRuntimeSupport(SimpleNamespace(mel_losses={"l1": 1.0}))
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_runtime_support.hparams",
            {
                "lambda_rhythm_dur": 1.0,
                "lambda_rhythm_op": 0.25,
                "lambda_rhythm_pref": 0.20,
                "lambda_rhythm_cons": 0.15,
                "lambda_rhythm_zero": 0.05,
                "lambda_rhythm_ortho": 0.02,
                "rhythm_v3_strict_target_alignment": False,
            },
            clear=True,
        ):
            config = support.build_duration_v3_target_build_config()
        self.assertEqual(config.lambda_cons, 0.15)
        self.assertEqual(config.lambda_ortho, 0.02)
        self.assertFalse(config.strict_target_alignment)

    def test_build_duration_v3_target_build_config_defaults_prompt_summary_bias_on(self) -> None:
        support = RhythmTaskRuntimeSupport(SimpleNamespace(mel_losses={"l1": 1.0}))
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_runtime_support.hparams",
            {
                "rhythm_v3_backbone": "prompt_summary",
                "lambda_rhythm_dur": 1.0,
                "lambda_rhythm_summary": 0.10,
                "lambda_rhythm_pref": 0.0,
                "lambda_rhythm_zero": 0.0,
                "lambda_rhythm_ortho": 0.0,
            },
            clear=True,
        ):
            config = support.build_duration_v3_target_build_config()
        self.assertEqual(config.lambda_bias, 0.20)

    def test_build_duration_v3_target_build_config_disables_prompt_summary_aux_by_default(self) -> None:
        support = RhythmTaskRuntimeSupport(SimpleNamespace(mel_losses={"l1": 1.0}))
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_runtime_support.hparams",
            {
                "rhythm_v3_backbone": "prompt_summary",
                "lambda_rhythm_dur": 1.0,
                "lambda_rhythm_pref": 0.0,
                "lambda_rhythm_zero": 0.0,
                "lambda_rhythm_ortho": 0.0,
            },
            clear=True,
        ):
            config = support.build_duration_v3_target_build_config()
        self.assertEqual(config.lambda_op, 0.0)

    def test_attach_acoustic_target_bundle_exposes_alignment_observability(self) -> None:
        class DummyOwner:
            mel_losses = {"l1": 1.0}

            @staticmethod
            def _align_acoustic_target_to_output(mel_out, acoustic_target, acoustic_weight):
                target_len = mel_out.size(1)
                return mel_out, acoustic_target[:, :target_len], acoustic_weight[:, :target_len]

        support = RhythmTaskRuntimeSupport(DummyOwner())
        output = {
            "mel_out": torch.zeros((1, 4, 3), dtype=torch.float32),
        }
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_runtime_support.hparams",
            {"rhythm_resample_retimed_target_to_output": False},
            clear=True,
        ):
            acoustic_target, acoustic_weight = support.attach_acoustic_target_bundle(
                output,
                acoustic_target=torch.ones((1, 6, 3), dtype=torch.float32),
                acoustic_target_is_retimed=True,
                acoustic_weight=torch.ones((1, 6), dtype=torch.float32),
                acoustic_target_source="cached",
                disable_source_pitch_supervision=False,
                disable_acoustic_train_path=False,
            )
        self.assertEqual(tuple(acoustic_target.shape), (1, 4, 3))
        self.assertEqual(tuple(acoustic_weight.shape), (1, 4))
        self.assertEqual(output["acoustic_target_length_frames_before_align"], 6.0)
        self.assertEqual(output["acoustic_output_length_frames_before_align"], 4.0)
        self.assertEqual(output["acoustic_target_length_delta_before_align"], 2.0)
        self.assertEqual(output["acoustic_target_length_mismatch_abs_before_align"], 2.0)
        self.assertEqual(output["acoustic_target_length_mismatch_present_before_align"], 1.0)
        self.assertAlmostEqual(output["acoustic_target_length_mismatch_ratio_before_align"], 2.0 / 6.0)
        self.assertEqual(output["acoustic_target_resampled_to_output"], 0.0)
        self.assertEqual(output["acoustic_target_trimmed_to_output"], 1.0)
        self.assertEqual(output["acoustic_target_length_frames_after_align"], 4.0)
        self.assertEqual(output["acoustic_output_length_frames_after_align"], 4.0)

    def test_attach_acoustic_target_bundle_aligns_retimed_pitch_targets_and_nonpadding(self) -> None:
        class DummyOwner:
            mel_losses = {"l1": 1.0}

            @staticmethod
            def _align_acoustic_target_to_output(mel_out, acoustic_target, acoustic_weight):
                target_len = mel_out.size(1)
                return mel_out, acoustic_target[:, :target_len], acoustic_weight[:, :target_len]

        support = RhythmTaskRuntimeSupport(DummyOwner())
        output = {
            "mel_out": torch.zeros((1, 4, 80), dtype=torch.float32),
            "tgt_nonpadding": torch.ones((1, 6, 1), dtype=torch.float32),
            "retimed_f0_tgt": torch.arange(6, dtype=torch.float32).view(1, 6),
            "retimed_uv_tgt": torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
        }
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_runtime_support.hparams",
            {
                "rhythm_resample_retimed_target_to_output": False,
                "rhythm_allow_source_pitch_fallback_when_retimed": False,
            },
            clear=True,
        ):
            support.attach_acoustic_target_bundle(
                output,
                acoustic_target=torch.zeros((1, 6, 80), dtype=torch.float32),
                acoustic_target_is_retimed=True,
                acoustic_weight=torch.ones((1, 6), dtype=torch.float32),
                acoustic_target_source="cached",
                disable_source_pitch_supervision=False,
                disable_acoustic_train_path=False,
            )
        self.assertEqual(tuple(output["retimed_f0_tgt"].shape), (1, 4))
        self.assertEqual(tuple(output["retimed_uv_tgt"].shape), (1, 4))
        self.assertEqual(tuple(output["tgt_nonpadding"].shape), (1, 4, 1))
        self.assertEqual(output["retimed_pitch_target_length_frames_before_align"], 6.0)
        self.assertEqual(output["retimed_pitch_target_length_mismatch_abs_before_align"], 2.0)
        self.assertEqual(output["retimed_pitch_target_resampled_to_output"], 0.0)
        self.assertEqual(output["retimed_pitch_target_trimmed_to_output"], 1.0)
        self.assertEqual(output["retimed_pitch_target_length_frames_after_align"], 4.0)

    def test_attach_acoustic_target_bundle_trims_retimed_pitch_targets_with_exact_values(self) -> None:
        class DummyOwner:
            mel_losses = {"l1": 1.0}

            @staticmethod
            def _align_acoustic_target_to_output(mel_out, acoustic_target, acoustic_weight):
                target_len = mel_out.size(1)
                return mel_out, acoustic_target[:, :target_len], acoustic_weight[:, :target_len]

        support = RhythmTaskRuntimeSupport(DummyOwner())
        output = {
            "mel_out": torch.zeros((1, 4, 80), dtype=torch.float32),
            "retimed_f0_tgt": torch.arange(6, dtype=torch.float32).unsqueeze(0),
            "retimed_uv_tgt": torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]], dtype=torch.float32),
        }
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_runtime_support.hparams",
            {"rhythm_resample_retimed_target_to_output": False},
            clear=True,
        ):
            support.attach_acoustic_target_bundle(
                output,
                acoustic_target=torch.zeros((1, 6, 80), dtype=torch.float32),
                acoustic_target_is_retimed=True,
                acoustic_weight=torch.ones((1, 6), dtype=torch.float32),
                acoustic_target_source="online",
                disable_source_pitch_supervision=False,
                disable_acoustic_train_path=False,
            )
        self.assertTrue(torch.allclose(output["retimed_f0_tgt"], torch.tensor([[0.0, 1.0, 2.0, 3.0]])))
        self.assertTrue(torch.allclose(output["retimed_uv_tgt"], torch.tensor([[0.0, 1.0, 0.0, 1.0]])))

    def test_attach_acoustic_target_bundle_resamples_retimed_pitch_targets_and_keeps_uv_binary(self) -> None:
        class DummyOwner:
            mel_losses = {"l1": 1.0}

            @staticmethod
            def _align_acoustic_target_to_output(mel_out, acoustic_target, acoustic_weight):
                target_len = mel_out.size(1)
                return mel_out, acoustic_target[:, :target_len], acoustic_weight[:, :target_len]

        support = RhythmTaskRuntimeSupport(DummyOwner())
        output = {
            "mel_out": torch.zeros((1, 4, 80), dtype=torch.float32),
            "retimed_f0_tgt": torch.arange(3, dtype=torch.float32).unsqueeze(0),
            "retimed_uv_tgt": torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
        }
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_runtime_support.hparams",
            {"rhythm_resample_retimed_target_to_output": False},
            clear=True,
        ):
            support.attach_acoustic_target_bundle(
                output,
                acoustic_target=torch.zeros((1, 4, 80), dtype=torch.float32),
                acoustic_target_is_retimed=True,
                acoustic_weight=torch.ones((1, 4), dtype=torch.float32),
                acoustic_target_source="online",
                disable_source_pitch_supervision=False,
                disable_acoustic_train_path=False,
            )
        self.assertEqual(tuple(output["retimed_f0_tgt"].shape), (1, 4))
        self.assertEqual(tuple(output["retimed_uv_tgt"].shape), (1, 4))
        self.assertTrue(torch.isfinite(output["retimed_f0_tgt"]).all())
        self.assertTrue(set(output["retimed_uv_tgt"].reshape(-1).tolist()).issubset({0.0, 1.0}))
        self.assertEqual(output["retimed_pitch_target_resampled_to_output"], 1.0)
        self.assertEqual(output["retimed_pitch_target_trimmed_to_output"], 0.0)

    def test_merge_retimed_weight_preserves_explicit_zero_confidence(self) -> None:
        merged = merge_retimed_weight(
            torch.ones((1, 3), dtype=torch.float32),
            torch.tensor([[0.0]], dtype=torch.float32),
            confidence_floor=0.05,
        )
        self.assertTrue(torch.allclose(merged, torch.zeros((1, 3), dtype=torch.float32)))

    def test_merge_retimed_weight_still_floors_positive_confidence(self) -> None:
        merged = merge_retimed_weight(
            torch.ones((1, 2), dtype=torch.float32),
            torch.tensor([[0.01]], dtype=torch.float32),
            confidence_floor=0.05,
        )
        self.assertTrue(torch.allclose(merged, torch.full((1, 2), 0.05, dtype=torch.float32)))

    def test_online_retimed_weight_does_not_shrink_clean_short_samples(self) -> None:
        planner = SimpleNamespace(
            speech_budget_win=torch.tensor([[0.2]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            raw_speech_budget_win=torch.tensor([[0.2]], dtype=torch.float32),
            raw_pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            effective_speech_budget_win=torch.tensor([[0.2]], dtype=torch.float32),
            effective_pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            feasible_total_budget_delta=torch.tensor([[0.0]], dtype=torch.float32),
        )
        execution = SimpleNamespace(planner=planner)
        sample = {
            "mels": torch.zeros((1, 3, 80), dtype=torch.float32),
            "rhythm_teacher_confidence": torch.tensor([[0.5]], dtype=torch.float32),
        }
        model_out = {
            "rhythm_execution": execution,
            "rhythm_online_retimed_mel_tgt": torch.ones((1, 3, 80), dtype=torch.float32),
            "rhythm_online_retimed_frame_weight": torch.ones((1, 3), dtype=torch.float32),
        }
        _, is_retimed, frame_weight, source = resolve_acoustic_target_post_model(
            sample,
            model_out,
            hparams={
                "rhythm_use_retimed_target_if_available": True,
                "rhythm_retimed_target_start_steps": 0,
                "rhythm_online_retimed_target_start_steps": 0,
                "rhythm_retimed_target_mode": "hybrid",
                "rhythm_retimed_confidence_floor": 0.05,
            },
            global_step=0,
            apply_rhythm_render=True,
            infer=False,
            test=False,
        )
        self.assertTrue(is_retimed)
        self.assertEqual(source, "online")
        self.assertTrue(torch.allclose(frame_weight, torch.full((1, 3), 0.5, dtype=torch.float32)))

    def test_online_retimed_weight_uses_confidence_and_repair_gate(self) -> None:
        planner = SimpleNamespace(
            speech_budget_win=torch.tensor([[3.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            raw_speech_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
            raw_pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            effective_speech_budget_win=torch.tensor([[3.0]], dtype=torch.float32),
            effective_pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            feasible_total_budget_delta=torch.tensor([[1.0]], dtype=torch.float32),
        )
        execution = SimpleNamespace(planner=planner)
        sample = {
            "mels": torch.zeros((1, 3, 80), dtype=torch.float32),
            "rhythm_teacher_confidence": torch.tensor([[0.5]], dtype=torch.float32),
        }
        model_out = {
            "rhythm_execution": execution,
            "rhythm_online_retimed_mel_tgt": torch.ones((1, 3, 80), dtype=torch.float32),
            "rhythm_online_retimed_frame_weight": torch.ones((1, 3), dtype=torch.float32),
        }
        _, is_retimed, frame_weight, source = resolve_acoustic_target_post_model(
            sample,
            model_out,
            hparams={
                "rhythm_use_retimed_target_if_available": True,
                "rhythm_retimed_target_start_steps": 0,
                "rhythm_online_retimed_target_start_steps": 0,
                "rhythm_retimed_target_mode": "hybrid",
                "rhythm_retimed_confidence_floor": 0.05,
            },
            global_step=0,
            apply_rhythm_render=True,
            infer=False,
            test=False,
        )
        self.assertTrue(is_retimed)
        self.assertEqual(source, "online")
        self.assertTrue(torch.allclose(frame_weight, torch.full((1, 3), 0.375, dtype=torch.float32)))

    def test_online_retimed_weight_also_uses_trace_reliability_gate(self) -> None:
        planner = SimpleNamespace(
            speech_budget_win=torch.tensor([[3.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            raw_speech_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
            raw_pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            effective_speech_budget_win=torch.tensor([[3.0]], dtype=torch.float32),
            effective_pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            feasible_total_budget_delta=torch.tensor([[1.0]], dtype=torch.float32),
            local_trace_path_weight=torch.tensor([0.20], dtype=torch.float32),
        )
        execution = SimpleNamespace(planner=planner)
        sample = {
            "mels": torch.zeros((1, 3, 80), dtype=torch.float32),
            "rhythm_teacher_confidence": torch.tensor([[0.5]], dtype=torch.float32),
        }
        model_out = {
            "rhythm_execution": execution,
            "rhythm_online_retimed_mel_tgt": torch.ones((1, 3, 80), dtype=torch.float32),
            "rhythm_online_retimed_frame_weight": torch.ones((1, 3), dtype=torch.float32),
        }
        _, is_retimed, frame_weight, source = resolve_acoustic_target_post_model(
            sample,
            model_out,
            hparams={
                "rhythm_use_retimed_target_if_available": True,
                "rhythm_retimed_target_start_steps": 0,
                "rhythm_online_retimed_target_start_steps": 0,
                "rhythm_retimed_target_mode": "hybrid",
                "rhythm_retimed_confidence_floor": 0.05,
            },
            global_step=0,
            apply_rhythm_render=True,
            infer=False,
            test=False,
        )
        self.assertTrue(is_retimed)
        self.assertEqual(source, "online")
        self.assertTrue(torch.allclose(frame_weight, torch.full((1, 3), 0.075, dtype=torch.float32)))
        self.assertTrue(torch.allclose(model_out["rhythm_online_retimed_trace_gate"], torch.tensor([[0.2]])))

    def test_attach_acoustic_target_bundle_disables_pitch_when_retimed_pitch_missing(self) -> None:
        support = RhythmTaskRuntimeSupport(self._runtime_owner())
        output = {
            "mel_out": torch.zeros((1, 3, 80), dtype=torch.float32),
        }
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_runtime_support.hparams",
            {"rhythm_allow_source_pitch_fallback_when_retimed": False},
            clear=True,
        ):
            support.attach_acoustic_target_bundle(
                output,
                acoustic_target=torch.zeros((1, 3, 80), dtype=torch.float32),
                acoustic_target_is_retimed=True,
                acoustic_weight=None,
                acoustic_target_source="cached",
                disable_source_pitch_supervision=False,
                disable_acoustic_train_path=False,
            )
        self.assertEqual(float(output["rhythm_pitch_supervision_disabled"]), 1.0)
        self.assertEqual(float(output["rhythm_missing_retimed_pitch_target"]), 1.0)

    def test_attach_acoustic_target_bundle_keeps_pitch_when_retimed_pitch_present(self) -> None:
        support = RhythmTaskRuntimeSupport(self._runtime_owner())
        output = {
            "mel_out": torch.zeros((1, 3, 80), dtype=torch.float32),
            "retimed_f0_tgt": torch.ones((1, 3), dtype=torch.float32),
            "retimed_uv_tgt": torch.zeros((1, 3), dtype=torch.float32),
        }
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_runtime_support.hparams",
            {"rhythm_allow_source_pitch_fallback_when_retimed": False},
            clear=True,
        ):
            support.attach_acoustic_target_bundle(
                output,
                acoustic_target=torch.zeros((1, 3, 80), dtype=torch.float32),
                acoustic_target_is_retimed=True,
                acoustic_weight=None,
                acoustic_target_source="online",
                disable_source_pitch_supervision=False,
                disable_acoustic_train_path=False,
            )
        self.assertEqual(float(output["rhythm_pitch_supervision_disabled"]), 0.0)
        self.assertEqual(float(output["rhythm_missing_retimed_pitch_target"]), 0.0)

    def test_attach_acoustic_target_bundle_allows_explicit_source_pitch_fallback(self) -> None:
        support = RhythmTaskRuntimeSupport(self._runtime_owner())
        output = {
            "mel_out": torch.zeros((1, 3, 80), dtype=torch.float32),
        }
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_runtime_support.hparams",
            {"rhythm_allow_source_pitch_fallback_when_retimed": True},
            clear=True,
        ):
            support.attach_acoustic_target_bundle(
                output,
                acoustic_target=torch.zeros((1, 3, 80), dtype=torch.float32),
                acoustic_target_is_retimed=True,
                acoustic_weight=None,
                acoustic_target_source="cached",
                disable_source_pitch_supervision=False,
                disable_acoustic_train_path=False,
            )
        self.assertEqual(float(output["rhythm_pitch_supervision_disabled"]), 0.0)
        self.assertEqual(float(output["rhythm_missing_retimed_pitch_target"]), 0.0)


if __name__ == "__main__":
    unittest.main()

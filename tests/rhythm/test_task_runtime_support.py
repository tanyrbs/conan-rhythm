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

from tasks.Conan.rhythm.runtime_modes import merge_retimed_weight
from tasks.Conan.rhythm.task_runtime_support import RhythmTaskRuntimeSupport


class RhythmTaskRuntimeSupportTests(unittest.TestCase):
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
                "rhythm_suppress_duplicate_primary_distill": True,
            },
            clear=True,
        ):
            config = support.build_rhythm_target_build_config()
        self.assertTrue(config.dedupe_primary_teacher_cache_distill)
        self.assertEqual(config.distill_exec_weight, 0.0)

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


if __name__ == "__main__":
    unittest.main()

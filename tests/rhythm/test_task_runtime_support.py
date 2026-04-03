from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


if __name__ == "__main__":
    unittest.main()

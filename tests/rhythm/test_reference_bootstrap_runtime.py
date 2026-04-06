from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.dataset_sample_builder import RhythmDatasetSampleAssembler
from tasks.Conan.rhythm.metrics import build_rhythm_metric_dict


class _AssemblerOwner:
    def __init__(self, *, require_external_reference: bool, allow_identity_pairs: bool = False) -> None:
        self.hparams = {
            "rhythm_require_external_reference": require_external_reference,
            "rhythm_allow_identity_pairs": allow_identity_pairs,
        }

    @staticmethod
    def _coerce_content_sequence(content) -> list[int]:
        return [int(x) for x in np.asarray(content).reshape(-1).tolist()]

    @staticmethod
    def _resolve_rhythm_target_mode() -> str:
        return "runtime_only"

    @staticmethod
    def _select_streaming_visible_tokens(tokens, *, item_name: str):
        return np.asarray(tokens)

    @staticmethod
    def _resolve_optional_sample_keys() -> tuple[str, ...]:
        return (
            "content_units",
            "dur_anchor_src",
            "open_run_mask",
            "sealed_mask",
            "sep_hint",
            "boundary_confidence",
            "ref_rhythm_stats",
            "ref_rhythm_trace",
            "rhythm_reference_is_self",
        )

    @staticmethod
    def _get_source_rhythm_cache(item, visible_tokens, *, target_mode: str):
        return {
            "content_units": np.asarray(item["content_units"]),
            "dur_anchor_src": np.asarray(item["dur_anchor_src"]),
            "open_run_mask": np.asarray(item["open_run_mask"]),
            "sealed_mask": np.asarray(item["sealed_mask"]),
            "sep_hint": np.asarray(item["sep_hint"]),
            "boundary_confidence": np.asarray(item["boundary_confidence"]),
        }

    @staticmethod
    def _should_export_streaming_offline_sidecars() -> bool:
        return False

    @staticmethod
    def _should_export_offline_teacher_aux() -> bool:
        return False

    @staticmethod
    def _should_export_streaming_prefix_meta() -> bool:
        return False

    @staticmethod
    def _should_use_self_rhythm_reference(item, *, target_mode: str) -> bool:
        return False

    @staticmethod
    def _get_reference_rhythm_conditioning(ref_item, sample, *, target_mode: str):
        return {
            "ref_rhythm_stats": np.asarray(ref_item["ref_rhythm_stats"]),
            "ref_rhythm_trace": np.asarray(ref_item["ref_rhythm_trace"]),
        }

    @staticmethod
    def _merge_rhythm_targets(item, source_cache, ref_conditioning, sample):
        return {}


class ReferenceBootstrapRuntimeTests(unittest.TestCase):
    @staticmethod
    def _build_item(name: str) -> dict:
        return {
            "item_name": name,
            "hubert": np.asarray([1, 2, 3], dtype=np.int64),
            "content_units": np.asarray([1, 2, 3], dtype=np.int64),
            "dur_anchor_src": np.asarray([1, 1, 1], dtype=np.int64),
            "open_run_mask": np.asarray([0, 0, 0], dtype=np.int64),
            "sealed_mask": np.asarray([1, 1, 1], dtype=np.float32),
            "sep_hint": np.asarray([0, 0, 1], dtype=np.int64),
            "boundary_confidence": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            "ref_rhythm_stats": np.asarray([0.2, 2.0, 4.0, 0.1, 0.3, 0.8], dtype=np.float32),
            "ref_rhythm_trace": np.zeros((8, 5), dtype=np.float32),
        }

    @staticmethod
    def _build_sample() -> dict:
        mel = torch.zeros((3, 80), dtype=torch.float32)
        return {"mel": mel, "ref_mel": mel.clone()}

    def test_sample_builder_marks_external_reference_usage(self) -> None:
        owner = _AssemblerOwner(require_external_reference=False)
        assembler = RhythmDatasetSampleAssembler(owner)
        item = self._build_item("src")
        ref_item = self._build_item("ref")

        sample = assembler.assemble(
            sample=self._build_sample(),
            item=item,
            ref_item=ref_item,
            item_name="src",
        )

        self.assertIn("rhythm_reference_is_self", sample)
        self.assertTrue(torch.allclose(sample["rhythm_reference_is_self"], torch.tensor([0.0])))

    def test_sample_builder_fail_fast_blocks_self_fallback_when_external_reference_is_required(self) -> None:
        owner = _AssemblerOwner(require_external_reference=True)
        assembler = RhythmDatasetSampleAssembler(owner)
        item = self._build_item("singleton")

        with self.assertRaises(RuntimeError):
            assembler.assemble(
                sample=self._build_sample(),
                item=item,
                ref_item=item,
                item_name="singleton",
            )

    def test_explicit_identity_pair_can_bypass_external_reference_fail_fast_when_enabled(self) -> None:
        owner = _AssemblerOwner(require_external_reference=True, allow_identity_pairs=True)
        assembler = RhythmDatasetSampleAssembler(owner)
        item = self._build_item("identity")
        sample = self._build_sample()
        sample["rhythm_pair_is_identity"] = torch.tensor([1.0], dtype=torch.float32)
        sample["_rhythm_pair_manifest_entry"] = True

        built = assembler.assemble(
            sample=sample,
            item=item,
            ref_item=item,
            item_name="identity",
        )

        self.assertIn("rhythm_reference_is_self", built)
        self.assertTrue(torch.allclose(built["rhythm_reference_is_self"], torch.tensor([1.0])))

    def test_metrics_expose_reference_self_and_external_rates(self) -> None:
        planner = type(
            "Planner",
            (),
            {
                "speech_budget_win": torch.tensor([[1.0]], dtype=torch.float32),
                "pause_budget_win": torch.tensor([[0.0]], dtype=torch.float32),
                "raw_speech_budget_win": torch.tensor([[1.0]], dtype=torch.float32),
                "raw_pause_budget_win": torch.tensor([[0.0]], dtype=torch.float32),
                "dur_shape_unit": torch.tensor([[0.0]], dtype=torch.float32),
                "pause_shape_unit": torch.tensor([[1.0]], dtype=torch.float32),
                "boundary_score_unit": torch.tensor([[0.0]], dtype=torch.float32),
                "trace_context": torch.zeros((1, 1, 3), dtype=torch.float32),
            },
        )()
        execution = type(
            "Execution",
            (),
            {
                "speech_duration_exec": torch.tensor([[1.0]], dtype=torch.float32),
                "blank_duration_exec": torch.tensor([[0.0]], dtype=torch.float32),
                "pause_after_exec": torch.tensor([[0.0]], dtype=torch.float32),
                "planner": planner,
                "commit_frontier": torch.tensor([1], dtype=torch.long),
            },
        )()
        output = {
            "rhythm_execution": execution,
            "rhythm_unit_batch": type(
                "UnitBatch",
                (),
                {
                    "unit_mask": torch.tensor([[1.0]], dtype=torch.float32),
                    "dur_anchor_src": torch.tensor([[1.0]], dtype=torch.float32),
                },
            )(),
        }

        metrics = build_rhythm_metric_dict(
            output,
            sample={"rhythm_reference_is_self": torch.tensor([[0.0]], dtype=torch.float32)},
        )

        self.assertTrue(torch.allclose(metrics["rhythm_metric_reference_self_rate"], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_reference_external_rate"], torch.tensor(1.0)))


if __name__ == "__main__":
    unittest.main()

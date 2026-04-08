from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.bridge import run_rhythm_frontend
from modules.Conan.rhythm.unit_frontend import RhythmUnitFrontend
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

    def test_from_precomputed_rebuilds_tail_open_masks_when_cache_sidecars_are_missing(self) -> None:
        frontend = RhythmUnitFrontend(tail_open_units=1)
        batch = frontend.from_precomputed(
            content_units=torch.tensor([[1, 2, 3]], dtype=torch.long),
            dur_anchor_src=torch.tensor([[1, 1, 1]], dtype=torch.long),
        )
        self.assertTrue(torch.equal(batch.open_run_mask, torch.tensor([[0, 0, 1]], dtype=torch.long)))
        self.assertTrue(torch.allclose(batch.sealed_mask, torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)))

    def test_run_rhythm_frontend_uses_open_tail_during_training_when_streaming_prefix_train_enabled(self) -> None:
        class _FakeFrontend:
            def __init__(self) -> None:
                self.mark_last_open = None

            def from_content_tensor(self, content, *, content_lengths=None, mark_last_open=True):
                self.mark_last_open = bool(mark_last_open)
                return type(
                    "Batch",
                    (),
                    {
                        "content_units": content.long(),
                        "dur_anchor_src": torch.ones_like(content, dtype=torch.long),
                        "unit_mask": torch.ones_like(content, dtype=torch.float32),
                        "open_run_mask": torch.zeros_like(content, dtype=torch.long),
                        "sealed_mask": torch.ones_like(content, dtype=torch.float32),
                        "sep_hint": torch.zeros_like(content, dtype=torch.long),
                        "boundary_confidence": torch.zeros_like(content, dtype=torch.float32),
                    },
                )()

        class _FakeModule:
            @staticmethod
            def build_reference_conditioning(*, ref_conditioning=None, ref_mel=None):
                return {"ref_rhythm_stats": torch.zeros((1, 6)), "ref_rhythm_trace": torch.zeros((1, 8, 5))}

            def __call__(self, **kwargs):
                return {"ok": True, "state": kwargs.get("state")}

        frontend = _FakeFrontend()
        module = _FakeModule()
        bundle = run_rhythm_frontend(
            rhythm_enable_v2=True,
            rhythm_unit_frontend=frontend,
            rhythm_module=module,
            content=torch.tensor([[1, 2, 3]], dtype=torch.long),
            ref=torch.zeros((1, 4, 80), dtype=torch.float32),
            infer=False,
            streaming_prefix_train=True,
        )
        self.assertTrue(bundle is not None)
        self.assertTrue(frontend.mark_last_open)

    def test_run_rhythm_frontend_inherits_module_phase_decoupled_default_without_override(self) -> None:
        class _FakeFrontend:
            def from_content_tensor(self, content, *, content_lengths=None, mark_last_open=True):
                return type(
                    "Batch",
                    (),
                    {
                        "content_units": content.long(),
                        "dur_anchor_src": torch.ones_like(content, dtype=torch.long),
                        "unit_mask": torch.ones_like(content, dtype=torch.float32),
                        "open_run_mask": torch.zeros_like(content, dtype=torch.long),
                        "sealed_mask": torch.ones_like(content, dtype=torch.float32),
                        "sep_hint": torch.zeros_like(content, dtype=torch.long),
                        "boundary_confidence": torch.zeros_like(content, dtype=torch.float32),
                    },
                )()

        class _FakeModule:
            phase_decoupled_timing = True
            enable_learned_offline_teacher = False

            def __init__(self) -> None:
                self.forward_kwargs = None

            @staticmethod
            def build_reference_conditioning(*, ref_conditioning=None, ref_mel=None):
                return {"ref_rhythm_stats": torch.zeros((1, 6)), "ref_rhythm_trace": torch.zeros((1, 8, 5))}

            def __call__(self, **kwargs):
                self.forward_kwargs = kwargs
                return {"ok": True, "state": kwargs.get("state")}

        frontend = _FakeFrontend()
        module = _FakeModule()
        run_rhythm_frontend(
            rhythm_enable_v2=True,
            rhythm_unit_frontend=frontend,
            rhythm_module=module,
            content=torch.tensor([[1, 2, 3]], dtype=torch.long),
            ref=torch.zeros((1, 4, 80), dtype=torch.float32),
            infer=False,
        )
        self.assertIsNotNone(module.forward_kwargs)
        self.assertTrue(bool(module.forward_kwargs["phase_decoupled_timing"]))

    def test_run_rhythm_frontend_teacher_as_main_does_not_forward_streaming_only_scheduler_overrides(self) -> None:
        class _FakeFrontend:
            def from_content_tensor(self, content, *, content_lengths=None, mark_last_open=True):
                return type(
                    "Batch",
                    (),
                    {
                        "content_units": content.long(),
                        "dur_anchor_src": torch.ones_like(content, dtype=torch.long),
                        "unit_mask": torch.ones_like(content, dtype=torch.float32),
                        "open_run_mask": torch.zeros_like(content, dtype=torch.long),
                        "sealed_mask": torch.ones_like(content, dtype=torch.float32),
                        "sep_hint": torch.zeros_like(content, dtype=torch.long),
                        "boundary_confidence": torch.zeros_like(content, dtype=torch.float32),
                    },
                )()

        class _FakeModule:
            phase_decoupled_timing = True
            enable_learned_offline_teacher = True

            def __init__(self) -> None:
                self.teacher_kwargs = None

            @staticmethod
            def build_reference_conditioning(*, ref_conditioning=None, ref_mel=None):
                return {"ref_rhythm_stats": torch.zeros((1, 6)), "ref_rhythm_trace": torch.zeros((1, 8, 5))}

            def forward_teacher(self, **kwargs):
                self.teacher_kwargs = kwargs
                return {"ok": True}, {"overall": torch.ones((1, 1), dtype=torch.float32)}

        module = _FakeModule()
        run_rhythm_frontend(
            rhythm_enable_v2=True,
            rhythm_unit_frontend=_FakeFrontend(),
            rhythm_module=module,
            content=torch.tensor([[1, 2, 3]], dtype=torch.long),
            ref=torch.zeros((1, 4, 80), dtype=torch.float32),
            infer=False,
            teacher_as_main=True,
            teacher_projector_force_full_commit=True,
            teacher_projector_soft_pause_selection=True,
            phase_decoupled_boundary_style_residual_scale=0.3,
            debt_control_scale=2.0,
            debt_pause_priority=0.2,
            debt_speech_priority=0.4,
            projector_debt_leak=0.1,
        )
        self.assertIsNotNone(module.teacher_kwargs)
        assert module.teacher_kwargs is not None
        self.assertNotIn("phase_decoupled_boundary_style_residual_scale", module.teacher_kwargs)
        self.assertNotIn("debt_control_scale", module.teacher_kwargs)
        self.assertNotIn("debt_pause_priority", module.teacher_kwargs)
        self.assertNotIn("debt_speech_priority", module.teacher_kwargs)
        self.assertIn("projector_debt_leak", module.teacher_kwargs)
        self.assertIn("projector_force_full_commit", module.teacher_kwargs)
        self.assertIn("projector_soft_pause_selection", module.teacher_kwargs)
        self.assertTrue(bool(module.teacher_kwargs["projector_soft_pause_selection"]))


if __name__ == "__main__":
    unittest.main()

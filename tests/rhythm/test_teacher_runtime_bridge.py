from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.bridge import run_rhythm_frontend
from modules.Conan.rhythm.factory import build_streaming_rhythm_module_from_hparams
from modules.Conan.rhythm.runtime_adapter import ConanRhythmAdapter


class TeacherRuntimeBridgeTests(unittest.TestCase):
    @staticmethod
    def _build_batch(content: torch.Tensor):
        return type(
            "Batch",
            (),
            {
                "content_units": content.long(),
                "dur_anchor_src": torch.ones_like(content, dtype=torch.long),
                "unit_mask": torch.ones_like(content, dtype=torch.float32),
                "open_run_mask": torch.tensor([[0, 1, 1]], dtype=torch.long).expand_as(content),
                "sealed_mask": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).expand_as(content),
                "sep_hint": torch.zeros_like(content, dtype=torch.long),
                "boundary_confidence": torch.zeros_like(content, dtype=torch.float32),
            },
        )()

    def test_run_rhythm_frontend_teacher_as_main_routes_to_forward_teacher(self) -> None:
        class _FakeFrontend:
            def from_content_tensor(self, content, *, content_lengths=None, mark_last_open=True):
                return TeacherRuntimeBridgeTests._build_batch(content)

        class _FakeModule:
            enable_learned_offline_teacher = True

            def __init__(self) -> None:
                self.teacher_kwargs = None

            @staticmethod
            def build_reference_conditioning(*, ref_conditioning=None, ref_mel=None):
                return {
                    "ref_rhythm_stats": torch.zeros((1, 6), dtype=torch.float32),
                    "ref_rhythm_trace": torch.zeros((1, 8, 5), dtype=torch.float32),
                }

            def forward_teacher(self, **kwargs):
                self.teacher_kwargs = kwargs
                return {"teacher": True}, {"overall": torch.tensor([[0.9]], dtype=torch.float32)}

            def __call__(self, **kwargs):
                raise AssertionError("teacher_as_main should not call the streaming path")

            def forward_dual(self, **kwargs):
                raise AssertionError("teacher_as_main should not call dual mode")

        module = _FakeModule()
        bundle = run_rhythm_frontend(
            rhythm_enable_v2=True,
            rhythm_unit_frontend=_FakeFrontend(),
            rhythm_module=module,
            content=torch.tensor([[1, 2, 3]], dtype=torch.long),
            ref=torch.zeros((1, 4, 80), dtype=torch.float32),
            infer=False,
            teacher_as_main=True,
        )
        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertTrue(bool(bundle["teacher_as_main"]))
        self.assertEqual(bundle["execution"], {"teacher": True})
        self.assertIsNone(bundle["offline_execution"])
        self.assertIsNone(bundle["algorithmic_teacher"])
        self.assertIn("overall", bundle["offline_confidence"])
        self.assertIsNotNone(module.teacher_kwargs)
        assert module.teacher_kwargs is not None
        self.assertNotIn("open_run_mask", module.teacher_kwargs)
        self.assertNotIn("sealed_mask", module.teacher_kwargs)

    def test_run_rhythm_frontend_teacher_as_main_requires_teacher_runtime(self) -> None:
        class _FakeFrontend:
            def from_content_tensor(self, content, *, content_lengths=None, mark_last_open=True):
                return TeacherRuntimeBridgeTests._build_batch(content)

        class _FakeModule:
            enable_learned_offline_teacher = False

            @staticmethod
            def build_reference_conditioning(*, ref_conditioning=None, ref_mel=None):
                return {
                    "ref_rhythm_stats": torch.zeros((1, 6), dtype=torch.float32),
                    "ref_rhythm_trace": torch.zeros((1, 8, 5), dtype=torch.float32),
                }

            def __call__(self, **kwargs):
                raise AssertionError("misconfigured teacher_as_main should fail before streaming fallback")

        with self.assertRaisesRegex(ValueError, "rhythm_teacher_as_main requires"):
            run_rhythm_frontend(
                rhythm_enable_v2=True,
                rhythm_unit_frontend=_FakeFrontend(),
                rhythm_module=_FakeModule(),
                content=torch.tensor([[1, 2, 3]], dtype=torch.long),
                ref=torch.zeros((1, 4, 80), dtype=torch.float32),
                infer=False,
                teacher_as_main=True,
            )

    def test_run_rhythm_frontend_teacher_as_main_is_disabled_during_infer(self) -> None:
        class _FakeFrontend:
            def from_content_tensor(self, content, *, content_lengths=None, mark_last_open=True):
                return TeacherRuntimeBridgeTests._build_batch(content)

        class _FakeModule:
            enable_learned_offline_teacher = True

            def __init__(self) -> None:
                self.forward_kwargs = None

            @staticmethod
            def build_reference_conditioning(*, ref_conditioning=None, ref_mel=None):
                return {
                    "ref_rhythm_stats": torch.zeros((1, 6), dtype=torch.float32),
                    "ref_rhythm_trace": torch.zeros((1, 8, 5), dtype=torch.float32),
                }

            def forward_teacher(self, **kwargs):
                raise AssertionError("infer=True should not activate teacher_as_main")

            def __call__(self, **kwargs):
                self.forward_kwargs = kwargs
                return {"streaming": True}

        module = _FakeModule()
        bundle = run_rhythm_frontend(
            rhythm_enable_v2=True,
            rhythm_unit_frontend=_FakeFrontend(),
            rhythm_module=module,
            content=torch.tensor([[1, 2, 3]], dtype=torch.long),
            ref=torch.zeros((1, 4, 80), dtype=torch.float32),
            infer=True,
            teacher_as_main=True,
        )
        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertFalse(bool(bundle["teacher_as_main"]))
        self.assertEqual(bundle["execution"], {"streaming": True})
        self.assertIsNotNone(module.forward_kwargs)

    def test_run_rhythm_frontend_dual_mode_routes_to_forward_dual(self) -> None:
        class _FakeFrontend:
            def from_content_tensor(self, content, *, content_lengths=None, mark_last_open=True):
                return TeacherRuntimeBridgeTests._build_batch(content)

        class _FakeModule:
            enable_learned_offline_teacher = True

            def __init__(self) -> None:
                self.dual_kwargs = None

            @staticmethod
            def build_reference_conditioning(*, ref_conditioning=None, ref_mel=None):
                return {
                    "ref_rhythm_stats": torch.zeros((1, 6), dtype=torch.float32),
                    "ref_rhythm_trace": torch.zeros((1, 8, 5), dtype=torch.float32),
                }

            def forward_dual(self, **kwargs):
                self.dual_kwargs = kwargs
                return {
                    "streaming_execution": {"streaming": True},
                    "offline_execution": {"teacher": True},
                    "offline_confidence": {"overall": torch.tensor([[0.8]], dtype=torch.float32)},
                    "algorithmic_teacher": None,
                }

            def __call__(self, **kwargs):
                raise AssertionError("dual mode should call forward_dual")

            def forward_teacher(self, **kwargs):
                raise AssertionError("dual mode should not call teacher_as_main path")

        module = _FakeModule()
        bundle = run_rhythm_frontend(
            rhythm_enable_v2=True,
            rhythm_unit_frontend=_FakeFrontend(),
            rhythm_module=module,
            content=torch.tensor([[1, 2, 3]], dtype=torch.long),
            ref=torch.zeros((1, 4, 80), dtype=torch.float32),
            infer=False,
            enable_dual_mode_teacher=True,
        )
        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertFalse(bool(bundle["teacher_as_main"]))
        self.assertEqual(bundle["execution"], {"streaming": True})
        self.assertEqual(bundle["offline_execution"], {"teacher": True})
        self.assertIn("overall", bundle["offline_confidence"])
        self.assertIsNone(bundle["algorithmic_teacher"])
        self.assertIsNotNone(module.dual_kwargs)

    def test_runtime_adapter_passes_teacher_runtime_overrides_to_bridge(self) -> None:
        hparams = {
            "hidden_size": 8,
            "rhythm_hidden_size": 8,
            "content_vocab_size": 32,
            "rhythm_enable_v2": True,
            "rhythm_stage": "teacher_offline",
            "rhythm_teacher_as_main": True,
            "rhythm_enable_learned_offline_teacher": True,
            "rhythm_runtime_enable_learned_offline_teacher": True,
            "rhythm_apply_mode": "off",
        }
        adapter = ConanRhythmAdapter(hparams, hidden_size=8)
        content = torch.tensor([[1, 2, 3]], dtype=torch.long)
        ref = torch.zeros((1, 4, 80), dtype=torch.float32)
        content_embed = torch.zeros((1, 3, 8), dtype=torch.float32)
        tgt_nonpadding = torch.ones((1, 3, 1), dtype=torch.float32)
        ret: dict = {}

        with (
            patch("modules.Conan.rhythm.runtime_adapter.run_rhythm_frontend") as run_mock,
            patch(
                "modules.Conan.rhythm.runtime_adapter.attach_rhythm_outputs",
                side_effect=lambda **kwargs: (kwargs["content_embed"], kwargs["tgt_nonpadding"]),
            ),
        ):
            run_mock.return_value = None
            adapter.forward(
                ret=ret,
                content=content,
                ref=ref,
                target=None,
                f0=None,
                uv=None,
                infer=False,
                global_steps=0,
                content_embed=content_embed,
                tgt_nonpadding=tgt_nonpadding,
                rhythm_runtime_overrides={
                    "teacher_projector_soft_pause_selection": True,
                    "teacher_projector_force_full_commit": False,
                    "phase_decoupled_phrase_boundary_threshold": 0.61,
                },
            )

        kwargs = run_mock.call_args.kwargs
        self.assertTrue(bool(kwargs["teacher_projector_soft_pause_selection"]))
        self.assertFalse(bool(kwargs["teacher_projector_force_full_commit"]))
        self.assertAlmostEqual(float(kwargs["phase_decoupled_phrase_gate_boundary_threshold"]), 0.61, places=6)

    def test_runtime_adapter_accepts_legacy_teacher_soft_pause_override_alias(self) -> None:
        hparams = {
            "hidden_size": 8,
            "rhythm_hidden_size": 8,
            "content_vocab_size": 32,
            "rhythm_enable_v2": True,
            "rhythm_stage": "teacher_offline",
            "rhythm_teacher_as_main": True,
            "rhythm_enable_learned_offline_teacher": True,
            "rhythm_runtime_enable_learned_offline_teacher": True,
            "rhythm_apply_mode": "off",
        }
        adapter = ConanRhythmAdapter(hparams, hidden_size=8)
        content = torch.tensor([[1, 2, 3]], dtype=torch.long)
        ref = torch.zeros((1, 4, 80), dtype=torch.float32)
        content_embed = torch.zeros((1, 3, 8), dtype=torch.float32)
        tgt_nonpadding = torch.ones((1, 3, 1), dtype=torch.float32)

        with (
            patch("modules.Conan.rhythm.runtime_adapter.run_rhythm_frontend") as run_mock,
            patch(
                "modules.Conan.rhythm.runtime_adapter.attach_rhythm_outputs",
                side_effect=lambda **kwargs: (kwargs["content_embed"], kwargs["tgt_nonpadding"]),
            ),
        ):
            run_mock.return_value = None
            adapter.forward(
                ret={},
                content=content,
                ref=ref,
                target=None,
                f0=None,
                uv=None,
                infer=False,
                global_steps=0,
                content_embed=content_embed,
                tgt_nonpadding=tgt_nonpadding,
                rhythm_runtime_overrides={
                    "teacher_projector_soft_pause_selection_override": True,
                },
            )

        kwargs = run_mock.call_args.kwargs
        self.assertTrue(bool(kwargs["teacher_projector_soft_pause_selection"]))

    def test_runtime_adapter_warns_when_teacher_override_is_unused_without_teacher_runtime(self) -> None:
        hparams = {
            "hidden_size": 8,
            "rhythm_hidden_size": 8,
            "content_vocab_size": 32,
            "rhythm_enable_v2": True,
            "rhythm_stage": "transitional",
            "rhythm_enable_learned_offline_teacher": False,
            "rhythm_runtime_enable_learned_offline_teacher": False,
            "rhythm_apply_mode": "off",
        }
        adapter = ConanRhythmAdapter(hparams, hidden_size=8)
        content = torch.tensor([[1, 2, 3]], dtype=torch.long)
        ref = torch.zeros((1, 4, 80), dtype=torch.float32)
        content_embed = torch.zeros((1, 3, 8), dtype=torch.float32)
        tgt_nonpadding = torch.ones((1, 3, 1), dtype=torch.float32)

        with (
            patch("modules.Conan.rhythm.runtime_adapter.run_rhythm_frontend") as run_mock,
            patch(
                "modules.Conan.rhythm.runtime_adapter.attach_rhythm_outputs",
                side_effect=lambda **kwargs: (kwargs["content_embed"], kwargs["tgt_nonpadding"]),
            ),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            run_mock.return_value = None
            adapter.forward(
                ret={},
                content=content,
                ref=ref,
                target=None,
                f0=None,
                uv=None,
                infer=False,
                global_steps=0,
                content_embed=content_embed,
                tgt_nonpadding=tgt_nonpadding,
                rhythm_runtime_overrides={
                    "teacher_projector_soft_pause_selection": True,
                },
            )

        self.assertTrue(
            any("teacher_projector_soft_pause_selection" in str(item.message) for item in caught),
            msg="expected unused teacher override warning when no teacher runtime branch is active",
        )

    def test_forward_teacher_passes_canonical_soft_pause_selection_to_projector(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "content_vocab_size": 32,
                "rhythm_trace_bins": 8,
                "rhythm_emit_reference_sidecar": True,
                "rhythm_stage": "teacher_offline",
                "rhythm_enable_learned_offline_teacher": True,
                "rhythm_runtime_enable_learned_offline_teacher": True,
            }
        )

        original_projector = module.projector

        class _CaptureProjector(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.kwargs = None

            def init_state(self, *, batch_size, device):
                return original_projector.init_state(batch_size=batch_size, device=device)

            def forward(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(trace_reliability=None)

        capture_projector = _CaptureProjector()
        module.projector = capture_projector
        stats = torch.tensor([[0.20, 2.0, 4.0, 0.10, 0.30, 0.80]], dtype=torch.float32)
        trace = torch.zeros((1, 8, 5), dtype=torch.float32)
        trace[:, :, 1] = torch.linspace(0.0, 1.0, 8)
        trace[:, :, 2] = torch.linspace(0.0, 1.0, 8)
        execution, confidence = module.forward_teacher(
            content_units=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            dur_anchor_src=torch.tensor([[2.0, 3.0, 1.0, 2.0]], dtype=torch.float32),
            ref_rhythm_stats=stats,
            ref_rhythm_trace=trace,
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            projector_soft_pause_selection=True,
        )
        self.assertIsNotNone(execution)
        self.assertIn("overall", confidence)
        self.assertIsNotNone(capture_projector.kwargs)
        assert capture_projector.kwargs is not None
        self.assertTrue(bool(capture_projector.kwargs["soft_pause_selection_override"]))


if __name__ == "__main__":
    unittest.main()

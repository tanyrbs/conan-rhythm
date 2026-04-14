from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_gen.conan_binarizer import VCBinarizer
from data_gen.tts.base_binarizer import BaseBinarizer
from modules.Conan.rhythm.stages import normalize_rhythm_stage
from tasks.Conan.rhythm.dataset_contracts import RhythmDatasetCacheContract
from tasks.Conan.rhythm.config_contract_stage_rules import detect_rhythm_profile
from tasks.Conan.rhythm.preflight_support import (
    _collect_processed_data_dir_findings,
    _compose_hparams_override,
    _inspect_indexed_split_arrays,
    _inspect_indexed_split_files,
    _inspect_minimal_v1_frontend_surface,
    _inspect_pitch_feature_readiness,
    _inspect_processed_data_dir,
    _run_dataset_and_model_dry_run,
)
from tasks.Conan.rhythm.task_config import validate_rhythm_training_hparams
from utils.commons.hparams import set_hparams


class _DummyPreflightDataset:
    def __init__(self, prefix: str, shuffle: bool = False) -> None:
        self._items = [{"item_name": f"{prefix}_0"}]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        return self._items[idx]

    def collater(self, items):
        return {"content": torch.ones((len(items), 1), dtype=torch.long)}


class _DummyTeacherOnlyPreflightTask:
    def build_tts_model(self) -> None:
        return None

    def run_model(self, batch, infer: bool = False):
        return {}, {
            "rhythm_execution": object(),
            "rhythm_stage": "teacher_offline",
            "rhythm_teacher_as_main": 0.0,
            "rhythm_teacher_only_stage": 1.0,
            "rhythm_module_only_objective": 1.0,
            "rhythm_skip_acoustic_objective": 1.0,
            "disable_acoustic_train_path": 1.0,
        }


class _DummyBrokenModuleOnlyPreflightTask:
    def build_tts_model(self) -> None:
        return None

    def run_model(self, batch, infer: bool = False):
        return {}, {
            "mel_out": torch.zeros((1, 2, 3), dtype=torch.float32),
            "rhythm_execution": object(),
            "rhythm_stage": "student_kd",
            "rhythm_module_only_objective": 1.0,
            "rhythm_skip_acoustic_objective": 0.0,
            "disable_acoustic_train_path": 0.0,
        }


class _DummyBrokenTeacherAsMainPreflightTask:
    def build_tts_model(self) -> None:
        return None

    def run_model(self, batch, infer: bool = False):
        return {}, {
            "rhythm_execution": object(),
            "rhythm_stage": "teacher_offline",
            "rhythm_teacher_as_main": 1.0,
            "rhythm_teacher_only_stage": 0.0,
            "rhythm_module_only_objective": 1.0,
            "rhythm_skip_acoustic_objective": 1.0,
            "disable_acoustic_train_path": 1.0,
        }


class PreflightReadinessTests(unittest.TestCase):
    def test_cached_only_stage_alias_maps_to_minimal_v1(self) -> None:
        self.assertEqual(normalize_rhythm_stage("cached_only"), "minimal_v1")

    def test_explicit_minimal_stage_triggers_minimal_profile(self) -> None:
        profile = detect_rhythm_profile(
            {
                "rhythm_stage": "minimal_v1",
                "rhythm_distill_surface": "cache",
            },
            config_path="egs/custom_stage.yaml",
        )
        self.assertEqual(profile, "minimal_v1")

    def test_minimal_stage_requires_v3_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimal_v1 must run on rhythm_v3 only"):
            validate_rhythm_training_hparams(
                {
                    "rhythm_stage": "minimal_v1",
                    "rhythm_enable_v2": True,
                    "rhythm_enable_v3": False,
                    "rhythm_response_rank": 4,
                }
            )

    def test_preflight_flags_empty_shell_and_missing_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "valid"
            prefix.with_suffix(".data").write_bytes(b"")
            prefix.with_suffix(".idx").write_bytes(b"idx")
            issues = _inspect_indexed_split_files(str(prefix))
            self.assertTrue(any("data file is empty" in issue for issue in issues))
            self.assertTrue(any("Missing lengths file" in issue for issue in issues))

    def test_preflight_flags_length_array_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "train"
            np.save(f"{prefix}_lengths.npy", np.asarray([1, 2, 3], dtype=np.int32))
            issues = _inspect_indexed_split_arrays(str(prefix), dataset_len=2)
            self.assertTrue(any("lengths mismatch" in issue for issue in issues))

    def test_preflight_flags_missing_f0_when_pitch_is_enabled(self) -> None:
        issues = _inspect_pitch_feature_readiness(
            [{"item_name": "ok", "f0": [1.0, 2.0]}, {"item_name": "bad"}],
            split="train",
            use_pitch_embed=True,
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("missing non-empty f0", issues[0])

    def test_preflight_skips_f0_check_when_pitch_is_disabled(self) -> None:
        issues = _inspect_pitch_feature_readiness(
            [{"item_name": "bad"}],
            split="valid",
            use_pitch_embed=False,
        )
        self.assertEqual(issues, [])

    def test_preflight_treats_missing_processed_dir_as_warning_only(self) -> None:
        issues = _inspect_processed_data_dir("")
        self.assertEqual(len(issues), 1)
        self.assertIn("cached-only preflight mainly validates binary cache readiness", issues[0])

    def test_preflight_can_escalate_missing_processed_dir_for_formal_training(self) -> None:
        warnings, errors = _collect_processed_data_dir_findings("", strict=True)
        self.assertEqual(warnings, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("Strict processed-data validation is enabled", errors[0])

    def test_preflight_compose_hparams_override_accepts_binary_and_processed_dirs(self) -> None:
        args = SimpleNamespace(
            hparams="foo=1",
            binary_data_dir="data/binary/demo",
            processed_data_dir="data/processed/demo",
        )
        override = _compose_hparams_override(args)
        self.assertEqual(
            override,
            "foo=1,binary_data_dir='data/binary/demo',processed_data_dir='data/processed/demo'",
        )

    def test_rhythm_cache_scalar_contract_rejects_vectors(self) -> None:
        with self.assertRaises(RuntimeError):
            RhythmDatasetCacheContract.extract_scalar(np.asarray([1, 2], dtype=np.int32))

    @unittest.skipUnless(os.name == "nt", "Windows-specific binarizer guard")
    def test_windows_binarizer_forces_single_process(self) -> None:
        old = os.environ.get("N_PROC")
        os.environ["N_PROC"] = "4"
        try:
            base = BaseBinarizer.__new__(BaseBinarizer)
            vc = VCBinarizer.__new__(VCBinarizer)
            self.assertEqual(BaseBinarizer.num_workers.fget(base), 1)
            self.assertEqual(VCBinarizer.num_workers.fget(vc), 1)
        finally:
            if old is None:
                os.environ.pop("N_PROC", None)
            else:
                os.environ["N_PROC"] = old

    def test_convert_range_does_not_mutate_input(self) -> None:
        base = BaseBinarizer.__new__(BaseBinarizer)
        base.item_names = ["a", "b", "c"]
        vc = VCBinarizer.__new__(VCBinarizer)
        vc.item_names = ["a", "b", "c"]
        original = [0, -1]

        base_range = base._convert_range(original)
        vc_range = vc._convert_range(original)

        self.assertEqual(original, [0, -1])
        self.assertEqual(base_range, [0, 3])
        self.assertEqual(vc_range, [0, 3])

    def test_preflight_model_dry_run_allows_teacher_only_stage_without_mel_out(self) -> None:
        dataset_module = ModuleType("tasks.Conan.dataset")
        dataset_module.ConanDataset = _DummyPreflightDataset
        task_module = ModuleType("tasks.Conan.Conan")
        task_module.ConanTask = _DummyTeacherOnlyPreflightTask
        context = SimpleNamespace(stage="teacher_offline", hparams={})
        with mock.patch.dict(
            sys.modules,
            {
                "tasks.Conan.dataset": dataset_module,
                "tasks.Conan.Conan": task_module,
            },
        ):
            errors = _run_dataset_and_model_dry_run("train", context=context, run_model=True)
        self.assertEqual(errors, [])

    def test_preflight_model_dry_run_rejects_module_only_without_skip_flag(self) -> None:
        dataset_module = ModuleType("tasks.Conan.dataset")
        dataset_module.ConanDataset = _DummyPreflightDataset
        task_module = ModuleType("tasks.Conan.Conan")
        task_module.ConanTask = _DummyBrokenModuleOnlyPreflightTask
        context = SimpleNamespace(stage="student_kd", hparams={})
        with mock.patch.dict(
            sys.modules,
            {
                "tasks.Conan.dataset": dataset_module,
                "tasks.Conan.Conan": task_module,
            },
        ):
            errors = _run_dataset_and_model_dry_run("train", context=context, run_model=True)
        self.assertTrue(
            any("rhythm_module_only_objective" in error and "rhythm_skip_acoustic_objective" in error for error in errors)
        )

    def test_preflight_model_dry_run_requires_mel_out_when_teacher_runs_as_main(self) -> None:
        dataset_module = ModuleType("tasks.Conan.dataset")
        dataset_module.ConanDataset = _DummyPreflightDataset
        task_module = ModuleType("tasks.Conan.Conan")
        task_module.ConanTask = _DummyBrokenTeacherAsMainPreflightTask
        context = SimpleNamespace(stage="teacher_offline", hparams={})
        with mock.patch.dict(
            sys.modules,
            {
                "tasks.Conan.dataset": dataset_module,
                "tasks.Conan.Conan": task_module,
            },
        ):
            errors = _run_dataset_and_model_dry_run("train", context=context, run_model=True)
        self.assertTrue(any("did not produce mel_out" in error for error in errors))

    def test_preflight_flags_minimal_v1_frontend_surface_when_silence_and_boundary_contract_are_unreachable(self) -> None:
        items = [
            {
                "item_name": "train_0",
                "hubert": np.asarray([71, 71, 72, 63], dtype=np.int64),
                "content_units": np.asarray([71, 72], dtype=np.int64),
                "dur_anchor_src": np.asarray([2, 2], dtype=np.int64),
                "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
                "sep_hint": np.asarray([0.0, 0.0], dtype=np.float32),
                "boundary_confidence": np.asarray([0.46, 0.48], dtype=np.float32),
                "source_boundary_cue": np.asarray([0.31, 0.44], dtype=np.float32),
            }
        ]
        summary, warnings, errors = _inspect_minimal_v1_frontend_surface(
            items,
            split="train",
            hparams={
                "rhythm_enable_v3": True,
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_gate_quality_strict": True,
                "silent_token": 57,
                "rhythm_v3_min_boundary_confidence_for_g": 0.5,
                "rhythm_source_phrase_threshold": 0.55,
            },
            profile="minimal_v1",
            strict_contract=True,
        )
        self.assertEqual(warnings, [])
        self.assertIsNotNone(summary)
        self.assertEqual(summary["raw_silent_token_items"], 0)
        self.assertEqual(summary["source_silence_items"], 0)
        self.assertEqual(summary["sep_nonzero_items"], 0)
        self.assertTrue(any("silent_token=57" in error for error in errors))
        self.assertTrue(any("min_boundary_confidence_for_g=0.500" in error for error in errors))
        self.assertTrue(any("source_phrase_threshold=0.550" in error for error in errors))

    def test_preflight_frontend_surface_downgrades_findings_to_warnings_without_strict_gate(self) -> None:
        items = [
            {
                "item_name": "train_0",
                "hubert": np.asarray([71, 71, 72, 63], dtype=np.int64),
                "content_units": np.asarray([71, 72], dtype=np.int64),
                "dur_anchor_src": np.asarray([2, 2], dtype=np.int64),
                "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
                "sep_hint": np.asarray([0.0, 0.0], dtype=np.float32),
                "boundary_confidence": np.asarray([0.46, 0.48], dtype=np.float32),
                "source_boundary_cue": np.asarray([0.31, 0.44], dtype=np.float32),
            }
        ]
        summary, warnings, errors = _inspect_minimal_v1_frontend_surface(
            items,
            split="train",
            hparams={
                "rhythm_enable_v3": True,
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_gate_quality_strict": False,
                "silent_token": 57,
                "rhythm_v3_min_boundary_confidence_for_g": 0.5,
                "rhythm_source_phrase_threshold": 0.55,
            },
            profile="minimal_v1",
            strict_contract=False,
        )
        self.assertEqual(errors, [])
        self.assertIsNotNone(summary)
        self.assertGreaterEqual(len(warnings), 3)

    def test_preflight_frontend_surface_uses_hparam_flag_even_when_profile_is_default(self) -> None:
        items = [
            {
                "item_name": "train_0",
                "hubert": np.asarray([71, 71, 72, 63], dtype=np.int64),
                "content_units": np.asarray([71, 72], dtype=np.int64),
                "dur_anchor_src": np.asarray([2, 2], dtype=np.int64),
                "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
                "sep_hint": np.asarray([0.0, 0.0], dtype=np.float32),
                "boundary_confidence": np.asarray([0.46, 0.48], dtype=np.float32),
                "source_boundary_cue": np.asarray([0.31, 0.44], dtype=np.float32),
            }
        ]
        summary, warnings, errors = _inspect_minimal_v1_frontend_surface(
            items,
            split="train",
            hparams={
                "rhythm_enable_v3": True,
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_gate_quality_strict": False,
                "silent_token": 57,
                "rhythm_v3_min_boundary_confidence_for_g": 0.5,
                "rhythm_source_phrase_threshold": 0.55,
            },
            profile="default",
            strict_contract=False,
        )
        self.assertIsNotNone(summary)
        self.assertEqual(errors, [])
        self.assertGreaterEqual(len(warnings), 3)

    def test_preflight_frontend_surface_accepts_acoustic_silence_sidecars_without_raw_silent_token(self) -> None:
        items = [
            {
                "item_name": "train_0",
                "hubert": np.asarray([71, 71, 71, 71, 71, 71], dtype=np.int64),
                "content_units": np.asarray([71, 71, 71], dtype=np.int64),
                "dur_anchor_src": np.asarray([2, 2, 2], dtype=np.int64),
                "source_silence_mask": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                "sep_hint": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
                "boundary_confidence": np.asarray([0.70, 0.84, 0.71], dtype=np.float32),
                "source_boundary_cue": np.asarray([0.62, 0.78, 0.63], dtype=np.float32),
            }
        ]
        summary, warnings, errors = _inspect_minimal_v1_frontend_surface(
            items,
            split="train",
            hparams={
                "rhythm_enable_v3": True,
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_gate_quality_strict": True,
                "silent_token": 57,
                "rhythm_v3_min_boundary_confidence_for_g": 0.5,
                "rhythm_source_phrase_threshold": 0.55,
            },
            profile="minimal_v1",
            strict_contract=True,
        )
        self.assertIsNotNone(summary)
        self.assertEqual(summary["raw_silent_token_items"], 0)
        self.assertEqual(summary["source_silence_items"], 1)
        self.assertEqual(summary["sep_nonzero_items"], 1)
        self.assertEqual(warnings, [])
        self.assertEqual(errors, [])

    def test_set_hparams_reset_ignores_saved_ckpt_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            cfg = tmp_dir / "cfg.yaml"
            cfg.write_text("foo: 1\n", encoding="utf-8")
            ckpt_dir = ROOT / "checkpoints" / "tmp_reset_contract_case"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = ckpt_dir / "config.yaml"
            cfg_path.write_text("foo: 2\n", encoding="utf-8")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    hp_reuse = set_hparams(
                        config=str(cfg),
                        exp_name="tmp_reset_contract_case",
                        print_hparams=False,
                        global_hparams=False,
                    )
                hp_reset = set_hparams(
                    config=str(cfg),
                    exp_name="tmp_reset_contract_case",
                    print_hparams=False,
                    global_hparams=False,
                    reset=True,
                )
                self.assertEqual(hp_reuse["foo"], 2)
                self.assertEqual(hp_reset["foo"], 1)
            finally:
                try:
                    cfg_path.unlink()
                except FileNotFoundError:
                    pass
                try:
                    ckpt_dir.rmdir()
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()

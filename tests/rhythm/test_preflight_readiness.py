from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_gen.conan_binarizer import VCBinarizer
from data_gen.tts.base_binarizer import BaseBinarizer
from modules.Conan.rhythm.stages import normalize_rhythm_stage
from tasks.Conan.rhythm.dataset_contracts import RhythmDatasetCacheContract
from tasks.Conan.rhythm.config_contract_stage_rules import detect_rhythm_profile
from tasks.Conan.rhythm.preflight_support import _inspect_indexed_split_arrays, _inspect_indexed_split_files
from utils.commons.hparams import set_hparams


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

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.commons.hparams import set_hparams


class StageWarmStartDefaultTests(unittest.TestCase):
    def test_maintained_and_experimental_stage_configs_pin_non_strict_load(self) -> None:
        config_paths = [
            "egs/conan_emformer_rhythm_v2_teacher_offline.yaml",
            "egs/conan_emformer_rhythm_v2_student_kd.yaml",
            "egs/conan_emformer_rhythm_v2_student_kd_context_match.yaml",
            "egs/conan_emformer_rhythm_v2_student_pairwise_ref_runtime_teacher.yaml",
            "egs/conan_emformer_rhythm_v2_student_ref_bootstrap.yaml",
            "egs/conan_emformer_rhythm_v2_student_retimed.yaml",
            "egs/conan_emformer_rhythm_v2_student_retimed_balanced.yaml",
        ]
        for config_path in config_paths:
            with self.subTest(config=config_path):
                hp = set_hparams(
                    config=config_path,
                    print_hparams=False,
                    global_hparams=False,
                    reset=True,
                )
                self.assertIs(
                    hp.get("load_ckpt_strict"),
                    False,
                    msg=f"{config_path} should keep partial warm-start loading non-strict.",
                )

    def test_maintained_stage3_config_enables_conservative_acoustic_ramp(self) -> None:
        hp = set_hparams(
            config="egs/conan_emformer_rhythm_v2_student_retimed.yaml",
            print_hparams=False,
            global_hparams=False,
            reset=True,
        )
        self.assertEqual(hp.get("rhythm_stage3_acoustic_weight_start"), 0.25)
        self.assertEqual(hp.get("rhythm_stage3_acoustic_weight_end"), 1.0)
        self.assertEqual(hp.get("rhythm_stage3_acoustic_ramp_steps"), 10000)

    def test_ref_bootstrap_config_externalizes_rhythm_supervision_without_cached_teacher(self) -> None:
        hp = set_hparams(
            config="egs/conan_emformer_rhythm_v2_student_ref_bootstrap.yaml",
            print_hparams=False,
            global_hparams=False,
            reset=True,
        )
        self.assertEqual(hp.get("rhythm_dataset_target_mode"), "runtime_only")
        self.assertEqual(hp.get("rhythm_cached_reference_policy"), "sample_ref")
        self.assertEqual(hp.get("rhythm_primary_target_surface"), "teacher")
        self.assertEqual(hp.get("rhythm_teacher_target_source"), "algorithmic")
        self.assertFalse(hp.get("rhythm_require_cached_teacher"))
        self.assertFalse(hp.get("rhythm_binarize_teacher_targets"))
        self.assertFalse(hp.get("rhythm_binarize_retimed_mel_targets"))
        self.assertTrue(hp.get("rhythm_require_external_reference"))
        self.assertTrue(hp.get("rhythm_allow_identity_pairs"))
        self.assertEqual(hp.get("lambda_rhythm_ref_descriptor_stats"), 0.10)
        self.assertEqual(hp.get("lambda_rhythm_ref_descriptor_trace"), 0.05)
        self.assertEqual(hp.get("lambda_rhythm_ref_group_contrastive"), 0.05)


if __name__ == "__main__":
    unittest.main()

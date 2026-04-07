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
            "egs/conan_emformer_rhythm_v2_student_retimed_hybrid_ablation.yaml",
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
        self.assertEqual(hp.get("rhythm_stage3_acoustic_weight_start"), 0.10)
        self.assertEqual(hp.get("rhythm_stage3_acoustic_weight_end"), 1.0)
        self.assertEqual(hp.get("rhythm_stage3_acoustic_ramp_steps"), 20000)
        self.assertEqual(hp.get("rhythm_retimed_target_mode"), "cached")
        self.assertEqual(hp.get("rhythm_online_retimed_target_start_steps"), 40000)
        self.assertTrue(hp.get("rhythm_stage3_scale_pitch_loss"))
        self.assertEqual(hp.get("rhythm_projector_pause_selection_mode"), "sparse")
        self.assertTrue(hp.get("rhythm_projector_use_boundary_commit_guard"))
        self.assertTrue(hp.get("rhythm_projector_build_render_plan"))
        self.assertEqual(hp.get("rhythm_loss_balance_mode"), "ema_group")

    def test_balanced_stage3_config_pushes_harder_against_acoustic_dominance(self) -> None:
        maintained_hp = set_hparams(
            config="egs/conan_emformer_rhythm_v2_student_retimed.yaml",
            print_hparams=False,
            global_hparams=False,
            reset=True,
        )
        balanced_hp = set_hparams(
            config="egs/conan_emformer_rhythm_v2_student_retimed_balanced.yaml",
            print_hparams=False,
            global_hparams=False,
            reset=True,
        )
        self.assertLess(
            balanced_hp.get("rhythm_stage3_acoustic_weight_start"),
            maintained_hp.get("rhythm_stage3_acoustic_weight_start"),
        )
        self.assertLess(
            balanced_hp.get("rhythm_stage3_acoustic_weight_end"),
            maintained_hp.get("rhythm_stage3_acoustic_weight_end"),
        )
        self.assertGreater(
            balanced_hp.get("rhythm_stage3_acoustic_ramp_steps"),
            maintained_hp.get("rhythm_stage3_acoustic_ramp_steps"),
        )
        self.assertEqual(balanced_hp.get("rhythm_loss_balance_mode"), "ema_group")
        self.assertEqual(balanced_hp.get("rhythm_retimed_target_mode"), "cached")
        self.assertEqual(balanced_hp.get("rhythm_online_retimed_target_start_steps"), 40000)
        self.assertLess(
            balanced_hp.get("rhythm_loss_balance_min_scale"),
            maintained_hp.get("rhythm_loss_balance_min_scale"),
        )
        self.assertGreater(
            balanced_hp.get("rhythm_loss_balance_max_scale"),
            maintained_hp.get("rhythm_loss_balance_max_scale"),
        )

    def test_hybrid_stage3_ablation_switches_target_mode_without_touching_warmstart_gate(self) -> None:
        maintained_hp = set_hparams(
            config="egs/conan_emformer_rhythm_v2_student_retimed.yaml",
            print_hparams=False,
            global_hparams=False,
            reset=True,
        )
        hybrid_hp = set_hparams(
            config="egs/conan_emformer_rhythm_v2_student_retimed_hybrid_ablation.yaml",
            print_hparams=False,
            global_hparams=False,
            reset=True,
        )
        self.assertEqual(maintained_hp.get("rhythm_retimed_target_mode"), "cached")
        self.assertEqual(hybrid_hp.get("rhythm_retimed_target_mode"), "hybrid")
        self.assertEqual(
            hybrid_hp.get("rhythm_online_retimed_target_start_steps"),
            maintained_hp.get("rhythm_online_retimed_target_start_steps"),
        )
        self.assertEqual(
            hybrid_hp.get("rhythm_stage3_acoustic_weight_start"),
            maintained_hp.get("rhythm_stage3_acoustic_weight_start"),
        )
        self.assertEqual(
            hybrid_hp.get("rhythm_stage3_acoustic_ramp_steps"),
            maintained_hp.get("rhythm_stage3_acoustic_ramp_steps"),
        )

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
        self.assertTrue(hp.get("rhythm_emit_reference_sidecar"))
        self.assertTrue(hp.get("rhythm_trace_reliability_enable"))
        self.assertTrue(hp.get("rhythm_trace_anchor_aware_sampling"))


if __name__ == "__main__":
    unittest.main()

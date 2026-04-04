import unittest

from scripts.integration_teacher_export_student_kd import _normalize_export_splits


class IntegrationTeacherExportStudentKDTests(unittest.TestCase):
    def test_normalize_export_splits_auto_adds_missing_eval_splits(self):
        self.assertEqual(
            _normalize_export_splits(
                ["train", "valid"],
                include_valid=True,
                include_test=True,
            ),
            ["train", "valid", "test"],
        )

    def test_normalize_export_splits_normalizes_aliases_and_dedupes(self):
        self.assertEqual(
            _normalize_export_splits(
                ["train", "val", "test", "validation", "train"],
                include_valid=True,
                include_test=True,
            ),
            ["train", "valid", "test"],
        )


if __name__ == "__main__":
    unittest.main()

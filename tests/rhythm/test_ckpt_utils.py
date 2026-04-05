from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.commons.ckpt_utils import load_ckpt, load_ckpt_emformer


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
        self.extra = torch.nn.Linear(3, 1)


class CkptUtilsTests(unittest.TestCase):
    def test_load_ckpt_reports_missing_and_unexpected_keys_in_non_strict_mode(self) -> None:
        model = _ToyModel()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "toy.ckpt"
            torch.save(
                {
                    "state_dict": {
                        "model.linear.weight": model.linear.weight.detach().clone(),
                        "model.linear.bias": model.linear.bias.detach().clone(),
                        "model.unused": torch.ones(1),
                    }
                },
                ckpt_path,
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                load_ckpt(model, str(ckpt_path), strict=False)
            output = buf.getvalue()
        self.assertIn("Missing keys while loading 'model': 2", output)
        self.assertIn("extra.weight", output)
        self.assertIn("Unexpected keys while loading 'model': 1", output)
        self.assertIn("unused", output)

    def test_load_ckpt_emformer_reports_missing_and_unexpected_keys_in_non_strict_mode(self) -> None:
        model = _ToyModel()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "toy_emformer.ckpt"
            torch.save(
                {
                    "state_dict": {
                        "linear.weight": model.linear.weight.detach().clone(),
                        "linear.bias": model.linear.bias.detach().clone(),
                        "unused": torch.ones(1),
                    }
                },
                ckpt_path,
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                load_ckpt_emformer(model, str(ckpt_path), strict=False)
            output = buf.getvalue()
        self.assertIn("Missing keys while loading 'model': 2", output)
        self.assertIn("extra.weight", output)
        self.assertIn("Unexpected keys while loading 'model': 1", output)
        self.assertIn("unused", output)


if __name__ == "__main__":
    unittest.main()

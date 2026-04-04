from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.commons.trainer_loop import TrainerLoopMixin


class _DummyTask(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gen = torch.nn.Parameter(torch.tensor([1.0]))
        self.disc = torch.nn.Parameter(torch.tensor([1.0]))
        self.seen_batches = []

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.seen_batches.append(batch)
        param = self.gen if optimizer_idx == 0 else self.disc
        return {
            "loss": (param ** 2).sum(),
            "progress_bar": {"optimizer_idx": float(optimizer_idx)},
            "tb_log": {"optimizer_idx": float(optimizer_idx)},
        }

    def on_before_optimization(self, opt_idx):
        return None

    def on_after_optimization(self, current_epoch, batch_idx, optimizer, opt_idx):
        return None


class _DummyTrainer(TrainerLoopMixin):
    def __init__(self) -> None:
        self.task = _DummyTask()
        self.optimizers = [
            torch.optim.SGD([self.task.gen], lr=0.1),
            torch.optim.SGD([self.task.disc], lr=0.1),
        ]
        self.prepare_calls = 0
        self.use_ddp = False
        self.amp = False
        self.on_gpu = False
        self.root_gpu = 0
        self.accumulate_grad_batches = 1
        self.global_step = 0
        self.current_epoch = 0
        self.print_nan_grads = False

    def get_task_ref(self):
        return self.task

    def _prepare_batch(self, batch):
        self.prepare_calls += 1
        return {"value": batch["value"].clone()}


class TrainerLoopTests(unittest.TestCase):
    def test_run_training_batch_prepares_batch_once_for_multi_optimizer(self) -> None:
        trainer = _DummyTrainer()
        trainer.run_training_batch(0, {"value": torch.tensor([1.0])})
        self.assertEqual(trainer.prepare_calls, 1)
        self.assertEqual(len(trainer.task.seen_batches), 2)
        self.assertIsNot(trainer.task.seen_batches[0], trainer.task.seen_batches[1])


if __name__ == "__main__":
    unittest.main()

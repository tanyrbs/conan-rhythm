from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.loss_balance import (
    AdaptiveRhythmLossBalancer,
    AdaptiveRhythmLossBalancerConfig,
)


class RhythmLossBalanceTests(unittest.TestCase):
    def test_balancer_is_noop_before_warmup(self) -> None:
        balancer = AdaptiveRhythmLossBalancer(
            AdaptiveRhythmLossBalancerConfig(mode="ema_group", warmup_steps=10)
        )
        losses = {
            "rhythm_exec_speech": torch.tensor(2.0, requires_grad=True),
            "rhythm_exec_stretch": torch.tensor(1.0, requires_grad=True),
            "rhythm_exec_pause": torch.tensor(2.0, requires_grad=True),
            "rhythm_budget": torch.tensor(1.0, requires_grad=True),
            "rhythm_prefix_state": torch.tensor(1.0, requires_grad=True),
        }
        balanced = balancer.apply(losses, global_step=0, training=True)
        self.assertTrue(torch.allclose(balanced["rhythm_exec_speech"], losses["rhythm_exec_speech"]))
        self.assertTrue(torch.allclose(balanced["rhythm_exec_stretch"], losses["rhythm_exec_stretch"]))
        self.assertNotIn("rhythm_loss_balance_exec_scale", balanced)

    def test_balancer_downweights_large_group_and_recenters_scales(self) -> None:
        balancer = AdaptiveRhythmLossBalancer(
            AdaptiveRhythmLossBalancerConfig(
                mode="ema_group",
                beta=0.0,
                alpha=1.0,
                warmup_steps=0,
                min_scale=0.25,
                max_scale=4.0,
            )
        )
        losses = {
            "rhythm_exec_speech": torch.tensor(4.0, requires_grad=True),
            "rhythm_exec_stretch": torch.tensor(0.5, requires_grad=True),
            "rhythm_exec_pause": torch.tensor(0.0, requires_grad=True),
            "rhythm_budget": torch.tensor(1.0, requires_grad=True),
            "rhythm_prefix_state": torch.tensor(0.0, requires_grad=True),
        }
        balanced = balancer.apply(losses, global_step=100, training=True)
        exec_scale = float(balanced["rhythm_loss_balance_exec_scale"].item())
        state_scale = float(balanced["rhythm_loss_balance_state_scale"].item())
        self.assertLess(exec_scale, 1.0)
        self.assertGreater(state_scale, 1.0)
        self.assertAlmostEqual((exec_scale + state_scale) / 2.0, 1.0, places=5)
        self.assertTrue(torch.allclose(balanced["rhythm_exec_speech"], losses["rhythm_exec_speech"] * exec_scale))
        self.assertTrue(torch.allclose(balanced["rhythm_exec_stretch"], losses["rhythm_exec_stretch"] * exec_scale))
        self.assertTrue(torch.allclose(balanced["rhythm_budget"], losses["rhythm_budget"] * state_scale))


if __name__ == "__main__":
    unittest.main()

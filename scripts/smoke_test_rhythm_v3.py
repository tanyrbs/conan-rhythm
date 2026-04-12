from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from utils.commons.single_thread_env import apply_single_thread_env, maybe_limit_torch_cpu_threads

apply_single_thread_env()

import torch

maybe_limit_torch_cpu_threads()
torch.manual_seed(1234)
np.random.seed(1234)

from modules.Conan.rhythm_v3.runtime_adapter import ConanDurationAdapter
from modules.Conan.rhythm_v3.source_cache import build_source_rhythm_cache_v3


def _build_prompt_conditioning():
    return {
        "prompt_content_units": torch.tensor([[1, 57, 2, 3]], dtype=torch.long),
        "prompt_duration_obs": torch.tensor([[3.0, 2.0, 4.0, 3.0]], dtype=torch.float32),
        "prompt_unit_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32),
        "prompt_valid_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32),
        "prompt_speech_mask": torch.tensor([[1.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structural smoke test for maintained Rhythm V3 / V1-G.")
    parser.parse_args()

    cache = build_source_rhythm_cache_v3(
        [1, 1, 57, 2, 2, 3, 3],
        silent_token=57,
        separator_aware=True,
        tail_open_units=1,
        emit_silence_runs=True,
        debounce_min_run_frames=2,
    )
    assert "source_silence_mask" in cache
    assert "source_boundary_cue" in cache
    assert "phrase_final_mask" in cache

    hparams = {
        "silent_token": 57,
        "rhythm_separator_aware": True,
        "rhythm_tail_open_units": 1,
        "rhythm_anchor_hidden_size": 32,
        "rhythm_anchor_min_frames": 1.0,
        "rhythm_anchor_max_frames": 6.0,
        "rhythm_hidden_size": 64,
        "rhythm_response_rank": 4,
        "rhythm_response_window_left": 2,
        "rhythm_response_window_right": 0,
        "rhythm_ref_coverage_floor": 0.05,
        "rhythm_max_logstretch": 0.8,
        "rhythm_streaming_mode": "strict",
        "rhythm_apply_mode": "always",
        "rhythm_enable_v2": False,
        "rhythm_enable_v3": True,
        "rhythm_v3_backbone": "prompt_summary",
        "rhythm_v3_warp_mode": "none",
        "rhythm_v3_allow_hybrid": False,
        "rhythm_v3_anchor_mode": "source_observed",
        "rhythm_v3_emit_silence_runs": True,
        "rhythm_v3_minimal_v1_profile": True,
        "rhythm_v3_rate_mode": "simple_global",
        "rhythm_v3_simple_global_stats": True,
        "rhythm_v3_use_log_base_rate": False,
        "rhythm_v3_use_reference_summary": False,
        "rhythm_v3_disable_learned_gate": True,
        "rhythm_v3_use_learned_residual_gate": False,
        "rhythm_v3_summary_pool_speech_only": True,
        "rhythm_v3_summary_use_unit_embedding": False,
        "rhythm_num_summary_slots": 1,
        "rhythm_v3_require_same_text_paired_target": True,
        "rhythm_v3_disallow_same_text_paired_target": False,
        "rhythm_v3_short_gap_silence_scale": 0.35,
        "rhythm_v3_leading_silence_scale": 0.0,
        "rhythm_v3_eval_mode": "analytic",
        "rhythm_v3_g_variant": "raw_median",
        "rhythm_v3_disable_local_residual": True,
        "rhythm_v3_disable_coarse_bias": True,
        "rhythm_v3_debug_export": True,
    }

    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    assert adapter.module.backbone_mode == "prompt_summary"
    assert adapter.module.rate_mode == "simple_global"
    assert adapter.unit_frontend.rate_mode == "simple_global"
    assert adapter.module.simple_global_stats is True
    assert adapter.module.use_log_base_rate is False
    assert adapter.module.use_learned_residual_gate is False
    assert adapter.module.eval_mode == "analytic"
    assert adapter.module.g_variant == "raw_median"

    ret = {}
    content = torch.tensor([[1, 1, 57, 57, 2, 2, 3, 3]], dtype=torch.long)
    adapter(
        ret=ret,
        content=content,
        ref=None,
        target=None,
        f0=None,
        uv=None,
        infer=True,
        global_steps=0,
        content_embed=torch.randn(content.size(0), content.size(1), 32),
        tgt_nonpadding=torch.ones(content.size(0), content.size(1), 1),
        content_lengths=torch.full((content.size(0),), int(content.size(1)), dtype=torch.long),
        rhythm_state=None,
        rhythm_ref_conditioning=_build_prompt_conditioning(),
        rhythm_apply_override=None,
        rhythm_runtime_overrides=None,
        rhythm_source_cache=None,
        rhythm_offline_source_cache=None,
        speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
    )
    assert ret["rhythm_v3_runtime_mode"] == "prompt_summary"
    assert ret["rhythm_v3_eval_mode"] == "analytic"
    assert ret["rhythm_v3_g_variant"] == "raw_median"
    assert torch.isfinite(ret["speech_duration_exec"]).all()
    assert ret["rhythm_execution"].local_response is not None
    assert torch.isfinite(ret["rhythm_execution"].local_response).all()
    assert ret["rhythm_execution"].global_bias_scalar is not None
    assert "rhythm_v3_debug" in ret
    assert ret["rhythm_v3_debug"]["eval_mode"] == "analytic"
    print("rhythm_v3 smoke ok")

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.commons.hparams import set_hparams

try:
    from inference.streaming_runtime import build_streaming_latency_report
except ImportError:  # pragma: no cover - supports direct script execution
    from streaming_runtime import build_streaming_latency_report


def main():
    parser = argparse.ArgumentParser(
        description="Report theoretical streaming latency / recompute budget without loading checkpoints."
    )
    parser.add_argument(
        "--config",
        default="egs/conan_emformer_rhythm_v2_student_kd.yaml",
        help="Config YAML to inspect.",
    )
    parser.add_argument("--exp_name", default="", help="Optional exp name if config is omitted.")
    parser.add_argument("--hparams", default="", help="Optional hparam overrides.")
    parser.add_argument(
        "--duration_seconds",
        type=float,
        default=None,
        help="Optional utterance duration for cumulative prefix-recompute estimation.",
    )
    args = parser.parse_args()

    hp = set_hparams(
        config=args.config,
        exp_name=args.exp_name,
        hparams_str=args.hparams,
        global_hparams=False,
        print_hparams=False,
    )
    report = build_streaming_latency_report(hp, duration_seconds=args.duration_seconds)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

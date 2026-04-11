from __future__ import annotations

from ..config_contract import collect_config_contract_evaluation


def validate_rhythm_v2_training_hparams(hparams) -> None:
    report = collect_config_contract_evaluation(hparams, model_dry_run=False).report
    if report.errors:
        raise ValueError("Invalid Rhythm V2 training config:\n- " + "\n- ".join(report.errors))
    if report.warnings:
        print("| Rhythm V2 config warnings:")
        for warning in report.warnings:
            print(f"|   - {warning}")


def validate_rhythm_training_hparams(hparams) -> None:
    validate_rhythm_v2_training_hparams(hparams)


__all__ = [
    "validate_rhythm_v2_training_hparams",
    "validate_rhythm_training_hparams",
]

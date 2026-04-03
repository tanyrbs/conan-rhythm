from __future__ import annotations

"""Backward-compatible wrapper for shared rhythm config contracts.

Single source of truth lives in tasks.Conan.rhythm.config_contract.
This module stays as a thin re-export layer so older imports keep working
without maintaining a second copy of the stage/profile rules.
"""

from tasks.Conan.rhythm.config_contract import (
    RhythmContractValidationResult,
    collect_rhythm_contract_issues,
    detect_rhythm_profile,
    validate_profile_contract,
    validate_stage_contract,
)

__all__ = [
    "RhythmContractValidationResult",
    "collect_rhythm_contract_issues",
    "detect_rhythm_profile",
    "validate_profile_contract",
    "validate_stage_contract",
]

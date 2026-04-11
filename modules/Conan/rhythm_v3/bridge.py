from __future__ import annotations


def resolve_rhythm_apply_mode(hparams, *, infer: bool = False, override=None) -> bool:
    if override is not None:
        return bool(override)
    mode = str(hparams.get("rhythm_apply_mode", "infer") or "infer").strip().lower()
    if mode in {"off", "none", "false"}:
        return False
    if mode in {"always", "all"}:
        return True
    if mode in {"infer", "inference", "test"}:
        return bool(infer)
    if mode in {"train", "training"}:
        return not bool(infer)
    return bool(infer)

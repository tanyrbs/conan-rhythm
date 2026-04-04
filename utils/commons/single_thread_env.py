from __future__ import annotations

import os


def apply_single_thread_env(threads: int = 1, *, force: bool = False) -> int:
    """Clamp common CPU thread pools for small diagnostics / CPU utilities."""

    threads = max(1, int(threads))
    setter = os.environ.__setitem__ if force else os.environ.setdefault
    value = str(threads)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TF_NUM_INTEROP_THREADS",
        "TF_NUM_INTRAOP_THREADS",
    ):
        setter(key, value)
    return threads


def _parse_thread_env_request(value: str | None, *, default_threads: int) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "0", "false", "off", "no", "none"}:
        return None
    if normalized in {"1", "true", "on", "yes", "auto"}:
        return max(1, int(default_threads))
    try:
        parsed = int(normalized)
    except ValueError as exc:  # pragma: no cover - defensive config guard
        raise ValueError(
            "CONAN_SINGLE_THREAD_ENV must be a positive integer or a truthy/falsey toggle."
        ) from exc
    if parsed <= 0:
        return None
    return parsed


def maybe_apply_single_thread_env_from_env(
    *,
    env_var: str = "CONAN_SINGLE_THREAD_ENV",
    default_threads: int = 1,
    force: bool = False,
) -> int | None:
    requested = _parse_thread_env_request(
        os.environ.get(env_var),
        default_threads=default_threads,
    )
    if requested is None:
        return None
    return apply_single_thread_env(requested, force=force)


def maybe_limit_torch_cpu_threads(threads: int = 1, *, force: bool = False) -> int:
    """Best-effort torch thread clamp for CPU-only probes / smoke tests."""

    threads = apply_single_thread_env(threads=threads, force=force)
    try:
        import torch
    except Exception:
        return threads

    try:
        current_threads = int(torch.get_num_threads())
    except Exception:
        current_threads = None
    if force or current_threads is None or current_threads > threads:
        try:
            torch.set_num_threads(threads)
        except Exception:
            pass

    try:
        current_interop = int(torch.get_num_interop_threads())
    except Exception:
        current_interop = None
    if force or current_interop is None or current_interop > threads:
        try:
            torch.set_num_interop_threads(threads)
        except RuntimeError:
            pass
        except Exception:
            pass
    return threads


__all__ = [
    "apply_single_thread_env",
    "maybe_apply_single_thread_env_from_env",
    "maybe_limit_torch_cpu_threads",
]

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


apply_single_thread_env()


__all__ = ["apply_single_thread_env", "maybe_limit_torch_cpu_threads"]

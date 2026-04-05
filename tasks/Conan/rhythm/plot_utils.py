from __future__ import annotations

"""Lightweight plotting helpers for rhythm-only validation paths."""


def _require_matplotlib_pyplot():
    try:
        import matplotlib

        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError(
            "matplotlib is required to render rhythm f0 plots during validation."
        ) from exc
    return plt


def _to_numpy(payload):
    if payload is None:
        return None
    if hasattr(payload, "detach"):
        payload = payload.detach()
    if hasattr(payload, "cpu"):
        payload = payload.cpu()
    if hasattr(payload, "numpy"):
        payload = payload.numpy()
    return payload


def f0_to_figure(f0_gt, f0_cwt=None, f0_pred=None):
    plt = _require_matplotlib_pyplot()
    fig = plt.figure(figsize=(12, 8))
    if f0_gt is not None:
        plt.plot(_to_numpy(f0_gt), color="r", label="gt")
    if f0_cwt is not None:
        plt.plot(_to_numpy(f0_cwt), color="b", label="ref")
    if f0_pred is not None:
        plt.plot(_to_numpy(f0_pred), color="green", label="pred")
    plt.legend()
    return fig


__all__ = ["f0_to_figure"]

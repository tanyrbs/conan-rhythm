from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest


_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_pytest_runtime"


def _iter_tmp_children(root: Path):
    if not root.exists():
        return []
    try:
        return list(root.iterdir())
    except Exception:
        return []


@pytest.fixture(scope="session", autouse=True)
def _clean_repo_tmp_root() -> Iterator[None]:
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    for child in _iter_tmp_children(_TMP_ROOT):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except Exception:
                pass
    yield


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Repo-local replacement for pytest tmp_path on Windows-restricted hosts."""

    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = _TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=False, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)

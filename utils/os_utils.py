from __future__ import annotations

import os
import shutil
from pathlib import Path


def _copy_path(from_path: Path, to_path: Path) -> None:
    if from_path.is_dir() and not from_path.is_symlink():
        shutil.copytree(from_path, to_path, dirs_exist_ok=True)
    else:
        to_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(from_path, to_path)


def link_file(from_file, to_file):
    source = Path(from_file)
    target = Path(to_file)
    if target.exists() or target.is_symlink():
        remove_file(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        relative_source = os.path.relpath(source, start=target.parent)
        os.symlink(relative_source, target, target_is_directory=source.is_dir())
    except OSError:
        _copy_path(source, target)


def move_file(from_file, to_file):
    source = Path(from_file)
    target = Path(to_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))


def copy_file(from_file, to_file):
    _copy_path(Path(from_file), Path(to_file))


def remove_file(*fns):
    for fn in fns:
        path = Path(fn)
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()

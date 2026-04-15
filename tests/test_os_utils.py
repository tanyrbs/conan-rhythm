from __future__ import annotations

from pathlib import Path

from utils.os_utils import copy_file, link_file, move_file, remove_file


def test_remove_file_handles_files_dirs_and_missing(tmp_path: Path):
    file_path = tmp_path / "demo.txt"
    file_path.write_text("demo", encoding="utf-8")
    dir_path = tmp_path / "tree"
    dir_path.mkdir()
    (dir_path / "nested.txt").write_text("nested", encoding="utf-8")

    remove_file(file_path, dir_path, tmp_path / "missing.txt")

    assert not file_path.exists()
    assert not dir_path.exists()


def test_copy_move_and_link_file_work_cross_platform(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("payload", encoding="utf-8")

    copied = tmp_path / "copied.txt"
    copy_file(source, copied)
    assert copied.read_text(encoding="utf-8") == "payload"

    linked = tmp_path / "linked.txt"
    link_file(source, linked)
    assert linked.read_text(encoding="utf-8") == "payload"

    moved = tmp_path / "moved.txt"
    move_file(copied, moved)
    assert not copied.exists()
    assert moved.read_text(encoding="utf-8") == "payload"

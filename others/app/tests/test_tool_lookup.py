"""ffmpeg/ffprobe: bundled copy wins over PATH, and a machine with neither still works.

The owner's target (2026-10-27) is a PC that is not a developer's: no Python, no ffmpeg, no PATH
editing, no admin. `install.bat` therefore unpacks a static ffmpeg into `others\\tools\\ffmpeg\\bin`,
and these tests pin the lookup order. They do NOT need a real ffmpeg binary on disk - a placeholder
file is enough to prove which path would be chosen, and the last arm proves we do not invent one.
"""

from __future__ import annotations

import os
import shutil

from parakeet_to_md import media, paths


def _bundled_exe(root):
    return root / paths.OTHERS_DIRNAME / paths.BUNDLED_SUBPATH / (
        "ffmpeg" + (".exe" if os.name == "nt" else "")
    )


def test_a_bundled_copy_wins_over_path(tmp_path, monkeypatch):
    """Measured failure this prevents: on the build machine ffmpeg is on PATH, so a test that only
    ever checks `shutil.which` passes while the shipped folder is empty and a fresh PC has nothing."""
    (tmp_path / paths.OTHERS_DIRNAME).mkdir()
    exe = _bundled_exe(tmp_path)
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"placeholder")
    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: r"C:\elsewhere\bin\%s.exe" % name)

    assert media.ffmpeg_path() == str(exe)


def test_a_flat_install_still_finds_its_own_copy(tmp_path, monkeypatch):
    """A copy installed before `others\\` existed must keep working: same lookup, no `others` level."""
    exe = tmp_path / paths.BUNDLED_SUBPATH / ("ffmpeg" + (".exe" if os.name == "nt" else ""))
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"placeholder")
    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: r"C:\elsewhere\bin\%s.exe" % name)

    assert media.ffmpeg_path() == str(exe)


def test_without_a_bundled_copy_the_path_is_used(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: r"C:\elsewhere\bin\%s.exe" % name)

    assert media.ffmpeg_path() == r"C:\elsewhere\bin\ffmpeg.exe"


def test_with_neither_the_answer_is_none_and_the_error_names_the_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    assert media.ffmpeg_path() is None
    try:
        media.require_tools()
    except media.MediaError as exc:
        assert "install.bat" in str(exc) and "ffmpeg" in str(exc)
    else:  # pragma: no cover - the arm exists so a silent None cannot pass
        raise AssertionError("require_tools() accepted a machine with no ffmpeg")

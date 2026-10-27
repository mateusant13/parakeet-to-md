"""The folder layout a non-developer sees: input\\, output\\, archive\\ and others\\.

The owner asked for exactly that on 2026-10-27, and the trap is that `others\\` itself contains
`models\\` and `tools\\` - so a root finder that accepts those as markers answers "others" and then
looks for `others\\others\\models`. These arms pin the marker, the data folders, and the flat-install
fallback that keeps an older copy running.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from parakeet_to_md import __version__, paths


def _fake_root(tmp_path: Path) -> Path:
    (tmp_path / "run.bat").write_text("@echo off\n")
    for name in ("input", "output", "archive"):
        (tmp_path / name).mkdir()
    return tmp_path


def test_the_root_is_the_folder_with_run_bat_not_others(tmp_path, monkeypatch):
    root = _fake_root(tmp_path)
    others = root / paths.OTHERS_DIRNAME
    (others / "models").mkdir(parents=True)
    (others / "tools" / "ffmpeg" / "bin").mkdir(parents=True)

    monkeypatch.setattr(paths, "project_root", lambda: root)
    assert paths.project_root() == root
    # the specific mistake: models/ and tools/ alone must not be read as "this is the root"
    assert paths.others_dir() == others
    assert (paths.others_dir() / "models") == others / "models"


def test_the_root_finder_ignores_a_deeper_models_folder(tmp_path):
    """Drives the real search: `others\\models` and `others\\tools` must not be read as the root."""
    root = _fake_root(tmp_path)
    others = root / paths.OTHERS_DIRNAME
    deep = others / "app" / "parakeet_to_md"
    deep.mkdir(parents=True)
    (others / "models").mkdir()
    (others / "tools" / "ffmpeg" / "bin").mkdir(parents=True)

    assert paths._find_root(deep) == root


def test_data_folders_never_move_into_others(tmp_path, monkeypatch):
    root = _fake_root(tmp_path)
    (root / paths.OTHERS_DIRNAME).mkdir()
    monkeypatch.setattr(paths, "project_root", lambda: root)

    assert paths.data_dir("output") == root / "output"
    assert paths.data_dir("archive") == root / "archive"


def test_a_flat_install_without_others_answers_the_root_itself(tmp_path, monkeypatch):
    root = _fake_root(tmp_path)
    monkeypatch.setattr(paths, "project_root", lambda: root)

    assert paths.others_dir() == root


def test_the_reported_version_is_the_pyproject_version():
    """`parakeet-to-md --version` printed 0.1.0 while the packaging metadata said 0.2.0: a number the
    user reads and a number the installer trusts must be the same one."""
    root = paths.project_root()
    for candidate in (root / paths.OTHERS_DIRNAME / "pyproject.toml", root / "pyproject.toml"):
        if candidate.exists():
            declared = tomllib.loads(candidate.read_text(encoding="utf-8"))["project"]["version"]
            break
    else:  # pragma: no cover
        raise AssertionError("no pyproject.toml found from %s" % root)

    assert declared == __version__

r"""Where the project's data lives, and where a bundled tool is looked for.

The user who double-clicks `install.bat` sees four folders - `input\`, `output\`, `archive\` and
`others\` - because a non-developer should not have to tell `models\` from `.venv\` from `app\`. The
three data folders stay at the root; everything technical lives in `others\`. Both layouts are read
here, and nothing else in the code is allowed to hard-code a path: this module is the only place that
knows where the project root is.

Two callers need this and they must not disagree: the engine (models/) and the media layer (the
ffmpeg that install.bat unpacks for a machine that has none). Resolving `parents[N]` broke the day
the code moved from `src/` to `app/`, so the search is marker-based.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

#: Where the technical half of the install lives. `install.bat` puts the venv, the models, the
#: ffmpeg copy and this source tree in here so the user's folder shows four entries, not nine.
OTHERS_DIRNAME = "others"

#: A non-developer PC has no ffmpeg on PATH and cannot be asked to install one, so install.bat
#: unpacks a static build here and the app looks here BEFORE PATH: the copy we ship is the one whose
#: behaviour was measured, and PATH on a stranger's machine is not ours to reason about.
BUNDLED_SUBPATH = Path("tools") / "ffmpeg" / "bin"


def _find_root(here: Path) -> Path:
    r"""Split out so a test can drive the real search from a temp tree (project_root() is pinned to
    this file's location by `__file__`, which no test can redirect)."""
    for parent in here.parents:
        if (parent / "run.bat").exists():
            return parent
    for parent in here.parents:  # scripts and tests run from a checkout with no run.bat
        if (parent / "input").is_dir() and (parent / "output").is_dir():
            return parent
    return here.parents[3]


def project_root() -> Path:
    """The folder the user double-clicks in - the one holding `run.bat`.

    `run.bat` is the marker because it exists in exactly one place, at the root, in both layouts.
    `others\\` also contains a `models\\` and a `tools\\`, so a marker test that accepted either of
    those would answer "others" and then look for `others\\others\\models`.
    """
    return _find_root(Path(__file__).resolve())


def others_dir() -> Path:
    """`<root>\\others` when it exists, else the root - so a pre-2026-10-27 flat install keeps working."""
    root = project_root()
    candidate = root / OTHERS_DIRNAME
    return candidate if candidate.is_dir() else root


def data_dir(name: str) -> Path:
    """`input` / `output` / `archive`: always at the root, never inside `others\\`."""
    return project_root() / name


def bundled(name: str) -> Path | None:
    """The shipped copy of `name` (ffmpeg/ffprobe), or None when this install did not unpack one."""
    exe = others_dir() / BUNDLED_SUBPATH / (name + (".exe" if os.name == "nt" else ""))
    return exe if exe.exists() else None


def tool(name: str) -> str | None:
    """Bundled first, PATH second. The caller gets a path it can run, never a guess."""
    shipped = bundled(name)
    if shipped is not None:
        return str(shipped)
    return shutil.which(name)

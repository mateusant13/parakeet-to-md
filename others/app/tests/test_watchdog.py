"""The silence guard: prove it by killing real processes, not by reading the code.

Measured 2026-10-27, the day these tests were written:

* `python -m parakeet_to_md --selftest --device cpu` produced **zero bytes** for 22.28 s on a pipe and
  then the whole report in one burst - Python block-buffers stdout when it is not a terminal, so a
  healthy run looked exactly like a wedge;
* `python -m parakeet_to_md --selftest --device trt` produced **zero bytes on stdout AND stderr for
  600 s** and had to be killed.

Silence is the failure mode, so each arm here is a real child process under a real ceiling:

a. nothing at all written -> exit **2**, and the sentence says "silence, not slowness";
b. one line written, then quiet -> exit **3**, and the sentence names the stage;
c. a step that prints regularly is **not** killed - the arm that proves the guard is not a timer.

Ceilings reach the driver through the environment (`PARAKEET_FIRST_OUTPUT_S`, `PARAKEET_QUIET_S`), so
these arms test the same wiring production uses, and every subprocess carries a hard `timeout=` so a
broken guard fails the arm instead of hanging the suite.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from parakeet_to_md import cli

SILENCE = "silence, not slowness - this is a wedge, not a slow model"

DRIVER_HEADER = """
    import sys
    import time

    from parakeet_to_md.cli import stage, start_watchdog, stop_watchdog

    start_watchdog()
"""


def write_driver(tmp_path, body):
    driver = tmp_path / "driver.py"
    driver.write_text(textwrap.dedent(DRIVER_HEADER) + textwrap.dedent(body), encoding="utf-8")
    env = dict(os.environ)
    return driver, env


def run_driver(tmp_path, body, *, first="5", quiet="5", timeout=30):
    """Run `body` in a child process with the guard armed from the environment.

    `sys.executable` is the venv python pytest already runs on, so the child imports the same installed
    package the user runs. `timeout=` is the reason a broken guard fails this arm instead of hanging the
    suite: the child's own sleeps are 10 s, so a guard that never fires still returns.
    """
    driver, env = write_driver(tmp_path, body)
    env["PARAKEET_FIRST_OUTPUT_S"] = first
    env["PARAKEET_QUIET_S"] = quiet
    return subprocess.run(
        [sys.executable, str(driver)],
        capture_output=True, text=True, timeout=timeout, env=env,
    )


def start_driver(tmp_path, body, *, first="5", quiet="5"):
    driver, env = write_driver(tmp_path, body)
    env["PARAKEET_FIRST_OUTPUT_S"] = first
    env["PARAKEET_QUIET_S"] = quiet
    proc = subprocess.Popen([sys.executable, str(driver)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env)
    return proc, env


def read_line(stream, *, timeout):
    """One line, or None when it does not arrive inside `timeout`.

    A thread rather than `select`, because Windows cannot select on a pipe: the ceiling on this read is
    what distinguishes "the bytes were flushed" from "the bytes were buffered until exit".
    """
    got: list[str | None] = []

    def pull():
        try:
            got.append(stream.readline())
        except Exception:  # noqa: BLE001 - a closed pipe reads as None above
            got.append(None)

    thread = threading.Thread(target=pull, daemon=True)
    thread.start()
    thread.join(timeout)
    return got[0] if got else None


# ---------------------------------------------------------------- (a) nothing ever written -> 2


def test_a_run_that_never_prints_exits_2_and_names_silence(tmp_path):
    """The measured `--device trt` case: no first output at all. Exit 1 would read as "a file failed";
    silence gets its own code so the message cannot be misread as a slow model."""
    started = time.monotonic()
    proc = run_driver(tmp_path, "time.sleep(10)\n", first="0.2", quiet="5")
    elapsed = time.monotonic() - started

    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stdout == "", "the guard fired, so the child must genuinely have written nothing"
    assert "no first output after 0.2s: " + SILENCE in proc.stderr, proc.stderr
    # It fired near its ceiling instead of waiting out the child's 10 s sleep.
    assert elapsed < 8, elapsed


# ---------------------------------------------------------------- (b) quiet after output -> 3


def test_a_run_that_goes_quiet_after_first_output_exits_3_and_names_the_stage(tmp_path):
    body = """
        print("started", flush=True)
        stage("transcribing the second block")
        time.sleep(10)
    """
    proc = run_driver(tmp_path, body, first="10", quiet="0.3")

    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    assert "started" in proc.stdout, "the first line did get out, so this is the quiet ceiling"
    assert "quiet for 0.3s while transcribing the second block" in proc.stderr, proc.stderr


# ---------------------------------------------------------------- (c) progress is never killed


def test_a_step_that_prints_regularly_is_never_killed(tmp_path):
    """The false-positive arm. 25 lines over ~1.25 s against a 0.5 s ceiling: a guard that cannot tell
    progress from silence would kill this, and a transcription that dies while it is working is worse
    than the wedge it was guarding against."""
    body = """
        for i in range(25):
            print("block %d" % i, flush=True)
            time.sleep(0.05)
        print("finished", flush=True)
    """
    proc = run_driver(tmp_path, body, first="0.5", quiet="0.5")

    assert proc.returncode == 0, (proc.returncode, proc.stdout[-400:], proc.stderr)
    assert "quiet for" not in proc.stderr, proc.stderr
    assert "no first output" not in proc.stderr, proc.stderr
    assert proc.stdout.count("block ") == 25, proc.stdout
    assert "finished" in proc.stdout


# ---------------------------------------------------------------- output arrives while running


def test_a_line_written_now_is_readable_now_not_when_the_process_exits(tmp_path):
    """The 22.28 s defect itself: a run that was printing all along looked byte-silent because stdout
    was block-buffered on a pipe. Here the child writes one line WITHOUT `flush=True` and then sleeps,
    and the line must reach the parent while the child is still alive - not in the exit flush."""
    body = """
        print("first line")
        time.sleep(3)
    """
    proc, env = start_driver(tmp_path, body, first="5", quiet="5")
    try:
        line = read_line(proc.stdout, timeout=1.5)
        assert line == "first line\n", ("no line before the process exited", line)
    finally:
        proc.kill()
        proc.wait(timeout=10)


# ---------------------------------------------------------------- measured limit of this design
#
# The guard is a Python thread, so it needs the GIL. A C extension that holds the GIL for a long call
# starves it: the ceiling then fires when the call RETURNS, not at the ceiling. That is measured, not
# theoretical - `--device trt` on 2026-10-27 emitted nothing for 386 s and the guard never fired,
# because that one call never returned. `xfail(strict=True)` keeps the limit visible: the day something
# makes this punctual (a supervisor process, a C-level timer that does not need the GIL), the arm turns
# XPASS and strict makes it RED, so the note has to be updated instead of quietly becoming false.


@pytest.mark.xfail(strict=True, reason="a Python thread cannot preempt a C call that holds the GIL")
def test_the_ceiling_is_punctual_inside_a_c_call_that_holds_the_gil(tmp_path):
    body = """
        print("started", flush=True)
        stage("a C call that holds the GIL")
        sum(range(60_000_000))
    """
    started = time.monotonic()
    proc = run_driver(tmp_path, body, first="10", quiet="0.05")
    elapsed = time.monotonic() - started

    assert proc.returncode == 3, (proc.returncode, proc.stderr)
    assert "quiet for 0.05s while a C call that holds the GIL" in proc.stderr, proc.stderr
    assert elapsed < 0.3, "fired only after the C call returned: %.2f s for a 0.05 s ceiling" % elapsed


# ---------------------------------------------------------------- the guard cannot be disabled


def test_a_typo_in_a_ceiling_is_refused_not_read_as_no_ceiling(monkeypatch):
    """`PARAKEET_QUIET_S=0` must not mean "off": a guard a typo disables is a comment."""
    for bad in ("0", "-3", "ninety", "  "):
        monkeypatch.setenv("PARAKEET_FIRST_OUTPUT_S", bad)
        monkeypatch.setenv("PARAKEET_QUIET_S", bad)
        assert cli._env_seconds("PARAKEET_FIRST_OUTPUT_S", 20.0) == 20.0, bad
        assert cli._env_seconds("PARAKEET_QUIET_S", 90.0) == 90.0, bad
    monkeypatch.setenv("PARAKEET_FIRST_OUTPUT_S", "3")
    assert cli._env_seconds("PARAKEET_FIRST_OUTPUT_S", 20.0) == 3.0


def test_the_default_kill_is_os_exit_so_nothing_can_swallow_it():
    """`os._exit` skips `except Exception`, non-daemon threads and atexit handlers - which is the whole
    reason a wedged interpreter can still be ended. A test that injected a fake killer everywhere would
    never notice this being swapped for a `return`."""
    assert cli.Watchdog().killer is os._exit


class _Sink:
    """A throwaway text stream that remembers what was written to it."""

    def __init__(self):
        import io

        self._stream = io.StringIO()
        self.reconfigure = None  # not a TextIOWrapper: `Watchdog.start` skips it, like a test double

    def write(self, data):
        return self._stream.write(data)

    def flush(self):
        pass

    def getvalue(self):
        return self._stream.getvalue()


def test_a_step_with_its_own_budget_survives_the_tighter_default():
    """The model load names its own budget (`PARAKEET_LOAD_QUIET_S`), because a first-run TensorRT
    engine build is legitimately silent for minutes. Without the per-step budget the 0.05 s default
    here would have fired."""
    out, err, killed = _Sink(), _Sink(), []
    watchdog = cli.Watchdog(out=out, err=err, first_output_s=5, quiet_s=0.05,
                            killer=killed.append).start()
    try:
        watchdog.out.write("first\n")
        watchdog.stage("loading the model", quiet_s=60)
        time.sleep(0.3)
        assert killed == [], "a step that declared its own budget must not be killed by the default"
        assert err.getvalue() == ""
    finally:
        watchdog.stop()

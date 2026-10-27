"""Command line: the only entry point the `.bat` files call.

Every wait here carries a ceiling that fires on SILENCE, never on slowness alone (`Watchdog` below).
That is not decoration: measured on 2026-10-27, `python -m parakeet_to_md --selftest --device cpu`
produced **zero bytes for 22.28 s** and then the whole report in one burst, and `--device trt`
produced zero bytes for **600 s** and had to be killed. The 22 s case was Python block-buffering
stdout on a pipe; the 600 s case was a wedge. Silence looks identical in both, so this file now
(a) flushes every byte as it is produced and (b) names the failure instead of waiting for it.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from . import __version__, paths
from .asr import DEVICES, default_model_dir, default_models_dir, load_engine
from .media import MediaError, ffmpeg_path, ffprobe_path, require_tools
from .pipeline import run_directory, summary
from .render import fmt_time, render
from .segments import Segment

REPO_ROOT = paths.project_root()

# ---------------------------------------------------------------- the silence guard
#
# Which hook: a `sys.stdout`/`sys.stderr` proxy, not a `probe.note(...)` call at each long step. It is
# the cheapest of the two - zero edits in `asr.py`, `media.py` and `pipeline.py`, because `print(...)`
# and `print(..., file=sys.stderr)` both resolve the module-level object at call time - and it is the
# only one that also fixes the buffering that made a healthy 22.28 s run look like a wedge.

#: Nothing written at all for this long means the process never got going. The model load itself
#: measured 12.94 s and 22.28 s on two CPU runs, and it happens AFTER the first line, so a run still
#: silent at 20 s is a wedge, not a slow load.
DEFAULT_FIRST_OUTPUT_S = 20.0

#: After the first byte, a step that reports progress gets this long between lines before it is called
#: stuck. Two steps are legitimately silent for longer and name their own budget in `_run`/`selftest`.
DEFAULT_QUIET_S = 90.0

#: The model load is one such step: measured 12.94 s and 22.28 s on CPU above, and a first-run TensorRT
#: engine build is slower by an unmeasured factor, so it gets ~13x the slower measurement - and stays
#: configurable through `PARAKEET_LOAD_QUIET_S`.
DEFAULT_LOAD_QUIET_S = 300.0


def _env_seconds(name: str, default: float) -> float:
    """The env value in seconds, or `default`.

    A zero, negative or garbage value is REFUSED rather than read as "no ceiling": a guard that a typo
    disables is not a guard, it is a comment.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _secs(value: float) -> str:
    """`3` for 3.0 and `0.2` for 0.2 - the number the operator typed, not `3.0`."""
    return "%g" % value


class _Stamped:
    """A stdout/stderr stand-in that tells the watchdog a byte left the program, then writes it.

    Flushing on every write is half the point: a block-buffered pipe is where 22.28 s of real progress
    turned into 22.28 s of apparent silence.
    """

    def __init__(self, raw, watchdog: "Watchdog") -> None:
        self._raw = raw
        self._watchdog = watchdog

    def write(self, data):
        if data:
            self._watchdog.touch()
        written = self._raw.write(data)
        self._raw.flush()
        return written

    def flush(self) -> None:
        self._raw.flush()

    def __getattr__(self, name):  # encoding, isatty, fileno, errors, buffer...
        return getattr(self._raw, name)


class Watchdog:
    """A thread that ends the process when the process goes SILENT.

    Two ceilings, both naming silence rather than a slow machine:

    * nothing written since start after `first_output_s` -> exit **2**;
    * nothing written since the last line for the running step's budget (`quiet_s` unless the step
      named its own) -> exit **3**.

    `os._exit` is deliberate: it is the one exit that no `except Exception`, no non-daemon thread and no
    atexit handler can turn back into a hang. The sentence is written first, with one unbuffered
    `os.write` straight to stderr's descriptor, so an exit code never arrives without its reason.
    """

    def __init__(self, *, out=None, err=None, first_output_s=None, quiet_s=None,
                 killer=os._exit) -> None:
        self.raw_out = sys.stdout if out is None else out
        self.raw_err = sys.stderr if err is None else err
        self.first_output_s = (DEFAULT_FIRST_OUTPUT_S if first_output_s is None
                               else float(first_output_s))
        self.quiet_s = DEFAULT_QUIET_S if quiet_s is None else float(quiet_s)
        self.killer = killer
        self.out: _Stamped | None = None
        self.err: _Stamped | None = None
        self.stage_name = "starting up"
        self.stage_quiet_s: float | None = None
        self._started = time.monotonic()
        self._last_output: float | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._cancelled = False

    def touch(self) -> None:
        """A byte was written: restart the quiet clock."""
        with self._lock:
            self._last_output = time.monotonic()

    def stage(self, name: str, quiet_s: float | None = None) -> None:
        """Name the step that is running, optionally with its own silence budget."""
        with self._lock:
            self.stage_name = name
            self.stage_quiet_s = quiet_s

    def start(self) -> "Watchdog":
        for raw in (self.raw_out, self.raw_err):
            reconfigure = getattr(raw, "reconfigure", None)
            if reconfigure is None:
                continue
            try:
                reconfigure(line_buffering=True, write_through=True)
            except (ValueError, OSError):  # detached, or not a TextIOWrapper
                pass
        self.out = _Stamped(self.raw_out, self)
        self.err = _Stamped(self.raw_err, self)
        if sys.stdout is self.raw_out:
            sys.stdout = self.out
        if sys.stderr is self.raw_err:
            sys.stderr = self.err
        self._thread = threading.Thread(target=self._watch, name="silence-watchdog", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Only for a clean return. A fired ceiling exits through `killer`, below."""
        self._cancelled = True
        self._stopped = True
        thread, self._thread = self._thread, None
        if thread is not None:
            # Wait for the guard to leave `_watch` BEFORE the interpreter starts finalizing: a daemon
            # thread that touches a buffered stream during finalization aborts the process with
            # `Fatal Python error: _enter_buffered_busy` instead of returning the caller's exit code.
            thread.join(0.5)
        if self.out is not None and sys.stdout is self.out:
            sys.stdout = self.raw_out
        if self.err is not None and sys.stderr is self.err:
            sys.stderr = self.raw_err

    def _tick(self) -> float:
        """Fast enough that a 0.2 s test ceiling fires near 0.2 s, slow enough never to spin a core."""
        return min(0.25, max(0.005, min(self.first_output_s, self.quiet_s) / 8.0))

    def _watch(self) -> None:
        while not self._stopped and not self._cancelled:
            with self._lock:
                last = self._last_output
                stage = self.stage_name
                budget = self.stage_quiet_s if self.stage_quiet_s is not None else self.quiet_s
            now = time.monotonic()
            if last is None:
                if now - self._started >= self.first_output_s:
                    self._fire(2, "no first output after %ss: silence, not slowness - this is a "
                                  "wedge, not a slow model" % _secs(self.first_output_s))
            elif now - last >= budget:
                self._fire(3, "quiet for %ss while %s" % (_secs(budget), stage))
            time.sleep(self._tick())

    def _fire(self, code: int, message: str) -> None:
        if self._cancelled:
            return
        self._stopped = True
        # NOT `self.raw_err.write(...)`: another thread may be inside that buffered writer, and a
        # daemon thread that touches it during interpreter finalization aborts the process with
        # `Fatal Python error: _enter_buffered_busy` - measured on 2026-10-27 while writing this guard
        # (exit code -1073740791 instead of 3). One unbuffered `os.write` to the descriptor cannot take
        # that lock, so the sentence survives and `os._exit` still decides the code.
        try:
            os.write(self.raw_err.fileno(), (message + "\n").encode("utf-8", "replace"))
        except Exception:  # noqa: BLE001 - a dead descriptor must not swallow the exit
            pass
        self.killer(code)


_WATCHDOG: Watchdog | None = None


def start_watchdog(*, out=None, err=None, first_output_s=None, quiet_s=None,
                   killer=os._exit) -> Watchdog:
    """Read the ceilings from the environment and start the guard.

    Called immediately after argument parsing, before any model work: nothing else in the program owns
    a clock, so nothing else can tell "slow" from "wedged".
    """
    global _WATCHDOG
    if first_output_s is None:
        first_output_s = _env_seconds("PARAKEET_FIRST_OUTPUT_S", DEFAULT_FIRST_OUTPUT_S)
    if quiet_s is None:
        quiet_s = _env_seconds("PARAKEET_QUIET_S", DEFAULT_QUIET_S)
    _WATCHDOG = Watchdog(out=out, err=err, first_output_s=first_output_s, quiet_s=quiet_s,
                         killer=killer).start()
    return _WATCHDOG


def stop_watchdog() -> None:
    global _WATCHDOG
    if _WATCHDOG is not None:
        _WATCHDOG.stop()
        _WATCHDOG = None


def stage(name: str, quiet_s: float | None = None) -> None:
    """Name the running step for the quiet message; a no-op when the guard is not running."""
    if _WATCHDOG is not None:
        _WATCHDOG.stage(name, quiet_s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="parakeet-to-md",
        description="Transcribe every audio/video file in a folder to readable markdown "
                    "(Parakeet TDT 0.6B v3 on ONNX Runtime: TensorRT/CUDA/DirectML, CPU fallback).",
    )
    p.add_argument("--input", default=str(paths.data_dir("input")), help="folder to watch for media")
    p.add_argument("--output", default=str(paths.data_dir("output")), help="where <name>.md is written")
    p.add_argument("--archive", default=str(paths.data_dir("archive")), help="where processed sources go")
    p.add_argument("--models-dir", default=str(default_models_dir()), help="model root (install.bat fills it)")
    p.add_argument("--model-dir", default=None, help="override the weight-set folder")
    p.add_argument("--device", default="auto", choices=list(DEVICES),
                   help="auto = TensorRT, CUDA, DirectML, then CPU (default); gpu = any GPU or fail; "
                        "trt/cuda/dml = that provider or fail; cpu = CPU only")
    p.add_argument("--threads", type=int, default=None,
                   help="intra-op threads per session (default: physical cores, capped at 8; "
                        "env PARAKEET_THREADS)")
    p.add_argument("--with-parts", action="store_true",
                   help="add a '## Part N' heading every 10 minutes (off: the transcript only)")
    p.add_argument("--block-seconds", type=float, default=600.0,
                   help="decode/ASR block size; sets the memory ceiling")
    p.add_argument("--only", default=None, help="process a single file by name or stem")
    p.add_argument("--keep-partial", action="store_true", help="keep resumable checkpoints after success")
    p.add_argument("--no-archive", action="store_true", help="leave the source in input/")
    p.add_argument("--selftest", action="store_true", help="check the environment without processing files")
    p.add_argument("--version", action="version", version="parakeet-to-md %s" % __version__)
    return p


def selftest(args) -> int:
    problems = 0

    def check(ok: bool, label: str, detail: str = "") -> None:
        nonlocal problems
        if not ok:
            problems += 1
        print("%s  %s%s" % ("PASS" if ok else "FAIL", label, ("  - " + detail) if detail and not ok else ""))

    check(ffmpeg_path() is not None, "ffmpeg (bundled or on PATH)",
          "run install.bat, which unpacks one into others\\tools\\ffmpeg\\bin")
    check(ffprobe_path() is not None, "ffprobe (bundled or on PATH)", "ships with ffmpeg")
    model_dir = Path(args.model_dir) if args.model_dir \
        else Path(args.models_dir) / default_model_dir().name
    check((model_dir / "vocab.txt").is_file() and (model_dir / "vad-model.onnx").is_file(),
          "Parakeet export present", str(model_dir))
    sets = [name for name in ("cpu", "gpu")
            if any((model_dir / name).glob("*.onnx"))]
    check(bool(sets), "weight set for this machine (cpu int4 / gpu fp16)", ", ".join(sets) or str(model_dir))

    if problems:
        print("\nselftest: %d problem(s) - run install.bat" % problems)
        return 2

    # The renderer is pure, so pin its contract before paying for a model load.
    sample = render(
        "selftest.wav",
        [Segment(0.5, 2.0, "First sentence."), Segment(2.1, 4.0, "Second sentence.")],
    )
    check("**[00:00:00]**" in sample or "**[00:00:01]**" in sample, "renderer emits timestamps")
    check(fmt_time(3725) == "01:02:05", "timestamp format is HH:MM:SS", fmt_time(3725))

    # The model load is the step measured silent for 600 s on `--device trt`, so it says it started and
    # it gets its own budget: a first-run TensorRT engine build is legitimately quiet for minutes.
    stage("loading the model (a first-run TensorRT engine build is slower)",
          _env_seconds("PARAKEET_LOAD_QUIET_S", DEFAULT_LOAD_QUIET_S))
    try:
        engine = load_engine(
            model_dir=model_dir, device=args.device, threads=args.threads,
        )
    except Exception as exc:  # noqa: BLE001 - an install problem is a message, not a traceback
        print("FAIL  model loads  - %s: %s" % (type(exc).__name__, exc))
        return 2
    check(True, "model loads (%s)" % engine.info.runtime)
    # The truth about where it runs: a session that silently fell back to CPU is the failure this
    # line exists to catch, so it prints the providers the sessions actually report.
    print("      %s | %d threads | %s"
          % (engine.info.describe(), engine.info.threads, engine.info.provider_summary()))
    for note in engine.info.notes():
        print("      note: %s" % note)

    import numpy as np

    stage("running the model on 3 s of silence")
    silence = np.zeros(16_000 * 3, dtype=np.float32)
    try:
        segments = engine.recognize_block(silence)
    except Exception as exc:  # noqa: BLE001
        print("FAIL  recognize on silence  - %s: %s" % (type(exc).__name__, exc))
        return 2
    check(len(segments) == 0, "3 s of silence produces no segments", "%d segments" % len(segments))

    if problems:
        print("\nselftest: %d problem(s)" % problems)
        return 2
    print("\nselftest: OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Before ANY model work: the two measured failures this file exists for were a model load that
    # never spoke, and nothing else in the program owns a clock.
    start_watchdog()
    try:
        return _run(args)
    finally:
        stop_watchdog()


def _run(args) -> int:
    if args.selftest:
        return selftest(args)

    models_dir = Path(args.models_dir)
    model_dir = Path(args.model_dir) if args.model_dir \
        else models_dir / default_model_dir().name

    try:
        require_tools()
    except MediaError as exc:
        print("environment: %s" % exc, file=sys.stderr)
        return 2

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print("environment: input folder %s does not exist" % input_dir, file=sys.stderr)
        return 2

    # Printed BEFORE the load, not after it: the load is the longest silent step in a normal run
    # (22.28 s measured on CPU) and the first-output ceiling measures exactly this gap.
    print("loading the model - paid once per run, not once per file: %s (%s)"
          % (model_dir.name, args.device), flush=True)
    stage("loading the model (a first-run TensorRT engine build is slower)",
          _env_seconds("PARAKEET_LOAD_QUIET_S", DEFAULT_LOAD_QUIET_S))
    loading = time.time()
    try:
        engine = load_engine(
            model_dir=model_dir, device=args.device, threads=args.threads,
        )
    except FileNotFoundError:
        return 2
    except Exception as exc:  # noqa: BLE001
        print("model: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2
    load_seconds = time.time() - loading

    # The block a lay user reads first. It answers the three questions he actually has - where do I
    # put the file, where does the .md appear, and why did 3 s of audio cost seconds - and it prints
    # the measured model-load time instead of letting him wonder whether the GPU was used.
    print("")
    print("parakeet-to-md %s" % __version__)
    print("  engine    %s" % engine.info.describe())
    print("            loaded in %.1f s - paid once per run, not once per file" % load_seconds)
    print("            %d threads | %s" % (engine.info.threads, engine.info.provider_summary()))
    for note in engine.info.notes():
        # Only when it changes what the user should expect: an explicit gpu request that did not
        # load, or `auto` with no GPU at all. A working DirectML run says nothing about CUDA.
        print("            note: %s" % note)
    print("  input     %s" % input_dir)
    print("  output    %s" % Path(args.output))
    print("  archive   %s" % ("(sources stay in input)" if args.no_archive else Path(args.archive)))
    print("")

    # `run_directory` prints once per BLOCK of audio, and a block is `--block-seconds` of audio (600 s
    # by default), so the longest legitimate silent stretch here is one block plus the decode around
    # it. Without this budget the 90 s default would kill a working CPU transcription of a long file -
    # a guard that fires on real progress is worse than no guard.
    stage("transcribing (a progress line is printed once per block of audio)",
          max(_env_seconds("PARAKEET_QUIET_S", DEFAULT_QUIET_S), args.block_seconds + 60.0))
    results = run_directory(
        engine,
        input_dir=input_dir,
        output_dir=Path(args.output),
        archive_dir=Path(args.archive),
        block_seconds=args.block_seconds,
        keep_partial=args.keep_partial,
        archive_source=not args.no_archive,
        only=args.only,
        with_parts=args.with_parts,
    )
    return summary(results, load_seconds=load_seconds)


if __name__ == "__main__":
    sys.exit(main())


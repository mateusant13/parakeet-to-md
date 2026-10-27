"""input -> archive -> output, one file at a time, resumable.

The unit of work is a *block* and each block is checkpointed to
`output/.partial/<stem>.jsonl` the moment it is recognized. That is the difference between "a 3-hour
file failed at minute 170" being a catastrophe and being a re-run that starts at minute 170: decoding
is cheap, the model pass is not, so nothing already paid for is ever thrown away.

Nothing is moved out of `input/` until its markdown has been written and read back. A file that fails
stays where the owner put it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .asr import Engine
from .media import SAMPLE_RATE, MediaError, iter_blocks, probe, require_tools
from .render import render
from .segments import Segment

MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".ts", ".mpg", ".mpeg", ".wmv", ".flv",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma", ".aiff", ".aif",
}


@dataclass
class Result:
    source: Path
    output: Path | None = None
    archived: Path | None = None
    segments: int = 0
    words: int = 0
    duration: float = 0.0
    blocks: int = 0
    resumed_from: float = 0.0
    elapsed: float = 0.0
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and self.output is not None


def media_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    out = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in MEDIA_EXTENSIONS
    ]
    return out


def _partial_path(output_dir: Path, stem: str) -> Path:
    return output_dir / ".partial" / ("%s.jsonl" % stem)


def _read_partial(path: Path) -> tuple[list[Segment], float, int]:
    """(segments so far, seconds already covered, last block index)."""
    segments: list[Segment] = []
    offset = 0.0
    last_index = -1
    if not path.exists():
        return segments, offset, last_index
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written last line from a kill: everything before it is trustworthy
            for start, end, text in row.get("segments") or []:
                segments.append(Segment(float(start), float(end), str(text)))
            offset = float(row.get("end_offset") or offset)
            last_index = max(last_index, int(row.get("i", last_index + 1)))
    return segments, offset, last_index


def _append_partial(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _unique(path: Path) -> Path:
    """`archive/name.mp4` -> `archive/name (2).mp4` if taken, so nothing is overwritten."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for n in range(2, 1000):
        candidate = path.with_name("%s (%d)%s" % (stem, n, suffix))
        if not candidate.exists():
            return candidate
    raise RuntimeError("cannot find a free archive name for %s" % path.name)


def process_file(
    engine: Engine,
    source: Path,
    output_dir: Path,
    archive_dir: Path,
    block_seconds: float = 600.0,
    keep_partial: bool = False,
    archive_source: bool = True,
    progress: bool = True,
    with_parts: bool = False,
) -> Result:
    started = time.time()
    result = Result(source=source)
    stem = source.stem
    partial = _partial_path(output_dir, stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        info = probe(source)
        result.duration = info["duration"]
        prior, covered, last_index = _read_partial(partial)
        if covered > 0:
            result.resumed_from = covered
            if progress:
                print(
                    "  resuming at %s (%d segments already done)"
                    % (time.strftime("%H:%M:%S", time.gmtime(covered)), len(prior)),
                    flush=True
                )

        segments: list[Segment] = list(prior)
        base_index = last_index + 1
        blocks = 0
        for rel_index, offset, audio in iter_blocks(
            source, block_seconds=block_seconds, start_offset_seconds=covered
        ):
            block_segments = engine.recognize_block(audio)
            absolute = [
                Segment(offset + s.start, offset + s.end, s.text) for s in block_segments
            ]
            end_offset = offset + audio.size / SAMPLE_RATE
            _append_partial(partial, {
                "i": base_index + rel_index,
                "offset": offset,
                "end_offset": end_offset,
                "segments": [[s.start, s.end, s.text] for s in absolute],
            })
            segments.extend(absolute)
            blocks += 1
            if progress and result.duration > block_seconds:
                # Only multi-block files get a progress line: a 3 s clip finished before the user could
                # read it, and printing "00:00:03 / 00:00:03 (100%)" there was noise, not information.
                done = end_offset / result.duration * 100.0 if result.duration else 0.0
                print(
                    "      block %d | %s of %s (%.0f%%) | %d segment(s) so far"
                    % (base_index + rel_index + 1, _clock(end_offset), _clock(result.duration),
                       done, len(segments)),
                    flush=True
                )

        result.blocks = blocks
        result.segments = len(segments)
        result.words = sum(len(s.text.split()) for s in segments)

        document = render(source.name, segments, with_parts=with_parts)

        target = output_dir / ("%s.md" % stem)
        if target.exists():
            # Keep the previous transcript instead of silently destroying it; archive/ is the folder
            # the owner already expects to hold history.
            archive_dir.mkdir(parents=True, exist_ok=True)
            backup = _unique(archive_dir / ("%s.md.bak" % stem))
            shutil.move(str(target), str(backup))
            result.notes.append("previous transcript kept at %s" % backup.name)

        tmp = target.with_suffix(".md.part")
        tmp.write_text(document, encoding="utf-8", newline="\n")
        readback = tmp.read_text(encoding="utf-8")
        if len(readback) != len(document):
            raise RuntimeError("written transcript does not read back the same length")
        os.replace(tmp, target)
        result.output = target

        if not segments:
            result.notes.append("no speech detected")

        if archive_source:
            archive_dir.mkdir(parents=True, exist_ok=True)
            destination = _unique(archive_dir / source.name)
            shutil.move(str(source), str(destination))
            result.archived = destination
        else:
            result.notes.append("source left in input/ (--no-archive)")

        if partial.exists() and not keep_partial:
            partial.unlink()

    except (MediaError, FileNotFoundError, RuntimeError, ValueError) as exc:
        result.error = "%s: %s" % (type(exc).__name__, exc)
    except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
        result.error = "%s: %s" % (type(exc).__name__, exc)

    result.elapsed = time.time() - started
    return result


def _clock(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def _speed(duration: float, elapsed: float) -> str:
    """`3.9x realtime` - the number that explains why a 3 s clip still costs seconds.

    Below 1x the run is slower than the audio itself, which is the case a user needs to see named
    rather than smoothed into "1x": one hour of video on a CPU-only machine takes longer than an hour.
    """
    if not duration or elapsed <= 0:
        return ""
    ratio = duration / elapsed
    if ratio < 1.0:
        return "%.2fx realtime - slower than the audio itself" % ratio
    return "%.1fx realtime" % ratio


def run_directory(
    engine: Engine,
    input_dir: Path,
    output_dir: Path,
    archive_dir: Path,
    block_seconds: float = 600.0,
    keep_partial: bool = False,
    archive_source: bool = True,
    only: str | None = None,
    with_parts: bool = False,
) -> list[Result]:
    require_tools()
    files = media_files(input_dir)
    if only:
        wanted = only.lower()
        files = [p for p in files if p.name.lower() == wanted or p.stem.lower() == wanted]
    if not files:
        print("nothing to do: no audio/video files in %s" % input_dir)
        return []

    print("%d file(s) to process in %s" % (len(files), input_dir))
    results: list[Result] = []
    for index, path in enumerate(files, start=1):
        size_mb = path.stat().st_size / (1024 * 1024)
        print("[%d/%d] %s (%.1f MB)" % (index, len(files), path.name, size_mb), flush=True)
        result = process_file(
            engine, path, output_dir, archive_dir,
            block_seconds=block_seconds, keep_partial=keep_partial, archive_source=archive_source,
            with_parts=with_parts,
        )
        results.append(result)
        if result.ok:
            speed = _speed(result.duration, result.elapsed)
            print(
                "      -> %s  |  %d segment(s), %d words  |  %.1f s for %s of audio%s"
                % (result.output, result.segments, result.words, result.elapsed,
                   _clock(result.duration), ("  (%s)" % speed) if speed else ""),
                flush=True
            )
            if result.archived:
                print("      source moved to %s" % result.archived, flush=True)
            for note in result.notes:
                print("      note: %s" % note, flush=True)
        else:
            print("      FAILED: %s" % result.error, file=sys.stderr, flush=True)
            print("      the file stays in %s, nothing was written" % input_dir, file=sys.stderr, flush=True)
    return results


def summary(results: list[Result], load_seconds: float = 0.0) -> int:
    done = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    if not results:
        return 0
    total = sum(r.duration for r in done)
    work = sum(r.elapsed for r in done)
    words = sum(r.words for r in done)
    speed = _speed(total, work)
    print("")
    print("done")
    print("  files    %d ok, %d failed" % (len(done), len(failed)))
    print("  audio    %s" % _clock(total))
    print("  words    %d" % words)
    if work:
        print("  work     %.1f s%s" % (work, ("  (%s)" % speed) if speed else ""))
    if load_seconds:
        # Printed as its own line because it is the whole answer to "why did a 3 s clip take
        # seconds": the model load is neither the audio nor the transcription.
        print("  total    %.1f s, of which %.1f s was loading the model" % (work + load_seconds, load_seconds))
    if failed:
        print("")
        print("failed:")
        for r in failed:
            print("  - %s: %s" % (r.source.name, r.error))
        return 1
    return 0

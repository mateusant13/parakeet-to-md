"""Media in, bounded 16 kHz mono PCM blocks out.

The owner's use case is a *hours-long* file, which decides two things:

1. **Blocks, not one array.** Three hours of 16 kHz float32 is 690 MB before the VAD even looks at
   it; a 600 s block is 38 MB and the model's attention stays bounded.
2. **Cut at the quietest frame, not at the boundary.** A hard cut at exactly 600.000 s lands inside a
   word, and the recognizer then emits a fragment for the tail of one block and another for the head
   of the next. Measured on a probe clip whose second utterance spans 4.58–8.32 s: a naive cut at 6 s
   splits it in half. So each block is read with `search_seconds` of lookahead and cut at the lowest
   energy 50 ms frame inside that window, which is where speech is not.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

from . import paths

SAMPLE_RATE = 16_000
FRAME_MS = 50
BYTES_PER_SAMPLE = 2


class MediaError(RuntimeError):
    """The file cannot be read by ffmpeg/ffprobe, or it has no audio stream."""


def ffmpeg_path() -> str | None:
    """Bundled copy first, PATH second (see `paths.tool`)."""
    return paths.tool("ffmpeg")


def ffprobe_path() -> str | None:
    return paths.tool("ffprobe")


def require_tools() -> None:
    missing = [name for name, path in (("ffmpeg", ffmpeg_path()), ("ffprobe", ffprobe_path())) if not path]
    if missing:
        raise MediaError(
            "missing %s - install.bat unpacks a copy into tools\\ffmpeg\\bin; or install it "
            "(winget install Gyan.FFmpeg) and reopen the terminal" % " and ".join(missing)
        )


def probe(path: Path) -> dict:
    """Duration in seconds and whether there is an audio stream. Raises MediaError if unreadable."""
    ffprobe = ffprobe_path()
    if not ffprobe:
        raise MediaError("ffprobe is not on PATH")
    cmd = [
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaError("ffprobe refused %s: %s" % (path.name, (proc.stderr or "").strip()[:300]))
    info = json.loads(proc.stdout or "{}")
    streams = info.get("streams") or []
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        raise MediaError("%s has no audio stream" % path.name)
    try:
        duration = float((info.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0.0:
        # Some containers only carry duration per stream.
        try:
            duration = float(audio[0].get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
    if duration <= 0.0:
        raise MediaError("%s reports no duration - refusing rather than guessing the end" % path.name)
    return {
        "duration": duration,
        "codec": audio[0].get("codec_name"),
        "channels": audio[0].get("channels"),
        "sample_rate": audio[0].get("sample_rate"),
    }


def _quietest_cut(pcm: np.ndarray, lo: int, hi: int, frame: int) -> int:
    """Sample index in [lo, hi) at the centre of the lowest-energy frame. `hi - lo` may be < frame."""
    if hi - lo <= frame:
        return max(lo, (lo + hi) // 2)
    window = pcm[lo:hi].astype(np.float32)
    frames = len(window) // frame
    if frames <= 0:
        return max(lo, (lo + hi) // 2)
    trimmed = window[: frames * frame].reshape(frames, frame)
    energy = np.sqrt(np.mean(trimmed * trimmed, axis=1))
    best = int(np.argmin(energy))
    return lo + int((best + 0.5) * frame)


def iter_blocks(
    path: Path,
    block_seconds: float = 600.0,
    search_seconds: float = 12.0,
    start_offset_seconds: float = 0.0,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield `(block_index, offset_seconds, float32 mono samples)` covering the file exactly once.

    `offset_seconds` is the absolute position of the first sample of the block in the source, which is
    what makes the timestamps in the transcript absolute. `block_index` counts from 0 for the first
    block produced from `start_offset_seconds`, so a resume continues the numbering the checkpoint
    used.
    """
    require_tools()
    ffmpeg = ffmpeg_path()
    block_n = int(block_seconds * SAMPLE_RATE)
    search_n = int(search_seconds * SAMPLE_RATE)
    frame_n = int(FRAME_MS * SAMPLE_RATE / 1000)
    if block_n <= search_n * 2:
        raise ValueError(
            "block_seconds=%.1f is too small: it must be more than twice search_seconds=%.1f so a "
            "quiet cut point always exists inside the block" % (block_seconds, search_seconds)
        )

    cmd = [ffmpeg, "-nostdin", "-v", "error"]
    if start_offset_seconds > 0:
        cmd += ["-ss", "%.3f" % start_offset_seconds]
    cmd += ["-i", str(path), "-vn", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    buf = bytearray()
    offset = float(start_offset_seconds)
    index = 0
    need_bytes = (block_n + search_n) * BYTES_PER_SAMPLE
    try:
        while True:
            while len(buf) < need_bytes:
                chunk = proc.stdout.read(max(4096, need_bytes - len(buf)))
                if not chunk:
                    break
                buf.extend(chunk)
            if not buf:
                break
            pcm = np.frombuffer(bytes(buf), dtype=np.int16)
            if pcm.size <= block_n:  # last block of the file: take all of it
                yield index, offset, pcm.astype(np.float32) / 32768.0
                break
            cut = _quietest_cut(pcm, block_n - search_n // 2, pcm.size, frame_n)
            yield index, offset, pcm[:cut].astype(np.float32) / 32768.0
            offset += cut / SAMPLE_RATE
            buf = bytearray(pcm[cut:].tobytes())
            index += 1
        err = b""
        try:
            err = proc.stderr.read() or b""
        except Exception:  # noqa: BLE001 - closing a pipe is not a failure of the read
            err = b""
    finally:
        if proc.poll() is None:
            proc.kill()
        if proc.returncode not in (0, None) and err:
            print(
                "media: ffmpeg exited %s: %s" % (proc.returncode, err.decode("utf-8", "replace")[:300]),
                file=sys.stderr,
            )

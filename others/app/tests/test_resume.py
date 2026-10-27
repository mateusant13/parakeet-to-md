"""Resume: a killed run must continue at the checkpoint, not restart at zero.

This is the claim the README makes for hours-long files ("a crash at minute 170 resumes at minute 170"),
and until this test existed nothing read the checkpoint back. The stub engine keeps the test free of the
model while still going through `process_file` - the same code the CLI uses.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from parakeet_to_md.pipeline import process_file
from parakeet_to_md.segments import Segment


class StubEngine:
    """Recognizes nothing, records what it was asked to recognize, and can die on a chosen block."""

    class info:  # noqa: N801 - mirrors the real Engine.info shape
        quantization = "int8"
        device = "cpu"
        providers = ["CPUExecutionProvider"]
        runtime = "stub"

    def __init__(self, die_after: int | None = None) -> None:
        self.calls: list[int] = []
        self.die_after = die_after

    def recognize_block(self, audio: np.ndarray) -> list[Segment]:
        self.calls.append(int(audio.size))
        if self.die_after is not None and len(self.calls) > self.die_after:
            raise RuntimeError("simulated crash")
        return []


def _wav(path: Path, seconds: float = 12.0) -> None:
    rate = 16_000
    t = np.arange(int(rate * seconds)) / rate
    tone = (np.sin(2 * np.pi * 220 * t) * 3000).astype(np.int16)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(tone.tobytes())


def test_a_crashed_run_resumes_at_the_checkpoint_instead_of_restarting(tmp_path: Path) -> None:
    source = tmp_path / "input" / "talk.wav"
    source.parent.mkdir(parents=True)
    _wav(source, 90.0)
    out, archive = tmp_path / "output", tmp_path / "archive"

    crashing = StubEngine(die_after=1)
    first = process_file(crashing, source, out, archive, block_seconds=60.0, archive_source=False,
                         progress=False)
    assert not first.ok, "the simulated crash must fail the run"
    assert source.exists(), "a failed run must leave the source in place"

    partial = out / ".partial" / "talk.jsonl"
    assert partial.exists(), "the checkpoint must survive the crash"
    rows = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    covered = float(rows[-1]["end_offset"])
    assert covered > 0.0

    resumed = StubEngine()
    second = process_file(resumed, source, out, archive, block_seconds=60.0, archive_source=False,
                          progress=False)
    assert second.ok, second.error
    assert second.resumed_from == covered, "the second run must start at the checkpoint"
    # 90 s of audio, 60 s blocks: a fresh run needs 2, so the resumed run must need fewer.
    assert len(resumed.calls) <= 1
    assert not partial.exists(), "a successful run clears its checkpoint"

"""The one data type that travels from the recognizer to the renderer."""

from __future__ import annotations

import re
from dataclasses import dataclass

# A segment whose text carries no letter or digit is not speech: Silero VAD emits short blips on
# music, room tone and clicks, and the recognizer turns those into "." or "♪". They cost a
# timestamp and a paragraph break, so they are dropped here rather than in the renderer.
_HAS_CONTENT = re.compile(r"[^\W_]", re.UNICODE)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?%])")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class Segment:
    """One utterance: absolute start/end in seconds and its text."""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def clean_text(text: str) -> str:
    """Collapse whitespace and repair the space-before-punctuation the model sometimes emits."""
    if not text:
        return ""
    out = _SPACES.sub(" ", text).strip()
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
    return out


def normalize(segments) -> list[Segment]:
    """Clean, drop non-speech, and enforce a monotonic timeline.

    The monotonic pass matters across block boundaries: the last segment of block N and the first of
    block N+1 can overlap by a few milliseconds because the cut is at a quiet frame, not at 600.000 s.
    An out-of-order timestamp in the `.md` is a defect the reader can see.
    """
    out: list[Segment] = []
    last_end = 0.0
    for seg in segments:
        text = clean_text(seg.text)
        if len(text) < 2 or not _HAS_CONTENT.search(text):
            continue
        start = max(float(seg.start), last_end)
        end = max(float(seg.end), start)
        if end - start <= 0.0:
            end = start + 0.01
        out.append(Segment(start=start, end=end, text=text))
        last_end = end
    return out

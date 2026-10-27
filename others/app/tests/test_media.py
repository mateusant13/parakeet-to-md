"""Block cutting: the cut must land where speech is not, and never outside the block."""

from __future__ import annotations

import numpy as np
import pytest

from parakeet_to_md.media import SAMPLE_RATE, _quietest_cut, iter_blocks


def _tone_then_silence(at: int, total: int, quiet: int = 800) -> np.ndarray:
    """Loud everywhere, except a `quiet`-sample hole starting at `at`."""
    pcm = (np.ones(total, dtype=np.int16) * 8000)
    pcm[at:at + quiet] = 0
    return pcm


def test_cut_lands_inside_the_quiet_hole() -> None:
    frame = 800  # 50 ms at 16 kHz
    total = SAMPLE_RATE * 2
    hole = SAMPLE_RATE + 4000
    pcm = _tone_then_silence(hole, total, quiet=frame)
    cut = _quietest_cut(pcm, SAMPLE_RATE, total, frame)
    assert hole <= cut <= hole + frame


def test_cut_stays_inside_the_search_window() -> None:
    frame = 800
    total = SAMPLE_RATE * 2
    lo, hi = SAMPLE_RATE, SAMPLE_RATE + 4000
    cut = _quietest_cut(_tone_then_silence(hi - 400, total, quiet=frame), lo, hi, frame)
    assert lo <= cut <= hi


def test_a_window_smaller_than_a_frame_does_not_crash_or_escape() -> None:
    frame = 800
    cut = _quietest_cut(np.zeros(SAMPLE_RATE, dtype=np.int16), 1000, 1200, frame)
    assert 1000 <= cut <= 1200


def test_block_seconds_must_exceed_the_search_window() -> None:
    with pytest.raises(ValueError):
        list(iter_blocks(__file__, block_seconds=10.0, search_seconds=12.0))

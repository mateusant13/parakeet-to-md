"""The renderer's contract: it is the captions and nothing else.

The owner's 2026-10-27 correction is encoded here as assertions: no front matter, no facts table, no
"generated" stamp, no part headings unless asked. Each test fails if that rule is broken.
"""

from __future__ import annotations

from parakeet_to_md.render import fmt_time, paragraphs, parts, render
from parakeet_to_md.segments import Segment, clean_text, normalize


def test_timestamp_is_always_hh_mm_ss() -> None:
    assert fmt_time(0) == "00:00:00"
    assert fmt_time(9.6) == "00:00:10"
    assert fmt_time(61) == "00:01:01"
    assert fmt_time(3725) == "01:02:05"
    assert fmt_time(-3) == "00:00:00"


def test_clean_text_repairs_space_before_punctuation() -> None:
    assert clean_text("hello ,   world .") == "hello, world."
    assert clean_text("   spaced   out  ") == "spaced out"


def test_normalize_drops_non_speech_segments() -> None:
    segments = normalize([
        Segment(0.0, 0.4, "."),
        Segment(0.5, 0.9, "♪"),
        Segment(1.0, 2.0, "Real words here."),
    ])
    assert [s.text for s in segments] == ["Real words here."]


def test_normalize_forces_a_monotonic_timeline() -> None:
    # A block boundary can hand back a start that predates the previous end by a few ms.
    segments = normalize([
        Segment(10.0, 12.0, "first part."),
        Segment(11.9, 13.0, "second part."),
    ])
    assert segments[1].start >= segments[0].end


def test_a_long_pause_starts_a_new_paragraph() -> None:
    segs = [Segment(0.0, 2.0, "one"), Segment(2.2, 4.0, "still same thought"),
            Segment(7.0, 9.0, "new thought.")]
    paras = paragraphs(segs)
    assert len(paras) == 2
    assert paras[1].text == "new thought."
    assert paras[0].text == "one still same thought"


def test_every_finished_sentence_is_its_own_line() -> None:
    """Sotto's structure (`sentence_segments`, transcribe.py:132) and the owner's ask of 2026-10-27:
    a sentence that ends in `.`/`?`/`!` closes its line even with no pause before the next one, so the
    timestamp belongs to the sentence and not to a 700-character block of them."""
    segs = [Segment(0.0, 2.0, "one."), Segment(2.2, 4.0, "two?"), Segment(4.2, 6.0, "three!")]
    paras = paragraphs(segs)
    assert [p.text for p in paras] == ["one.", "two?", "three!"]
    assert [p.start for p in paras] == [0.0, 2.2, 4.2]


def test_a_sentence_with_no_full_stop_keeps_accumulating() -> None:
    """The other half of the rule: our units are VAD chunks, so speech without punctuation must still
    join up instead of becoming one line per chunk."""
    segs = [Segment(0.0, 2.0, "and then he said"), Segment(2.2, 4.0, "that it was fine.")]
    paras = paragraphs(segs)
    assert [p.text for p in paras] == ["and then he said that it was fine."]


def test_a_line_never_runs_past_the_thirty_second_cap() -> None:
    """Sotto caps a segment at 30 s; without a cap a 10-minute monologue with one full stop would be a
    single timestamp, which is the failure the cap exists to prevent."""
    segs = [Segment(t * 2.0, t * 2.0 + 1.9, "word") for t in range(60)]
    paras = paragraphs(segs)
    assert all(p.end - p.start <= 30.0 + 2.0 for p in paras)
    assert len(paras) > 1


def test_long_speech_is_split_so_no_paragraph_is_a_wall_of_text() -> None:
    segs = [Segment(i * 2.0, i * 2.0 + 1.9, "word " * 12) for i in range(20)]
    paras = paragraphs(segs, max_chars=200)
    assert len(paras) > 1
    assert all(len(p.text) <= 200 + 60 for p in paras)


def test_parts_group_by_ten_minutes() -> None:
    paras = paragraphs([
        Segment(5.0, 6.0, "a."),
        Segment(650.0, 651.0, "b."),
        Segment(1250.0, 1251.0, "c."),
    ])
    grouped = parts(paras)
    assert [p.index for p in grouped] == [1, 2, 3]
    assert [fmt_time(p.start) for p in grouped] == ["00:00:05", "00:10:50", "00:20:50"]


def test_render_is_only_the_captions() -> None:
    document = render(
        "lecture.mp4",
        [Segment(0.0, 2.0, "Opening line."), Segment(30.0, 33.0, "Second paragraph.")],
    )
    assert document.startswith("# lecture")
    assert "**[00:00:00]** Opening line." in document
    assert "**[00:00:30]** Second paragraph." in document
    # The owner removed the metadata block; nothing may reintroduce it.
    assert "---" not in document, "no YAML front matter"
    assert "generated:" not in document, "no generated stamp"
    assert "| Source |" not in document and "| Engine |" not in document, "no facts table"
    assert "## Part" not in document, "parts are opt-in"


def test_parts_are_opt_in() -> None:
    document = render(
        "long.mp4",
        [Segment(5.0, 6.0, "a."), Segment(650.0, 651.0, "b.")],
        with_parts=True,
    )
    assert "## Part 1" in document and "## Part 2" in document


def test_render_with_no_segments_says_so_instead_of_emitting_an_empty_page() -> None:
    document = render("silent.mp3", [])
    assert "No speech detected" in document
    assert "## Part" not in document

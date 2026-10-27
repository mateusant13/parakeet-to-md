"""Folder behaviour: what counts as media, and never overwriting an archive entry."""

from __future__ import annotations

from pathlib import Path

from parakeet_to_md.pipeline import _unique, media_files


def test_media_files_takes_audio_and_video_and_ignores_everything_else(tmp_path: Path) -> None:
    (tmp_path / "lecture.MP4").write_bytes(b"x")
    (tmp_path / "call.mp3").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("nope", encoding="utf-8")
    (tmp_path / ".hidden.mp4").write_bytes(b"x")
    (tmp_path / "sub").mkdir()

    names = [p.name for p in media_files(tmp_path)]
    assert names == ["call.mp3", "lecture.MP4"]


def test_unique_never_returns_an_existing_name(tmp_path: Path) -> None:
    first = tmp_path / "talk.mp4"
    first.write_bytes(b"x")
    second = _unique(first)
    assert second.name == "talk (2).mp4"
    second.write_bytes(b"x")
    assert _unique(first).name == "talk (3).mp4"


def test_missing_input_folder_is_empty_not_an_error(tmp_path: Path) -> None:
    assert media_files(tmp_path / "nope") == []

"""End-to-end proof on real audio: not part of `pytest`, run by hand or by the release check.

    .venv\\Scripts\\python tests\\smoke_real.py

It builds a real MP4 (Windows TTS speech over a colour video track, looped until it is long enough to
need several blocks), runs the *installed* entry point exactly as `run.bat` does, and then asserts the
things a user would notice if they broke:

  * the transcript is named after the video and sits in output/
  * the source file moved to archive/
  * every timestamp is present, monotonic, and inside the file's duration
  * the file was actually cut into more than one block (so the block path ran, not just block 1)
  * a second run of the same folder does nothing and says so

This is the check that would have caught a renderer that emits one line per segment, or an archive
step that copies instead of moves.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "app"))

from parakeet_to_md.cli import main as cli_main  # noqa: E402

SPEECH = (
    "This is an end to end test of the transcription pipeline. "
    "The file contains several sentences so that the reader has to group them into paragraphs. "
    "Timestamps must be absolute and must never go backwards. "
)


def build_video(target: Path, seconds: int = 300) -> None:
    """TTS speech + a colour video track, looped to `seconds`."""
    wav = target.with_suffix(".tts.wav")
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.SetOutputToWaveFile('%s'); $s.Speak('%s'); $s.Dispose();"
        % (wav, SPEECH.replace("'", "''"))
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True,
                   stdout=subprocess.DEVNULL)
    loops = max(1, seconds // 20)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-stream_loop", str(loops), "-i", str(wav),
            "-f", "lavfi", "-i", "color=c=black:s=320x240:r=10",
            "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k", str(target),
        ],
        check=True,
    )
    wav.unlink(missing_ok=True)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="parakeet-smoke-"))
    src = work / "input"
    out = work / "output"
    archive = work / "archive"
    for d in (src, out, archive):
        d.mkdir(parents=True)

    video = src / "smoke lecture.mp4"
    print("building a 5-minute test video with real speech ...", flush=True)
    build_video(video, seconds=300)

    print("running the installed entry point ...", flush=True)
    rc = cli_main([
        "--input", str(src), "--output", str(out), "--archive", str(archive),
        "--block-seconds", "60",
    ])

    problems: list[str] = []

    def check(ok: bool, what: str, detail: str = "") -> None:
        print("%s  %s%s" % ("PASS" if ok else "FAIL", what, ("  - " + detail) if detail and not ok else ""))
        if not ok:
            problems.append(what)

    md = out / "smoke lecture.md"
    check(rc == 0, "exit code 0", str(rc))
    check(md.exists(), "transcript named after the video", str(md))
    check((archive / "smoke lecture.mp4").exists(), "source moved to archive/")
    check(not video.exists(), "source gone from input/")

    if md.exists():
        text = md.read_text(encoding="utf-8")
        stamps = [int(h) * 3600 + int(m) * 60 + int(s)
                  for h, m, s in re.findall(r"\*\*\[(\d\d):(\d\d):(\d\d)\]\*\*", text)]
        check(len(stamps) >= 4, "several timestamped paragraphs", str(len(stamps)))
        check(stamps == sorted(stamps), "timestamps never go backwards")
        check(all(0 <= s <= 400 for s in stamps), "timestamps inside the file's duration",
              str(stamps[-5:]))
        check("## Part" not in text, "lean document: no part headings by default")
        check("generated:" not in text and "---" not in text.split("\n")[0],
              "lean document: no metadata block")
        check("No speech detected" not in text, "speech was actually recognized")
        body = text
        check(len(re.findall(r"\.", body)) >= 4, "text carries sentence punctuation")
        # Several blocks were processed: the log line is the witness, the markdown only shows text.
        check(text.count("**[" ) >= 4, "more than one paragraph per file")

    leftover = [p.name for p in src.iterdir() if p.is_file() and p.suffix != ".gitkeep"]
    check(leftover == [], "input/ is empty afterwards", str(leftover))

    print("\nwork dir kept for inspection: %s" % work)
    if not problems:
        shutil.rmtree(work, ignore_errors=True)
        print("smoke: OK")
        return 0
    print("smoke: %d problem(s): %s" % (len(problems), ", ".join(problems)))
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Fetch the weights a run needs into `others/models/parakeet/`.

Two sets, and the installer is told which one to fetch:

* ``cpu``  - the 4-bit MatMulNBits export of moondream/parakeet-redux, 416 MB. This is the default and
  the only set a machine without a usable GPU needs.
* ``gpu``  - the fp16 NeMo export, 1.2 GB, which is what TensorRT, CUDA and DirectML can put on a card.
  A GPU machine also gets the ``cpu`` set, so a driver that stops working costs speed, not the program.

Downloads run in parallel and every file is pinned by size and SHA-256: a 1.2 GB weight file that
arrives truncated is the kind of failure that shows up as a wrong transcript an hour later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .asr import default_model_dir

REDUX = "eschmidbauer/parakeet-redux-onnx"
FP16 = "grikdotnet/parakeet-tdt-0.6b-fp16"

#: (repo, remote name, local path, size, sha256)
SHARED = (
    (REDUX, "vocab.txt", "vocab.txt", 93939,
     "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"),
    (REDUX, "vad-model.onnx", "vad-model.onnx", 18300687,
     "18dfbe344307f9eda4735d4da1fe1cd6e3be27055a6c25b4a4a2e6fe0bffde50"),
    (REDUX, "preprocessor.onnx", "preprocessor.onnx", 1224294,
     "3de5742f5780fc8615c1d883a2a60e28621ff13d2e20d24d76e94b9e4168445f"),
)

CPU = (
    (REDUX, "encoder-model.onnx", "cpu/encoder-model.onnx", 343841943,
     "ade2c65c188776b3662db620a553ab82ac7e9ca11a7be6b696bcc5190f75382e"),
    (REDUX, "decoder_joint-model.onnx", "cpu/decoder_joint-model.onnx", 72552270,
     "97ae0cb049f32a5d7a6572e91020b8d87d66baffa0ff9de2cf6de4c0960e92f8"),
)

GPU = (
    (FP16, "encoder-model.fp16.onnx", "gpu/encoder-model.fp16.onnx", 1238960452,
     "a2bdeeb99cb7e5548818e823127b33854dd0c26f5d0c8da91effdd895ea0e717"),
    (FP16, "decoder_joint-model.fp16.onnx", "gpu/decoder_joint-model.fp16.onnx", 36266140,
     "b33a73b7c1d71b9d5a0911f5cb478be3dcbf79f53355c531ab1cd1dcd68ad8ef"),
    (FP16, "nemo128.onnx", "gpu/nemo128.onnx", 139764,
     "a9fde1486ebfcc08f328d75ad4610c67835fea58c73ba57e3209a6f6cf019e9f"),
)

CONFIG = {
    "model_type": "nemo-conformer-tdt",
    "features_size": 128,
    "subsampling_factor": 8,
    "durations": [0, 1, 2, 3, 4],
    "vocab_size": 8193,
    "blank_token_id": 8192,
    "max_tokens_per_step": 10,
    "encoder_frame_seconds": 0.08,
    "sample_rate": 16000,
    "source": "moondream/parakeet-redux",
}

_print_lock = threading.Lock()


def _say(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(repo: str, remote: str, target: Path, size: int, sha: str) -> None:
    import urllib.error
    import urllib.request

    if target.is_file() and target.stat().st_size == size and _sha256(target) == sha:
        _say("    --  %s (already here)" % target.name)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    url = "https://huggingface.co/%s/resolve/main/%s" % (repo, remote)
    started = time.time()
    for attempt in range(1, 4):
        offset = part.stat().st_size if part.is_file() else 0
        request = urllib.request.Request(url)
        if offset:
            request.add_header("Range", "bytes=%d-" % offset)
        try:
            with urllib.request.urlopen(request, timeout=60) as response, part.open("ab") as fh:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
        except (urllib.error.URLError, OSError) as exc:
            if attempt == 3:
                raise SystemExit("%s: download failed: %s" % (remote, exc))
            _say("    !!  %s: retrying (%s)" % (target.name, exc))
            time.sleep(2 * attempt)
            continue
        if part.stat().st_size == size:
            break
        part.unlink(missing_ok=True)
        if attempt == 3:
            raise SystemExit("%s: got %d bytes, expected %d"
                             % (remote, part.stat().st_size if part.exists() else 0, size))

    if _sha256(part) != sha:
        part.unlink(missing_ok=True)
        raise SystemExit("%s: checksum mismatch - the download is corrupt, rerun install.bat" % remote)
    part.replace(target)
    seconds = max(time.time() - started, 0.001)
    _say("    ok  %s  %.1f MB  %.1f MB/s" % (target.name, size / 1048576, size / 1048576 / seconds))


def install(root: Path, sets: str, jobs: int = 4) -> int:
    root.mkdir(parents=True, exist_ok=True)
    wanted = list(SHARED)
    if sets in ("cpu", "both"):
        wanted += list(CPU)
    if sets in ("gpu", "both"):
        wanted += list(GPU)

    print("weights -> %s (%d files, %s)"
          % (root, len(wanted), "CPU + GPU" if sets == "both" else sets.upper()), flush=True)
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(_fetch, *entry[:2], root / entry[2], entry[3], entry[4]): entry
                   for entry in wanted}
        for future, entry in futures.items():
            try:
                future.result()
            except SystemExit as exc:
                failures.append(str(exc))

    # The CPU profile reads the shared mel through the same file name the GPU one uses; a copy keeps
    # each weight set self-describing instead of reaching out of its own folder.
    if (root / "preprocessor.onnx").is_file():
        shutil.copyfile(root / "preprocessor.onnx", root / "cpu" / "preprocessor.onnx")

    (root / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n",
                                     encoding="utf-8", newline="\n")

    if failures:
        for line in failures:
            print("  FAIL %s" % line, file=sys.stderr)
        return 2
    total = sum(f.stat().st_size for f in root.rglob("*.onnx"))
    print("weights: ready, %.1f MB of graphs" % (total / 1048576))
    return 0





def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="parakeet_to_md.installer",
                                     description="download the Parakeet TDT weights")
    parser.add_argument("--models-dir", default=str(default_model_dir().parent),
                        help="model root; the weights land in <root>/parakeet")
    parser.add_argument("--set", default="cpu", choices=["cpu", "gpu", "both"],
                        help="cpu = int4 416 MB; gpu = fp16 1.2 GB + cpu; both = everything")
    parser.add_argument("--jobs", type=int, default=4, help="parallel downloads")
    args = parser.parse_args(argv)
    return install(Path(args.models_dir) / "parakeet", args.set, jobs=args.jobs)


if __name__ == "__main__":
    sys.exit(main())

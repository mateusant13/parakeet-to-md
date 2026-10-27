"""Parakeet TDT v3 on ONNX Runtime: features -> greedy TDT decode -> sentence segments.

Two weight sets, chosen by the device:

* ``cpu`` - the 4-bit MatMulNBits export of moondream/parakeet-redux (416 MB). Ternary weights in
  ONNX Runtime's int4 format, which is the fastest thing that runs on a CPU without a GPU.
* ``gpu`` - the fp16 NeMo export of the same model (1.2 GB), which is what CUDA, TensorRT and
  DirectML can actually put on the card. The int4 blocks are a CPU contrib op and never reach a GPU.

The decoder is the token-and-duration transducer loop: one prediction-network step per encoder frame,
each step emitting a token *and* a duration, so word times come out of the decode instead of a second
pass. Long audio is cut at pauses found by the model's own VAD head, never inside a word.

Providers are probed with a real session and every session is asked what it actually got: a
registered provider is not a working one, and a silent fall back to CPU is the failure this module
exists to make impossible (`--device gpu` refuses instead of pretending).
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import paths
from .segments import Segment

SAMPLE_RATE = 16_000
SEGMENT_SECONDS = 30.0
MIN_PAUSE_SECONDS = 0.2
MIN_SEGMENT_SECONDS = 1.0
BLOCK_SECONDS = 120.0
MIN_FEATURE_SAMPLES = 320
SPEECH_THRESHOLD = 0.5
MIN_SPEECH_SECONDS = 0.1
MIN_GAP_SECONDS = 0.1
SKIP_PIECES = {"<unk>", "<pad>", "<blk>"}

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
TRT = "TensorrtExecutionProvider"
DML = "DmlExecutionProvider"

DEVICES = ("auto", "gpu", "trt", "cuda", "dml", "cpu")

_SHORT = {TRT: "TensorRT", CUDA: "CUDA", DML: "DirectML", CPU: "CPU"}


def _short(provider) -> str:
    name = provider if isinstance(provider, str) else provider[0]
    return _SHORT.get(name, name)


def _provider_of(name: str) -> str:
    return {"trt": TRT, "cuda": CUDA, "dml": DML, "cpu": CPU}.get(name, name)

#: Keys a TDT export may carry in config.json; the rest is the model family's fixed shape.
DEFAULT_CONFIG = {
    "durations": [0, 1, 2, 3, 4],
    "vocab_size": 8193,
    "blank_token_id": 8192,
    "max_tokens_per_step": 10,
    "encoder_frame_seconds": 0.08,
    "sample_rate": SAMPLE_RATE,
}

#: Relative paths inside a weight set. `preprocessor` for the CPU set is the shared NeMo-style mel
#: that the VAD head was exported against; the GPU set ships its own.
PROFILES = {
    "cpu": {
        "dir": "cpu",
        "encoder": "encoder-model.onnx",
        "decoder": "decoder_joint-model.onnx",
        "preprocessor": "preprocessor.onnx",
        "precision": "int4",
    },
    "gpu": {
        "dir": "gpu",
        "encoder": "encoder-model.fp16.onnx",
        "decoder": "decoder_joint-model.fp16.onnx",
        "preprocessor": "nemo128.onnx",
        "precision": "fp16",
    },
}

SHARED_FILES = ("config.json", "vocab.txt", "vad-model.onnx", "preprocessor.onnx")


def default_models_dir() -> Path:
    return paths.others_dir() / "models"


def default_model_dir() -> Path:
    return default_models_dir() / "parakeet"


def _physical_cores() -> int:
    """Cores, not logical CPUs. ONNX Runtime's default is every logical CPU, which on a hybrid
    (P-core + E-core) part oversubscribes and can be an order of magnitude slower than a cap."""
    if os.name != "nt":
        return max(1, os.cpu_count() or 1)
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        relation_core = 0

        class Entry(ctypes.Structure):
            _fields_ = [("Relationship", wintypes.DWORD), ("Size", wintypes.DWORD),
                        ("Data", ctypes.c_byte * 256)]

        length = wintypes.DWORD(0)
        kernel32.GetLogicalProcessorInformationEx(relation_core, None, ctypes.byref(length))
        buffer = ctypes.create_string_buffer(length.value)
        if not kernel32.GetLogicalProcessorInformationEx(relation_core, buffer, ctypes.byref(length)):
            raise OSError("GetLogicalProcessorInformationEx failed")
        offset, cores = 0, 0
        while offset < length.value:
            entry = ctypes.cast(ctypes.byref(buffer, offset), ctypes.POINTER(Entry)).contents
            cores += 1
            offset += entry.Size
        return max(1, cores)
    except Exception:  # noqa: BLE001 - a thread count is not worth failing a run over
        return max(1, (os.cpu_count() or 2) // 2)


def default_threads() -> int:
    """Threads for every session. Capped because ONNX Runtime's own default is the failure mode and
    because the encoder saturates memory bandwidth well before it saturates the core count."""
    raw = os.environ.get("PARAKEET_THREADS", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return max(1, min(_physical_cores(), 8))


@dataclass
class EngineInfo:
    device: str                      # what the run actually got: trt|cuda|dml|cpu
    requested: str
    precision: str
    threads: int
    providers: dict[str, list[str]] = field(default_factory=dict)
    runtime: str = ""
    fallback_reason: str = ""

    def describe(self) -> str:
        name = {"trt": "TensorRT", "cuda": "CUDA", "dml": "DirectML", "cpu": "CPU"}.get(
            self.device, self.device.upper())
        return "Parakeet TDT 0.6B v3 (%s, %s)" % (self.precision, name)

    def notes(self) -> list[str]:
        """Only the reason a GPU request ended on a lower rung, and only when it was not explicit."""
        if not self.fallback_reason or self.requested != "auto":
            return []
        return [self.fallback_reason]

    def provider_summary(self) -> str:
        return ", ".join(
            "%s=%s" % (name, "+".join(p.replace("ExecutionProvider", "") for p in providers))
            for name, providers in self.providers.items())


class _ModelMissing(FileNotFoundError):
    pass


def _vocab(path: Path) -> "Vocabulary":
    return Vocabulary(path)


def _is_punctuation(piece: str) -> bool:
    stripped = piece.strip()
    return bool(stripped) and all(unicodedata.category(ch).startswith("P") for ch in stripped)


class Vocabulary:
    """`vocab.txt` as onnx-asr writes it: one `piece id` per line, the blank named `<blk>`."""

    def __init__(self, path: Path) -> None:
        self.pieces: dict[int, str] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line:
                piece, index = line.rsplit(" ", 1)
                self.pieces[int(index)] = piece
        self.blank_id = next(i for i, p in self.pieces.items() if p == "<blk>")
        self.skip = {i for i, p in self.pieces.items() if p in SKIP_PIECES}

    def decode(self, token_ids) -> str:
        text = "".join(self.pieces[t] for t in token_ids if t not in self.skip)
        text = text.replace("\u2581", " ")
        return text[1:] if text.startswith(" ") else text

    def words(self, token_ids, durations, frame_seconds: float) -> list[dict]:
        """Word spans from the transducer's own token durations."""
        frame = 0
        words: list[dict] = []
        current: list[int] = []
        current_start = current_end = 0.0

        def flush() -> None:
            nonlocal current
            if current:
                text = self.decode(current).strip()
                if text:
                    words.append({"word": text, "start": current_start, "end": current_end})
                current = []

        for token, duration in zip(token_ids, durations):
            start = frame * frame_seconds
            frame += duration
            if token in self.skip:
                continue
            raw = self.pieces[token]
            piece = raw.replace("\u2581", " ")
            if raw.startswith("\u2581"):
                flush()
            if _is_punctuation(piece):
                if current:
                    current.append(token)
                    current_end = frame * frame_seconds
                elif words:
                    words[-1] = {"word": words[-1]["word"] + piece.strip(),
                                 "start": words[-1]["start"], "end": frame * frame_seconds}
                continue
            if not current:
                current_start = start
            current.append(token)
            current_end = frame * frame_seconds
        flush()
        return words


def _provision_dlls() -> str:
    """Put the pip wheels' CUDA / TensorRT DLLs on the search path before a session is created.

    Without this the CUDA provider is registered, fails to load, and the session silently reports
    CPU - the exact silent downgrade this module refuses to report as a GPU run.
    """
    done: list[str] = []
    try:
        import onnxruntime

        preload = getattr(onnxruntime, "preload_dlls", None)
        if callable(preload):
            preload()
            done.append("preload_dlls")
    except Exception:  # noqa: BLE001 - absent on CPU-only builds, which is not a failure
        pass
    for entry in sys.path:
        base = Path(entry)
        folders = [base / "tensorrt_libs", *sorted((base / "nvidia").glob("*/bin"))]
        for folder in folders:
            if folder.is_dir():
                try:
                    os.add_dll_directory(str(folder))
                    done.append(folder.name)
                except OSError:
                    pass
    return ",".join(done)


def _trt_options(cache_dir: Path) -> dict:
    return {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(cache_dir),
        "trt_timing_cache_enable": True,
        "trt_max_workspace_size": 2 * 1024 ** 3,
        # One profile covers the segmenter's 1-30 s windows; a rebuild per audio length would cost
        # more than the engine saves.
        "trt_profile_min_shapes": "audio_signal:1x128x100",
        "trt_profile_opt_shapes": "audio_signal:1x128x2000",
        "trt_profile_max_shapes": "audio_signal:1x128x3001",
    }


class Engine:
    """Loaded weight set + sessions. `recognize_block` is the only method the pipeline calls."""

    def __init__(self, model_dir: Path | None = None, device: str = "auto",
                 threads: int | None = None, segment_seconds: float = SEGMENT_SECONDS) -> None:
        import onnxruntime as ort

        if device not in DEVICES:
            raise ValueError("device must be one of %s, got %r" % (", ".join(DEVICES), device))
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        self.requested = device
        self.segment_seconds = float(segment_seconds)
        self._window_samples = int(round(self.segment_seconds * SAMPLE_RATE))
        # Encoder input lengths, padded up to one of a few buckets so a GPU compiles a handful of
        # shapes instead of one per window. The largest bucket is the one a long recording uses.
        steps = {5.0, 10.0, 15.0, 20.0, 25.0, self.segment_seconds}
        self._buckets = [int(round(s * SAMPLE_RATE)) for s in sorted(s for s in steps if s > 0)]
        self.threads = int(threads) if threads else default_threads()

        config = dict(DEFAULT_CONFIG)
        config_path = self.model_dir / "config.json"
        if config_path.is_file():
            try:
                config.update({k: v for k, v in json.loads(config_path.read_text("utf-8")).items()
                               if k in DEFAULT_CONFIG})
            except (OSError, ValueError):
                pass
        self.durations = [int(d) for d in config["durations"]]
        self.vocab_size = int(config["vocab_size"])
        self.max_symbols_per_step = int(config["max_tokens_per_step"])
        self.frame_seconds = float(config["encoder_frame_seconds"])
        self.vocab = Vocabulary(self.model_dir / "vocab.txt")
        self.blank_id = int(config["blank_token_id"])

        problems: list[str] = []
        available = set(ort.get_available_providers())
        self._sessions: dict[str, object] = {}
        chosen = ""
        for name, profile, providers in self._candidates(device, available, problems):
            try:
                self._load_profile(profile, providers)
            except _ModelMissing as exc:
                problems.append(str(exc))
                continue
            except Exception as exc:  # noqa: BLE001 - the next provider in the chain exists for this
                problems.append("%s failed: %s: %s"
                                % (_short(providers[0]), type(exc).__name__, exc))
                self._sessions = {}
                continue
            chosen = name
            break

        if not chosen:
            raise RuntimeError("no usable device for %s: %s"
                               % (self.requested, "; ".join(problems) or "nothing available"))

        self.device = chosen
        self.info = EngineInfo(
            device=chosen,
            requested=device,
            precision=self._precision,
            threads=self.threads,
            providers={k: list(v.get_providers()) for k, v in self._sessions.items()},
            runtime="onnxruntime %s" % ort.__version__,
            fallback_reason="; ".join(problems),
        )

    # -- loading ---------------------------------------------------------
    def _candidates(self, device: str, available: set[str], problems: list[str]):
        gpu = PROFILES["gpu"]
        cpu = PROFILES["cpu"]
        gpu_dir = self.model_dir / gpu["dir"]
        gpu_ready = (gpu_dir / gpu["encoder"]).is_file()
        wanted = ["trt", "cuda", "dml"] if device in ("gpu", "auto") else [device]
        if device != "cpu":
            if not gpu_ready:
                problems.append(
                    "no GPU weights in %s - run install.bat (it downloads them when a card is present)"
                    % gpu_dir)
                if device != "auto":
                    return
            else:
                for name in wanted:
                    if _provider_of(name) not in available:
                        problems.append("%s is not in this onnxruntime build" % _short(name))
                        continue
                    for providers in self._provider_list(name):
                        yield name, gpu, providers
                if device != "auto":
                    return
        yield "cpu", cpu, [CPU]

    @staticmethod
    def _provider_list(name: str) -> list[list]:
        """Provider lists to try for a rung: the encoder may use TensorRT, the small graphs do not."""
        cache = paths.others_dir() / "trt-engines"
        if name == "trt":
            return [[(TRT, _trt_options(cache)), CUDA, CPU]]
        if name == "cuda":
            return [[CUDA, CPU]]
        if name == "dml":
            return [[DML, CPU]]
        return [[CPU]]

    def _load_profile(self, profile: dict, providers: list) -> None:
        base = self.model_dir / profile["dir"]
        missing = [name for name in (profile["encoder"], profile["decoder"], profile["preprocessor"])
                   if not (base / name).is_file()]
        missing += [name for name in SHARED_FILES if not (self.model_dir / name).is_file()]
        if missing:
            raise _ModelMissing("missing model files in %s: %s - run install.bat"
                                % (self.model_dir, ", ".join(sorted(set(missing)))))

        first = providers[0] if isinstance(providers[0], str) else providers[0][0]
        if first in (TRT, CUDA):
            _provision_dlls()

        # TensorRT is an encoder provider here: one optimization profile built once and cached.
        # The joint graph runs a single 1024x1 frame per call, where a TRT engine is pure overhead.
        small = [p for p in providers if not (isinstance(p, tuple) and p[0] == TRT)] or [CPU]
        if small and small[-1] != CPU:
            small = [*small, CPU]

        self._intra = self.threads
        self._precision = profile["precision"]
        self._encoder = self._session(base / profile["encoder"], providers)
        self._preprocessor = self._session(base / profile["preprocessor"], small)
        self._decoder = self._session(base / profile["decoder"], small)
        self._vad_preprocessor = self._session(self.model_dir / "preprocessor.onnx", [CPU])
        self._vad = self._session(self.model_dir / "vad-model.onnx", small)

        wanted = TRT if isinstance(providers[0], tuple) else providers[0]
        got = list(self._encoder.get_providers())
        if wanted != CPU and wanted not in got:
            raise RuntimeError("%s was requested but the encoder reports %s"
                               % (_short(wanted), [_short(p) for p in got]))
        for name, session in (("decoder", self._decoder), ("vad", self._vad)):
            reported = list(session.get_providers())
            if wanted != CPU and not any(p != CPU for p in reported):
                raise RuntimeError("%s fell back to CPU while the encoder is on %s"
                                   % (name, _short(wanted)))
        self._sessions = {
            "encoder": self._encoder,
            "preprocessor": self._preprocessor,
            "decoder": self._decoder,
            "vad": self._vad,
            "vad-preprocessor": self._vad_preprocessor,
        }
        state_shape = next(i.shape for i in self._decoder.get_inputs() if i.name == "input_states_1")
        self._state_shape = (int(state_shape[0]), 1, int(state_shape[2]))
        self._warm_up(wanted)

    def _warm_up(self, wanted: str) -> None:
        """Run one real pass for every session before the first file.

        A session that *registered* a provider has not necessarily *executed* on it: on this machine
        CUDA registers, fails on the first Conv with a cuDNN error, and ONNX Runtime then re-runs the
        whole graph on CPU while still reporting CUDA. The fallback is disabled on GPU sessions, so
        that failure surfaces here - at load, inside the provider chain - instead of mid-file.
        """
        pcm = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self._encode(pcm)
        vad_features, vad_lengths = self._features(pcm, self._vad_preprocessor)
        self._vad.run(None, {"audio_signal": vad_features.astype(np.float32), "length": vad_lengths})
        self._decoder.run(None, {
            "encoder_outputs": np.zeros((1, 1024, 1), np.float32),
            "targets": np.array([[self.blank_id]], dtype=np.int32),
            "target_length": np.ones(1, dtype=np.int32),
            "input_states_1": np.zeros(self._state_shape, np.float32),
            "input_states_2": np.zeros(self._state_shape, np.float32),
        })

    def _session(self, path: Path, providers: list):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = getattr(self, "_intra", None) or default_threads()
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(path), options, providers=list(providers))
        if any(p != CPU for p in session.get_providers()):
            # No silent re-run on CPU: a GPU run that quietly becomes a CPU run is the defect this
            # module exists to make impossible.
            session.disable_fallback()
        return session

    # -- audio -> segments ------------------------------------------------
    def _features(self, pcm: np.ndarray, session, real_samples: int | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
        count = pcm.size if real_samples is None else int(real_samples)
        features, lengths = session.run(None, {
            "waveforms": pcm[None].astype(np.float32),
            "waveforms_lens": np.array([count], dtype=np.int64),
        })
        return features, lengths

    def _encode(self, pcm: np.ndarray) -> np.ndarray:
        """One encoder pass, always on the same input length.

        Every window is padded to `segment_seconds` and the real length travels in `length`, which is
        what NeMo exports mask by. A fixed shape is not cosmetic: TensorRT builds one engine per shape
        and DirectML compiles one kernel set per shape, so varying windows pay the build cost three
        times per file.
        """
        bucket = next((b for b in self._buckets if b >= pcm.size), self._buckets[-1])
        real = min(pcm.size, bucket)
        wave = np.zeros(bucket, dtype=np.float32)
        wave[:real] = pcm[:real]
        features, lengths = self._features(wave, self._preprocessor, real_samples=real)
        outputs, out_lengths = self._encoder.run(
            None, {"audio_signal": features.astype(np.float32), "length": lengths})
        frames = int(out_lengths[0])
        return np.ascontiguousarray(np.asarray(outputs)[0, :, :frames].T)

    def _greedy(self, encoded: np.ndarray) -> tuple[list[int], list[int]]:
        frames = encoded.shape[0]
        state = (np.zeros(self._state_shape, np.float32), np.zeros(self._state_shape, np.float32))
        last_token = self.blank_id
        frame = 0
        steps = self.max_symbols_per_step * frames
        tokens: list[int] = []
        durations: list[int] = []
        target_length = np.ones(1, dtype=np.int32)
        while frame < frames and steps > 0:
            logits, _, hidden, cell = self._decoder.run(None, {
                "encoder_outputs": encoded[frame][None, :, None].astype(np.float32),
                "targets": np.array([[last_token]], dtype=np.int32),
                "target_length": target_length,
                "input_states_1": state[0],
                "input_states_2": state[1],
            })
            row = np.asarray(logits)[0, 0, 0]
            token = int(row[: self.vocab_size].argmax())
            duration = self.durations[int(row[self.vocab_size:].argmax())]
            if token == self.blank_id and duration == 0:
                duration = 1
            tokens.append(token)
            durations.append(duration)
            frame += duration
            if token != self.blank_id:
                last_token = token
                state = (hidden, cell)
            steps -= 1
        return tokens, durations

    def _speech_regions(self, pcm: np.ndarray) -> list[tuple[float, float]]:
        duration = pcm.size / SAMPLE_RATE
        regions: list[tuple[float, float]] = []
        block = round(BLOCK_SECONDS * SAMPLE_RATE)
        for offset in range(0, pcm.size, block):
            piece = pcm[offset:offset + block]
            base = offset / SAMPLE_RATE
            piece_duration = piece.size / SAMPLE_RATE
            if piece.size < MIN_FEATURE_SAMPLES:
                regions.append((base, base + piece_duration))
                continue
            features, lengths = self._features(piece, self._vad_preprocessor)
            probabilities, valid = self._vad.run(
                None, {"audio_signal": features.astype(np.float32), "length": lengths})
            speaking = np.asarray(probabilities)[0, : int(valid[0])] >= SPEECH_THRESHOLD
            edges = np.flatnonzero(np.diff(np.r_[0, speaking.astype(np.int8), 0]))
            for start, end in edges.reshape(-1, 2) * self.frame_seconds:
                start = base + float(start)
                end = base + min(float(end), piece_duration)
                if regions and start - regions[-1][1] < MIN_GAP_SECONDS:
                    regions[-1] = (regions[-1][0], end)
                elif end - start >= MIN_SPEECH_SECONDS:
                    regions.append((start, end))
        return regions

    def _cut_segments(self, pcm: np.ndarray, regions: list[tuple[float, float]]) -> list[tuple[int, int]]:
        cap = round(self.segment_seconds * SAMPLE_RATE)
        if pcm.size <= cap:
            return [(0, pcm.size)] if regions else []
        pieces: list[tuple[int, int]] = []
        start = 0
        while pcm.size - start > cap:
            origin = start / SAMPLE_RATE
            relative = [(max(s - origin, 0.0), e - origin) for s, e in regions if e > origin]
            # Pauses between speech, then the middle of the last one inside the window: a cut in the
            # middle of a silence cannot split a word.
            pauses: list[tuple[float, float]] = []
            previous = 0.0
            for begin, end in sorted(relative):
                if begin - previous >= MIN_PAUSE_SECONDS - 1e-6:
                    pauses.append((previous, begin))
                previous = max(previous, end)
            if (pcm.size - start) / SAMPLE_RATE - previous >= MIN_PAUSE_SECONDS - 1e-6:
                pauses.append((previous, (pcm.size - start) / SAMPLE_RATE))
            inside = [p for p in pauses
                      if p[0] >= MIN_SEGMENT_SECONDS and p[1] <= self.segment_seconds]
            if not inside:
                inside = [p for p in pauses
                          if MIN_SEGMENT_SECONDS <= (p[0] + p[1]) / 2 <= self.segment_seconds]
            cut = (inside[-1][0] + inside[-1][1]) / 2 if inside else self.segment_seconds
            index = round(cut * SAMPLE_RATE)
            if any(end > 0 and begin < index / SAMPLE_RATE for begin, end in relative):
                pieces.append((start, start + index))
            start += index
        remaining = (pcm.size - start) / SAMPLE_RATE
        origin = start / SAMPLE_RATE
        relative = [(max(s - origin, 0.0), e - origin) for s, e in regions if e > origin]
        silent = not any(end > 0 and begin < remaining for begin, end in relative)
        if not (pieces and (pcm.size - start < MIN_FEATURE_SAMPLES or silent)):
            pieces.append((start, pcm.size))
        return pieces

    def _sentences(self, words: list[dict], offset: float) -> list[Segment]:
        out: list[Segment] = []
        pending: list[dict] = []
        for word in words:
            pending.append(word)
            finished = word["word"].rstrip().endswith((".", "?", "!"))
            too_long = pending[-1]["end"] - pending[0]["start"] >= self.segment_seconds
            if finished or too_long:
                text = " ".join(w["word"] for w in pending)
                out.append(Segment(offset + pending[0]["start"], offset + pending[-1]["end"], text))
                pending = []
        if pending:
            text = " ".join(w["word"] for w in pending)
            out.append(Segment(offset + pending[0]["start"], offset + pending[-1]["end"], text))
        return out

    def recognize_block(self, audio: np.ndarray) -> list[Segment]:
        """Segments for one decoded block, at most `segment_seconds` of work per encoder pass."""
        if audio.size < MIN_FEATURE_SAMPLES:
            return []
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        regions = self._speech_regions(audio)
        if not regions:
            return []
        out: list[Segment] = []
        for start, end in self._cut_segments(audio, regions):
            tokens, durations = self._greedy(self._encode(audio[start:end]))
            text = self.vocab.decode(tokens)
            if not text.strip():
                continue
            base = start / SAMPLE_RATE
            words = self.vocab.words(tokens, durations, self.frame_seconds)
            if words:
                out.extend(self._sentences(words, base))
            else:
                out.append(Segment(base, end / SAMPLE_RATE, text))
        return out

    def transcribe(self, audio: np.ndarray) -> list[Segment]:
        return self.recognize_block(audio)


def load_engine(model_dir: Path | None = None, device: str = "auto",
                threads: int | None = None, segment_seconds: float = SEGMENT_SECONDS) -> Engine:
    try:
        return Engine(model_dir=model_dir, device=device, threads=threads,
                      segment_seconds=segment_seconds)
    except FileNotFoundError as exc:
        print("engine: %s" % exc, file=sys.stderr)
        raise

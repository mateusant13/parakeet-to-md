"""Which provider did we ACTUALLY get?

The failure these tests exist for: ONNX Runtime accepts a provider, fails to load it or fails on the
first node, and quietly re-runs the graph on the CPU while the session still reports the GPU. A tool
that believes its own request reports "GPU" while burning CPU minutes.

So the device decision is exercised against a fake runtime, without a model: what the sessions report
decides the device, an explicit request that cannot be honoured raises, and `auto` walks the chain and
keeps the reason it walked.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from parakeet_to_md import asr as asr_mod

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
TRT = "TensorrtExecutionProvider"
DML = "DmlExecutionProvider"


def names(providers):
    return [p if isinstance(p, str) else p[0] for p in providers]


class FakeSession:
    def __init__(self, providers, behaviour):
        self._requested = names(providers)
        self._providers = list(behaviour(self._requested))

    def get_providers(self):
        return list(self._providers)

    def disable_fallback(self):
        pass

    def get_inputs(self):
        return [types.SimpleNamespace(name="input_states_1", shape=(2, "N", 640)),
                types.SimpleNamespace(name="input_states_2", shape=(2, "N", 640))]

    def end_profiling(self):
        return ""

    def run(self, output_names, feed):
        if "waveforms" in feed:
            samples = int(np.asarray(feed["waveforms_lens"])[0])
            frames = samples // 160 + 1
            return [np.zeros((1, 128, frames), np.float32), np.array([frames], np.int64)]
        if "audio_signal" in feed:
            frames = int(feed["audio_signal"].shape[2])
            out = max(1, frames // 8)
            return [np.zeros((1, 1024, out), np.float32), np.array([out], np.int64)]
        if "encoder_outputs" in feed:
            return [np.zeros((1, 1, 1, 8198), np.float32), None,
                    np.zeros((2, 1, 640), np.float32), np.zeros((2, 1, 640), np.float32)]
        frames = int(feed["audio_signal"].shape[2]) // 8
        return [np.zeros((1, max(1, frames)), np.float32), np.array([max(1, frames)], np.int64)]


def fake_runtime(monkeypatch, behaviour, calls=None):
    module = types.ModuleType("onnxruntime")
    module.__version__ = "0.0-test"

    class SessionOptions:
        def __init__(self):
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0
            self.log_severity_level = 0
            self.graph_optimization_level = 0

    class GraphOptimizationLevel:
        ORT_ENABLE_ALL = 99

    class InferenceSession(FakeSession):
        def __init__(self, path, options=None, providers=None):
            if calls is not None:
                calls.append(names(providers or []))
            super().__init__(providers or [], behaviour)

    module.SessionOptions = SessionOptions
    module.GraphOptimizationLevel = GraphOptimizationLevel
    module.InferenceSession = InferenceSession
    module.get_available_providers = lambda: [TRT, CUDA, DML, CPU]
    monkeypatch.setitem(sys.modules, "onnxruntime", module)


def fake_weights(tmp_path):
    root = tmp_path / "parakeet"
    root.mkdir()
    (root / "config.json").write_text('{"durations": [0, 1, 2, 3, 4]}', encoding="utf-8")
    (root / "vocab.txt").write_text("a 0\n<blk> 8192\n", encoding="utf-8")
    for name in ("vad-model.onnx", "preprocessor.onnx"):
        (root / name).write_bytes(b"x")
    for set_name, files in (("cpu", ("encoder-model.onnx", "decoder_joint-model.onnx",
                                     "preprocessor.onnx")),
                            ("gpu", ("encoder-model.fp16.onnx", "decoder_joint-model.fp16.onnx",
                                     "nemo128.onnx"))):
        folder = root / set_name
        folder.mkdir()
        for name in files:
            (folder / name).write_bytes(b"x")
    return root


def build(tmp_path, monkeypatch, *, device, behaviour, calls=None):
    fake_runtime(monkeypatch, behaviour, calls)
    return asr_mod.load_engine(model_dir=fake_weights(tmp_path), device=device)


def always(providers):
    return providers


def test_every_device_name_the_cli_offers_exists():
    assert set(asr_mod.DEVICES) == {"auto", "gpu", "trt", "cuda", "dml", "cpu"}


def test_tensorrt_is_first_and_asks_for_fp16_and_a_cached_engine():
    providers = asr_mod.Engine._provider_list("trt")[0]
    assert names(providers)[0] == TRT
    options = providers[0][1]
    assert options["trt_fp16_enable"] is True
    assert options["trt_engine_cache_enable"] is True
    assert options["trt_engine_cache_path"].endswith("trt-engines")


def test_auto_asks_for_tensorrt_first(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    engine = build(tmp_path, monkeypatch, device="auto", behaviour=always, calls=calls)
    assert engine.info.device == "trt"
    assert calls[0][0] == TRT


def test_a_reported_cpu_under_tensorrt_moves_to_the_next_rung_and_says_why(tmp_path, monkeypatch):
    def behaviour(providers):
        return [CUDA, CPU] if CUDA in providers else [CPU]

    engine = build(tmp_path, monkeypatch, device="auto", behaviour=behaviour)
    assert engine.info.device == "cuda"
    assert "tensorrt" in engine.info.fallback_reason.lower()
    assert engine.info.precision == "fp16"


def test_an_explicit_request_that_cannot_load_raises_instead_of_downgrading(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError) as excinfo:
        build(tmp_path, monkeypatch, device="trt", behaviour=lambda providers: [CPU])
    assert "TensorRT" in str(excinfo.value)


def test_an_explicit_cuda_request_that_cannot_load_raises(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError) as excinfo:
        build(tmp_path, monkeypatch, device="cuda", behaviour=lambda providers: [CPU])
    assert "CUDA" in str(excinfo.value)


def test_directml_is_a_supported_rung_for_amd_and_intel(tmp_path, monkeypatch):
    engine = build(tmp_path, monkeypatch, device="dml",
                   behaviour=lambda providers: [DML, CPU] if DML in providers else [CPU])
    assert engine.info.device == "dml"
    assert "DirectML" in engine.info.describe()


def test_no_gpu_loads_lands_on_cpu_with_the_reason_kept(tmp_path, monkeypatch):
    engine = build(tmp_path, monkeypatch, device="auto", behaviour=lambda providers: [CPU])
    assert engine.info.device == "cpu"
    assert engine.info.precision == "int4"
    assert "TensorRT" in engine.info.fallback_reason
    notes = engine.info.notes()
    assert notes and "TensorRT" in notes[0]


def test_cpu_never_asks_for_a_gpu_provider(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    engine = build(tmp_path, monkeypatch, device="cpu", behaviour=always, calls=calls)
    assert engine.info.device == "cpu"
    assert all(call == [CPU] for call in calls)


def test_a_gpu_request_with_no_gpu_weights_names_the_installer(tmp_path, monkeypatch):
    root = fake_weights(tmp_path)
    for path in (root / "gpu").iterdir():
        path.unlink()
    fake_runtime(monkeypatch, always)
    with pytest.raises(RuntimeError) as excinfo:
        asr_mod.load_engine(model_dir=root, device="gpu")
    assert "install.bat" in str(excinfo.value)


def test_a_working_gpu_run_reports_no_fallback_reason(tmp_path, monkeypatch):
    engine = build(tmp_path, monkeypatch, device="auto", behaviour=always)
    assert engine.info.fallback_reason == ""
    assert engine.info.notes() == []
    assert engine.info.describe() == "Parakeet TDT 0.6B v3 (fp16, TensorRT)"


def test_threads_are_capped_and_overridable_by_the_environment(monkeypatch):
    monkeypatch.setenv("PARAKEET_THREADS", "3")
    assert asr_mod.default_threads() == 3
    monkeypatch.setenv("PARAKEET_THREADS", "junk")
    assert 1 <= asr_mod.default_threads() <= 8
    monkeypatch.delenv("PARAKEET_THREADS")
    assert 1 <= asr_mod.default_threads() <= 8

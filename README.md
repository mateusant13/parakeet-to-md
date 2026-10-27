# parakeet-to-md

Turn a folder of videos and audio into readable, timestamped markdown — locally, offline, with
[Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

Drop files in `input\`, double-click `run.bat`, read the transcript in `output\`. Runs on an NVIDIA
GPU, an AMD or Intel GPU, or the CPU, and needs no account, no API key and no internet after install.

```
# lecture-03

**[00:00:04]** Today we finish the section on transducers.
**[00:00:11]** The decoder emits one token and one duration per frame.
**[00:00:24]** That is why word timings come for free.
```

## Install

Windows 10/11, no administrator rights, nothing added to `PATH`.

1. Download or clone this repository.
2. Double-click **`install.bat`**.

It fetches a portable Python if the machine has none, creates `others\.venv`, installs the runtime,
and downloads the weights:

| Machine | Runtime installed | Weights |
| --- | --- | --- |
| NVIDIA (driver present) | `onnxruntime-gpu` + TensorRT (DLLs from pip wheels) | fp16 1.2 GB + int4 420 MB |
| AMD / Intel | `onnxruntime-directml` | fp16 1.2 GB + int4 420 MB |
| CPU only | `onnxruntime` | int4 420 MB |

`install.bat --cpu` forces the CPU-only install, `install.bat --gpu` forces the GPU stack.
Every download is pinned by size and SHA-256.

## Use

Put files in `input\` and double-click **`run.bat`**. Each source is transcribed to `output\<name>.md`
and then moved to `archive\`. A file that fails stays in `input\` and the reason is printed.

A run resumes: the work is checkpointed per decoded block in `output\.partial\`, so a long file that
was interrupted continues where it stopped.

## Command line

```
others\.venv\Scripts\python.exe -m parakeet_to_md [options]

  --input DIR        folder to read          (default: input\)
  --output DIR       folder to write         (default: output\)
  --archive DIR      folder to move sources  (default: archive\)
  --only NAME        one file, by name or stem
  --device DEVICE    auto (default) | gpu | trt | cuda | dml | cpu
  --threads N        intra-op threads per session (default: physical cores, max 8)
  --block-seconds N  decode block size       (default: 600)
  --with-parts       add a "## Part N" heading every 10 minutes
  --no-archive       leave sources in input\
  --keep-partial     keep the checkpoint files after success
  --selftest         check the install without touching any file
```

`--device auto` walks TensorRT → CUDA → DirectML → CPU and reports the provider that actually ran.
An explicit `--device trt|cuda|dml|cpu` never downgrades: if the session does not come up on that
provider, the run stops with the reason instead of quietly using the CPU.

## Output

One sentence per timestamped line, a blank line between paragraphs, no front matter. Sentence
boundaries come from the model's own token durations, and long recordings are cut at pauses (never
inside a word) before each encoder pass, so the timestamps stay tight over hours of audio.

## Speed

Parakeet TDT is a 0.6B-parameter model. The encoder is ~95% of the work, so it is the only graph that
needs the fastest available provider; the decoder runs one tiny step per 80 ms frame.

Published numbers for this model family, on other machines:

| Runtime | Hardware | Speed |
| --- | --- | --- |
| int4 (this project's CPU set) | Apple M2 Max CPU | ~25–50x realtime ([redux export](https://huggingface.co/eschmidbauer/parakeet-redux-onnx)) |
| int4, Photon runtime | 8-core EPYC | ~113x realtime ([moondream](https://huggingface.co/moondream/parakeet-redux)) |
| int8 CPU | desktop CPU | ~18–30x realtime ([onnx-asr](https://github.com/istupakov/onnx-asr)) |
| CUDA | Tesla T4 | ~58x realtime |
| TensorRT fp16 | Tesla T4 / RTX-class | ~237x / ~329x realtime ([onnx-asr benchmarks](https://github.com/istupakov/onnx-asr#benchmarks)) |

Your own number is printed by the tool — every finished file ends with `x realtime` on the same line
as its transcript path, and `--selftest` prints the provider and the model load time.

Measured on real transcripts, the things that move the number most, in order: the provider
(TensorRT ≫ CUDA ≫ DirectML ≫ CPU), the thread count (ONNX Runtime defaults to every logical CPU,
which oversubscribes hybrid P-core/E-core parts — this project caps it at the physical core count,
max 8), and the decoder step count (this project's decode does one joint-network call per encoder
frame, which is the minimum the architecture allows).

## Models

| Set | Source | Files | Size |
| --- | --- | --- | --- |
| `cpu/` | [eschmidbauer/parakeet-redux-onnx](https://huggingface.co/eschmidbauer/parakeet-redux-onnx) | 4-bit `MatMulNBits` export of [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux) (ternary weights, exact) | 416 MB |
| `gpu/` | [grikdotnet/parakeet-tdt-0.6b-fp16](https://huggingface.co/grikdotnet/parakeet-tdt-0.6b-fp16) | NeMo fp16 export of the same model | 1.2 GB |
| shared | `eschmidbauer/parakeet-redux-onnx` | mel preprocessor, VAD head, tokenizer | 20 MB |

`MatMulNBits` is an ONNX Runtime CPU contrib op: it does not run on CUDA, TensorRT or DirectML. That is
why the GPU path has its own fp16 weight set. Both sets are the same model (same tokenizer, same
outputs), so switching `--device` changes speed, not the transcript.

Licences: model weights CC-BY-4.0 (NVIDIA), code MIT.

## Requirements

Python 3.10+ (install.bat provisions 3.12 if needed), `numpy`, `onnxruntime` — or
`onnxruntime-gpu` / `onnxruntime-directml` instead. `ffmpeg` is bundled by `install.bat`; an existing
one on `PATH` is used if present.

## Development

```
cd others
.venv\Scripts\python.exe -m pytest app\tests
```

`--selftest` is the install check: it loads the engine, prints the providers every session actually
reports, and transcribes 3 seconds of silence expecting no segments.

## Licence

MIT. Model weights keep their own licence (CC-BY-4.0, NVIDIA).

# matalu

Local, real-time speech-to-text **service** for Apple Silicon. Captures your
microphone, transcribes on-device with NVIDIA **Parakeet** via Apple's **MLX**
framework (GPU/ANE-accelerated), and streams `partial`/`final` transcripts over
a local **WebSocket**.

Rust owns capture, resampling, transport, and the WebSocket server; the MLX model
(Python-only) runs in a supervised **sidecar** process. Punctuation and casing
come from the model itself. Text passes through a pluggable
[`Corrector`](src/corrector.rs) seam — a no-op today, the drop-in point for a
context-aware grammar/formatting LLM later.

## Architecture

```
 cpal mic thread          Rust process                    Python sidecar
 ───────────────         ──────────────                  ────────────────
 device callback ─f32─▶  resample→16k mono ─raw f32──▶   parakeet-mlx streaming
 (48 kHz native)         (own thread)      (child stdin)  + RMS VAD segmentation
                                                          │  JSON lines (stdout)
                          TranscriptEvent ◀───────────────┘
                          (corrector) → tokio broadcast ─▶ WebSocket clients
```

- **Audio** (`src/audio.rs`): cpal capture, downmix to mono, streaming resample
  to 16 kHz, on its own thread (a mic-permission stall never blocks the server).
- **Sidecar** (`src/sidecar.rs` + `sidecar/matalu_sidecar.py`): Rust spawns the
  Python process and bridges two pipes — audio in (raw f32), JSON events out. The
  Python side runs `parakeet-mlx` streaming and an RMS silence VAD, emitting
  `partial` frames live and a `final` after ~700 ms of quiet.
- **Server** (`src/server.rs`): `axum` WebSocket at `/ws`, `broadcast` fan-out;
  `/health` for liveness.

## Setup

Requires Rust, Python 3, and macOS (Apple Silicon).

```bash
# 1. Python deps (parakeet-mlx pulls in mlx). A venv is recommended:
python3 -m pip install parakeet-mlx

# 2. Build
cargo build --release
```

The MLX model (`mlx-community/parakeet-tdt-0.6b-v2`, ~1.2 GB) is downloaded and
cached automatically by the sidecar on first run.

## Run

```bash
cargo run --release
# → starting parakeet-mlx sidecar ...
# → WebSocket server listening (connect to ws://127.0.0.1:8765/ws)
```

Grant **microphone permission** to your terminal on first run
(System Settings → Privacy & Security → Microphone).

Consume the stream:

```bash
# with websocat (brew install websocat):
websocat ws://127.0.0.1:8765/ws

# or Python (pip install websockets):
python3 -c '
import asyncio, websockets
async def main():
    async with websockets.connect("ws://127.0.0.1:8765/ws") as ws:
        while True:
            print(await ws.recv())
asyncio.run(main())'
```

Frames:

```json
{ "type": "partial", "text": "hello wor",     "ts_ms": 1200 }
{ "type": "final",   "text": "Hello, world.", "ts_ms": 1600 }
```

### Configuration (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `MATALU_BIND` | `127.0.0.1:8765` | Server bind address |
| `MATALU_PYTHON` | `python3` | Interpreter that runs the sidecar |
| `MATALU_SIDECAR` | `sidecar/matalu_sidecar.py` | Sidecar script path |
| `MATALU_MLX_MODEL` | `mlx-community/parakeet-tdt-0.6b-v2` | HF id or local path |
| `MATALU_SILENCE_MS` | `700` | Silence (ms) that finalizes an utterance |
| `MATALU_VAD_RMS` | `0.010` | RMS threshold for speech vs silence |
| `MATALU_DEMO` | _(unset)_ | If set, emit synthetic events (no mic/model needed) |

## Testing

```bash
# WebSocket protocol end-to-end, synthetic events (no mic/model):
MATALU_DEMO=1 cargo run

# Sidecar streaming smoke test on a 16 kHz WAV:
python3 sidecar/test_stream.py some_audio_16k.wav

# Resampler unit tests:
cargo test
```

## Roadmap

- **Grammar/formatting LLM**: implement `Corrector` (e.g. `OllamaCorrector` →
  `localhost:11434`, or in-process `llama.cpp`) for context-aware cleanup on
  finalized utterances or a rolling window.
- **Latency tuning**: adjust the streaming `context_size` / chunk granularity in
  the sidecar for lower latency vs accuracy.

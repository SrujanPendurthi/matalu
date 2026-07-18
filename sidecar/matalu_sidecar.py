"""matalu ASR sidecar — parakeet-mlx streaming engine.

Protocol (kept deliberately dumb so the Rust side owns capture + transport):
  stdin  : a continuous stream of raw little-endian float32 samples,
           mono, 16 kHz. (Rust captures the mic, resamples, and pipes it here.)
  stdout : newline-delimited JSON events, one per line:
             {"type":"partial","text":"...","ts_ms":123}
             {"type":"final",  "text":"...","ts_ms":456}
  stderr : human-readable logs (Rust forwards these to its logger).

Utterance segmentation is done here with a simple RMS silence VAD: `partial`
frames grow live as audio streams; a `final` is emitted after SILENCE_MS of
quiet, and the streaming context is reset for the next utterance.
"""
import json
import os
import sys
import numpy as np
import mlx.core as mx

from parakeet_mlx import from_pretrained

SR = 16000

# Model source resolution prefers a local/bundled copy over the Hub, so a normal
# run never touches the network. A parakeet-mlx model dir is just a folder with
# `config.json` + the `*.safetensors` weights; `from_pretrained` accepts either
# such a path or a Hub repo ID.
REPO_ID = "mlx-community/parakeet-tdt-0.6b-v2"
_SIDECAR_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLED_DIR = os.path.join(_SIDECAR_DIR, "models", "parakeet-tdt-0.6b-v2")


def resolve_model() -> str:
    """Pick the model source, preferring a local/bundled dir over the Hub.

    Priority:
      1. MATALU_MLX_MODEL — explicit override (a local path or a Hub repo ID).
      2. A bundled model dir next to the sidecar (fully offline; the norm).
      3. The Hub repo ID — downloads + caches on first run (the fallback).
    """
    override = os.environ.get("MATALU_MLX_MODEL")
    if override:
        return override
    if os.path.isdir(_BUNDLED_DIR):
        return _BUNDLED_DIR
    return REPO_ID


SILENCE_MS = int(os.environ.get("MATALU_SILENCE_MS", "700"))
VAD_RMS = float(os.environ.get("MATALU_VAD_RMS", "0.010"))
CHUNK = SR // 2  # 0.5 s processing granularity

# Streaming attention context (left, right) in encoder frames. This is the main
# speed/accuracy knob: the default (256,256) decodes BELOW real time (~0.6x RTF)
# on an M4, while (64,64) runs ~1.4x RTF with no measurable accuracy loss on our
# tests. Larger = more accurate but slower; too small (e.g. 32,16) collapses.
_ctx = os.environ.get("MATALU_MLX_CONTEXT", "64,64").split(",")
CONTEXT_SIZE = (int(_ctx[0]), int(_ctx[1]))


def log(msg: str) -> None:
    sys.stderr.write(f"[sidecar] {msg}\n")
    sys.stderr.flush()


def emit(kind: str, text: str, ts_ms: int) -> None:
    sys.stdout.write(json.dumps({"type": kind, "text": text, "ts_ms": ts_ms}) + "\n")
    sys.stdout.flush()


def main() -> None:
    model_src = resolve_model()
    kind = "local dir" if os.path.isdir(model_src) else "Hub repo"
    log(f"loading {model_src} ({kind}) ...")
    model = from_pretrained(model_src)
    if model.preprocessor_config.sample_rate != SR:
        log(f"WARNING: model sample_rate={model.preprocessor_config.sample_rate}, expected {SR}")

    # Warm up: run one throwaway inference so the Metal kernels compile now,
    # while the mic is still gated, rather than stalling the first real words.
    with model.transcribe_stream(context_size=CONTEXT_SIZE) as warm:
        warm.add_audio(mx.array(np.zeros(SR, dtype=np.float32)))
        _ = warm.result.text
    log("model loaded and warmed; ready")

    # Tell the Rust supervisor it's safe to start mic capture (avoids the
    # startup backpressure that otherwise drops audio during model load).
    sys.stdout.write(json.dumps({"type": "ready"}) + "\n")
    sys.stdout.flush()

    stdin = sys.stdin.buffer
    processed = 0        # total samples consumed (for timestamps)
    silence_ms = 0
    in_utterance = False

    ctx = model.transcribe_stream(context_size=CONTEXT_SIZE)
    tx = ctx.__enter__()

    try:
        while True:
            # Block until a full 0.5 s chunk arrives (or EOF/partial at shutdown).
            data = stdin.read(CHUNK * 4)
            if not data:
                break
            n = len(data) // 4
            if n == 0:
                continue
            samples = np.frombuffer(data[: n * 4], dtype=np.float32)
            processed += n
            ts_ms = processed * 1000 // SR

            rms = float(np.sqrt(np.mean(np.square(samples)))) if n else 0.0
            # Always feed audio so the streaming cache stays continuous and
            # trailing words flush during the silence tail.
            tx.add_audio(mx.array(samples))

            if rms > VAD_RMS:
                in_utterance = True
                silence_ms = 0
                emit("partial", tx.result.text, ts_ms)
            elif in_utterance:
                silence_ms += n * 1000 // SR
                emit("partial", tx.result.text, ts_ms)
                if silence_ms >= SILENCE_MS:
                    emit("final", tx.result.text, ts_ms)
                    # Reset streaming state for the next utterance.
                    ctx.__exit__(None, None, None)
                    ctx = model.transcribe_stream(context_size=CONTEXT_SIZE)
                    tx = ctx.__enter__()
                    in_utterance = False
                    silence_ms = 0
    finally:
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass
    log("stdin closed; exiting")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, BrokenPipeError):
        # Parent shutting down (Ctrl-C / closed pipe) — exit quietly.
        pass

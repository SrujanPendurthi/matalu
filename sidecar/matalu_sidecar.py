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
# Force a `final` + context reset after this much *continuous* speech even
# without a silence gap. Dictation rarely hits it; meeting audio (monologues,
# overlapping speakers) can run for minutes with no 700 ms gap, which would
# otherwise grow one endless partial and let the streaming context balloon.
MAX_UTTERANCE_MS = int(os.environ.get("MATALU_MAX_UTTERANCE_MS", "12000"))
# Reset the streaming context after this much *continuous idle silence* (no
# active utterance). In meeting mode the gate feeds real audio non-stop, so we
# keep calling add_audio through quiet stretches; without this the context (and
# MLX memory) grows unbounded on silence, since a `final` — which is what resets
# it — only fires after speech. Bounds the silence pre-roll the context holds.
IDLE_RESET_MS = int(os.environ.get("MATALU_IDLE_RESET_MS", "3000"))
CHUNK = SR // 2  # 0.5 s processing granularity

# Streaming attention context (left, right) in encoder frames. This is the main
# speed/accuracy knob. Measured on this hardware (5.7s utterance):
#   (64,64)   RTF 0.9-1.4x, but DROPS the utterance start (too little warmup ctx)
#   (128,128) RTF ~1.1x, start recovered — real-time and clearly better (default)
#   (256,256) same accuracy as 128 but RTF ~0.7x (below real time — it lags)
# Larger = more accurate but slower; too small (e.g. 32,16) collapses.
_ctx = os.environ.get("MATALU_MLX_CONTEXT", "128,128").split(",")
CONTEXT_SIZE = (int(_ctx[0]), int(_ctx[1]))


def log(msg: str) -> None:
    sys.stderr.write(f"[sidecar] {msg}\n")
    sys.stderr.flush()


def emit(kind: str, text: str, ts_ms: int) -> None:
    sys.stdout.write(json.dumps({"type": kind, "text": text, "ts_ms": ts_ms}) + "\n")
    sys.stdout.flush()


def new_stream(model):
    """Open a fresh streaming context and return (ctx, tx). Callers exit the old
    ctx first. Also release MLX's pooled buffers so the reset actually frees
    memory (weights stay resident; macOS can't reclaim the pool itself)."""
    mx.clear_cache()
    ctx = model.transcribe_stream(context_size=CONTEXT_SIZE)
    return ctx, ctx.__enter__()


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

    # Diagnostic: when MATALU_DUMP_WAV is set, write exactly the audio this
    # process receives (post-resample, post-gate) to a 16 kHz mono WAV, so a bad
    # live transcript can be replayed/analyzed offline — isolates a mic/capture
    # problem (the WAV itself sounds wrong) from a streaming one (WAV is fine).
    dump = None
    dump_path = os.environ.get("MATALU_DUMP_WAV")
    if dump_path:
        import wave
        dump = wave.open(dump_path, "wb")
        dump.setnchannels(1)
        dump.setsampwidth(2)
        dump.setframerate(SR)
        log(f"dumping received audio to {dump_path}")

    stdin = sys.stdin.buffer
    processed = 0        # total samples consumed (for timestamps)
    silence_ms = 0
    utterance_ms = 0     # continuous-speech duration since the last final
    idle_ms = 0          # continuous idle-silence duration (no active utterance)
    in_utterance = False

    ctx, tx = new_stream(model)

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
            step_ms = n * 1000 // SR

            if dump is not None:
                dump.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())

            rms = float(np.sqrt(np.mean(np.square(samples)))) if n else 0.0
            # Always feed audio so the streaming cache stays continuous and
            # trailing words flush during the silence tail.
            tx.add_audio(mx.array(samples))

            if rms > VAD_RMS:
                in_utterance = True
                silence_ms = 0
                idle_ms = 0
                utterance_ms += step_ms
            elif in_utterance:
                silence_ms += step_ms
                utterance_ms += step_ms

            if in_utterance:
                emit("partial", tx.result.text, ts_ms)
                # Commit + reset on a silence gap (utterance ended) OR the
                # max-utterance cap (long continuous speech that never pauses,
                # e.g. a meeting monologue) so the transcript keeps flowing and
                # the streaming context stays bounded. new_stream() also clears
                # MLX's pooled buffers so the reset frees memory.
                if silence_ms >= SILENCE_MS or utterance_ms >= MAX_UTTERANCE_MS:
                    emit("final", tx.result.text, ts_ms)
                    ctx.__exit__(None, None, None)
                    ctx, tx = new_stream(model)
                    in_utterance = False
                    silence_ms = 0
                    utterance_ms = 0
            else:
                # Idle silence with no active utterance. We keep feeding real
                # audio (meeting mode never gates to NONE), so periodically drop
                # the accumulated silence context to keep MLX memory bounded.
                idle_ms += step_ms
                if idle_ms >= IDLE_RESET_MS:
                    ctx.__exit__(None, None, None)
                    ctx, tx = new_stream(model)
                    idle_ms = 0
    finally:
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass
        if dump is not None:
            dump.close()
    log("stdin closed; exiting")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, BrokenPipeError):
        # Parent shutting down (Ctrl-C / closed pipe) — exit quietly.
        pass

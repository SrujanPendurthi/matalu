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

**Partials stream; finals are decoded at full context.** Streaming attention is
a large accuracy tax — measured 6.12% WER streaming vs 1.49% full-context on
LibriSpeech test-clean (published: 1.69%). The app buffers a whole dictation and
pastes once, so partials are only cosmetic (the pill and transcript window); the
text that actually gets injected comes from the `final`. So each utterance's raw
samples are buffered and re-decoded in one pass when the VAD ends it. See
`finalize()`.
"""
import json
import os
import sys
import numpy as np
import mlx.core as mx

from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import get_logmel

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

# Weight quantization for the ASR model. 8 is free (measured: no WER change,
# -383 MB, 1.37x faster); 0 disables and keeps bf16. See quantize_model().
ASR_BITS = int(os.environ.get("MATALU_ASR_BITS", "8"))

# Decode each `final` from the buffered utterance at full context instead of
# taking the streaming result. Measured 4.2x more accurate AND 3.1x faster.
# `0` restores the streaming-result behavior. See finalize().
FULL_CONTEXT_FINAL = os.environ.get("MATALU_FULL_CONTEXT_FINAL", "1") != "0"


def log(msg: str) -> None:
    sys.stderr.write(f"[sidecar] {msg}\n")
    sys.stderr.flush()


def emit(kind: str, text: str, ts_ms: int) -> None:
    sys.stdout.write(json.dumps({"type": kind, "text": text, "ts_ms": ts_ms}) + "\n")
    sys.stdout.flush()


def finalize(model, buffered, streamed_text: str) -> str:
    """Text for a `final`: one full-context decode over the buffered utterance.

    Streaming attention costs ~4x WER (6.12% vs 1.49% on LibriSpeech test-clean,
    published 1.69%), and the app only injects the `final` — partials are
    cosmetic. So the utterance's raw samples are re-decoded in a single pass,
    the same path `model.transcribe()` takes internally.

    Best-effort: any failure returns the streaming result, which is what this
    used to emit. A worse transcript beats a lost utterance.
    """
    if not FULL_CONTEXT_FINAL or not buffered:
        return streamed_text
    audio = np.concatenate(buffered)
    # get_logmel needs at least one hop, and a sub-hop clip would raise rather
    # than return empty. Nothing useful to decode there anyway.
    if len(audio) < model.preprocessor_config.hop_length:
        return streamed_text
    try:
        mel = get_logmel(mx.array(audio), model.preprocessor_config)
        return model.generate(mel)[0].text
    except Exception as e:  # noqa: BLE001 — never lose the utterance
        log(f"WARNING: full-context decode failed ({e!r}); using streaming result")
        return streamed_text


def new_stream(model):
    """Open a fresh streaming context and return (ctx, tx). Callers exit the old
    ctx first. Also release MLX's pooled buffers so the reset actually frees
    memory (weights stay resident; macOS can't reclaim the pool itself)."""
    mx.clear_cache()
    ctx = model.transcribe_stream(context_size=CONTEXT_SIZE)
    return ctx, ctx.__enter__()


def quantize_model(model, bits: int) -> None:
    """Quantize the encoder/decoder Linear layers in place (MLX affine, group 64).

    `parakeet-mlx` has no quantization of its own and loads bf16, so this is
    applied after `from_pretrained`. **The self-attention linears must be
    skipped**: `transcribe_stream()` swaps in a `rel_pos_local_attn` module and
    copies weights across with `load_weights()`, which rejects the extra
    `scales`/`biases` a quantized Linear carries. Skipping them leaves ~100 of
    220 Linear layers quantized — mostly the feed-forward blocks, which is where
    the parameters are.

    Measured on LibriSpeech test-clean (60 utts, streaming, context 128,128):

        bf16   1236 MB   10.32% WER   1.01x RTF
        8-bit   853 MB   10.32% WER   1.39x RTF   <- default: free
        6-bit   751 MB   11.51% WER   1.32x RTF
        4-bit   648 MB   15.48% WER   1.34x RTF   <- +50% relative, do not use

    8-bit is strictly better than bf16 on every axis. Anything below it trades
    real accuracy, so re-run the WER sweep before changing this default.
    """
    import mlx.nn as nn

    skipped = 0

    def should_quantize(path, module):
        nonlocal skipped
        if not isinstance(module, nn.Linear):
            return False
        if "self_attn" in path:
            skipped += 1
            return False
        return True

    nn.quantize(model, group_size=64, bits=bits, class_predicate=should_quantize)
    mx.eval(model.parameters())
    log(f"quantized to {bits}-bit (group 64), skipped {skipped} self_attn linears")


def main() -> None:
    model_src = resolve_model()
    kind = "local dir" if os.path.isdir(model_src) else "Hub repo"
    log(f"loading {model_src} ({kind}) ...")
    model = from_pretrained(model_src)
    if model.preprocessor_config.sample_rate != SR:
        log(f"WARNING: model sample_rate={model.preprocessor_config.sample_rate}, expected {SR}")

    # Best-effort: a quantization failure must not cost us the whole ASR stage,
    # so fall back to the bf16 weights that are already loaded and keep going.
    if ASR_BITS:
        try:
            quantize_model(model, ASR_BITS)
        except Exception as e:  # noqa: BLE001
            log(f"WARNING: quantization failed ({e!r}); continuing at bf16")

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
    utt_buf = []         # raw samples of the current utterance, for finalize()
    prev_chunk = None    # one chunk of pre-roll, so a quiet onset isn't clipped

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
            speech = rms > VAD_RMS

            if speech:
                if not in_utterance:
                    # Start the buffer one chunk early. The VAD fires on RMS, so
                    # a word that begins quietly often starts in the *previous*
                    # chunk; without this pre-roll the full-context decode clips
                    # the utterance's first syllable.
                    utt_buf = [prev_chunk] if prev_chunk is not None else []
                in_utterance = True
                silence_ms = 0
                idle_ms = 0
                utterance_ms += step_ms
            elif in_utterance:
                silence_ms += step_ms
                utterance_ms += step_ms

            if in_utterance:
                utt_buf.append(samples)
                # Feed the encoder only while there is speech. Silence used to be
                # fed so trailing words flushed into the streaming result, but the
                # `final` now comes from finalize(), so encoding the silence tail
                # is pure waste — and it was the expensive half of the drain.
                # Gated on the same flag: with full-context finals off, the
                # streaming result *is* the final and still needs that flush, so
                # MATALU_FULL_CONTEXT_FINAL=0 restores the old behavior exactly.
                if speech or not FULL_CONTEXT_FINAL:
                    tx.add_audio(mx.array(samples))
                    emit("partial", tx.result.text, ts_ms)
                # Commit + reset on a silence gap (utterance ended) OR the
                # max-utterance cap (long continuous speech that never pauses,
                # e.g. a meeting monologue) so the transcript keeps flowing and
                # the streaming context stays bounded. new_stream() also clears
                # MLX's pooled buffers so the reset frees memory.
                if silence_ms >= SILENCE_MS or utterance_ms >= MAX_UTTERANCE_MS:
                    emit("final", finalize(model, utt_buf, tx.result.text), ts_ms)
                    utt_buf = []
                    ctx.__exit__(None, None, None)
                    ctx, tx = new_stream(model)
                    in_utterance = False
                    silence_ms = 0
                    utterance_ms = 0
            else:
                # Idle silence with no active utterance. We keep feeding real
                # audio (meeting mode never gates to NONE), so periodically drop
                # the accumulated silence context to keep MLX memory bounded.
                tx.add_audio(mx.array(samples))
                idle_ms += step_ms
                if idle_ms >= IDLE_RESET_MS:
                    ctx.__exit__(None, None, None)
                    ctx, tx = new_stream(model)
                    idle_ms = 0

            prev_chunk = samples
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

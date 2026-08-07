# matalu

Local, real-time dictation for Apple Silicon. Hold a hotkey, speak, release —
cleaned text is typed into whatever app has focus. Nothing leaves the machine.

Two on-device models via Apple **MLX**:

| | | |
|---|---|---|
| **Parakeet TDT 0.6B** | 8-bit | audio → verbatim text |
| **Qwen2.5-1.5B-Instruct** | 4-bit + QLoRA adapter | verbatim → cleaned text |

The cleanup stage removes fillers, stutters, and false starts — "so um I think
we should uh ship it Tuesday, no wait, Thursday" becomes "So, I think we should
ship it on Thursday." It is **deletion-only**: it never paraphrases, and if the
model's output isn't drawn from your words it is rejected and the raw
transcript is used instead.

## Architecture

Rust owns capture, resampling, injection, and process supervision. MLX is
Python-only, so each model runs in its own supervised sidecar over pipes.

```
 cpal mic          Rust                     Python sidecars
 ────────         ──────                   ─────────────────
 48 kHz  ─f32─▶  resample → 16k ─raw f32─▶ parakeet-mlx  (partials stream,
                 audio gate                              finals decoded at
                                            │             full context)
                 TranscriptEvent  ◀──JSON───┘
                        │
                 buffer whole utterance
                        │
                        └─────────text──────▶ Qwen + LoRA (cleanup)
                                   ◀──────────
                 guards → clipboard + CGEvent paste
```

**Nothing types while you speak.** The whole press→release window is buffered
and cleaned once, which is what lets a self-correction spanning two sentences
be fixed at all — you can't repair "Tuesday" after it's already typed.

- `crates/core/` — mic capture, resampling, sidecar supervision, `TranscriptEvent`s
- `src-tauri/` — Tauri v2 app: hotkey, session state machine, cleanup, injection, tray, pill
- `sidecar/` — the two Python sidecars plus post-hoc diarization
- `training/` — QLoRA dataset build, training config, and the eval harnesses

## Setup

Requires Rust, Python 3, and macOS on Apple Silicon.

```bash
python3 -m pip install parakeet-mlx mlx-lm
cargo tauri dev
```

Models (~2 GB total) download and cache on first run. Grant **Microphone** and
**Accessibility** permission — the latter is what allows typing into other apps.

The app is menu-bar only: no dock icon. A floating pill appears while dictating;
the transcript window is reachable from the tray.

## Quality

Measured, not estimated. `training/eval_pipeline.py` and `training/eval_adapter.py`
reproduce these.

| | |
|---|---|
| ASR, LibriSpeech test-clean | **1.49% WER** (published: 1.69%) |
| Cleanup, 250 held-out pairs | **3.27% WER** vs 13.67% doing nothing |
| Disfluencies removed | 54.4% |
| Content words preserved | 99.3% |
| Post-release latency | ~510 ms |

The cleanup model is fine-tuned on
[DisfluencySpeech](https://huggingface.co/datasets/amaai-lab/DisfluencySpeech)
(`transcript_a` → `transcript_c`). The adapter is committed; without it the app
silently falls back to a much weaker base model, so check the sidecar's startup
log for `system prompt: short (tuned)`.

## Also included

- **Meeting transcription** (tray → Start Meeting) — continuous capture via an
  Aggregate Device (mic + BlackHole loopback), with post-hoc speaker diarization
  through sherpa-onnx. Best-effort: no models, no labels, transcript still works.
- **`MATALU_CLEANUP_LOG=<path>`** — appends `{raw, cleaned}` pairs per dictation,
  for fine-tuning on your own voice rather than a public corpus.

Configuration is env-var only; see `CLAUDE.md` for the full list and for the
measured reasoning behind the defaults.

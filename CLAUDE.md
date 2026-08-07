# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`matalu` is a local, real-time speech-to-text system for Apple Silicon. It captures the mic and transcribes on-device with NVIDIA **Parakeet**, then cleans the transcript (fillers, stutters, false starts) with a QLoRA-tuned **Qwen2.5-1.5B**. Both run via Apple **MLX** — Python-only, so each lives in its own supervised **sidecar** process while Rust owns capture, resampling, and everything downstream.

**Two models, deliberately at different precisions:** Parakeet 0.6B at 8-bit, Qwen 1.5B at 4-bit + adapter. ASR errors go straight into WER; a constrained deletion task has slack. Neither is fine-tuned except the cleanup adapter, and they are not merged (see the end-to-end note under conventions).

The project is a **Cargo workspace**: a desktop app over a shared core library.

- **`matalu-app`** (`src-tauri/`) — the primary goal: a Wispr-Flow-style **Tauri desktop app** that types transcripts into whatever app has focus (global hotkey + system-wide text injection).
- **`matalu`** core lib (`crates/core/`) — mic capture, resampling, sidecar supervision, `TranscriptEvent`s. Transport-agnostic; the app depends on it.

## Commands

```bash
cargo build                              # build the whole workspace
cargo build -p matalu-app                # build just the desktop app
cargo test --workspace                   # 30 tests: 28 app + 2 core
cargo test -p matalu                     # core unit tests (LinearResampler, crates/core/src/audio.rs)
cargo test -p matalu streaming_matches_one_shot   # a single test by name

cargo tauri dev                          # run the desktop app (from src-tauri/, or with --config)
MATALU_DEMO=1 ./target/debug/matalu-app  # boot the app UI with synthetic events (no mic/model)


python3 sidecar/clean_sidecar.py --selftest      # cleanup sidecar's pure text handling (no model)
echo '{"text":"so um i think we should uh ship it"}' | python3 sidecar/clean_sidecar.py  # cleanup smoke test
cargo test -p matalu-app -- --ignored --nocapture # cleanup integration: real sidecar spawn + timeout fallback

# Quality measurement (training/). These are how any model/decoding change is judged.
python3 training/eval_pipeline.py        # ASR alone / cleanup alone / end-to-end, vs DisfluencySpeech ground truth
python3 training/eval_adapter.py         # base vs adapter on 250 held-out pairs, through the real sidecar
N_UTTS=50 python3 training/eval_pipeline.py       # quicker pass
python3 training/build_dataset.py        # rebuild training/data from DisfluencySpeech
python3 -m mlx_lm lora --train -c training/lora_config.yaml   # retrain the adapter (~10 min; stops early)

sidecar/build_sidecar.sh                 # PyInstaller-build the self-contained sidecar → src-tauri/binaries/ (before `cargo tauri build`)
```

Python deps for the sidecars: `python3 -m pip install parakeet-mlx mlx-lm` (pulls in mlx; a venv is recommended). `mlx-lm` powers the cleanup sidecar and also trains its QLoRA adapter (`mlx_lm.lora` trains LoRA directly on 4-bit weights — that *is* QLoRA, no PyTorch or bitsandbytes). Without `mlx-lm` the cleanup stage is simply unavailable and dictation types live and raw. The MLX model (~1.2 GB) downloads and caches automatically on first sidecar run. Grant microphone permission on first real run. The desktop app additionally needs **Accessibility** permission (for text injection). Sidecar path: the **app** (`cargo tauri dev`) runs with CWD `src-tauri/`, so `pipeline.rs` anchors the dev script to `CARGO_MANIFEST_DIR/../sidecar/matalu_sidecar.py` — no CWD requirement; override with `MATALU_SIDECAR`.

In `cargo tauri dev` the `main` transcript window no longer appears — it launches hidden. The visible surfaces are the **menu-bar tray** (Settings…, Show/Hide, Quit) and the floating **pill** (shown only while dictating); reveal the transcript window via the tray's "Show / Hide Window".

## Architecture

The **core pipeline** (`crates/core/`) flows one direction through stages, each in its own thread/process so a stall in one never blocks the others, terminating in a `tokio::broadcast` of `TranscriptEvent`s that the app consumes:

```
cpal mic (48 kHz) → resample→16k mono → Python parakeet-mlx sidecar → broadcast<TranscriptEvent> → frontend
   audio.rs           audio.rs (own thread)   sidecar.rs + matalu_sidecar.py                    └ app: cleanup + injection + UI (src-tauri/)
```

- **`crates/core/src/audio.rs`**: cpal capture on a dedicated OS thread (cpal `Stream` is `!Send`, and macOS `build_input_stream` blocks on mic-permission resolution). Downmixes to mono and resamples with a stateful `LinearResampler` that carries fractional position + last sample across buffers. The realtime callback **drops** buffers via `try_send` rather than blocking when the worker falls behind.
- **`crates/core/src/sidecar.rs`**: spawns the sidecar child — either a self-contained bundled executable (`cfg.sidecar_bin`, e.g. a PyInstaller build) run directly, or `python_bin sidecar_script` in dev — and two bridge threads — a *writer* (audio `Vec<f32>` → child stdin as raw little-endian f32) and a *reader* (child stdout newline-JSON → `TranscriptEvent` → broadcast). The reader intercepts a `{"type":"ready"}` control line to fire the readiness signal. The `Child` must be kept alive; dropping it stops transcription. Consumers **gate mic capture on the `ready` signal** (up to 180 s) — feeding audio while the model loads backs up the pipe and drops it.
- **`sidecar/matalu_sidecar.py`**: runs `parakeet-mlx` streaming + an RMS silence VAD. Deliberately dumb protocol (Rust owns capture/transport). Warms the model with a throwaway inference before signaling `ready`. Emits `partial` frames live; after `SILENCE_MS` of quiet emits a `final` and **resets the streaming context** for the next utterance.
- **`crates/core/src/events.rs`**: `TranscriptEvent` enum, `#[serde(tag = "type", rename_all = "lowercase")]` → `{"type":"partial","text":"...","ts_ms":123}`. The same shape is *deserialized* from sidecar stdout and *serialized* onward as Tauri IPC `transcript` events.
- **`src-tauri/src/cleaner.rs` + `sidecar/clean_sidecar.py`**: the downstream **cleanup LLM** (Qwen2.5-1.5B-Instruct 4-bit via **mlx-lm**), a second supervised sidecar. See below.

**Frontend:**
- **`src-tauri/`** (`matalu-app`): Tauri v2 shell. `lib.rs::run()` builds the app; `pipeline.rs::start()` spawns the core pipeline plus an **audio-gate** thread and routes each `TranscriptEvent` through the session to injection + UI (`app.emit("transcript", …)`). The gate is **3-state** (`pipeline::gate` — `REAL`/`SILENCE`/`NONE`): Listening forwards real mic audio, Draining forwards silence (so the sidecar's VAD flushes the trailing `final`), and Idle forwards **nothing** — the sidecar's blocking stdin read parks so it stops running inference between dictations (the "explicit start/stop" without a protocol change).
  - `session.rs` — activation state machine (Idle/Listening/Draining), gates the audio feed, drives injection. Not ready until the sidecar is warm + mic live (`mark_ready`), so the "listening" indicator can't lie.
  - `injector.rs` — system-wide text injection: diff-based live partials (unit-tested `DiffState`) pasted via clipboard (`arboard`) + **CoreGraphics `CGEvent`** keystrokes. Do **not** use enigo here — its Unicode path SIGTRAPs when called off the main thread.
  - `hotkey.rs` — global activation via `tauri-plugin-global-shortcut` (Pressed/Released → push-to-talk or toggle). The `fn` (Globe) preset can't be a plugin binding, so it's driven by `fnkey.rs` instead — a **CGEventTap** on its own thread + `CFRunLoop` that edge-detects the secondary-`fn` flag on `FlagsChanged` and calls the same `Session::on_press`/`on_release`. It's listen-only (fn passes through) and needs Accessibility/Input Monitoring; switching to/from `fn` in settings takes effect on the next launch (a run-loop tap can't be started mid-session).
  - `settings.rs` + `commands.rs` — persisted `Settings` (activation mode + hotkey preset) as JSON in the app config dir; Tauri commands for the settings window (`ui/settings.html`) and Accessibility onboarding.
  - `tray.rs` — menu-bar tray (Settings…, **Start/Stop Meeting Transcript**, Show/Hide, Quit). Frontend is static HTML/JS in `ui/` (no bundler; `withGlobalTauri` exposes `window.__TAURI__`).
  - **Meeting-transcript mode** (`session.rs::start_meeting`/`stop_meeting`, toggled from the tray): capture runs continuously (`gate::REAL`), transcripts fill the `main` window but are **not** injected (`on_event` early-returns while `meeting`), and the dictation hotkey is inert. It reuses the whole existing capture→sidecar→broadcast→UI path; the only capture change is pointing `MATALU_INPUT_DEVICE` at an **Aggregate Device** (mic + BlackHole loopback) so both sides are heard as one merged stream. `ui/index.html` accumulates finals with **Save** (→ `commands::save_transcript`, writes `~/Documents/matalu-transcript-<ts>.txt`) / **Clear** buttons.
  - **Speaker tags (post-hoc diarization).** On meeting **Stop**, the merged audio (recorded to a temp WAV by `pipeline::MeetingRecorder` off the audio-gate thread, via `hound`) is diarized once by **`sidecar/diarize.py`** (sherpa-onnx, onnxruntime — *not* PyTorch; offline ONNX models under `sidecar/models/diarization/`, overridable with `MATALU_DIARIZE_SEG_MODEL`/`MATALU_DIARIZE_EMB_MODEL`; auto speaker-count). `diarize.rs` runs the script, aligns each timed line to the max-overlap speaker segment, and emits `meeting_result`; the UI re-renders as `Speaker N: text` with click-to-rename. **Best-effort:** if sherpa-onnx or the models are absent, `diarize.py` exits non-zero and the transcript stays **unlabeled** — meeting mode never breaks. The diarizer is transient (loads on Stop, exits) so it costs no resident RAM. Setup: `pip install sherpa-onnx` + drop the segmentation/embedding `.onnx` files in `sidecar/models/diarization/`. **Adding a window = declaring it in `tauri.conf.json` with a `url` pointing at a `ui/*.html` file** — Tauri serves those directly, so there's no build step and nothing to import; the files are plain HTML/JS.
  - **Menu-bar-only presence:** `lib.rs::run()` sets `ActivationPolicy::Accessory` (no dock icon), and the `main` transcript window is **hidden on launch** (dev-only; reveal via tray "Show / Hide Window"). The signature surface is a floating **pill** — a transparent, always-on-top, non-activating (`focus:false` + `set_ignore_cursor_events`) window (`ui/pill.html`, needs `macos-private-api`), parked bottom-center by `lib.rs::setup_pill`. `session.rs::set_pill` shows it for the whole dictation window (Listening + Draining) and hides it on return to Idle; it mirrors the same app-wide `transcript`/`status` emits the main window uses.

## Key conventions & decisions

- **The cleanup model is QLoRA fine-tuned, and the adapter is NOT in git.** `sidecar/adapters/cleanup/` is gitignored (build artifact), so a fresh clone silently runs the far weaker base model. Rebuild in ~15 min:
  ```bash
  python3 training/build_dataset.py                        # DisfluencySpeech a -> c, 4500/250/250
  python3 -m mlx_lm lora --train -c training/lora_config.yaml
  python3 training/eval_adapter.py                         # base vs adapter, held-out
  ```
  Check the sidecar's startup log — it prints `system prompt: short (tuned)` vs `full (base model)`, which is how you tell which one is actually loaded.
  Measured on 250 held-out pairs, through the real sidecar with production guards:

  | | WER | disfluencies removed | content kept | guard fallback |
  |---|---|---|---|---|
  | no cleanup | 13.67% | 0% | 100% | — |
  | base model | 11.95% | 21.8% | 96.7% | 4.8% |
  | **+ adapter** | **3.27%** | **54.4%** | **99.3%** | **0.0%** |

  Removal more than doubled *while* preservation rose — those normally trade off, so it learned the deletion-only transform rather than "edit harder". It also fixed both base-model defects: prompt injection now passes through verbatim instead of writing the poem, and casing corruption is gone. Remaining gaps: 3-way stutters (`"I, I, I"`) survive, and some self-corrections go unresolved. Both are it erring conservative, which is the requested direction — see the deletion-only convention below.
  **Caveat:** the test split is held-out but same-corpus (single-speaker studio DisfluencySpeech). Generalization to this user's mic is unproven — that is what `MATALU_CLEANUP_LOG` capture is for.
- **Cleanup decoding uses prompt-lookup speculative decoding (`MATALU_CLEAN_NO_LOOKUP=1` disables).** The drafter copies the continuation of a matching n-gram straight out of the prompt — no second model, no extra memory — because cleanup is deletion-only so nearly every output token already appears in the input. **79% of drafted tokens are accepted**, and the measured win through the real sidecar is **1.74x (595 → 342 ms) with byte-identical output**.
  - **It is exact, and that is the entire justification.** A drafted token is kept only where it matches what the model would have produced anyway; the first mismatch is replaced by the model's own token. Verified against plain greedy on 45 utterances plus the held-out eval (2.84% WER, 77.0% capture — unchanged).
  - **`n=2, k=4` (`MATALU_CLEAN_DRAFT_N`/`_K`).** Bigger drafts lose: a rejected token wastes the rest of its batch, so k=4 accepts 79% where k=16 accepts 33%.
  - **The first implementation reported 1.45x while producing *different* text** — it ran past EOS, because a batch of accepted drafts can overshoot the stop token in a way greedy never does. Exactness is not a nice-to-have here; an unchecked speculative decoder is just a fast wrong answer.
  - The startup self-check compares **plain greedy** against lookup and against cache+lookup. Checking the two optimized paths against each other would pass while both were wrong — keep the un-optimized reference.
- **Do NOT `mlx_lm.fuse` the cleanup adapter. Four fusion variants were measured; all four lose.** Numbers below are on a common 100-pair subset where the shipping config scores 2.84%:

  | | WER | removed | kept | fallback | mem | latency |
  |---|---|---|---|---|---|---|
  | base model | 11.27% | 21.8% | 96.7% | 4.0% | 860 MB | 485 ms |
  | **base + adapter** | **2.84%** | **54.0%** | **99.0%** | **0.0%** | **860 MB** | 595 ms |
  | fused, 4-bit requant | 9.67% | 34.2% | 96.4% | 5.2% | 839 MB | 508 ms |
  | fused → mixed 4/8-bit | 3.15% | 53.1% | 99.0% | 0.0% | 1200 MB | 723 ms |
  | fused → 8-bit | 3.62% | 54.4% | 99.0% | 0.4% | 1500 MB | 853 ms |
  | fused `--dequantize` | 3.29% | 54.4% | 99.3% | 0.0% | 3087 MB | 1381 ms |

  - **4-bit requant destroys the tune.** The delta is full-precision and small relative to the 4-bit step, so it rounds away.
  - **`--dequantize` preserves it exactly**, which proves requantization is the culprit — but bf16 reads 3087 MB per token, making it 2x *slower* than the adapter.
  - **Mixed 4/8-bit is the best fusion**, built with a custom `quant_predicate` protecting the 112 LoRA-touched modules (layers 12–27, all seven projections) at 8-bit and the rest at 4-bit — 6.441 bits/weight. It recovers nearly all the quality (9.67% → 3.15%) and beats AWQ's premise, since we *know* which weights changed rather than inferring salience from calibration data. It still loses on all three axes.

  **The reason is worth internalizing: the adapter already is the optimal mixed-precision scheme.** 4-bit base + a full-precision delta stored separately = 4.5 effective bits + 21 MB. Every requantization scheme is trying to approximate, inside quantized weights, information the adapter simply keeps. Don't re-litigate without new evidence.
- **Training uses `TUNED_SYSTEM_PROMPT` (short), inference picks by adapter presence.** The base model needs the full instruction block or it hijacks and over-edits; the adapter encodes the behavior, so it gets a ~10-token marker instead. Training on the long block made the run **4x slower for no signal** (`mask_prompt` already zeroes its loss). `training/build_dataset.py` imports the prompt from the sidecar rather than copying it, so the two cannot drift. Val loss plateaued by iter 100–200, so the 1200-iter config stops early on purpose.
- **Dictation cleanup is buffered, Wispr-Flow style.** With cleanup on, **nothing types while you speak**: `session.rs` accumulates committed finals in `utterance_buf`, and on release the whole press→release window is cleaned once and pasted as **one** edit. Full-window context is the point — a self-correction spanning two sentences ("ship it Tuesday, no wait, Thursday") is unfixable once "Tuesday" is already typed. The UI and pill still show raw partials live via the app-wide `transcript` emit. Both drain exits (trailing `final` **and** the watchdog) must route through `Session::finish_dictation` — skipping the watchdog path silently drops the whole utterance. `MAX_BUFFER_CHARS` (600) forces a mid-session flush so a marathon dictation isn't minutes of nothing.
- **Cleanup is deletion-only, and that is enforced mechanically.** User requirement: prefer word-for-word over omission — dropping a content word is worse than leaving an "um". Stock Qwen 1.5B does not honor this by prompt alone; measured, it obeys dictated text that reads like an instruction ("ignore all previous instructions…" → it writes the poem) and lowercases words Parakeet already capitalized (`Q1` → `q1`). Neither is caught by a length check. So `cleaner.rs` vets every candidate with pure, unit-tested guards: **word-overlap ≥ 0.90** (every output word must already exist in the input — catches hijacks and summarization), **casing restoration** (restore capitals the model dropped; never strip ones it added), and a **two-sided length ratio** (0.5–1.3). Any rejection, timeout, or dead child ⇒ paste the **raw** text. Losing the user's words is the one unacceptable outcome.
- **The cleanup timeout scales with input length.** Generation cost grows with length, so a fixed timeout would make long dictations *always* fall back to raw, precisely when cleanup is worth most — hence `MATALU_CLEAN_TIMEOUT_MS` (2000) is a **base** plus `TIMEOUT_PER_CHAR` (12 ms/char, ~2x the measured slope). Note this interacts with tests: a timeout test must use a budget no speedup can beat (base `0` + a 2-char input = 24 ms), or it silently starts passing for the wrong reason.
- **The system prompt is KV-cached across requests; do not undo this.** Profiling found latency was dominated not by the model but by re-reading our own 263-token system block every request — **612 ms of a 973 ms** call (290 prefill tokens vs 12 decode tokens). `clean_sidecar.py` prefills that block into a `make_prompt_cache` once at warm-up, sends only the per-request suffix, and `trim_prompt_cache`s back afterwards. Measured **2.0–2.3x** (e.g. 1021 → 451 ms) with byte-identical output. Consequences to respect:
  - The **prompt-prefix invariant** (`full[:len(PREFIX)] == PREFIX`) is checked on every request. A model or chat-template swap can break it, and the failure mode is *silently wrong text*, not an error — so the warm-up runs the probe **both cached and uncached and compares**, disabling the cache on mismatch. Keep that self-check.
  - Any cached-path exception drops to the uncached path and rebuilds the cache. `MATALU_CLEAN_NO_CACHE=1` forces uncached.
  - This uses `stream_generate`, not `generate`, because trimming needs the generated-token count.
  - Because prefill is now nearly free, **shortening the system prompt buys almost no speed** — do it for clarity if at all, not for latency.
- **The ASR model is quantized to 8-bit at load (`MATALU_ASR_BITS`, default 8; `0` = bf16).** `parakeet-mlx` loads bf16 and has no quantization of its own, so `matalu_sidecar.py::quantize_model` applies `nn.quantize` (MLX affine, group 64, weight-only) after `from_pretrained`. Measured on LibriSpeech test-clean (60 utts, streaming): **8-bit is strictly better than bf16 — same WER, −383 MB, 1.39x RTF** (real drain 969 → 679 ms). 6-bit costs +1.19 WER, **4-bit costs +5.16 WER (+50% relative) and must not be used.** That sweep sampled **shortest-first**, which inflates every absolute WER ~2.5x, so only its *deltas* are quotable — the model's real numbers are in the streaming/full-context table below.
  - **`self_attn` linears must be skipped** (~120 of 220). `transcribe_stream()` swaps in a `rel_pos_local_attn` module and copies weights with `load_weights()`, which rejects the `scales`/`biases` a quantized Linear carries. Quantizing them raises `Received 10 parameters not in model`.
  - **This is aggregate-neutral, not output-identical** — unlike the prompt cache and the drain burst, individual transcripts *do* change; the justification is equal WER over a corpus, not an equal string. Any change to the model, bit width, or streaming context needs a fresh WER sweep, not a spot check: a single TTS sentence made 4-bit look like a punctuation difference when it was actually +50% relative WER.
- **Streaming costs ~4x WER, and it is the largest quality lever in the system.** Measured on a *random* 60-utterance LibriSpeech test-clean sample, 8-bit:

  | config | WER |
  |---|---|
  | full context (`model.transcribe`, non-streaming) | **1.49%** |
  | streaming `context_size=(128,128)` — what the app runs | **6.12%** |
  | nvidia published, test-clean | 1.69% |

  At full context we match published, so the model and the quantization are fine — the loss is entirely the streaming configuration. `(256,256)` measured identical to `(128,128)`, so context size within streaming is not the knob. For scale: streaming costs ~4.6 WER points while the entire cleanup LLM buys 1.33 (see `training/eval_pipeline.py`).
- **Partials stream; `final`s are decoded at full context (`MATALU_FULL_CONTEXT_FINAL`, default 1).** This is how the streaming tax above is avoided. The buffered UX means partials are purely cosmetic (pill + transcript window) — the only text the app injects is the `final` — so `matalu_sidecar.py` buffers each utterance's raw samples and re-decodes them in one pass via `get_logmel` + `model.generate` when its VAD ends the utterance. Needed **no protocol change and no Rust change**: the sidecar already receives every sample.
  - **It is faster, not slower, despite adding a decode.** Once the `final` no longer comes from the streaming result, the trailing silence no longer needs to go through the encoder — and that was the expensive half of the drain. Measured through the real sidecar: **353 → 169 ms (2.09x)**, and the output went from `'So am I. I think we should ship it on Tuesday. No wait. Thursday.'` to verbatim-correct `'So um I think we should ship it on Tuesday, no wait, Thursday.'`
  - **One chunk of pre-roll is buffered** because the VAD fires on RMS and a quiet word onset often begins in the *previous* chunk. Without it the decode clips the first syllable — the most likely way this breaks subtly.
  - The skip-silence behavior is gated on the **same** flag, so `MATALU_FULL_CONTEXT_FINAL=0` restores the old path exactly. Splitting them produces a third, worse path where the streaming final never gets its flush.
  - The burst drain (below) still matters and still composes: the sidecar must *observe* silence chunks to detect the end of an utterance, it just no longer encodes them.
- **The drain silence is burst, not paced — do not "fix" it back to mic cadence.** On release the gate switches to `gate::SILENCE`, and the sidecar needs `SILENCE_MS` of quiet (counted in 0.5 s chunks) before it emits the trailing `final` that starts cleanup. Feeding those zeros at *microphone cadence* made that wait real-time — measured **1509 ms**, larger than the cleanup itself. `burst_drain_silence` pushes the whole drain at once instead: **491 ms, transcript byte-identical** (Parakeet's streaming decode is sample-driven, not clock-driven, so the same zeros delivered faster decode the same). Two things to preserve:
  - **Queued real audio is flushed first.** Buffers still in `capture_rx` when the gate flips were captured *before* release — the user's last words. Silence jumping ahead of them makes the VAD finalize early and those words are lost (the gate goes `NONE` moments later) or split into the next utterance. Unit-tested (`queued_real_audio_is_flushed_before_the_silence_burst`).
  - Burst size comes from `Config::silence_ms`, never hardcoded, and `try_send` is kept throughout so a full queue truncates rather than blocking the gate — the paced silence path after it is the fallback.
- **The two models cannot be weight-merged, but a single end-to-end model is a live open question.** SLERP and friends need identical architecture and tensor shapes; Parakeet (audio-in RNN-T/TDT conformer) shares no correspondence with Qwen (text-in decoder-only), and merging two N-param models yields one N-param model anyway — it combines capabilities, not speed. What *would* pay is a **different** thing: one model trained audio → cleaned text, removing the whole second stage (~340 ms and ~860 MB).
  The argument for it is the error decomposition (`training/eval_pipeline.py`, 200 DisfluencySpeech utts, current config):

  | | WER |
  |---|---|
  | ASR alone (vs verbatim) | 4.37% |
  | cleanup alone (perfect input) | 4.36% |
  | **end-to-end** | **9.00%** |

  4.37 + 4.36 ≈ 9.00 — **the two error sources compound almost additively**, and content preservation drops 98.8% → 95.0% between clean input and ASR input purely from the cleanup stage mishandling ASR errors. That cascade is structural to any two-stage design and no amount of tuning either stage removes it. A single model has one error source.
  Costs that keep it unbuilt: it **destroys the guard architecture** (no raw transcript to check the output against, so word-for-word becomes unenforceable), `parakeet-mlx` is inference-only so it means Whisper+LoRA or NeMo+CUDA, and 9.00% is now a genuinely good bar to beat. `amaai-lab/DisfluencySpeech` ships `audio` alongside `transcript_a/c`, so the training data exists.
- **The cleanup sidecar warms eagerly and unloads only after 30 min — do not tighten this without re-measuring cold start.** `Cleaner::clean` does **not** wait for the child: if it isn't up, it returns `None` and the caller pastes raw text. Cold start is **~5.5 s** (model load plus the prompt-cache and lookup self-checks, each of which runs a probe inference). So any dictation shorter than that, on a cold cleaner, silently loses cleanup — and short dictations are the common case.
  Two things keep that from happening: `warm()` fires from `Session::mark_ready` (pipeline up, mic live) rather than only on hotkey press, so the first dictation of the app's life isn't a guaranteed miss; and `MATALU_CLEAN_IDLE_MS` defaults to **1800000 (30 min)**, which covers a work session. It was 180000, which meant the first dictation after any short break lost cleanup. Cleanup is worth 76% of the achievable improvement — reclaiming 860 MB three minutes sooner is not worth trading that away every session.
  `warm()` never blocks the caller, and a not-yet-warm `clean()` logs `cleanup sidecar not warm yet` so it is distinguishable from a guard rejection — the two want opposite fixes.
- **Config is env-var only** (`crates/core/src/config.rs`, `Config::from_env`). Notable vars: `MATALU_PYTHON`, `MATALU_SIDECAR`, `MATALU_SIDECAR_BIN` (optional bundled sidecar executable — overrides python+script), `MATALU_MLX_MODEL`, `MATALU_SILENCE_MS` (700), `MATALU_VAD_RMS` (0.010), `MATALU_INPUT_DEVICE` (optional — capture a named input device by its `Display` name instead of the default mic; used for meeting mode), `MATALU_MAX_UTTERANCE_MS` (12000; sidecar-read — forces a `final`+context reset after this much continuous speech), `MATALU_IDLE_RESET_MS` (3000; sidecar-read — resets the streaming context + clears the MLX cache after this much continuous idle silence, so meeting mode's non-stop feed can't grow the cache pool to GBs), `MATALU_ASR_BITS` (8; sidecar-read — ASR weight quantization, `0` = bf16), `MATALU_DUMP_WAV` (optional; sidecar-read — write exactly the audio the sidecar receives, post-resample and post-gate, to a 16 kHz mono WAV), `MATALU_DEMO`.
  App-only debug overrides (read in `src-tauri/`, not part of `Config`): `MATALU_MODE` (`toggle`/`pushtotalk` — overrides the persisted activation mode), `MATALU_NO_INJECT` (run the pipeline and UI without typing into other apps — useful for testing without Accessibility).
  Cleanup-stage vars (read in `src-tauri/src/cleaner.rs` / `sidecar/clean_sidecar.py`): `MATALU_CLEAN_MODEL` (override; else a bundled `sidecar/models/Qwen2.5-1.5B-Instruct-4bit/` dir, else the Hub repo — same three-tier shape as the ASR model), `MATALU_CLEAN_ADAPTER` (QLoRA adapter dir; defaults to `sidecar/adapters/cleanup/` if present), `MATALU_CLEAN_SIDECAR`, `MATALU_CLEAN_TIMEOUT_MS` (2000 base), `MATALU_CLEAN_IDLE_MS` (1800000), `MATALU_CLEAN_NO_LOOKUP`, `MATALU_CLEAN_DRAFT_N`/`_K`, `MATALU_CLEAN_NO_CACHE`, `MATALU_NO_CLEANUP` (force the raw live-partial path; also implied by `MATALU_DEMO`). The user-facing on/off toggle is a persisted setting (`settings.cleanup`), not an env var.
- **Packaging bundles the sidecar as an executable** (M6). `tauri.conf.json` `bundle.externalBin` ships `binaries/matalu-sidecar-<triple>`; `sidecar/build_sidecar.sh` + `matalu-sidecar.spec` build it with **PyInstaller** (bundling mlx/parakeet-mlx is the fragile part — verify it loads on a clean machine). A committed **placeholder stub** at that path keeps `cargo build`/`cargo tauri dev` working and errors loudly if a packaged build ships without the real binary; the build script overwrites it (don't commit the built binary). At runtime `pipeline.rs` prefers a `matalu-sidecar` next to the app executable and falls back to the dev `python3` path — same prefer-bundled-else-dev shape as the model resolution.
- **The sidecar prefers a local/bundled model over the Hub** (`resolve_model()` in `matalu_sidecar.py`). Precedence: `MATALU_MLX_MODEL` (explicit override — local path or Hub repo ID) → a bundled `sidecar/models/parakeet-tdt-0.6b-v2/` dir if present (fully offline; the norm) → the Hub repo ID `mlx-community/parakeet-tdt-0.6b-v2` (downloads + caches on first run). Drop the model folder into `sidecar/models/` to run offline with no env var. The sidecar additionally reads **`MATALU_MLX_CONTEXT`** (default `128,128`) — the streaming attention context `(left,right)`, the main speed/accuracy knob. Measured here: `64,64` runs fast but **drops the utterance start** (too little warmup context); `128,128` recovers it and stays real-time (~1.1x RTF); `256,256` is no more accurate and decodes below real time (~0.7x, lags). Too small (e.g. `32,16`) collapses. `MATALU_DUMP_WAV=<path>` makes the sidecar write exactly the audio it receives (post-resample, post-gate) to a 16 kHz mono WAV — the diagnostic for separating a mic/capture problem from a streaming one. The app will move user-facing prefs into a settings store; ASR/sidecar params stay env-driven.
- **Never block the audio callback or the async runtime.** Bounded/lossy channels are intentional — preserve the drop-don't-block behavior.
- **UI updates use app-wide `app.emit(...)`, never `window.emit(...)`.** A single `app.emit("transcript"|"status", …)` fans out to every window (main + pill) at once, which is what keeps the pill and the transcript window in sync. Switching to a per-window emit would silently desync the pill — keep emits app-wide.
- **Workspace uses `[workspace.dependencies]`** (root `Cargo.toml`); crates reference shared deps with `.workspace = true`. `tokio`'s `time` feature is enabled per-crate where needed (the app uses it; core's feature set is minimal).

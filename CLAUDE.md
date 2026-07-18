# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`matalu` is a local, real-time speech-to-text system for Apple Silicon. It captures the mic and transcribes on-device with NVIDIA **Parakeet** via Apple **MLX** (Python-only, run in a supervised **sidecar** process); Rust owns capture, resampling, and everything downstream. The project is a **Cargo workspace** with two frontends over one shared core:

- **`matalu-app`** (`src-tauri/`) — the primary goal: a Wispr-Flow-style **Tauri desktop app** that types transcripts into whatever app has focus (global hotkey + system-wide text injection). Under active construction; see the plan at `~/.claude/plans/rippling-leaping-moonbeam.md`.
- **`matalu-headless`** (`crates/headless/`) — the original headless **WebSocket service** (`/ws` + browser viewer at `/`), kept as a dev/debug harness for the pipeline.
- **`matalu`** core lib (`crates/core/`) — mic capture, resampling, sidecar supervision, `TranscriptEvent`s. Transport-agnostic; both frontends depend on it.

## Commands

```bash
cargo build                              # build the whole workspace
cargo build -p matalu-app                # build just the desktop app
cargo test -p matalu                     # core unit tests (LinearResampler, crates/core/src/audio.rs)
cargo test -p matalu streaming_matches_one_shot   # a single test by name

cargo tauri dev                          # run the desktop app (from src-tauri/, or with --config)
MATALU_DEMO=1 ./target/debug/matalu-app  # boot the app UI with synthetic events (no mic/model)

cargo run -p matalu-headless             # run the headless WebSocket service (mic + sidecar + /ws)
MATALU_DEMO=1 cargo run -p matalu-headless        # WebSocket path with synthetic events
python3 sidecar/test_stream.py some_audio_16k.wav # sidecar streaming smoke test on a 16 kHz WAV
```

Python deps for the sidecar: `python3 -m pip install parakeet-mlx` (pulls in mlx; a venv is recommended). The MLX model (~1.2 GB) downloads and caches automatically on first sidecar run. Grant microphone permission on first real run. The desktop app additionally needs **Accessibility** permission (for text injection). Default sidecar path (`sidecar/matalu_sidecar.py`) is relative to CWD, so run from the workspace root in dev.

In `cargo tauri dev` the `main` transcript window no longer appears — it launches hidden. The visible surfaces are the **menu-bar tray** (Settings…, Show/Hide, Quit) and the floating **pill** (shown only while dictating); reveal the transcript window via the tray's "Show / Hide Window".

## Architecture

The **core pipeline** (`crates/core/`) flows one direction through stages, each in its own thread/process so a stall in one never blocks the others, terminating in a `tokio::broadcast` of `TranscriptEvent`s that a frontend consumes:

```
cpal mic (48 kHz) → resample→16k mono → Python parakeet-mlx sidecar → broadcast<TranscriptEvent> → frontend
   audio.rs           audio.rs (own thread)   sidecar.rs + matalu_sidecar.py                    ├ headless: WebSocket /ws (server.rs)
                                                                                                 └ app: text injection + UI (src-tauri/)
```

- **`crates/core/src/audio.rs`**: cpal capture on a dedicated OS thread (cpal `Stream` is `!Send`, and macOS `build_input_stream` blocks on mic-permission resolution). Downmixes to mono and resamples with a stateful `LinearResampler` that carries fractional position + last sample across buffers. The realtime callback **drops** buffers via `try_send` rather than blocking when the worker falls behind.
- **`crates/core/src/sidecar.rs`**: spawns the Python child and two bridge threads — a *writer* (audio `Vec<f32>` → child stdin as raw little-endian f32) and a *reader* (child stdout newline-JSON → `TranscriptEvent` → broadcast). The reader intercepts a `{"type":"ready"}` control line to fire the readiness signal. The `Child` must be kept alive; dropping it stops transcription. Consumers **gate mic capture on the `ready` signal** (up to 180 s) — feeding audio while the model loads backs up the pipe and drops it.
- **`sidecar/matalu_sidecar.py`**: runs `parakeet-mlx` streaming + an RMS silence VAD. Deliberately dumb protocol (Rust owns capture/transport). Warms the model with a throwaway inference before signaling `ready`. Emits `partial` frames live; after `SILENCE_MS` of quiet emits a `final` and **resets the streaming context** for the next utterance.
- **`crates/core/src/events.rs`**: `TranscriptEvent` enum, `#[serde(tag = "type", rename_all = "lowercase")]` → `{"type":"partial","text":"...","ts_ms":123}`. The same shape is *deserialized* from sidecar stdout and *serialized* onward (WS frames, or Tauri IPC `transcript` events).
- **`crates/core/src/corrector.rs`**: post-ASR text seam. See below.

**Frontends:**
- **`crates/headless/`** (`matalu-headless`): `main.rs` wires the pipeline to `server.rs` — `axum` `/ws` (per-client broadcast subscription), `/health`, and `/` (serves `index.html` via `include_str!`).
- **`src-tauri/`** (`matalu-app`): Tauri v2 shell. `lib.rs::run()` builds the app; `pipeline.rs::start()` spawns the core pipeline plus an **audio-gate** thread (feeds real mic audio to the sidecar only while listening, silence otherwise) and routes each `TranscriptEvent` through the session to injection + UI (`app.emit("transcript", …)`).
  - `session.rs` — activation state machine (Idle/Listening/Draining), gates the audio feed, drives injection. Not ready until the sidecar is warm + mic live (`mark_ready`), so the "listening" indicator can't lie.
  - `injector.rs` — system-wide text injection: diff-based live partials (unit-tested `DiffState`) pasted via clipboard (`arboard`) + **CoreGraphics `CGEvent`** keystrokes. Do **not** use enigo here (see [[matalu-injection-coregraphics]]).
  - `hotkey.rs` — global activation via `tauri-plugin-global-shortcut` (Pressed/Released → push-to-talk or toggle).
  - `settings.rs` + `commands.rs` — persisted `Settings` (activation mode + hotkey preset) as JSON in the app config dir; Tauri commands for the settings window (`ui/settings.html`) and Accessibility onboarding.
  - `tray.rs` — menu-bar tray (Settings…, Show/Hide, Quit). Frontend is static HTML/JS in `ui/` (no bundler; `withGlobalTauri` exposes `window.__TAURI__`). **Adding a window = declaring it in `tauri.conf.json` with a `url` pointing at a `ui/*.html` file** — Tauri serves those directly, so there's no build step and nothing to import; the files are plain HTML/JS.
  - **Menu-bar-only presence:** `lib.rs::run()` sets `ActivationPolicy::Accessory` (no dock icon), and the `main` transcript window is **hidden on launch** (dev-only; reveal via tray "Show / Hide Window"). The signature surface is a floating **pill** — a transparent, always-on-top, non-activating (`focus:false` + `set_ignore_cursor_events`) window (`ui/pill.html`, needs `macos-private-api`), parked bottom-center by `lib.rs::setup_pill`. `session.rs::set_pill` shows it for the whole dictation window (Listening + Draining) and hides it on return to Idle; it mirrors the same app-wide `transcript`/`status` emits the main window uses.

## Key conventions & decisions

- **The `Corrector` stays `PassThrough`.** Parakeet emits punctuation/casing itself, so there is no separate grammar/formatting LLM in v1. The trait is the extension point for a future `OllamaCorrector`/`llama.cpp` pass, but **do not build one unless the user asks.**
- **Config is env-var only** (`crates/core/src/config.rs`, `Config::from_env`). Notable vars: `MATALU_BIND` (`127.0.0.1:8765`, headless only), `MATALU_PYTHON`, `MATALU_SIDECAR`, `MATALU_MLX_MODEL`, `MATALU_SILENCE_MS` (700), `MATALU_VAD_RMS` (0.010), `MATALU_DEMO`. The sidecar additionally reads **`MATALU_MLX_CONTEXT`** (default `64,64`) — the streaming attention context `(left,right)`, the main speed/accuracy knob (`64,64` ≈ 1.4x RTF on M4; `256,256` decodes below real time; too small collapses). The app will move user-facing prefs into a settings store; ASR/sidecar params stay env-driven.
- **Never block the audio callback or the async runtime.** Bounded/lossy channels are intentional — preserve the drop-don't-block behavior.
- **UI updates use app-wide `app.emit(...)`, never `window.emit(...)`.** A single `app.emit("transcript"|"status", …)` fans out to every window (main + pill) at once, which is what keeps the pill and the transcript window in sync. Switching to a per-window emit would silently desync the pill — keep emits app-wide.
- **Workspace uses `[workspace.dependencies]`** (root `Cargo.toml`); crates reference shared deps with `.workspace = true`. `tokio`'s `time` feature is enabled per-crate where needed (the app uses it; core's feature set is minimal).

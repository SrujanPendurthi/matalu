//! Wire the `matalu` core pipeline into the Tauri app: mic → **audio gate** →
//! sidecar → transcript fan-out → ([`Session`] → injector) + UI.
//!
//! The gate sits between capture and the sidecar so the [`Session`] can decide,
//! per buffer, whether the model hears real audio (listening) or silence (idle).

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Duration;

use matalu::config::{Config, TARGET_SAMPLE_RATE};
use matalu::corrector::PassThrough;
use matalu::events::TranscriptEvent;
use matalu::{audio, sidecar};
use tauri::{AppHandle, Emitter};
use tokio::sync::broadcast;

use crate::injector;
use crate::session::{Mode, Session};

/// Tauri event name carrying a serialized [`TranscriptEvent`] to the UI.
pub const TRANSCRIPT_EVENT: &str = "transcript";

/// Audio-gate modes shared between the [`Session`] and the gate thread. The
/// session writes; the gate reads (`Relaxed` — a one-buffer lag is harmless).
///
/// This is the "explicit start/stop" mechanism: rather than feeding the sidecar
/// silence forever, [`NONE`] feeds it *nothing*, so its blocking `stdin.read`
/// parks and it stops running inference on silence between dictations.
pub mod gate {
    /// Idle: forward nothing — the sidecar blocks on stdin and idles (no inference).
    pub const NONE: u8 = 0;
    /// Draining: forward silence so the sidecar's VAD flushes the trailing `final`.
    pub const SILENCE: u8 = 1;
    /// Listening: forward real mic audio.
    pub const REAL: u8 = 2;
}

/// What [`start`] hands back to the app: the session (also placed in managed
/// state so the hotkey handler can reach it) and the sidecar child to keep alive.
pub struct Started {
    pub session: Arc<Session>,
    pub child: Option<std::process::Child>,
}

/// Start the pipeline and return the [`Session`] + sidecar child. `mode` is the
/// initial activation mode (from persisted settings / env override).
pub fn start(app: AppHandle, mode: Mode) -> anyhow::Result<Started> {
    let mut cfg = Config::from_env()?;
    // Prefer a bundled sidecar binary shipped next to the app executable
    // (packaged build) unless the dev env vars pin python/script explicitly.
    if cfg.sidecar_bin.is_none() && std::env::var_os("MATALU_SIDECAR").is_none() {
        if let Some(bundled) = bundled_sidecar_path() {
            tracing::info!(path = %bundled.display(), "using bundled sidecar binary");
            cfg.sidecar_bin = Some(bundled.to_string_lossy().into_owned());
        } else {
            // Dev fallback: `cargo tauri dev` runs with CWD = src-tauri/, so the
            // default relative script path (`sidecar/matalu_sidecar.py`) misses.
            // Anchor to the crate dir at compile time → workspace-root/sidecar.
            cfg.sidecar_script =
                concat!(env!("CARGO_MANIFEST_DIR"), "/../sidecar/matalu_sidecar.py").to_string();
        }
    }
    tracing::info!(?cfg, ?mode, "starting matalu pipeline");
    let silence_ms = cfg.silence_ms;

    // Text injection is on by default (gated by the session); MATALU_NO_INJECT
    // disables it for pure UI testing.
    let inject = if std::env::var("MATALU_NO_INJECT").is_ok() {
        None
    } else {
        if !injector::accessibility_trusted() {
            tracing::warn!(
                "Accessibility permission not granted; text injection will silently \
                 no-op until it is enabled in System Settings → Privacy & Security → Accessibility"
            );
        }
        match injector::spawn() {
            Ok(h) => Some(h),
            Err(e) => {
                tracing::error!(error = %e, "failed to start injector; continuing without injection");
                None
            }
        }
    };

    // Session owns the listening state and the audio gate mode (starts Idle =
    // feed nothing, so the sidecar idles until the first dictation).
    let gate_mode = Arc::new(AtomicU8::new(gate::NONE));
    let session = Session::new(app.clone(), inject, gate_mode.clone(), mode, silence_ms);

    // Transcript fan-out: one producer, consumers = the forwarder.
    let (events_tx, events_rx) = broadcast::channel::<TranscriptEvent>(256);
    forward(app, events_rx, session.clone());

    // Demo mode: synthetic events only (still gated by the session, so hold the
    // hotkey to see them type). No mic/model needed — ready immediately.
    if std::env::var("MATALU_DEMO").is_ok() {
        tracing::info!("MATALU_DEMO enabled: emitting synthetic transcript events");
        session.mark_ready();
        spawn_demo(events_tx);
        return Ok(Started { session, child: None });
    }

    // capture --(gate)--> sidecar. Both legs bounded + lossy (drop, never block).
    let (capture_tx, capture_rx) = crossbeam_channel::bounded::<Vec<f32>>(64);
    let (sidecar_tx, sidecar_rx) = crossbeam_channel::bounded::<Vec<f32>>(64);
    spawn_audio_gate(capture_rx, sidecar_tx, gate_mode);

    let corrector = Arc::new(PassThrough);
    let sc = sidecar::spawn(cfg, sidecar_rx, events_tx, corrector)?;
    let sidecar_ready = sc.ready;

    // Gate mic capture on sidecar readiness (model load/warm), bounded, off the
    // UI thread so the window comes up immediately. Once capture is live, mark
    // the session ready so activation is honored (and the indicator stops lying).
    let session_ready = session.clone();
    std::thread::spawn(move || {
        match sidecar_ready.recv_timeout(Duration::from_secs(180)) {
            Ok(()) => tracing::info!("sidecar ready; starting microphone capture"),
            Err(_) => tracing::warn!("sidecar not ready after 180s; starting capture anyway"),
        }
        audio::spawn_capture(TARGET_SAMPLE_RATE, capture_tx);
        session_ready.mark_ready();
    });

    Ok(Started { session, child: Some(sc.child) })
}

/// A bundled sidecar executable sitting next to the app binary in a packaged
/// macOS `.app`, if present. Tauri's `externalBin` copies it into
/// `Contents/MacOS/` (suffix stripped) at bundle time — but it *also* copies it
/// next to the **dev** binary (`target/debug/`) on a plain `cargo build`, so we
/// must gate on actually running from `…/Contents/MacOS/` or `cargo tauri dev`
/// would exec the placeholder stub instead of falling back to the Python script.
fn bundled_sidecar_path() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    if dir.file_name()?.to_str()? != "MacOS" {
        return None; // not a packaged .app (e.g. dev target/debug) — use Python
    }
    let candidate = dir.join("matalu-sidecar");
    candidate.exists().then_some(candidate)
}

/// Forward mic buffers to the sidecar according to the session's [`gate`] mode:
/// [`gate::REAL`] passes mic audio, [`gate::SILENCE`] substitutes a same-length
/// silent buffer (so the sidecar's VAD flushes a trailing `final`), and
/// [`gate::NONE`] forwards *nothing* — the sidecar's blocking read parks and it
/// stops inferring between dictations. Lossy like the capture callback.
fn spawn_audio_gate(
    capture_rx: crossbeam_channel::Receiver<Vec<f32>>,
    sidecar_tx: crossbeam_channel::Sender<Vec<f32>>,
    gate_mode: Arc<AtomicU8>,
) {
    std::thread::Builder::new()
        .name("audio-gate".into())
        .spawn(move || {
            let mut prev = gate::NONE;
            let mut real_samples: u64 = 0;
            let mut peak: f32 = 0.0;
            while let Ok(buf) = capture_rx.recv() {
                let mode = gate_mode.load(Ordering::Relaxed);
                if mode != prev {
                    match mode {
                        gate::REAL => {
                            tracing::debug!("gate: now forwarding real mic audio");
                            real_samples = 0;
                            peak = 0.0;
                        }
                        gate::SILENCE => tracing::debug!("gate: forwarding silence (draining)"),
                        _ => tracing::debug!("gate: idle (forwarding nothing)"),
                    }
                    prev = mode;
                }
                let out = match mode {
                    gate::REAL => {
                        // Track signal level of forwarded audio (debug: mic-level diag).
                        for &s in &buf {
                            peak = peak.max(s.abs());
                        }
                        real_samples += buf.len() as u64;
                        if real_samples >= TARGET_SAMPLE_RATE as u64 / 2 {
                            tracing::debug!(peak, "gate: real-audio level (last ~0.5s, 1.0=max)");
                            real_samples = 0;
                            peak = 0.0;
                        }
                        buf
                    }
                    gate::SILENCE => vec![0.0f32; buf.len()], // silence, same cadence
                    _ => continue,                            // NONE: drop, feed nothing
                };
                let _ = sidecar_tx.try_send(out);
            }
            tracing::info!("audio gate: capture channel closed");
        })
        .expect("spawn audio-gate thread");
}

/// Subscribe to the transcript broadcast; route each event through the session
/// (which drives injection) and mirror it to the UI.
fn forward(app: AppHandle, mut rx: broadcast::Receiver<TranscriptEvent>, session: Arc<Session>) {
    tauri::async_runtime::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(ev) => {
                    match &ev {
                        TranscriptEvent::Partial { text, .. } => {
                            tracing::debug!(kind = "partial", %text, "transcript")
                        }
                        TranscriptEvent::Final { text, .. } => {
                            tracing::debug!(kind = "final", %text, "transcript")
                        }
                    }
                    session.on_event(&ev);
                    if let Err(e) = app.emit(TRANSCRIPT_EVENT, &ev) {
                        tracing::warn!(error = %e, "failed to emit transcript to UI");
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!(skipped = n, "UI forwarder lagged; dropped events");
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
        tracing::info!("transcript broadcast closed; forwarder stopping");
    });
}

/// Emit a looping synthetic utterance so the pipeline can be exercised without
/// audio or the ASR model.
fn spawn_demo(tx: broadcast::Sender<TranscriptEvent>) {
    tauri::async_runtime::spawn(async move {
        let steps = [
            "hello",
            "hello world",
            "hello world this",
            "hello world this is a demo.",
        ];
        let mut ts: u64 = 0;
        loop {
            for (i, text) in steps.iter().enumerate() {
                tokio::time::sleep(Duration::from_millis(400)).await;
                ts += 400;
                let ev = if i == steps.len() - 1 {
                    TranscriptEvent::Final { text: text.to_string(), ts_ms: ts }
                } else {
                    TranscriptEvent::Partial { text: text.to_string(), ts_ms: ts }
                };
                let _ = tx.send(ev);
            }
        }
    });
}

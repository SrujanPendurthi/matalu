//! Wire the `matalu` core pipeline into the Tauri app: mic → **audio gate** →
//! sidecar → transcript fan-out → ([`Session`] → injector) + UI.
//!
//! The gate sits between capture and the sidecar so the [`Session`] can decide,
//! per buffer, whether the model hears real audio (listening) or silence (idle).

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use matalu::config::{Config, TARGET_SAMPLE_RATE};
use matalu::events::TranscriptEvent;
use matalu::{audio, sidecar};
use tauri::{AppHandle, Emitter};
use tokio::sync::broadcast;

use crate::cleaner::Cleaner;
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

/// Records the merged 16 kHz mono meeting audio to a WAV for post-hoc
/// diarization, and tracks meeting-relative elapsed samples. Shared between the
/// audio-gate thread (writes) and the [`Session`] (start/stop + elapsed time).
pub struct MeetingRecorder {
    writer: Mutex<Option<hound::WavWriter<BufWriter<File>>>>,
    path: Mutex<Option<std::path::PathBuf>>,
    samples: AtomicU64,
}

impl MeetingRecorder {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            writer: Mutex::new(None),
            path: Mutex::new(None),
            samples: AtomicU64::new(0),
        })
    }

    /// Begin recording to `path` (overwrites); resets the sample counter.
    pub fn start(&self, path: &Path) -> anyhow::Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: TARGET_SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let writer = hound::WavWriter::create(path, spec)?;
        self.samples.store(0, Ordering::Relaxed);
        *self.path.lock().unwrap() = Some(path.to_path_buf());
        *self.writer.lock().unwrap() = Some(writer);
        Ok(())
    }

    /// Append mono f32 samples (called from the gate thread on `REAL` audio).
    /// No-op unless recording.
    ///
    /// ponytail: this does buffered file I/O on the gate thread — cheap via
    /// `BufWriter`; a disk stall would slow forwarding, which the lossy sidecar
    /// channel absorbs. Fine for personal meeting capture; move to a dedicated
    /// writer thread if it ever bites.
    pub fn write(&self, buf: &[f32]) {
        let mut guard = self.writer.lock().unwrap();
        if let Some(w) = guard.as_mut() {
            for &s in buf {
                let _ = w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16);
            }
            self.samples.fetch_add(buf.len() as u64, Ordering::Relaxed);
        }
    }

    /// Finalize the WAV and stop recording; returns the written path (if any).
    pub fn stop(&self) -> Option<std::path::PathBuf> {
        if let Some(w) = self.writer.lock().unwrap().take() {
            let _ = w.finalize();
        }
        self.path.lock().unwrap().take()
    }

    /// Meeting-relative elapsed milliseconds (from samples recorded so far).
    pub fn elapsed_ms(&self) -> u64 {
        self.samples.load(Ordering::Relaxed) * 1000 / TARGET_SAMPLE_RATE as u64
    }
}

/// What [`start`] hands back to the app: the session (also placed in managed
/// state so the hotkey handler can reach it) and the sidecar child to keep alive.
pub struct Started {
    pub session: Arc<Session>,
    pub child: Option<std::process::Child>,
}

/// Start the pipeline and return the [`Session`] + sidecar child. `mode` is the
/// initial activation mode (from persisted settings / env override).
pub fn start(app: AppHandle, mode: Mode, cleanup: bool) -> anyhow::Result<Started> {
    let mut cfg = Config::from_env();
    // Prefer a bundled sidecar binary shipped next to the app executable
    // (packaged build) unless the dev env vars pin python/script explicitly.
    if cfg.sidecar_bin.is_none() && std::env::var_os("MATALU_SIDECAR").is_none() {
        if let Some(bundled) = bundled_binary("matalu-sidecar") {
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
    let input_device = cfg.input_device.clone();

    // Text injection is on by default (gated by the session); MATALU_NO_INJECT
    // disables it for pure UI testing.
    let inject = if std::env::var("MATALU_NO_INJECT").is_ok() {
        None
    } else {
        // Ask, don't just log. Untrusted means every keystroke is discarded with
        // no error anywhere the user can see — the app transcribes perfectly and
        // types nothing. The system dialog has a direct "Open System Settings"
        // button, which is the only self-service path out of that state.
        if !injector::prompt_for_accessibility() {
            tracing::warn!(
                "Accessibility permission not granted; text injection will silently \
                 no-op until it is enabled in System Settings → Privacy & Security → \
                 Accessibility. Prompted the user; a restart may be needed after granting."
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

    // Downstream filler/grammar cleanup. Best-effort and lazily loaded: it warms
    // on hotkey press and unloads when idle, so it costs no resident RAM between
    // dictations. Skipped for MATALU_NO_CLEANUP and for demo mode (which exists
    // to exercise the UI without a mic or a model), both of which fall back to
    // the raw live-partial path.
    let no_cleanup =
        std::env::var("MATALU_NO_CLEANUP").is_ok() || std::env::var("MATALU_DEMO").is_ok();
    let cleaner = if no_cleanup { None } else { Some(Cleaner::new()) };

    // Session owns the listening state and the audio gate mode (starts Idle =
    // feed nothing, so the sidecar idles until the first dictation).
    let gate_mode = Arc::new(AtomicU8::new(gate::NONE));
    let recorder = MeetingRecorder::new();
    let session = Session::new(
        app.clone(),
        inject,
        cleaner,
        cleanup,
        gate_mode.clone(),
        recorder.clone(),
        mode,
        silence_ms,
    );

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
    // The gate runs for the life of the app. Its drop counter is diagnostic —
    // asserted on in the tests below, and the only evidence a drop leaves.
    let (_gate_drops, _gate) =
        spawn_audio_gate(capture_rx, sidecar_tx, gate_mode, recorder, silence_ms);

    let sc = sidecar::spawn(cfg, sidecar_rx, events_tx)?;
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
        audio::spawn_capture(TARGET_SAMPLE_RATE, input_device, capture_tx);
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
///
/// Shared by both sidecars (`matalu-sidecar`, `matalu-cleaner`).
pub(crate) fn bundled_binary(name: &str) -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    if dir.file_name()?.to_str()? != "MacOS" {
        return None; // not a packaged .app (e.g. dev target/debug) — use Python
    }
    let candidate = dir.join(name);
    candidate.exists().then_some(candidate)
}

/// The sidecar consumes audio in 0.5 s units (`CHUNK = SR // 2`), so its VAD
/// counts silence in 500 ms steps.
const SIDECAR_STEP_MS: u64 = 500;

/// Forward mic buffers to the sidecar according to the session's [`gate`] mode:
/// [`gate::REAL`] passes mic audio, [`gate::SILENCE`] substitutes a same-length
/// silent buffer (so the sidecar's VAD flushes a trailing `final`), and
/// [`gate::NONE`] forwards *nothing* — the sidecar's blocking read parks and it
/// stops inferring between dictations. Lossy like the capture callback.
///
/// On entering [`gate::SILENCE`] the drain silence is **burst** rather than paced
/// at mic cadence — see [`burst_drain_silence`]. That is the single biggest term
/// in the post-release wait: measured 2017 ms paced vs 407 ms burst.
fn spawn_audio_gate(
    capture_rx: crossbeam_channel::Receiver<Vec<f32>>,
    sidecar_tx: crossbeam_channel::Sender<Vec<f32>>,
    gate_mode: Arc<AtomicU8>,
    recorder: Arc<MeetingRecorder>,
    silence_ms: u64,
) -> (Arc<AtomicU64>, std::thread::JoinHandle<()>) {
    let dropped = Arc::new(AtomicU64::new(0));
    let dropped_thread = Arc::clone(&dropped);
    let handle = std::thread::Builder::new()
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
                        gate::SILENCE => {
                            tracing::debug!("gate: draining (bursting silence)");
                            // `buf` was captured before the flip — it is part of
                            // the user's last words, so it goes through as real
                            // audio ahead of the burst.
                            recorder.write(&buf);
                            forward_to_sidecar(&sidecar_tx, buf, &dropped_thread);
                            burst_drain_silence(&capture_rx, &sidecar_tx, &recorder, silence_ms);
                            prev = mode;
                            continue;
                        }
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
                        // Persist meeting audio for post-hoc diarization (no-op
                        // unless a meeting is recording).
                        recorder.write(&buf);
                        buf
                    }
                    // The burst above already crossed the VAD threshold; this
                    // just keeps the cadence until the session leaves Draining,
                    // and covers a burst buffer the lossy channel dropped.
                    gate::SILENCE => vec![0.0f32; buf.len()],
                    _ => continue, // NONE: drop, feed nothing
                };
                forward_to_sidecar(&sidecar_tx, out, &dropped_thread);
            }
            tracing::info!("audio gate: capture channel closed");
        })
        .expect("spawn audio-gate thread");
    (dropped, handle)
}

/// Number of zero samples to burst so the sidecar's VAD crosses `SILENCE_MS`.
/// It counts in [`SIDECAR_STEP_MS`] steps and fires at `>=`, so this is the
/// steps needed plus one for the partially-filled step at the boundary.
fn drain_burst_samples(silence_ms: u64) -> usize {
    let steps = silence_ms.div_ceil(SIDECAR_STEP_MS) + 1;
    (steps * SIDECAR_STEP_MS * TARGET_SAMPLE_RATE as u64 / 1000) as usize
}

/// Forward one buffer to the sidecar, counting buffers the queue could not take.
///
/// `try_send`, never `send`: the gate must not block, so a full queue drops the
/// buffer rather than stalling capture behind it. What it must *not* do is drop
/// it silently — in meeting mode the gate is the only path to the transcript,
/// so a discarded buffer is speech that never reaches the model while the
/// meeting WAV keeps it (`MeetingRecorder::write` runs first). The transcript
/// would just be missing words, with nothing to say so.
///
/// Measured 2026-09-04: this never fires on this hardware. `finalize()` runs at
/// ~28x realtime (431 ms for a 12 s utterance, the `MATALU_MAX_UTTERANCE_MS`
/// cap), so the sidecar stays ~0.5 s *ahead* of a realtime feed and the 64-slot
/// queue plus the 64 KB stdin pipe (1.02 s @16 kHz) is never approached. The
/// counter exists because that margin is hardware- and model-dependent and
/// nothing else would report it shrinking. Same idiom as the other lossy hop,
/// `matalu::audio`'s capture callback.
fn forward_to_sidecar(
    sidecar_tx: &crossbeam_channel::Sender<Vec<f32>>,
    buf: Vec<f32>,
    dropped: &AtomicU64,
) {
    if sidecar_tx.try_send(buf).is_err() {
        let n = dropped.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 100 == 1 {
            tracing::warn!(dropped = n, "sidecar behind; dropping audio buffers");
        }
    }
}

/// Push the whole drain silence at once instead of pacing it at mic cadence.
///
/// Pacing silence in real time is an artifact of reusing the capture loop, not a
/// requirement: Parakeet's streaming decode is sample-driven, not clock-driven,
/// so the same zeros delivered faster yield the same transcript — this is an
/// exact optimization, not a heuristic. Measured 2017 ms → 407 ms from release to
/// the trailing `final`.
///
/// **Ordering matters and is the whole risk here.** Buffers still sitting in
/// `capture_rx` when the gate flipped were captured *before* release — they are
/// the user's last words. Bursting silence ahead of them would reorder the
/// stream, so the VAD would finalize early and those words would be lost (the
/// gate goes `NONE` moments later) or split into the next utterance. So the
/// queue is flushed as real audio first.
fn burst_drain_silence(
    capture_rx: &crossbeam_channel::Receiver<Vec<f32>>,
    sidecar_tx: &crossbeam_channel::Sender<Vec<f32>>,
    recorder: &MeetingRecorder,
    silence_ms: u64,
) {
    let mut flushed = 0usize;
    while let Ok(pending) = capture_rx.try_recv() {
        flushed += pending.len();
        recorder.write(&pending);
        let _ = sidecar_tx.try_send(pending);
    }

    // One buffer per sidecar step, so a dropped `try_send` costs only one step.
    let total = drain_burst_samples(silence_ms);
    let per_step = (SIDECAR_STEP_MS * TARGET_SAMPLE_RATE as u64 / 1000) as usize;
    let mut sent = 0usize;
    while sent < total {
        let n = per_step.min(total - sent);
        if sidecar_tx.try_send(vec![0.0f32; n]).is_err() {
            // Never block the gate. The paced silence path still runs after
            // this, so the VAD flushes anyway — just at the old speed.
            tracing::warn!(sent, total, "drain burst truncated; sidecar queue full");
            break;
        }
        sent += n;
    }
    tracing::debug!(flushed_real = flushed, silence = sent, "gate: drain burst");
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors the sidecar's VAD: it accumulates `SIDECAR_STEP_MS` per whole
    /// chunk read and finalizes at `silence_ms >= SILENCE_MS`. The burst must
    /// carry enough samples to reach that, or the drain silently falls back to
    /// the slow paced path.
    fn steps_until_final(burst_samples: usize, silence_ms: u64) -> Option<u64> {
        let per_step = (SIDECAR_STEP_MS * TARGET_SAMPLE_RATE as u64 / 1000) as usize;
        let mut accumulated = 0u64;
        for step in 1..=(burst_samples / per_step) as u64 {
            accumulated += SIDECAR_STEP_MS;
            if accumulated >= silence_ms {
                return Some(step);
            }
        }
        None
    }

    #[test]
    fn burst_always_crosses_the_vad_threshold() {
        for silence_ms in [200, 500, 700, 1000, 1500, 2000, 3000] {
            let samples = drain_burst_samples(silence_ms);
            assert!(
                steps_until_final(samples, silence_ms).is_some(),
                "burst of {samples} samples never finalizes at silence_ms={silence_ms}"
            );
        }
    }

    #[test]
    fn burst_does_not_massively_overshoot() {
        // One spare step of headroom past the threshold, no more — every extra
        // step is a full chunk of wasted inference on the critical path.
        for silence_ms in [500, 700, 1000] {
            let samples = drain_burst_samples(silence_ms);
            let step = steps_until_final(samples, silence_ms).unwrap();
            let per_step = (SIDECAR_STEP_MS * TARGET_SAMPLE_RATE as u64 / 1000) as usize;
            let sent_steps = (samples / per_step) as u64;
            assert_eq!(sent_steps, step + 1, "silence_ms={silence_ms}");
        }
    }

    #[test]
    fn default_silence_bursts_1500ms() {
        // 700 ms threshold -> 2 steps to cross, +1 spare = 1.5 s @ 16 kHz.
        assert_eq!(drain_burst_samples(700), 24_000);
    }

    /// The correctness risk of bursting: audio already queued when the gate
    /// flips was captured *before* release and is the user's last words. If
    /// silence jumped ahead of it the VAD would finalize early and those words
    /// would be lost — the gate goes NONE moments later.
    #[test]
    fn queued_real_audio_is_flushed_before_the_silence_burst() {
        let (cap_tx, cap_rx) = crossbeam_channel::bounded::<Vec<f32>>(64);
        let (side_tx, side_rx) = crossbeam_channel::bounded::<Vec<f32>>(64);
        let recorder = MeetingRecorder::new();

        // Two buffers of real speech still in flight at the moment of release.
        cap_tx.send(vec![0.5f32; 100]).unwrap();
        cap_tx.send(vec![-0.5f32; 100]).unwrap();

        burst_drain_silence(&cap_rx, &side_tx, &recorder, 700);
        drop(side_tx);

        let forwarded: Vec<Vec<f32>> = side_rx.iter().collect();
        let real: Vec<&Vec<f32>> =
            forwarded.iter().filter(|b| b.iter().any(|s| *s != 0.0)).collect();
        assert_eq!(real.len(), 2, "both queued speech buffers must be forwarded");

        let first_silence = forwarded.iter().position(|b| b.iter().all(|s| *s == 0.0)).unwrap();
        let last_real = forwarded.iter().rposition(|b| b.iter().any(|s| *s != 0.0)).unwrap();
        assert!(
            last_real < first_silence,
            "silence at index {first_silence} jumped ahead of real audio at {last_real}"
        );

        let silence: usize =
            forwarded[first_silence..].iter().map(|b| b.len()).sum();
        assert_eq!(silence, drain_burst_samples(700));
    }

    #[test]
    fn burst_with_nothing_queued_sends_only_silence() {
        let (_cap_tx, cap_rx) = crossbeam_channel::bounded::<Vec<f32>>(64);
        let (side_tx, side_rx) = crossbeam_channel::bounded::<Vec<f32>>(64);
        burst_drain_silence(&cap_rx, &side_tx, &MeetingRecorder::new(), 700);
        drop(side_tx);
        let forwarded: Vec<Vec<f32>> = side_rx.iter().collect();
        assert!(forwarded.iter().all(|b| b.iter().all(|s| *s == 0.0)));
        assert_eq!(forwarded.iter().map(|b| b.len()).sum::<usize>(), 24_000);
    }

    /// The gate must never block, so a full downstream queue truncates the
    /// burst rather than waiting. The paced silence path then still flushes.
    #[test]
    fn full_sidecar_queue_truncates_instead_of_blocking() {
        let (_cap_tx, cap_rx) = crossbeam_channel::bounded::<Vec<f32>>(64);
        let (side_tx, side_rx) = crossbeam_channel::bounded::<Vec<f32>>(1);
        burst_drain_silence(&cap_rx, &side_tx, &MeetingRecorder::new(), 700);
        assert_eq!(side_rx.len(), 1, "should have stopped at the queue limit");
    }

    #[test]
    fn dropped_buffers_are_counted_not_silent() {
        let (side_tx, _side_rx) = crossbeam_channel::bounded::<Vec<f32>>(1);
        let dropped = AtomicU64::new(0);

        forward_to_sidecar(&side_tx, vec![0.1f32; 10], &dropped);
        assert_eq!(dropped.load(Ordering::Relaxed), 0, "a buffer that fits is not a drop");

        for _ in 0..5 {
            forward_to_sidecar(&side_tx, vec![0.1f32; 10], &dropped);
        }
        assert_eq!(dropped.load(Ordering::Relaxed), 5, "every buffer past capacity counts");
    }

    /// The counter is the only evidence a dropped buffer leaves, so the *gate*
    /// has to keep routing through it — a unit test on `forward_to_sidecar`
    /// alone would still pass if the gate went back to a bare `let _ =
    /// try_send`. Drives the real thread and asserts on what it counted.
    #[test]
    fn gate_counts_buffers_the_sidecar_queue_rejects() {
        let (cap_tx, cap_rx) = crossbeam_channel::bounded::<Vec<f32>>(64);
        let (side_tx, _side_rx) = crossbeam_channel::bounded::<Vec<f32>>(2);
        let (dropped, gate) = spawn_audio_gate(
            cap_rx,
            side_tx,
            Arc::new(AtomicU8::new(gate::REAL)),
            MeetingRecorder::new(),
            700,
        );

        // Nothing drains `_side_rx`, so the queue takes 2 and rejects the rest.
        for _ in 0..12 {
            cap_tx.send(vec![0.5f32; 10]).unwrap();
        }
        drop(cap_tx); // closing capture ends the gate loop, so join is deterministic
        gate.join().expect("gate thread panicked");

        assert_eq!(
            dropped.load(Ordering::Relaxed),
            10,
            "12 buffers into a queue of 2 must count 10 drops"
        );
    }
}

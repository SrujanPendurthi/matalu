//! Dictation session state machine: turns hotkey presses into "listening"
//! windows, gates whether real audio reaches the sidecar, and routes transcript
//! events to the text injector.
//!
//! States:
//! - **Idle**: the audio gate feeds *silence* to the sidecar (keeps it warm and
//!   lets its VAD stay reset); transcript events are ignored — no typing.
//! - **Listening**: real audio flows; partials type live, a VAD `final` commits
//!   an utterance mid-session.
//! - **Draining**: the user ended the session; we switch the gate back to
//!   silence but keep routing to the injector briefly so the sidecar's trailing
//!   `final` (flushed by the incoming silence) commits the last words. A
//!   watchdog forces Idle if that `final` never arrives.
//!
//! Idle feeds *silence* rather than cutting the stream so we never send real
//! speech to the model when the user isn't dictating (privacy), while the
//! sidecar's existing silence-VAD segmentation keeps working unchanged.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use matalu::events::TranscriptEvent;
use tauri::{AppHandle, Emitter, Manager};

use crate::diarize::{self, Line};
use crate::injector::InjectorHandle;
use crate::pipeline::{gate, MeetingRecorder};

/// Tauri event carrying `{ "listening": bool }` for the UI status indicator.
pub const STATUS_EVENT: &str = "status";

/// How the hotkey drives sessions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Hold to talk: press starts, release ends.
    PushToTalk,
    /// Tap to toggle listening on/off.
    Toggle,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Idle,
    Listening,
    Draining,
}

pub struct Session {
    state: Mutex<State>,
    mode: Mutex<Mode>,
    /// True once the sidecar is warm and mic capture is live. Activation before
    /// this is ignored (the pipeline can't hear anything yet).
    ready: Arc<AtomicBool>,
    /// Meeting-transcript mode: capture runs continuously, transcripts fill the
    /// window but are **not** injected, and the dictation hotkey is ignored.
    meeting: AtomicBool,
    /// Records the meeting audio to a WAV and tracks meeting-relative time, for
    /// post-hoc diarization on stop.
    recorder: Arc<MeetingRecorder>,
    /// Committed meeting lines with timing, accumulated while `meeting`.
    meeting_lines: Mutex<Vec<Line>>,
    /// Read by the audio gate thread; one of [`gate::NONE`]/[`gate::SILENCE`]/
    /// [`gate::REAL`]. Drives what the sidecar hears per dictation phase.
    gate: Arc<AtomicU8>,
    /// None when injection is disabled/unavailable (UI still updates).
    injector: Option<InjectorHandle>,
    app: AppHandle,
    /// Last uncommitted partial text (for the drain watchdog fallback).
    last_partial: Mutex<Option<String>>,
    /// Invalidates stale drain watchdogs across rapid start/stop.
    drain_gen: AtomicU64,
    /// How long to wait in Draining for the trailing `final` before forcing Idle.
    drain_timeout: Duration,
}

impl Session {
    pub fn new(
        app: AppHandle,
        injector: Option<InjectorHandle>,
        gate: Arc<AtomicU8>,
        recorder: Arc<MeetingRecorder>,
        mode: Mode,
        silence_ms: u64,
    ) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State::Idle),
            mode: Mutex::new(mode),
            ready: Arc::new(AtomicBool::new(false)),
            meeting: AtomicBool::new(false),
            recorder,
            meeting_lines: Mutex::new(Vec::new()),
            gate,
            injector,
            app,
            last_partial: Mutex::new(None),
            drain_gen: AtomicU64::new(0),
            // Give the sidecar time to flush a trailing `final` after we start
            // feeding silence (its VAD needs SILENCE_MS of quiet first).
            drain_timeout: Duration::from_millis(silence_ms + 800),
        })
    }

    pub fn set_mode(&self, mode: Mode) {
        *self.mode.lock().unwrap() = mode;
    }

    /// Mark the pipeline ready (sidecar warm + mic capturing). Until this is
    /// called, activation is ignored so the "listening" indicator can't lie.
    pub fn mark_ready(&self) {
        self.ready.store(true, Ordering::Relaxed);
        let _ = self
            .app
            .emit(STATUS_EVENT, serde_json::json!({ "listening": false, "state": "ready" }));
        tracing::info!("session: ready (mic live, model warm)");
    }

    // --- meeting transcript mode ---------------------------------------------

    pub fn is_meeting(&self) -> bool {
        self.meeting.load(Ordering::Relaxed)
    }

    /// Start meeting transcription: capture continuously (gate always `REAL`),
    /// show the transcript window, and stop injecting. No-op until the pipeline
    /// is ready (mic live + model warm).
    pub fn start_meeting(self: &Arc<Self>) {
        if !self.ready.load(Ordering::Relaxed) {
            tracing::warn!("meeting ignored: pipeline still warming up (model loading)");
            let _ = self.app.emit(
                STATUS_EVENT,
                serde_json::json!({ "listening": false, "state": "warming" }),
            );
            return;
        }
        self.meeting.store(true, Ordering::Relaxed);
        *self.state.lock().unwrap() = State::Listening;
        self.meeting_lines.lock().unwrap().clear();
        // Record the meeting audio for post-hoc diarization (best-effort — if
        // recording can't start, the meeting still transcribes, just unlabeled).
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let wav = std::env::temp_dir().join(format!("matalu-meeting-{ts}.wav"));
        if let Err(e) = self.recorder.start(&wav) {
            tracing::warn!(error = %e, "failed to start meeting recorder; transcript will be unlabeled");
        }
        self.gate.store(gate::REAL, Ordering::Relaxed);
        if let Some(win) = self.app.get_webview_window("main") {
            let _ = win.show();
            let _ = win.set_focus();
        }
        let _ = self
            .app
            .emit(STATUS_EVENT, serde_json::json!({ "listening": true, "state": "meeting" }));
        tracing::info!("session: meeting transcript started");
    }

    /// Stop meeting transcription: idle the capture, keep the window up, and
    /// kick off post-hoc diarization on a background thread (loads models,
    /// takes seconds) which emits the labeled transcript when done.
    pub fn stop_meeting(self: &Arc<Self>) {
        self.meeting.store(false, Ordering::Relaxed);
        *self.state.lock().unwrap() = State::Idle;
        self.gate.store(gate::NONE, Ordering::Relaxed);
        self.emit_status(false);
        tracing::info!("session: meeting transcript stopped");

        let wav = self.recorder.stop();
        let lines = std::mem::take(&mut *self.meeting_lines.lock().unwrap());
        match wav {
            Some(wav) if !lines.is_empty() => {
                let app = self.app.clone();
                std::thread::spawn(move || diarize::run(app, wav, lines));
            }
            _ => tracing::info!("no meeting audio/lines to diarize"),
        }
    }

    // --- hotkey entry points -------------------------------------------------

    /// Hotkey pressed.
    pub fn on_press(self: &Arc<Self>) {
        if self.meeting.load(Ordering::Relaxed) {
            return; // dictation hotkey is inert while a meeting is recording
        }
        let mode = *self.mode.lock().unwrap();
        match mode {
            Mode::PushToTalk => self.start(),
            Mode::Toggle => {
                let listening = matches!(*self.state.lock().unwrap(), State::Listening);
                if listening {
                    self.stop();
                } else {
                    self.start();
                }
            }
        }
    }

    /// Hotkey released (only meaningful for push-to-talk).
    pub fn on_release(self: &Arc<Self>) {
        if self.meeting.load(Ordering::Relaxed) {
            return;
        }
        if *self.mode.lock().unwrap() == Mode::PushToTalk {
            self.stop();
        }
    }

    // --- transitions ---------------------------------------------------------

    fn start(self: &Arc<Self>) {
        if !self.ready.load(Ordering::Relaxed) {
            tracing::warn!("activation ignored: pipeline still warming up (model loading)");
            let _ = self.app.emit(
                STATUS_EVENT,
                serde_json::json!({ "listening": false, "state": "warming" }),
            );
            return;
        }
        let mut st = self.state.lock().unwrap();
        if *st == State::Listening {
            return; // idempotent: ignore key-repeat / double-press
        }
        *st = State::Listening;
        self.drain_gen.fetch_add(1, Ordering::Relaxed); // cancel any pending drain
        *self.last_partial.lock().unwrap() = None;
        if let Some(i) = &self.injector {
            i.reset();
        }
        self.gate.store(gate::REAL, Ordering::Relaxed);
        drop(st);
        self.emit_status(true);
        self.set_pill(true);
        tracing::info!("session: listening");
    }

    fn stop(self: &Arc<Self>) {
        let mut st = self.state.lock().unwrap();
        if *st != State::Listening {
            return;
        }
        *st = State::Draining;
        self.gate.store(gate::SILENCE, Ordering::Relaxed); // sidecar now gets silence
        let gen = self.drain_gen.fetch_add(1, Ordering::Relaxed) + 1;
        drop(st);
        self.emit_status(false);
        tracing::info!("session: draining");
        self.spawn_drain_watchdog(gen);
    }

    /// After `drain_timeout`, if still Draining under the same generation, commit
    /// whatever partial we last saw and force Idle (the trailing `final` was lost).
    fn spawn_drain_watchdog(self: &Arc<Self>, gen: u64) {
        let me = Arc::clone(self);
        let timeout = self.drain_timeout;
        std::thread::spawn(move || {
            std::thread::sleep(timeout);
            let mut st = me.state.lock().unwrap();
            if *st != State::Draining || me.drain_gen.load(Ordering::Relaxed) != gen {
                return; // superseded by a newer session, or already finished
            }
            *st = State::Idle;
            me.gate.store(gate::NONE, Ordering::Relaxed); // stop feeding the sidecar
            drop(st);
            if let Some(i) = &me.injector {
                if let Some(text) = me.last_partial.lock().unwrap().take() {
                    i.commit(text);
                } else {
                    i.reset();
                }
            }
            me.emit_status(false);
            me.set_pill(false);
            tracing::info!("session: drain watchdog forced idle");
        });
    }

    // --- transcript routing --------------------------------------------------

    /// Feed a transcript event through the session. No-op while Idle.
    pub fn on_event(&self, ev: &TranscriptEvent) {
        if self.meeting.load(Ordering::Relaxed) {
            // Meeting mode: the UI receives every event via pipeline::forward()
            // (no injection). Record each committed line with its meeting-relative
            // time span so post-hoc diarization can label it by speaker.
            if let TranscriptEvent::Final { text, .. } = ev {
                let end_ms = self.recorder.elapsed_ms();
                let mut lines = self.meeting_lines.lock().unwrap();
                let start_ms = lines.last().map(|l| l.end_ms).unwrap_or(0);
                lines.push(Line { start_ms, end_ms, text: text.clone() });
            }
            return;
        }
        let mut st = self.state.lock().unwrap();
        if *st == State::Idle {
            return;
        }
        match ev {
            TranscriptEvent::Partial { text, .. } => {
                if let Some(i) = &self.injector {
                    i.partial(text.clone());
                }
                *self.last_partial.lock().unwrap() = Some(text.clone());
            }
            TranscriptEvent::Final { text, .. } => {
                if let Some(i) = &self.injector {
                    i.commit(text.clone());
                }
                *self.last_partial.lock().unwrap() = None;
                if *st == State::Draining {
                    *st = State::Idle;
                    self.gate.store(gate::NONE, Ordering::Relaxed); // stop feeding the sidecar
                    self.drain_gen.fetch_add(1, Ordering::Relaxed);
                    drop(st);
                    self.emit_status(false);
                    self.set_pill(false);
                    tracing::info!("session: idle (trailing final committed)");
                }
                // In Listening, a VAD final just segments an utterance; stay on.
            }
        }
    }

    fn emit_status(&self, listening: bool) {
        let _ = self
            .app
            .emit(STATUS_EVENT, serde_json::json!({ "listening": listening }));
    }

    /// Show/hide the floating pill overlay. Visible for the whole dictation
    /// window (Listening + Draining); hidden once we return to Idle.
    fn set_pill(&self, visible: bool) {
        if let Some(win) = self.app.get_webview_window("pill") {
            let _ = if visible { win.show() } else { win.hide() };
        }
    }
}

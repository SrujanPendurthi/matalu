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
//!
//! ## Buffered cleanup (the Wispr-Flow shape)
//!
//! With the [`crate::cleaner`] stage enabled, nothing types while you speak.
//! Committed utterances accumulate in [`Session::utterance_buf`] and the whole
//! press→release window is cleaned and pasted as **one** edit on release. That
//! full-window context is what lets a self-correction spanning two sentences be
//! repaired at all — "ship it Tuesday, no wait, Thursday" is unfixable once
//! "Tuesday" has already been typed. The UI and pill still show raw partials
//! live (they read the app-wide `transcript` emit), so there is still feedback.
//!
//! With cleanup off, the original live-partial behavior is used verbatim.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use matalu::events::TranscriptEvent;
use tauri::{AppHandle, Emitter, Manager};

use crate::cleaner::Cleaner;
use crate::diarize::{self, Line};
use crate::injector::InjectorHandle;
use crate::pipeline::{gate, MeetingRecorder};

/// Flush and paste mid-session once the buffer passes this many characters, so
/// a long dictation isn't minutes of nothing appearing. Costs the full-context
/// benefit at the seam, which is why it sits well above a normal utterance
/// (~600 chars is roughly 100 words, or 40 s of continuous speech).
///
/// It also bounds cleanup latency: generation cost scales with length, so at
/// this cap a flush cleans in ~4 s rather than the ~13 s a 2000-char buffer
/// would need. See `TIMEOUT_PER_CHAR` in [`crate::cleaner`].
const MAX_BUFFER_CHARS: usize = 600;

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
    /// Downstream filler/grammar cleanup. `None` when unavailable — the
    /// dictation then types live and raw, exactly as before this stage existed.
    cleaner: Option<Arc<Cleaner>>,
    /// User setting. Off restores the live-partial behavior even with a cleaner.
    cleanup: AtomicBool,
    /// Committed utterances for the current dictation, joined and cleaned as one
    /// unit on release. Only used while cleanup is active.
    utterance_buf: Mutex<Vec<String>>,
    app: AppHandle,
    /// Last uncommitted partial text (for the drain watchdog fallback).
    last_partial: Mutex<Option<String>>,
    /// Invalidates stale drain watchdogs across rapid start/stop.
    drain_gen: AtomicU64,
    /// How long to wait in Draining for the trailing `final` before forcing Idle.
    drain_timeout: Duration,
}

impl Session {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        app: AppHandle,
        injector: Option<InjectorHandle>,
        cleaner: Option<Arc<Cleaner>>,
        cleanup: bool,
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
            cleaner,
            cleanup: AtomicBool::new(cleanup),
            utterance_buf: Mutex::new(Vec::new()),
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

    pub fn set_cleanup(&self, on: bool) {
        self.cleanup.store(on, Ordering::Relaxed);
    }

    /// Whether this dictation buffers for cleanup instead of typing live.
    /// Requires both a working cleaner and the user setting.
    fn cleanup_on(&self) -> bool {
        self.cleaner.is_some() && self.cleanup.load(Ordering::Relaxed)
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
        self.utterance_buf.lock().unwrap().clear();
        if let Some(i) = &self.injector {
            i.reset();
        }
        // Load the cleanup model now: the seconds spent speaking hide the load,
        // so the cleanup at the end of this utterance is already warm.
        if let Some(c) = &self.cleaner {
            c.warm();
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
            // The trailing `final` never arrived; the last partial is the best
            // record of what was said. Route it through the same exit as the
            // normal path so buffered text still gets cleaned and pasted.
            let salvaged = me.last_partial.lock().unwrap().take();
            match (me.cleanup_on(), salvaged) {
                (true, Some(text)) => me.utterance_buf.lock().unwrap().push(text),
                (false, Some(text)) => {
                    if let Some(i) = &me.injector {
                        i.commit(text);
                    }
                }
                (_, None) => {
                    if let Some(i) = &me.injector {
                        i.reset();
                    }
                }
            }
            me.emit_status(false);
            me.finish_dictation();
            tracing::info!("session: drain watchdog forced idle");
        });
    }

    /// The single exit from a dictation window, reached from both the trailing
    /// `final` and the drain watchdog. With cleanup on, joins the buffered
    /// utterances, cleans them, and pastes the result as one edit; otherwise the
    /// text is already in the target app and this only hides the pill.
    ///
    /// Runs the clean off-thread — it blocks on the model for a few hundred ms.
    fn finish_dictation(self: &Arc<Self>) {
        let Some(cleaner) = self.cleaner.clone().filter(|_| self.cleanup_on()) else {
            self.set_pill(false);
            return;
        };
        let raw = self.take_buffer();
        if raw.trim().is_empty() {
            self.set_pill(false);
            return;
        }
        self.spawn_clean_and_paste(cleaner, raw, true);
    }

    /// Mid-session flush so a long dictation isn't minutes of nothing appearing.
    /// Same clean-and-paste path, but the session stays live.
    fn flush_long_session(self: &Arc<Self>) {
        let Some(cleaner) = self.cleaner.clone().filter(|_| self.cleanup_on()) else {
            return;
        };
        let raw = self.take_buffer();
        if raw.trim().is_empty() {
            return;
        }
        tracing::info!(chars = raw.chars().count(), "session: flushing long dictation");
        self.spawn_clean_and_paste(cleaner, raw, false);
    }

    fn take_buffer(&self) -> String {
        std::mem::take(&mut *self.utterance_buf.lock().unwrap()).join(" ")
    }

    /// Clean `raw` on a worker thread and paste the result. A rejected or failed
    /// clean pastes `raw` unchanged — the user's words are never dropped.
    fn spawn_clean_and_paste(
        self: &Arc<Self>,
        cleaner: Arc<Cleaner>,
        raw: String,
        hide_pill: bool,
    ) {
        let me = Arc::clone(self);
        std::thread::spawn(move || {
            let _ = me.app.emit(
                STATUS_EVENT,
                serde_json::json!({ "listening": false, "state": "cleaning" }),
            );
            let started = std::time::Instant::now();
            let cleaned = cleaner.clean(&raw);
            let used_cleanup = cleaned.is_some();
            log_pair(&raw, cleaned.as_deref());
            let text = cleaned.unwrap_or(raw);
            tracing::info!(
                ms = started.elapsed().as_millis() as u64,
                cleaned = used_cleanup,
                "dictation cleanup finished"
            );
            if let Some(i) = &me.injector {
                i.commit(text);
            }
            if hide_pill {
                me.set_pill(false);
            }
        });
    }

    // --- transcript routing --------------------------------------------------

    /// Feed a transcript event through the session. No-op while Idle.
    pub fn on_event(self: &Arc<Self>, ev: &TranscriptEvent) {
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
        let buffering = self.cleanup_on();
        match ev {
            TranscriptEvent::Partial { text, .. } => {
                // While buffering, partials stay off the target app — the whole
                // window is pasted once at the end. They still reach the UI and
                // pill via the app-wide emit in `pipeline::forward`.
                if !buffering {
                    if let Some(i) = &self.injector {
                        i.partial(text.clone());
                    }
                }
                *self.last_partial.lock().unwrap() = Some(text.clone());
            }
            TranscriptEvent::Final { text, .. } => {
                if buffering {
                    self.utterance_buf.lock().unwrap().push(text.clone());
                } else if let Some(i) = &self.injector {
                    i.commit(text.clone());
                }
                *self.last_partial.lock().unwrap() = None;
                if *st == State::Draining {
                    *st = State::Idle;
                    self.gate.store(gate::NONE, Ordering::Relaxed); // stop feeding the sidecar
                    self.drain_gen.fetch_add(1, Ordering::Relaxed);
                    drop(st);
                    self.emit_status(false);
                    self.finish_dictation();
                    tracing::info!("session: idle (trailing final committed)");
                } else {
                    // In Listening, a VAD final just segments an utterance; stay
                    // on. Only a very long dictation forces an early paste.
                    let over_cap = buffer_over_cap(&self.utterance_buf.lock().unwrap());
                    drop(st);
                    if over_cap {
                        self.flush_long_session();
                    }
                }
            }
        }
    }

    fn emit_status(&self, listening: bool) {
        let _ = self
            .app
            .emit(STATUS_EVENT, serde_json::json!({ "listening": listening }));
    }

    /// Show/hide the floating pill overlay. Visible for the whole dictation
    /// window (Listening + Draining + cleaning); hidden once the text is pasted.
    fn set_pill(&self, visible: bool) {
        if let Some(win) = self.app.get_webview_window("pill") {
            let _ = if visible { win.show() } else { win.hide() };
        }
    }
}

/// Append one `{raw, cleaned, ts}` line to `MATALU_CLEANUP_LOG`, when set.
///
/// This is the training-data capture for the QLoRA fine-tune: use matalu
/// normally, then review the file and correct the `cleaned` side to get pairs
/// in exactly the right distribution (real Parakeet output, this user's
/// speech). Rejected cleanups log `cleaned: null` — those are the interesting
/// ones. Cleanups that changed nothing are kept too: they become the identity
/// pairs that teach the model to leave already-clean text alone.
///
/// Off unless the env var is set, and failures are silent — capture must never
/// interfere with a dictation.
fn log_pair(raw: &str, cleaned: Option<&str>) {
    let Ok(path) = std::env::var("MATALU_CLEANUP_LOG") else {
        return;
    };
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let line = serde_json::json!({ "ts": ts, "raw": raw, "cleaned": cleaned }).to_string();
    let appended = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, format!("{line}\n").as_bytes()));
    if let Err(e) = appended {
        tracing::warn!(error = %e, path, "failed to append to cleanup log");
    }
}

/// Whether the buffered dictation has grown past [`MAX_BUFFER_CHARS`], counting
/// the spaces the join will add. Pure so the threshold is testable without an
/// `AppHandle`.
fn buffer_over_cap(buf: &[String]) -> bool {
    if buf.is_empty() {
        return false;
    }
    let chars: usize = buf.iter().map(|s| s.chars().count()).sum();
    chars + buf.len() - 1 > MAX_BUFFER_CHARS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utterance(chars: usize) -> String {
        "a".repeat(chars)
    }

    #[test]
    fn empty_and_short_buffers_do_not_flush() {
        assert!(!buffer_over_cap(&[]));
        assert!(!buffer_over_cap(&[utterance(10), utterance(20)]));
        // Exactly at the cap stays — only *past* it flushes.
        assert!(!buffer_over_cap(&[utterance(MAX_BUFFER_CHARS)]));
    }

    #[test]
    fn long_dictation_flushes() {
        assert!(buffer_over_cap(&[utterance(MAX_BUFFER_CHARS + 1)]));
    }

    #[test]
    fn joining_spaces_count_toward_the_cap() {
        // Two halves sum to exactly the cap, but joining adds a space -> over.
        let halves = vec![utterance(MAX_BUFFER_CHARS / 2), utterance(MAX_BUFFER_CHARS / 2)];
        assert!(buffer_over_cap(&halves));
        // One char shorter, and the joined string lands exactly on the cap.
        let under = vec![utterance(MAX_BUFFER_CHARS / 2 - 1), utterance(MAX_BUFFER_CHARS / 2)];
        assert!(!buffer_over_cap(&under));
    }

    #[test]
    fn multibyte_utterances_count_chars_not_bytes() {
        // "é" is two bytes; a byte-based cap would flush ~2x too early.
        let text = "é".repeat(MAX_BUFFER_CHARS);
        assert!(!buffer_over_cap(&[text]));
    }
}

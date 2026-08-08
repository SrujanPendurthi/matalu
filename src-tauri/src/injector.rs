//! System-wide text injection into the focused app.
//!
//! Strategy (see the plan): **live partials** delivered via **clipboard paste**.
//! As Parakeet partials grow/revise, we insert only the *diff* — backspace the
//! changed tail of what we already typed, then paste the new tail — so the
//! common (append-only) case pastes just the new suffix with no backspacing.
//!
//! The keyboard/clipboard side effects run on a **dedicated thread** so
//! injections are serialized and never touch the audio/async paths. The pure
//! diff logic lives in [`DiffState`] and is unit-tested without a display.

use std::sync::mpsc::{self, Sender};
use std::time::Duration;

use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};

/// macOS ANSI virtual key codes.
const KEY_V: u16 = 9; // kVK_ANSI_V
const KEY_BACKSPACE: u16 = 51; // kVK_Delete

/// Delay after ⌘V before restoring the previous clipboard, so the paste lands
/// first. Tunable later via settings.
const RESTORE_DELAY: Duration = Duration::from_millis(120);
/// Delay after writing the clipboard before pressing ⌘V (pasteboard settle).
const PRE_PASTE_DELAY: Duration = Duration::from_millis(20);

/// The concrete edit to apply to the focused app for one transcript update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertPlan {
    /// Number of Backspace presses to delete the changed tail of prior text.
    pub backspaces: usize,
    /// Text to paste after backspacing (may be empty).
    pub insert: String,
}

impl InsertPlan {
    fn is_noop(&self) -> bool {
        self.backspaces == 0 && self.insert.is_empty()
    }
}

/// Per-utterance diff state: tracks what we've inserted so we can compute the
/// minimal edit for each new partial/final.
#[derive(Default)]
pub struct DiffState {
    /// Text currently present in the target app for the active utterance.
    inserted: String,
    /// Emit a leading space before the next utterance's first insertion, so
    /// consecutive committed utterances don't run together.
    pending_separator: bool,
}

impl DiffState {
    /// Compute the edit to converge the target app from `inserted` to `next`.
    /// When `is_final`, the utterance is committed: state resets so a later
    /// utterance never backspaces into already-committed text.
    pub fn step(&mut self, next: &str, is_final: bool) -> InsertPlan {
        let common = common_prefix_chars(&self.inserted, next);
        let inserted_len = self.inserted.chars().count();
        let backspaces = inserted_len - common;

        let mut insert: String = next.chars().skip(common).collect();
        // Leading separator only at the very start of a fresh utterance.
        if self.inserted.is_empty() && self.pending_separator && !next.is_empty() {
            insert.insert(0, ' ');
            self.pending_separator = false;
        }

        if is_final {
            // Arm a separator before the *next* utterance if this one had text.
            if !next.is_empty() {
                self.pending_separator = true;
            }
            self.inserted.clear();
        } else {
            self.inserted = next.to_string();
        }

        InsertPlan { backspaces, insert }
    }

    /// Abandon the current utterance's tracking (e.g. session stopped) without
    /// touching the target app. Keeps any armed separator.
    pub fn reset(&mut self) {
        self.inserted.clear();
    }
}

/// Longest common prefix length in **chars** (not bytes) of `a` and `b`.
fn common_prefix_chars(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

/// Commands sent to the injector thread.
enum Cmd {
    /// A transcript update (`is_final` marks utterance commit).
    Text { text: String, is_final: bool },
    /// Drop the current utterance's tracking without editing the app.
    Reset,
}

/// Cloneable handle to the injector thread.
#[derive(Clone)]
pub struct InjectorHandle {
    tx: Sender<Cmd>,
}

impl InjectorHandle {
    /// Apply a live partial.
    pub fn partial(&self, text: impl Into<String>) {
        let _ = self.tx.send(Cmd::Text { text: text.into(), is_final: false });
    }
    /// Commit a final utterance.
    pub fn commit(&self, text: impl Into<String>) {
        let _ = self.tx.send(Cmd::Text { text: text.into(), is_final: true });
    }
    /// Reset per-utterance tracking (session stop).
    pub fn reset(&self) {
        let _ = self.tx.send(Cmd::Reset);
    }
}

/// Start the injector thread. Keyboard + clipboard live here and are used only
/// from this thread. Returns a handle for enqueuing edits.
pub fn spawn() -> anyhow::Result<InjectorHandle> {
    let (tx, rx) = mpsc::channel::<Cmd>();

    std::thread::Builder::new()
        .name("text-injector".into())
        .spawn(move || {
            let mut clipboard = match arboard::Clipboard::new() {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!(error = %e, "failed to open clipboard; text injection disabled");
                    return;
                }
            };
            let mut diff = DiffState::default();
            // Re-checked here, not just at startup: the user may grant the
            // permission while the app is running, and a silent no-op is
            // indistinguishable from "the model produced nothing".
            let mut warned_untrusted = false;

            while let Ok(cmd) = rx.recv() {
                if !matches!(cmd, Cmd::Reset) && !accessibility_trusted() && !warned_untrusted {
                    warned_untrusted = true;
                    tracing::error!(
                        "discarding injected text: Accessibility is not granted for this app. \
                         Enable it in System Settings → Privacy & Security → Accessibility, \
                         then relaunch. (Launching the binary from a terminal can mask this — \
                         the terminal's own grant is used instead.)"
                    );
                }
                match cmd {
                    Cmd::Reset => diff.reset(),
                    Cmd::Text { text, is_final } => {
                        let plan = diff.step(&text, is_final);
                        if plan.is_noop() {
                            continue;
                        }
                        apply_plan(&mut clipboard, &plan);
                    }
                }
            }
            tracing::info!("injector channel closed; thread stopping");
        })?;

    Ok(InjectorHandle { tx })
}

/// Execute one [`InsertPlan`]: backspace the changed tail, then paste the new
/// tail via the clipboard (saving and restoring the user's clipboard).
fn apply_plan(clipboard: &mut arboard::Clipboard, plan: &InsertPlan) {
    for _ in 0..plan.backspaces {
        post_key(KEY_BACKSPACE, CGEventFlags::CGEventFlagNull);
    }
    if plan.insert.is_empty() {
        return;
    }

    // Save the user's clipboard, borrow it for the paste, then restore.
    let saved = clipboard.get_text().ok();
    if let Err(e) = clipboard.set_text(plan.insert.clone()) {
        tracing::warn!(error = %e, "failed to set clipboard for paste");
        return;
    }
    std::thread::sleep(PRE_PASTE_DELAY);
    post_key(KEY_V, CGEventFlags::CGEventFlagCommand); // ⌘V
    std::thread::sleep(RESTORE_DELAY);
    if let Some(prev) = saved {
        let _ = clipboard.set_text(prev);
    }
}

/// Post a key down+up via CoreGraphics with explicit modifier `flags`. Using a
/// fixed virtual keycode avoids the keyboard-layout lookup (TIS/UCKeyTranslate)
/// that crashes off the main thread, and setting flags explicitly means a
/// physically-held modifier (e.g. Option during push-to-talk) can't pollute the
/// synthesized combo — `CGEventFlagCommand` alone yields a clean ⌘V.
fn post_key(keycode: u16, flags: CGEventFlags) {
    let source = match CGEventSource::new(CGEventSourceStateID::CombinedSessionState) {
        Ok(s) => s,
        Err(()) => {
            tracing::warn!("failed to create CGEventSource; cannot inject keystroke");
            return;
        }
    };
    for down in [true, false] {
        match CGEvent::new_keyboard_event(source.clone(), keycode, down) {
            Ok(event) => {
                event.set_flags(flags);
                event.post(CGEventTapLocation::HID);
            }
            Err(()) => tracing::warn!(keycode, down, "failed to create CGEvent keyboard event"),
        }
    }
}

/// Whether this process is trusted for Accessibility (required to synthesize
/// keystrokes / paste into other apps). Injection silently no-ops without it.
pub fn accessibility_trusted() -> bool {
    #[cfg(target_os = "macos")]
    {
        // AXIsProcessTrusted lives in the ApplicationServices umbrella framework.
        extern "C" {
            fn AXIsProcessTrusted() -> bool;
        }
        unsafe { AXIsProcessTrusted() }
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

/// Same check, but shows the system "grant Accessibility" dialog when untrusted.
///
/// Without this the app is silently useless on a fresh install: the hotkey
/// fires, the model transcribes, and every keystroke is discarded because
/// `CGEvent.post` is a no-op for an untrusted process. Nothing surfaces —
/// there is no error, just no text.
///
/// **This is why launching from a terminal hides the bug.** macOS attributes
/// the TCC check to the *responsible* process, so a terminal-spawned build
/// inherits the terminal's grant and looks trusted; the same bundle launched
/// from Finder is not. Test permissions by opening the `.app`, never by running
/// its binary from a shell.
pub fn prompt_for_accessibility() -> bool {
    #[cfg(target_os = "macos")]
    {
        use core_foundation::base::TCFType;
        use core_foundation::boolean::CFBoolean;
        use core_foundation::dictionary::CFDictionary;
        use core_foundation::string::{CFString, CFStringRef};

        extern "C" {
            fn AXIsProcessTrustedWithOptions(options: *const std::ffi::c_void) -> bool;
            static kAXTrustedCheckOptionPrompt: CFStringRef;
        }
        unsafe {
            let key = CFString::wrap_under_get_rule(kAXTrustedCheckOptionPrompt);
            let options =
                CFDictionary::from_CFType_pairs(&[(key, CFBoolean::true_value().as_CFType())]);
            AXIsProcessTrustedWithOptions(options.as_concrete_TypeRef() as *const _)
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(bs: usize, ins: &str) -> InsertPlan {
        InsertPlan { backspaces: bs, insert: ins.to_string() }
    }

    #[test]
    fn append_only_partials_paste_just_the_suffix() {
        let mut d = DiffState::default();
        assert_eq!(d.step("hello", false), plan(0, "hello"));
        assert_eq!(d.step("hello wor", false), plan(0, " wor"));
        assert_eq!(d.step("hello world", false), plan(0, "ld"));
    }

    #[test]
    fn mid_utterance_revision_backspaces_changed_tail() {
        let mut d = DiffState::default();
        d.step("teh", false);
        // "teh" -> "the": common prefix "t", delete "eh" (2), type "he".
        assert_eq!(d.step("the", false), plan(2, "he"));
    }

    #[test]
    fn final_resets_and_next_utterance_gets_leading_space() {
        let mut d = DiffState::default();
        d.step("hello", false);
        // Final converges (no change) and commits.
        assert_eq!(d.step("hello", true), plan(0, ""));
        // Next utterance starts fresh with a separating space.
        assert_eq!(d.step("world", false), plan(0, " world"));
    }

    #[test]
    fn final_with_punctuation_change_edits_then_commits() {
        let mut d = DiffState::default();
        d.step("hello world", false);
        // final "Hello, world." shares only "" (capital H) -> delete all 11, retype.
        assert_eq!(d.step("Hello, world.", true), plan(11, "Hello, world."));
        // separator armed for the next utterance
        assert_eq!(d.step("again", false), plan(0, " again"));
    }

    #[test]
    fn empty_final_does_not_arm_separator() {
        let mut d = DiffState::default();
        assert_eq!(d.step("", true), plan(0, ""));
        assert_eq!(d.step("first", false), plan(0, "first"));
    }

    #[test]
    fn unicode_prefix_counts_by_char_not_byte() {
        let mut d = DiffState::default();
        d.step("café", false);
        // extend by one char; must not miscount the multi-byte 'é'
        assert_eq!(d.step("café!", false), plan(0, "!"));
    }
}

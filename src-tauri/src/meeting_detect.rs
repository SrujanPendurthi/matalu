//! Detect when the user is in a video call, so the tray can offer to record it.
//!
//! macOS already knows: CoreAudio's process objects report which processes are
//! capturing microphone input right now — the same fact behind the orange
//! menu-bar dot. So "Zoom is recording you" is something we read, not something
//! we guess at from window titles or process lists (Zoom being *open* says
//! nothing about whether you are in a meeting).
//!
//! **This only suggests.** It never starts or stops a recording. Auto-starting
//! on a false positive would make the dictation hotkey inert
//! ([`Session::on_press`] early-returns while `meeting`), write a WAV to temp
//! unasked, and steal window focus mid-call — three silent surprises to save one
//! click. A wrong tray label costs nothing.
//!
//! Best-effort, like diarization: the process-object API is macOS 14.4+, and if
//! it is unavailable the thread logs once and exits with everything else intact.

use std::ffi::c_void;
use std::sync::Arc;

use core_foundation::base::TCFType;
use core_foundation::string::{CFString, CFStringRef};

use crate::session::Session;

/// How often the mic-capture state is sampled. Nothing pushes this at us, so it
/// has to be polled. With [`Detector::CONFIRM`] this puts the tray ~10 s behind
/// a call starting, which is fine for something you act on by clicking.
const POLL: std::time::Duration = std::time::Duration::from_secs(5);

/// Conferencing apps, by bundle ID, with the name to show.
///
/// An allowlist rather than "any other process holding the mic" so Voice Memos,
/// QuickTime, and Photo Booth never trigger it. The browsers are here for
/// Google Meet and are the one real source of false positives — any page
/// capturing the mic looks like a call. Acceptable precisely because this only
/// suggests.
const CONFERENCING_APPS: &[(&str, &str)] = &[
    ("us.zoom.xos", "Zoom"),
    ("com.microsoft.teams", "Teams"),
    ("com.microsoft.teams2", "Teams"),
    ("com.tinyspeck.slackmacgap", "Slack"),
    ("com.hnc.Discord", "Discord"),
    ("com.cisco.webexmeetingsapp", "Webex"),
    ("Cisco-Systems.Spark", "Webex"),
    ("com.apple.FaceTime", "FaceTime"),
    ("com.google.Chrome", "Chrome"),
    ("com.apple.Safari", "Safari"),
    ("company.thebrowser.Browser", "Arc"),
    ("org.mozilla.firefox", "Firefox"),
    ("com.microsoft.edgemac", "Edge"),
    ("com.brave.Browser", "Brave"),
];

/// The first allowlisted app among the bundle IDs currently capturing input.
fn conferencing_app(bundle_ids: &[String]) -> Option<&'static str> {
    bundle_ids.iter().find_map(|id| {
        CONFERENCING_APPS
            .iter()
            .find(|(bundle, _)| bundle == id)
            .map(|(_, name)| *name)
    })
}

/// Debounces raw samples into transitions.
///
/// A single sample is not enough to move the tray: conferencing apps drop and
/// re-add their input stream while joining, and a browser can start capturing a
/// moment before the call is really up. Requiring the same answer twice running
/// costs one poll of latency and removes the flapping.
struct Detector {
    /// The state actually reported to the caller.
    reported: Option<&'static str>,
    /// The candidate being counted toward, and how many samples back it.
    pending: Option<&'static str>,
    streak: u8,
}

impl Detector {
    /// Consecutive identical samples needed before a change is reported.
    const CONFIRM: u8 = 2;

    fn new() -> Self {
        Self { reported: None, pending: None, streak: 0 }
    }

    /// Feed one sample. Returns the new state **only when it changes**, so the
    /// caller can emit and re-label without tracking anything itself.
    fn observe(&mut self, now: Option<&'static str>) -> Option<Option<&'static str>> {
        if now == self.pending {
            self.streak = self.streak.saturating_add(1);
        } else {
            self.pending = now;
            self.streak = 1;
        }
        if self.streak >= Self::CONFIRM && self.reported != now {
            self.reported = now;
            return Some(now);
        }
        None
    }
}

/// Start the detector on its own thread. Same shape as [`crate::fnkey::spawn`]:
/// it holds the `Arc<Session>` directly rather than going through Tauri state.
pub fn spawn(session: Arc<Session>) {
    std::thread::Builder::new()
        .name("meeting-detect".into())
        .spawn(move || run(session))
        .expect("spawn meeting-detect thread");
}

fn run(session: Arc<Session>) {
    let mut detector = Detector::new();
    loop {
        std::thread::sleep(POLL);
        let Some(ids) = capturing_bundle_ids() else {
            tracing::warn!(
                "CoreAudio process list unavailable (needs macOS 14.4+); \
                 meeting auto-detection disabled"
            );
            return;
        };
        // Every capturing app that is *not* on the allowlist, at debug level:
        // when a conferencing app stops being recognized (bundle IDs drift
        // between versions — Teams already needs two), this is what makes it
        // diagnosable from a log instead of by guessing.
        if tracing::enabled!(tracing::Level::DEBUG) {
            for id in &ids {
                if conferencing_app(std::slice::from_ref(id)).is_none() {
                    tracing::debug!(bundle_id = %id, "capturing the mic, not on the allowlist");
                }
            }
        }
        if let Some(state) = detector.observe(conferencing_app(&ids)) {
            match state {
                Some(app) => tracing::info!(app, "meeting detected"),
                None => tracing::info!("meeting ended"),
            }
            session.set_meeting_detected(state);
        }
    }
}

// --- CoreAudio ---------------------------------------------------------------
//
// Hand-declared rather than pulling in `objc2-core-audio`: that crate is in
// Cargo.lock only transitively (cpal -> coreaudio-rs), so using it means a new
// dependency *and* a second CFString type alongside the `core-foundation` this
// file already needs. Same approach as `injector::accessibility_trusted`, which
// declares `AXIsProcessTrustedWithOptions` itself.
//
// Selector values verified against objc2-core-audio-0.3.2's generated
// AudioHardware.rs; they are FourCC codes and are stable API.

type AudioObjectID = u32;
type OSStatus = i32;

#[repr(C)]
struct AudioObjectPropertyAddress {
    selector: u32,
    scope: u32,
    element: u32,
}

const SYSTEM_OBJECT: AudioObjectID = 1;
const SCOPE_GLOBAL: u32 = 0x676c_6f62; // 'glob'
const ELEMENT_MAIN: u32 = 0;

const PROCESS_OBJECT_LIST: u32 = 0x7072_7323; // 'prs#'
const PROCESS_PID: u32 = 0x7070_6964; // 'ppid'
const PROCESS_BUNDLE_ID: u32 = 0x7062_6964; // 'pbid'
const PROCESS_IS_RUNNING_INPUT: u32 = 0x7069_7269; // 'piri'

#[link(name = "CoreAudio", kind = "framework")]
extern "C" {
    fn AudioObjectGetPropertyDataSize(
        object: AudioObjectID,
        address: *const AudioObjectPropertyAddress,
        qualifier_size: u32,
        qualifier: *const c_void,
        out_size: *mut u32,
    ) -> OSStatus;

    fn AudioObjectGetPropertyData(
        object: AudioObjectID,
        address: *const AudioObjectPropertyAddress,
        qualifier_size: u32,
        qualifier: *const c_void,
        io_size: *mut u32,
        out_data: *mut c_void,
    ) -> OSStatus;
}

fn address(selector: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress { selector, scope: SCOPE_GLOBAL, element: ELEMENT_MAIN }
}

/// Read a fixed-size property. `None` on any CoreAudio error, which for a
/// per-process property routinely just means the process went away mid-poll.
fn property<T: Copy>(object: AudioObjectID, selector: u32) -> Option<T> {
    let addr = address(selector);
    // SAFETY: every T used here is POD (u32, i32, a raw pointer) for which an
    // all-zero bit pattern is valid; CoreAudio overwrites it on success, and on
    // failure the value is discarded. `Default` is not an option — raw pointers
    // do not implement it.
    let mut value: T = unsafe { std::mem::zeroed() };
    let mut size = std::mem::size_of::<T>() as u32;
    // SAFETY: `addr` and `value` outlive the call, and `size` matches `value`'s
    // size, which is what CoreAudio writes into.
    let status = unsafe {
        AudioObjectGetPropertyData(
            object,
            &addr,
            0,
            std::ptr::null(),
            &mut size,
            &mut value as *mut T as *mut c_void,
        )
    };
    (status == 0).then_some(value)
}

/// The system's list of audio process objects, or `None` if the API is
/// unavailable (pre-14.4) — which the caller treats as "disable detection".
fn process_objects() -> Option<Vec<AudioObjectID>> {
    let addr = address(PROCESS_OBJECT_LIST);
    let mut size: u32 = 0;
    // SAFETY: `addr` and `size` outlive the call.
    let status =
        unsafe { AudioObjectGetPropertyDataSize(SYSTEM_OBJECT, &addr, 0, std::ptr::null(), &mut size) };
    if status != 0 {
        return None;
    }
    let count = size as usize / std::mem::size_of::<AudioObjectID>();
    let mut ids = vec![0 as AudioObjectID; count];
    if count == 0 {
        return Some(ids);
    }
    // SAFETY: `ids` has room for exactly `size` bytes, which is what we pass.
    let status = unsafe {
        AudioObjectGetPropertyData(
            SYSTEM_OBJECT,
            &addr,
            0,
            std::ptr::null(),
            &mut size,
            ids.as_mut_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return None;
    }
    ids.truncate(size as usize / std::mem::size_of::<AudioObjectID>());
    Some(ids)
}

/// Bundle IDs of every process currently capturing microphone input, excluding
/// this one.
///
/// `None` means the API itself is unavailable. An empty `Vec` means nothing else
/// is recording — a real answer, not a failure.
fn capturing_bundle_ids() -> Option<Vec<String>> {
    let ours = std::process::id() as i32;
    let mut out = Vec::new();
    for object in process_objects()? {
        // A UInt32 boolean in CoreAudio, not a C bool.
        if property::<u32>(object, PROCESS_IS_RUNNING_INPUT) != Some(1) {
            continue;
        }
        // Skip ourselves by PID rather than bundle ID: matalu holds the mic
        // continuously, and the PID comparison also works unbundled (dev runs).
        if property::<i32>(object, PROCESS_PID) == Some(ours) {
            continue;
        }
        let Some(cf) = property::<CFStringRef>(object, PROCESS_BUNDLE_ID) else {
            continue;
        };
        if cf.is_null() {
            continue;
        }
        // SAFETY: CoreAudio hands back a +1 CFString for this property, so the
        // create rule is correct — wrapping it under the get rule would leak.
        let id = unsafe { CFString::wrap_under_create_rule(cf) }.to_string();
        // Unbundled processes report an *empty* string here rather than null —
        // an unbundled `cargo tauri dev` build of this very app does. Nothing
        // without a bundle ID can match the allowlist, and keeping them would
        // only put blanks in the debug log.
        if !id.is_empty() {
            out.push(id);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{conferencing_app, Detector};

    /// The allowlist is the whole reason a recording app doesn't look like a
    /// call, so the negative case matters more than the positive one.
    #[test]
    fn only_allowlisted_conferencing_apps_count() {
        assert_eq!(conferencing_app(&["us.zoom.xos".into()]), Some("Zoom"));
        assert_eq!(
            conferencing_app(&["com.apple.VoiceMemos".into(), "com.google.Chrome".into()]),
            Some("Chrome"),
        );

        assert_eq!(conferencing_app(&["com.apple.VoiceMemos".into()]), None);
        assert_eq!(conferencing_app(&[]), None);
    }

    /// One sample must not move the tray, and a steady state must not re-report
    /// — the detector polls forever, so a state that re-fires every 5 s would
    /// re-emit and re-label continuously.
    #[test]
    fn a_transition_needs_two_agreeing_samples_and_reports_once() {
        let mut d = Detector::new();

        assert_eq!(d.observe(Some("Zoom")), None, "one sample is not enough");
        assert_eq!(d.observe(Some("Zoom")), Some(Some("Zoom")));
        assert_eq!(d.observe(Some("Zoom")), None, "steady state must not re-report");

        // A single blip back to nothing is not a transition.
        assert_eq!(d.observe(None), None);
        assert_eq!(d.observe(Some("Zoom")), None);
        assert_eq!(d.observe(Some("Zoom")), None, "never actually left the reported state");

        assert_eq!(d.observe(None), None);
        assert_eq!(d.observe(None), Some(None), "call ended");
    }
}

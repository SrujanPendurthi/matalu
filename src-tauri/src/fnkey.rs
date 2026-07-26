//! fn-key (Globe) global hold activation via a CGEventTap.
//!
//! The `tauri-plugin-global-shortcut` path can't bind a modifier-only key like
//! `fn`, so this taps the low-level CoreGraphics event stream instead: we watch
//! `FlagsChanged` events for the secondary-`fn` flag and edge-detect its
//! down/up to drive the [`Session`] like any other press/release. It runs push-
//! to-talk (hold to dictate) or toggle exactly as the plugin hotkey does, since
//! it routes through the same [`Session::on_press`]/[`Session::on_release`].
//!
//! The tap lives on a dedicated thread with its own `CFRunLoop`. It is
//! **listen-only** (never consumes the event, so `fn` keeps its normal OS
//! behavior) and needs **Accessibility** (or Input Monitoring) permission —
//! without it `CGEventTapCreate` returns null and we log and give up.

use std::cell::Cell;
use std::sync::Arc;

use core_foundation::runloop::CFRunLoop;
use core_graphics::event::{
    CGEvent, CGEventFlags, CGEventTap, CGEventTapLocation, CGEventTapOptions, CGEventTapPlacement,
    CGEventTapProxy, CGEventType, CallbackResult,
};

use crate::session::Session;

/// Start the fn-key tap on its own thread. Best-effort: logs and returns if the
/// tap can't be created (missing permission). The thread runs for the app's life.
pub fn spawn(session: Arc<Session>) {
    std::thread::Builder::new()
        .name("fn-key-tap".into())
        .spawn(move || run(session))
        .expect("spawn fn-key-tap thread");
}

fn run(session: Arc<Session>) {
    // The callback is `Fn` (called repeatedly) but needs to remember the last
    // fn state to edge-detect. It only ever runs on this thread, so a `Cell` is
    // enough (no cross-thread sharing).
    let fn_down = Cell::new(false);

    let callback = move |_proxy: CGEventTapProxy, _etype: CGEventType, event: &CGEvent| {
        // FlagsChanged fires for *any* modifier; edge-detect only the fn bit so
        // shift/ctrl/etc. changes don't toggle dictation.
        let is_fn = event
            .get_flags()
            .contains(CGEventFlags::CGEventFlagSecondaryFn);
        if is_fn != fn_down.get() {
            fn_down.set(is_fn);
            if is_fn {
                session.on_press();
            } else {
                session.on_release();
            }
        }
        CallbackResult::Keep // listen-only: pass fn through untouched
    };

    let result = CGEventTap::with_enabled(
        CGEventTapLocation::HID,
        CGEventTapPlacement::HeadInsertEventTap,
        CGEventTapOptions::ListenOnly,
        vec![CGEventType::FlagsChanged],
        callback,
        || {
            tracing::info!("fn-key tap active (hold fn / Globe to dictate)");
            // Blocks this thread, servicing the tap, for the app's lifetime.
            CFRunLoop::run_current();
        },
    );

    if result.is_err() {
        tracing::error!(
            "failed to create fn-key event tap; grant Accessibility (or Input Monitoring) \
             permission in System Settings and restart. fn-key activation is disabled."
        );
    }
}

//! Global activation hotkey.
//!
//! Uses `tauri-plugin-global-shortcut`, which reports both `Pressed` and
//! `Released` — so one binding drives both push-to-talk (hold) and toggle (tap).
//! The plugin handler routes those edges into the [`Session`] state machine.
//!
//! Default binding is `Alt+Space`. Modifier-only / `fn`-key holds aren't cleanly
//! supported by the plugin and are a documented stretch (CGEventTap).

use std::sync::Arc;

use tauri::{App, Manager};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

use crate::session::Session;

/// Build the global-shortcut plugin with a handler that drives the session.
/// The handler looks the session up from managed state (registered in setup).
pub fn plugin<R: tauri::Runtime>() -> tauri::plugin::TauriPlugin<R> {
    tauri_plugin_global_shortcut::Builder::new()
        .with_handler(|app, _shortcut, event| {
            tracing::debug!(state = ?event.state, "hotkey event");
            if let Some(session) = app.try_state::<Arc<Session>>() {
                match event.state {
                    ShortcutState::Pressed => session.inner().on_press(),
                    ShortcutState::Released => session.inner().on_release(),
                }
            } else {
                tracing::warn!("hotkey fired but session not in managed state yet");
            }
        })
        .build()
}

/// Register the given activation shortcut. Call from `setup` after the session
/// is in managed state.
pub fn register(app: &App, shortcut: Shortcut) -> Result<(), Box<dyn std::error::Error>> {
    app.global_shortcut().register(shortcut)?;
    Ok(())
}

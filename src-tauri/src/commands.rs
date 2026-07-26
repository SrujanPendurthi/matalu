//! Tauri commands invoked by the settings window.

use std::sync::Arc;

use serde::Serialize;
use tauri::{AppHandle, Manager, State};
use tauri_plugin_global_shortcut::GlobalShortcutExt;

use crate::session::Session;
use crate::settings::{self, Settings};

/// Current persisted settings.
#[tauri::command]
pub fn get_settings(app: AppHandle) -> Settings {
    settings::load(&app)
}

/// Apply and persist new settings: update the session's activation mode and
/// re-register the global hotkey.
#[tauri::command]
pub fn set_settings(
    app: AppHandle,
    session: State<'_, Arc<Session>>,
    settings: Settings,
) -> Result<(), String> {
    session.set_mode(settings.activation_mode.into());

    // Clear any existing plugin hotkey, then (re)register unless the new preset
    // is Fn. Fn is a CGEventTap started at launch; switching *to* Fn just drops
    // the plugin hotkey now and the tap comes up on the next launch (starting a
    // run-loop tap mid-session isn't supported here).
    let gs = app.global_shortcut();
    let _ = gs.unregister_all();
    if let Some(shortcut) = settings.hotkey.shortcut() {
        gs.register(shortcut)
            .map_err(|e| format!("failed to register hotkey: {e}"))?;
    }

    settings::save(&app, &settings).map_err(|e| e.to_string())?;
    Ok(())
}

#[derive(Serialize)]
pub struct Permissions {
    /// Accessibility trust — required to synthesize keystrokes / paste.
    pub accessibility: bool,
}

/// Current permission status for the onboarding UI.
#[tauri::command]
pub fn get_permissions() -> Permissions {
    Permissions {
        accessibility: crate::injector::accessibility_trusted(),
    }
}

/// Open the macOS Accessibility privacy pane so the user can grant permission.
#[tauri::command]
pub fn open_accessibility_settings() {
    let _ = std::process::Command::new("open")
        .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")
        .spawn();
}

/// Show (and focus) the settings window.
#[tauri::command]
pub fn show_settings_window(app: AppHandle) {
    show_settings(&app);
}

/// Reveal the predefined settings window.
pub fn show_settings(app: &AppHandle) {
    if let Some(win) = app.get_webview_window("settings") {
        let _ = win.show();
        let _ = win.set_focus();
    } else {
        tracing::warn!("settings window not found");
    }
}

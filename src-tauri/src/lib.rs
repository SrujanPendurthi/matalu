//! Matalu desktop app (Tauri v2 shell).
//!
//! Reuses the `matalu` core pipeline (mic capture → resample → parakeet-mlx
//! sidecar → [`TranscriptEvent`]s) and, instead of the headless WebSocket
//! fan-out, routes transcripts through a [`session::Session`] to system-wide
//! text injection, activated by a global hotkey.

mod commands;
mod hotkey;
mod injector;
mod pipeline;
mod session;
mod settings;
mod tray;

use tauri::Manager;

use crate::session::Mode;

/// App-lifetime state kept in Tauri's managed store. Holds the sidecar child so
/// it is not dropped (dropping it stops transcription).
pub struct AppState {
    /// The supervised parakeet-mlx sidecar process (kept alive for the app's life).
    pub _sidecar_child: std::sync::Mutex<Option<std::process::Child>>,
}

/// Tauri entry point.
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "matalu=info,matalu_app_lib=info".into()),
        )
        .init();

    tauri::Builder::default()
        .plugin(hotkey::plugin())
        .invoke_handler(tauri::generate_handler![
            commands::get_settings,
            commands::set_settings,
            commands::get_permissions,
            commands::open_accessibility_settings,
            commands::show_settings_window,
        ])
        .setup(|app| {
            let handle = app.handle().clone();

            // Persisted settings drive activation mode + hotkey. MATALU_MODE
            // still overrides the mode for debugging.
            let cfg = settings::load(&handle);
            let mode = match std::env::var("MATALU_MODE").as_deref() {
                Ok("toggle") => Mode::Toggle,
                Ok("pushtotalk") => Mode::PushToTalk,
                _ => cfg.activation_mode.into(),
            };

            // Start the core pipeline; get the session + sidecar child.
            let started = pipeline::start(handle, mode)?;

            // The hotkey handler resolves the session from managed state.
            app.manage(started.session);
            app.manage(AppState {
                _sidecar_child: std::sync::Mutex::new(started.child),
            });

            hotkey::register(app, cfg.hotkey.shortcut())?;
            tray::build(app)?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running matalu");
}

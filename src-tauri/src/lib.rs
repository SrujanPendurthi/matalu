//! Matalu desktop app (Tauri v2 shell).
//!
//! Reuses the `matalu` core pipeline (mic capture → resample → parakeet-mlx
//! sidecar → [`TranscriptEvent`]s) and, instead of the headless WebSocket
//! fan-out, routes transcripts through a [`session::Session`] to system-wide
//! text injection, activated by a global hotkey.

mod commands;
mod fnkey;
mod hotkey;
mod injector;
mod pipeline;
mod session;
mod settings;
mod tray;

use crate::settings::HotkeyPreset;

use tauri::{LogicalPosition, Manager};

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
            // Menu-bar-only: no dock icon, no app window in ⌘-Tab. The UI lives
            // in the tray; the transcript window is dev-only (Show via tray).
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);

            // The floating pill is display-only: click-through so it never steals
            // clicks (or focus) from the app being dictated into, parked at the
            // bottom-center of the primary screen.
            setup_pill(app.handle());

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

            // The hotkey handler resolves the session from managed state; keep a
            // clone for the fn-key tap (which holds the session directly).
            let session = started.session.clone();
            app.manage(started.session);
            app.manage(AppState {
                _sidecar_child: std::sync::Mutex::new(started.child),
            });

            // Fn (Globe) is driven by a CGEventTap; every other preset is a
            // plugin global shortcut. (Switching to/from Fn takes effect on the
            // next launch — see commands::set_settings.)
            match cfg.hotkey.shortcut() {
                Some(shortcut) => hotkey::register(app, shortcut)?,
                None if cfg.hotkey == HotkeyPreset::Fn => fnkey::spawn(session),
                None => {}
            }
            tray::build(app)?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running matalu");
}

/// Make the pill click-through and park it at the bottom-center of the primary
/// monitor. Best-effort: if the window or monitor info isn't available we leave
/// the pill at its configured default position.
fn setup_pill(app: &tauri::AppHandle) {
    let Some(pill) = app.get_webview_window("pill") else {
        tracing::warn!("pill window not found; skipping placement");
        return;
    };
    // Display-only overlay: pass clicks through to whatever is behind it.
    let _ = pill.set_ignore_cursor_events(true);

    // Bottom-center of the primary monitor (logical coords avoid scale math).
    match pill.primary_monitor() {
        Ok(Some(monitor)) => {
            let scale = monitor.scale_factor();
            let m_size = monitor.size().to_logical::<f64>(scale);
            let m_pos = monitor.position().to_logical::<f64>(scale);
            const PILL_W: f64 = 260.0;
            const PILL_H: f64 = 52.0;
            const BOTTOM_GAP: f64 = 96.0;
            let x = m_pos.x + (m_size.width - PILL_W) / 2.0;
            let y = m_pos.y + m_size.height - PILL_H - BOTTOM_GAP;
            let _ = pill.set_position(LogicalPosition::new(x, y));
        }
        _ => tracing::warn!("no primary monitor; leaving pill at default position"),
    }
}

//! Menu-bar tray icon and menu.

use std::sync::Arc;

use tauri::menu::{MenuBuilder, MenuItem, MenuItemBuilder};
use tauri::tray::TrayIconBuilder;
use tauri::{App, AppHandle, Manager, Wry};

use crate::session::Session;

/// The meeting menu item, in managed state so anything can re-label it.
///
/// Without this it is reachable only from the tray's own click handler, and both
/// things that now drive the label from outside — meeting detection, and
/// `start_meeting` refusing while the model is still loading — would leave it
/// stale.
pub struct MeetingItem(MenuItem<Wry>);

/// Re-label the meeting item from the session's actual state.
///
/// Always derive it, never set it from the branch just taken: `start_meeting`
/// no-ops while the pipeline is warming, and the old code's unconditional
/// "Stop Meeting Transcript" left the menu claiming a recording that never
/// started — the next click then tried to start it again instead of stopping.
pub fn sync_label(app: &AppHandle) {
    let (Some(item), Some(session)) =
        (app.try_state::<MeetingItem>(), app.try_state::<Arc<Session>>())
    else {
        return;
    };
    let text = if session.is_meeting() {
        "Stop Meeting Transcript"
    } else if session.meeting_detected() {
        "Meeting detected — Start Transcript"
    } else {
        "Start Meeting Transcript"
    };
    let _ = item.0.set_text(text);
}

/// Build the tray icon + menu. For M2 this is Show/Hide + Quit; activation and
/// status items arrive with the hotkey/session milestones.
pub fn build(app: &App) -> tauri::Result<()> {
    let settings = MenuItemBuilder::with_id("settings", "Settings…").build(app)?;
    let meeting = MenuItemBuilder::with_id("meeting", "Start Meeting Transcript").build(app)?;
    let toggle = MenuItemBuilder::with_id("toggle_window", "Show / Hide Window").build(app)?;
    let quit = MenuItemBuilder::with_id("quit", "Quit Matalu").build(app)?;
    let menu = MenuBuilder::new(app)
        .items(&[&settings, &meeting, &toggle, &quit])
        .build()?;

    app.manage(MeetingItem(meeting.clone()));

    let icon = app
        .default_window_icon()
        .cloned()
        .expect("app has a default window icon");

    let _tray = TrayIconBuilder::with_id("main-tray")
        .icon(icon)
        .tooltip("Matalu")
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(move |app, event| match event.id().as_ref() {
            "quit" => app.exit(0),
            "settings" => crate::commands::show_settings(app),
            "meeting" => {
                if let Some(session) = app.try_state::<Arc<Session>>() {
                    if session.is_meeting() {
                        session.stop_meeting();
                    } else {
                        session.start_meeting();
                    }
                }
                sync_label(app);
            }
            "toggle_window" => {
                if let Some(win) = app.get_webview_window("main") {
                    match win.is_visible() {
                        Ok(true) => {
                            let _ = win.hide();
                        }
                        _ => {
                            let _ = win.show();
                            let _ = win.set_focus();
                        }
                    }
                }
            }
            _ => {}
        })
        .build(app)?;

    Ok(())
}

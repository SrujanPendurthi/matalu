//! Menu-bar tray icon and menu.

use std::sync::Arc;

use tauri::menu::{MenuBuilder, MenuItemBuilder};
use tauri::tray::TrayIconBuilder;
use tauri::{App, Manager};

use crate::session::Session;

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

    // Clone the meeting item into the handler so it can flip its own label.
    let meeting_item = meeting.clone();

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
                        let _ = meeting_item.set_text("Start Meeting Transcript");
                    } else {
                        session.start_meeting();
                        let _ = meeting_item.set_text("Stop Meeting Transcript");
                    }
                }
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

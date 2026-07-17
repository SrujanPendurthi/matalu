//! Persisted user settings (activation mode + hotkey), stored as JSON in the
//! app config dir. Hotkeys are a fixed set of presets for v1 — arbitrary
//! key-capture is a later polish item.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};
use tauri_plugin_global_shortcut::{Code, Modifiers, Shortcut};

use crate::session::Mode;

/// How the hotkey drives dictation.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum ActivationMode {
    PushToTalk,
    Toggle,
}

impl From<ActivationMode> for Mode {
    fn from(m: ActivationMode) -> Self {
        match m {
            ActivationMode::PushToTalk => Mode::PushToTalk,
            ActivationMode::Toggle => Mode::Toggle,
        }
    }
}

/// A fixed set of activation-hotkey choices. (Modifier-only / arbitrary capture
/// is deferred; those need a CGEventTap or key-capture UI.)
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum HotkeyPreset {
    AltSpace,
    CtrlSpace,
    CmdShiftSpace,
    F5,
}

impl HotkeyPreset {
    pub fn shortcut(self) -> Shortcut {
        match self {
            HotkeyPreset::AltSpace => Shortcut::new(Some(Modifiers::ALT), Code::Space),
            HotkeyPreset::CtrlSpace => Shortcut::new(Some(Modifiers::CONTROL), Code::Space),
            HotkeyPreset::CmdShiftSpace => {
                Shortcut::new(Some(Modifiers::SUPER | Modifiers::SHIFT), Code::Space)
            }
            HotkeyPreset::F5 => Shortcut::new(None, Code::F5),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct Settings {
    pub activation_mode: ActivationMode,
    pub hotkey: HotkeyPreset,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            activation_mode: ActivationMode::PushToTalk,
            hotkey: HotkeyPreset::AltSpace,
        }
    }
}

fn settings_path(app: &AppHandle) -> Option<PathBuf> {
    app.path()
        .app_config_dir()
        .ok()
        .map(|d| d.join("settings.json"))
}

/// Load settings, falling back to defaults on any missing/corrupt file.
pub fn load(app: &AppHandle) -> Settings {
    let Some(p) = settings_path(app) else {
        return Settings::default();
    };
    match std::fs::read_to_string(&p) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
            tracing::warn!(error = %e, "settings.json unparseable; using defaults");
            Settings::default()
        }),
        Err(_) => Settings::default(),
    }
}

/// Persist settings to the app config dir.
pub fn save(app: &AppHandle, settings: &Settings) -> anyhow::Result<()> {
    if let Some(p) = settings_path(app) {
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&p, serde_json::to_string_pretty(settings)?)?;
        tracing::info!(?settings, "settings saved");
    }
    Ok(())
}

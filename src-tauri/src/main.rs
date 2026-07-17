// Prevent a second console window on Windows (harmless on macOS).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    matalu_app_lib::run();
}

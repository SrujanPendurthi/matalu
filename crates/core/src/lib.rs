//! matalu core — the local real-time speech-to-text pipeline.
//!
//! Pipeline: cpal mic → streaming resample to 16 kHz ([`audio`]) → Python
//! `parakeet-mlx` sidecar (streaming ASR + VAD segmentation, [`sidecar`]) →
//! [`events::TranscriptEvent`]s on a broadcast channel. Text passes through a
//! pluggable [`corrector`] seam.
//!
//! This crate is transport-agnostic: it produces `TranscriptEvent`s and leaves
//! delivery to the host. `matalu-headless` fans them out over a WebSocket; the
//! Tauri desktop app routes them to system-wide text injection.

pub mod audio;
pub mod config;
pub mod corrector;
pub mod events;
pub mod sidecar;

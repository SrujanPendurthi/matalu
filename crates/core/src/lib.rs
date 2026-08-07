//! matalu core — the local real-time speech-to-text pipeline.
//!
//! Pipeline: cpal mic → streaming resample to 16 kHz ([`audio`]) → Python
//! `parakeet-mlx` sidecar (streaming ASR + VAD segmentation, [`sidecar`]) →
//! [`events::TranscriptEvent`]s on a broadcast channel.
//!
//! This crate is transport-agnostic: it produces `TranscriptEvent`s and leaves
//! delivery to the host. The Tauri desktop app routes them to text cleanup and
//! system-wide injection.

pub mod audio;
pub mod config;
pub mod events;
pub mod sidecar;

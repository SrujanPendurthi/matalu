//! matalu — local real-time speech-to-text service.
//!
//! Pipeline: cpal mic → streaming resample to 16 kHz ([`audio`]) → Python
//! `parakeet-mlx` sidecar (streaming ASR + VAD segmentation, [`sidecar`]) →
//! tokio broadcast → WebSocket fan-out ([`server`]). Text passes through a
//! pluggable [`corrector`] seam.

pub mod audio;
pub mod config;
pub mod corrector;
pub mod events;
pub mod server;
pub mod sidecar;

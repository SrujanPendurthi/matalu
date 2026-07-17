//! ASR engine = a supervised Python `parakeet-mlx` sidecar process.
//!
//! MLX is Python-only, so the model runs in a child process. Rust owns
//! everything else (capture, resample, transport). Two threads bridge the pipes:
//!   - writer: 16 kHz mono f32 audio → child stdin (raw little-endian f32)
//!   - reader: child stdout (newline JSON) → [`TranscriptEvent`] → broadcast
//! The child's stderr is inherited so its logs appear alongside ours.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;

use anyhow::{Context, Result};
use crossbeam_channel::Receiver;
use tokio::sync::broadcast;

use crate::config::Config;
use crate::corrector::Corrector;
use crate::events::TranscriptEvent;

/// Handle returned by [`spawn`]: the child process plus a one-shot receiver
/// that fires once the sidecar has loaded + warmed the model and is ready to
/// consume audio. The [`Child`] must be kept alive; dropping it stops transcription.
pub struct Sidecar {
    pub child: Child,
    pub ready: std::sync::mpsc::Receiver<()>,
}

/// Spawn the sidecar and the two bridge threads.
pub fn spawn(
    cfg: Config,
    audio_rx: Receiver<Vec<f32>>,
    events_tx: broadcast::Sender<TranscriptEvent>,
    corrector: Arc<dyn Corrector>,
) -> Result<Sidecar> {
    tracing::info!(
        python = %cfg.python_bin,
        script = %cfg.sidecar_script,
        model = %cfg.mlx_model,
        "starting parakeet-mlx sidecar"
    );

    let mut child = Command::new(&cfg.python_bin)
        .arg(&cfg.sidecar_script)
        .env("MATALU_MLX_MODEL", &cfg.mlx_model)
        .env("MATALU_SILENCE_MS", cfg.silence_ms.to_string())
        .env("MATALU_VAD_RMS", cfg.vad_rms_threshold.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("failed to spawn sidecar: {} {}", cfg.python_bin, cfg.sidecar_script))?;

    let mut stdin = child.stdin.take().expect("piped stdin");
    let stdout = child.stdout.take().expect("piped stdout");

    // Writer: audio frames → child stdin as raw little-endian f32.
    std::thread::Builder::new()
        .name("sidecar-writer".into())
        .spawn(move || {
            let mut bytes: Vec<u8> = Vec::new();
            while let Ok(samples) = audio_rx.recv() {
                bytes.clear();
                bytes.reserve(samples.len() * 4);
                for s in &samples {
                    bytes.extend_from_slice(&s.to_le_bytes());
                }
                if stdin.write_all(&bytes).is_err() {
                    tracing::warn!("sidecar stdin closed; stopping audio writer");
                    break;
                }
            }
        })
        .expect("spawn sidecar-writer");

    // Reader: child stdout (newline JSON) → ready signal | corrector → broadcast.
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<()>();
    std::thread::Builder::new()
        .name("sidecar-reader".into())
        .spawn(move || {
            let reader = BufReader::new(stdout);
            let mut ready_tx = Some(ready_tx);
            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(e) => {
                        tracing::error!(error = %e, "sidecar stdout read error");
                        break;
                    }
                };
                if line.trim().is_empty() {
                    continue;
                }
                // A `{"type":"ready"}` control line signals model readiness;
                // everything else is a transcript event.
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    if v.get("type").and_then(|t| t.as_str()) == Some("ready") {
                        if let Some(tx) = ready_tx.take() {
                            let _ = tx.send(());
                        }
                        continue;
                    }
                }
                match serde_json::from_str::<TranscriptEvent>(&line) {
                    Ok(ev) => {
                        let refined = corrector.refine(ev.text());
                        let _ = events_tx.send(ev.with_text(refined));
                    }
                    Err(e) => tracing::warn!(error = %e, line = %line, "unparseable sidecar output"),
                }
            }
            tracing::info!("sidecar stdout ended; reader stopping");
        })
        .expect("spawn sidecar-reader");

    Ok(Sidecar { child, ready: ready_rx })
}

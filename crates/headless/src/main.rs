//! matalu binary: wire mic capture → ASR worker → WebSocket server.

use std::sync::Arc;

use anyhow::Result;
use tokio::sync::broadcast;
use tracing_subscriber::EnvFilter;

use matalu::config::{Config, TARGET_SAMPLE_RATE};
use matalu::corrector::{Corrector, PassThrough};
use matalu::events::TranscriptEvent;
use matalu::{audio, sidecar};

mod server;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "matalu=info".into()))
        .init();

    let cfg = Config::from_env()?;
    tracing::info!(?cfg, "starting matalu");

    // Transcript fan-out: one producer (ASR worker), N consumers (WS clients).
    let (events_tx, _events_rx) = broadcast::channel(256);

    // Optional synthetic stream for testing the WS path without mic/model.
    if std::env::var("MATALU_DEMO").is_ok() {
        tracing::info!("MATALU_DEMO enabled: emitting synthetic transcript events");
        spawn_demo(events_tx.clone());
    }

    // Audio -> ASR: bounded so a stalled worker can't grow memory without limit;
    // the capture callback drops (never blocks) when this is full.
    let (audio_tx, audio_rx) = crossbeam_channel::bounded::<Vec<f32>>(64);

    let corrector: Arc<dyn Corrector> = Arc::new(PassThrough);
    // ASR runs in a supervised Python parakeet-mlx sidecar; keep the child alive.
    let sc = sidecar::spawn(cfg.clone(), audio_rx, events_tx.clone(), corrector)?;
    let _child = sc.child;

    // Gate mic capture on sidecar readiness: starting the mic while the model is
    // still loading backs up the pipe and drops audio. Wait for the ready signal
    // (bounded) on a helper thread so the server can come up immediately.
    std::thread::spawn(move || {
        match sc.ready.recv_timeout(std::time::Duration::from_secs(180)) {
            Ok(()) => tracing::info!("sidecar ready; starting microphone capture"),
            Err(_) => tracing::warn!("sidecar not ready after 180s; starting capture anyway"),
        }
        audio::spawn_capture(TARGET_SAMPLE_RATE, audio_tx);
    });

    // Serve until Ctrl-C.
    tokio::select! {
        r = server::serve(cfg.bind_addr, events_tx) => r?,
        _ = tokio::signal::ctrl_c() => tracing::info!("shutting down"),
    }
    Ok(())
}

/// Emit a looping synthetic utterance (partials growing into a final) so the
/// WebSocket protocol can be exercised without audio or the ASR model.
fn spawn_demo(tx: broadcast::Sender<TranscriptEvent>) {
    tokio::spawn(async move {
        let steps = [
            "hello",
            "hello world",
            "hello world this",
            "hello world this is a demo.",
        ];
        let mut ts: u64 = 0;
        loop {
            for (i, text) in steps.iter().enumerate() {
                tokio::time::sleep(std::time::Duration::from_millis(400)).await;
                ts += 400;
                let ev = if i == steps.len() - 1 {
                    TranscriptEvent::Final { text: text.to_string(), ts_ms: ts }
                } else {
                    TranscriptEvent::Partial { text: text.to_string(), ts_ms: ts }
                };
                let _ = tx.send(ev);
            }
        }
    });
}

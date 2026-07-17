//! WebSocket/HTTP server. Each `/ws` client subscribes to the transcript
//! broadcast and receives one JSON frame per [`TranscriptEvent`].

use std::net::SocketAddr;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use axum::Router;
use tokio::sync::broadcast;

use matalu::events::TranscriptEvent;

/// Live transcription viewer page, served at `/`.
const INDEX_HTML: &str = include_str!("index.html");

#[derive(Clone)]
struct AppState {
    tx: broadcast::Sender<TranscriptEvent>,
}

/// Run the server until the process is shut down.
pub async fn serve(
    bind: SocketAddr,
    tx: broadcast::Sender<TranscriptEvent>,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/", get(|| async { Html(INDEX_HTML) }))
        .route("/health", get(|| async { "ok" }))
        .route("/ws", get(ws_handler))
        .with_state(AppState { tx });

    let listener = tokio::net::TcpListener::bind(bind).await?;
    tracing::info!(%bind, "WebSocket server listening (connect to ws://{bind}/ws)");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| client_loop(socket, state.tx.subscribe()))
}

async fn client_loop(mut socket: WebSocket, mut rx: broadcast::Receiver<TranscriptEvent>) {
    tracing::debug!("websocket client connected");
    loop {
        match rx.recv().await {
            Ok(event) => {
                let json = match serde_json::to_string(&event) {
                    Ok(j) => j,
                    Err(e) => {
                        tracing::error!(%e, "failed to serialize event");
                        continue;
                    }
                };
                if socket.send(Message::Text(json.into())).await.is_err() {
                    break; // client disconnected
                }
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                tracing::warn!(skipped = n, "client lagged; dropped events");
            }
            Err(broadcast::error::RecvError::Closed) => break,
        }
    }
    tracing::debug!("websocket client disconnected");
}

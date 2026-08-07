//! Transcript events streamed to WebSocket clients.

use serde::{Deserialize, Serialize};

/// A single message on the transcript stream.
///
/// `partial` frames grow/refine live as audio streams in; a `final` frame is
/// emitted once an utterance is committed (after a silence gap). Serialized as
/// `{"type":"partial","text":"...","ts_ms":123}`. `Deserialize` is used to parse
/// the same shape back from the ASR sidecar's stdout.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum TranscriptEvent {
    Partial { text: String, ts_ms: u64 },
    Final { text: String, ts_ms: u64 },
}


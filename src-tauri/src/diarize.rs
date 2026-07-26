//! Post-hoc speaker diarization: run `sidecar/diarize.py` (sherpa-onnx) on a
//! recorded meeting WAV, then align its speaker segments to the transcript
//! lines and emit a labeled result to the UI.
//!
//! This is best-effort: if the diarizer is missing (no sherpa-onnx / models) or
//! fails, we emit the transcript **unlabeled** so meeting mode never breaks.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

/// Tauri event carrying the diarized, speaker-labeled meeting transcript.
pub const MEETING_RESULT_EVENT: &str = "meeting_result";

/// A committed transcript line with its meeting-relative time span (ms).
#[derive(Clone, Debug, Serialize)]
pub struct Line {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

/// A transcript line tagged with a speaker cluster (`-1` = unknown).
#[derive(Clone, Debug, Serialize)]
pub struct LabeledLine {
    pub speaker: i32,
    pub text: String,
}

/// One diarization segment from `diarize.py` (seconds).
#[derive(Clone, Debug, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub speaker: i32,
}

/// Assign each line the speaker whose segment overlaps it most in time; `-1`
/// when no segment overlaps (silence/unknown).
pub fn align(lines: &[Line], segments: &[Segment]) -> Vec<LabeledLine> {
    lines
        .iter()
        .map(|l| {
            let (ls, le) = (l.start_ms as f64 / 1000.0, l.end_ms as f64 / 1000.0);
            let mut best = -1i32;
            let mut best_overlap = 0.0f64;
            for s in segments {
                let overlap = (le.min(s.end) - ls.max(s.start)).max(0.0);
                if overlap > best_overlap {
                    best_overlap = overlap;
                    best = s.speaker;
                }
            }
            LabeledLine { speaker: best, text: l.text.clone() }
        })
        .collect()
}

/// Run diarization on `wav`, align to `lines`, emit `meeting_result`, and delete
/// the temp WAV. Intended to run on a background thread (loads models, seconds).
pub fn run(app: AppHandle, wav: PathBuf, lines: Vec<Line>) {
    let labeled = match diarize_wav(&wav) {
        Ok(segments) => {
            tracing::info!(segments = segments.len(), "diarization complete");
            align(&lines, &segments)
        }
        Err(e) => {
            tracing::warn!(error = %e, "diarization failed; emitting unlabeled transcript");
            lines
                .iter()
                .map(|l| LabeledLine { speaker: -1, text: l.text.clone() })
                .collect()
        }
    };
    if let Err(e) = app.emit(MEETING_RESULT_EVENT, &labeled) {
        tracing::warn!(error = %e, "failed to emit meeting_result");
    }
    let _ = std::fs::remove_file(&wav);
}

fn diarize_wav(wav: &Path) -> anyhow::Result<Vec<Segment>> {
    let python = std::env::var("MATALU_PYTHON").unwrap_or_else(|_| "python3".into());
    // Dev path: anchored to the crate dir like the ASR sidecar (cargo tauri dev
    // runs with CWD = src-tauri/). Diarization is a dev/personal feature in v1.
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/../sidecar/diarize.py");
    let out = Command::new(&python).arg(script).arg(wav).output()?;
    if !out.status.success() {
        anyhow::bail!(
            "diarize.py exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(serde_json::from_slice(&out.stdout)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(start_ms: u64, end_ms: u64) -> Line {
        Line { start_ms, end_ms, text: format!("{start_ms}-{end_ms}") }
    }

    #[test]
    fn align_picks_max_overlap_speaker() {
        let lines = vec![line(0, 2000), line(2000, 4000), line(10_000, 11_000)];
        let segments = vec![
            Segment { start: 0.0, end: 2.5, speaker: 0 },  // covers line 0, most of 1
            Segment { start: 2.5, end: 4.0, speaker: 1 },  // tail of line 1
        ];
        let labeled = align(&lines, &segments);
        assert_eq!(labeled[0].speaker, 0, "line 0 (0-2s) fully inside speaker 0");
        assert_eq!(labeled[1].speaker, 1, "line 1 (2-4s) overlaps speaker 1 (1.5s) > speaker 0 (0.5s)");
        assert_eq!(labeled[2].speaker, -1, "line 2 (10-11s) overlaps no segment → unknown");
    }
}

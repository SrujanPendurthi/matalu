//! Runtime configuration, sourced from environment variables with sane defaults.

use std::net::SocketAddr;

/// Target sample rate the ASR model expects (parakeet-mlx = 16 kHz).
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

#[derive(Clone, Debug)]
pub struct Config {
    /// Address the WebSocket/HTTP server binds to.
    pub bind_addr: SocketAddr,
    /// Python interpreter used to run the sidecar.
    pub python_bin: String,
    /// Path to the parakeet-mlx sidecar script.
    pub sidecar_script: String,
    /// A self-contained sidecar executable (e.g. a PyInstaller bundle). When
    /// `Some`, it is run directly and `python_bin`/`sidecar_script` are ignored;
    /// when `None`, the sidecar is launched as `python_bin sidecar_script`. The
    /// packaged app sets this to the bundled binary next to the app executable.
    pub sidecar_bin: Option<String>,
    /// HuggingFace id (or local path) of the MLX model.
    pub mlx_model: String,
    /// Silence duration (ms) that finalizes an utterance (passed to the sidecar).
    pub silence_ms: u64,
    /// RMS threshold below which a chunk is silence (passed to the sidecar).
    pub vad_rms_threshold: f32,
}

impl Config {
    /// Build config from env vars, falling back to defaults.
    ///
    /// - `MATALU_BIND`        (default `127.0.0.1:8765`)
    /// - `MATALU_PYTHON`      (default `python3`)
    /// - `MATALU_SIDECAR`     (default `sidecar/matalu_sidecar.py`)
    /// - `MATALU_SIDECAR_BIN` (optional; a bundled sidecar executable — overrides python+script)
    /// - `MATALU_MLX_MODEL`   (default `mlx-community/parakeet-tdt-0.6b-v2`)
    /// - `MATALU_SILENCE_MS`  (default `700`)
    /// - `MATALU_VAD_RMS`     (default `0.010`)
    pub fn from_env() -> anyhow::Result<Self> {
        fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
            std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
        }
        fn env_str(key: &str, default: &str) -> String {
            std::env::var(key).unwrap_or_else(|_| default.to_string())
        }

        let bind_addr: SocketAddr = env_str("MATALU_BIND", "127.0.0.1:8765").parse()?;

        Ok(Self {
            bind_addr,
            python_bin: env_str("MATALU_PYTHON", "python3"),
            sidecar_script: env_str("MATALU_SIDECAR", "sidecar/matalu_sidecar.py"),
            sidecar_bin: std::env::var("MATALU_SIDECAR_BIN").ok().filter(|s| !s.is_empty()),
            mlx_model: env_str("MATALU_MLX_MODEL", "mlx-community/parakeet-tdt-0.6b-v2"),
            silence_ms: env_or("MATALU_SILENCE_MS", 700),
            vad_rms_threshold: env_or("MATALU_VAD_RMS", 0.010_f32),
        })
    }
}

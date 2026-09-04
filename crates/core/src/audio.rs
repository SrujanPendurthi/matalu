//! Microphone capture (cpal) + streaming downmix/resample to 16 kHz mono f32.
//!
//! The cpal callback runs on a realtime audio thread, so it must not block or
//! allocate unboundedly. It downmixes to mono, resamples to 16 kHz with a
//! stateful linear resampler, and `try_send`s the result to the ASR worker,
//! dropping data (rather than blocking) if the worker ever falls behind.

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{ErrorKind, FromSample, SampleFormat, SizedSample, Stream};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};

/// Stateful streaming linear resampler (arbitrary input rate -> `out_rate`).
///
/// Maintains fractional read position and the last input sample across calls so
/// successive buffers interpolate seamlessly at their boundaries. No
/// anti-aliasing filter — adequate for 16 kHz speech ASR (the model does its own
/// mel/log front-end); revisit with a polyphase filter if downsampling artifacts
/// ever hurt accuracy.
struct LinearResampler {
    ratio: f64, // input samples consumed per output sample
    pos: f64,   // fractional position, in input-sample units, of next output
    last: f32,  // final input sample of the previous buffer (virtual index -1)
}

impl LinearResampler {
    fn new(in_rate: u32, out_rate: u32) -> Self {
        Self {
            ratio: in_rate as f64 / out_rate as f64,
            pos: 0.0,
            last: 0.0,
        }
    }

    #[inline]
    fn sample_at(&self, i: isize, input: &[f32]) -> f32 {
        if i < 0 {
            self.last
        } else {
            input[i as usize]
        }
    }

    /// Resample one mono buffer, appending outputs to `out`.
    fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if input.is_empty() {
            return;
        }
        let n = input.len() as f64;
        let mut pos = self.pos;
        // Emit while both bracketing samples (floor(pos), floor(pos)+1) are available.
        while pos + 1.0 < n {
            let i = pos.floor();
            let frac = (pos - i) as f32;
            let a = self.sample_at(i as isize, input);
            let b = self.sample_at(i as isize + 1, input);
            out.push(a + (b - a) * frac);
            pos += self.ratio;
        }
        // Carry state: next buffer's index -1 is this buffer's last sample.
        self.last = *input.last().unwrap();
        self.pos = pos - n;
    }
}

/// Delay before rebuilding a lost stream, and the retry interval when the
/// device is not there at all. Long enough that a device transition settles
/// first; short enough that a reconnect feels immediate.
const REOPEN_DELAY: std::time::Duration = std::time::Duration::from_millis(1000);

/// How often a follow-the-default capture re-checks which device the system
/// considers the default input. Nothing pushes this at us, so it has to be
/// polled; 2 s is imperceptible against a human plugging a headset in and costs
/// one cheap CoreAudio query off the realtime thread.
const DEFAULT_POLL: std::time::Duration = std::time::Duration::from_secs(2);

/// Start microphone capture on a dedicated OS thread.
///
/// cpal's `Stream` is `!Send` and must be created and kept alive on one thread,
/// and on macOS `build_input_stream` blocks in CoreAudio until microphone
/// permission is resolved. Doing this on its own thread keeps that latency (and
/// any permission stall) off the server's startup path — `/health` and `/ws`
/// come up immediately regardless of mic state.
pub fn spawn_capture(target_rate: u32, device_name: Option<String>, tx: Sender<Vec<f32>>) {
    std::thread::Builder::new()
        .name("audio-capture".into())
        .spawn(move || capture_loop(target_rate, device_name, tx))
        .expect("failed to spawn audio-capture thread");
}

/// Hold a stream open for the life of the process, rebuilding it whenever the
/// device goes away.
///
/// Without this a disconnect is silent and terminal: CoreAudio simply stops
/// calling the data callback, no audio reaches the sidecar, and the session
/// still reports ready — "listening" with a dead mic, the same symptom class as
/// a denied TCC grant. cpal cannot restart a dead stream, so recovery means
/// building a new one, which has to happen on this thread because `Stream` is
/// `!Send`.
///
/// Reopening re-resolves the device, which is what makes the recovery useful: an
/// unnamed capture follows the system default (Bluetooth mic gone, built-in
/// takes over), and the resampler is rebuilt at whatever rate the new device
/// reports. That second part matters on its own — `StreamInvalidated` fires when
/// a device changes sample rate underneath us, and the old ratio would resample
/// every buffer wrong.
fn capture_loop(target_rate: u32, device_name: Option<String>, tx: Sender<Vec<f32>>) {
    let mut failures: u64 = 0;
    loop {
        // Capacity 1: one signal is all a rebuild needs, and a full channel
        // means one is already pending.
        let (lost_tx, lost_rx) = crossbeam_channel::bounded::<()>(1);
        match open_stream(target_rate, device_name.as_deref(), tx.clone(), lost_tx) {
            Ok((stream, opened)) => {
                failures = 0;
                tracing::info!(device = %opened, "microphone capture running");
                // Block here for the life of the stream. Callbacks fire on
                // CoreAudio's own thread meanwhile and `stream` must stay alive
                // for them.
                match wait_for_rebuild(&lost_rx, device_name.as_deref(), &opened) {
                    Rebuild::DeviceLost => tracing::warn!("input device lost; reopening"),
                    Rebuild::DefaultChanged(to) => {
                        tracing::info!(from = %opened, %to, "system default input changed; reopening")
                    }
                }
                drop(stream);
            }
            Err(e) => {
                failures += 1;
                // The device can be absent for minutes — a headset in its case,
                // a lid shut — and one line per second is noise, not evidence.
                if failures == 1 || failures.is_multiple_of(60) {
                    tracing::error!(
                        error = %e,
                        attempts = failures,
                        "failed to start microphone capture; retrying"
                    );
                }
            }
        }
        std::thread::sleep(REOPEN_DELAY);
    }
}

/// Why [`wait_for_rebuild`] returned.
enum Rebuild {
    /// The stream is dead: a fatal error, or it dropped its sender.
    DeviceLost,
    /// The stream is healthy but the system default input moved elsewhere.
    DefaultChanged(String),
}

/// Block until something warrants rebuilding the stream.
///
/// Two reasons, and the second is why this polls instead of just blocking on the
/// channel: a default-device change leaves our stream perfectly healthy, so cpal
/// reports nothing at all. Without the poll we would keep capturing the old
/// device forever — audio, from the wrong microphone, with no error and no log
/// line to say so. That is the same failure shape as the disconnect this
/// function also handles, just quieter.
///
/// A **named** device never follows the default: meeting mode pins an Aggregate
/// Device by name and must not drift off it.
fn wait_for_rebuild(lost_rx: &Receiver<()>, device_name: Option<&str>, opened: &str) -> Rebuild {
    if device_name.is_some() {
        let _ = lost_rx.recv();
        return Rebuild::DeviceLost;
    }
    loop {
        match lost_rx.recv_timeout(DEFAULT_POLL) {
            // Disconnected means the stream dropped its sender — also a rebuild.
            Ok(()) | Err(RecvTimeoutError::Disconnected) => return Rebuild::DeviceLost,
            Err(RecvTimeoutError::Timeout) => {
                let current = default_input_name();
                if should_follow_default(device_name, opened, current.as_deref()) {
                    return Rebuild::DefaultChanged(current.unwrap_or_default());
                }
            }
        }
    }
}

/// Display name of the system default input, or `None` if there isn't one.
fn default_input_name() -> Option<String> {
    // cpal 0.18 exposes the device name via `Display`, not a `name()` method.
    cpal::default_host().default_input_device().map(|d| d.to_string())
}

/// Whether an open stream should be torn down and reopened on a different device.
///
/// `current` of `None` means the system reports no input device at all. There is
/// nothing better to move to, so keep whatever is still working rather than
/// rebuilding into a retry loop.
fn should_follow_default(device_name: Option<&str>, opened: &str, current: Option<&str>) -> bool {
    device_name.is_none() && matches!(current, Some(c) if c != opened)
}

/// Whether a stream error means the stream is dead and must be rebuilt.
///
/// Deliberately narrow. `DeviceChanged` documents that cpal rerouted us and the
/// stream is still live, and `Xrun`/`RealtimeDenied` are glitches — rebuilding
/// on any of those would tear down a working capture, and a recurring one would
/// loop.
fn is_fatal(kind: ErrorKind) -> bool {
    matches!(
        kind,
        ErrorKind::DeviceNotAvailable | ErrorKind::StreamInvalidated
    )
}

/// Build + start the input stream (blocking on macOS mic permission).
///
/// `device_name` selects a specific input device by name (e.g. an Aggregate
/// Device merging mic + system audio); `None` uses the system default mic.
fn open_stream(
    target_rate: u32,
    device_name: Option<&str>,
    tx: Sender<Vec<f32>>,
    lost_tx: Sender<()>,
) -> Result<(Stream, String)> {
    let host = cpal::default_host();
    let device = match device_name {
        Some(name) => {
            let mut found = None;
            // cpal 0.18 exposes the device name via `Display`, not a `name()` method.
            for d in host.input_devices().context("failed to enumerate input devices")? {
                if d.to_string() == name {
                    found = Some(d);
                    break;
                }
            }
            found.ok_or_else(|| anyhow!("input device not found: {name}"))?
        }
        None => host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device (microphone) found"))?,
    };
    // Record what we actually opened, so a later default-device change can be
    // recognized as a change rather than compared against the request.
    let name = device.to_string();
    let supported = device
        .default_input_config()
        .context("failed to read default input config")?;

    let sample_format = supported.sample_format();
    let in_rate = supported.sample_rate();
    let channels = supported.channels() as usize;
    let config: cpal::StreamConfig = supported.config();

    tracing::info!(
        in_rate,
        channels,
        ?sample_format,
        target_rate,
        "opening input stream"
    );

    let stream = match sample_format {
        SampleFormat::F32 => {
            build_stream::<f32>(&device, &config, channels, in_rate, target_rate, tx, lost_tx)
        }
        SampleFormat::I16 => {
            build_stream::<i16>(&device, &config, channels, in_rate, target_rate, tx, lost_tx)
        }
        SampleFormat::U16 => {
            build_stream::<u16>(&device, &config, channels, in_rate, target_rate, tx, lost_tx)
        }
        other => Err(anyhow!("unsupported sample format: {other:?}")),
    }?;

    stream.play().context("failed to start input stream")?;
    Ok((stream, name))
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    in_rate: u32,
    target_rate: u32,
    tx: Sender<Vec<f32>>,
    lost_tx: Sender<()>,
) -> Result<Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let mut resampler = LinearResampler::new(in_rate, target_rate);
    let mut mono: Vec<f32> = Vec::new();
    let mut dropped: u64 = 0;

    let data_cb = move |data: &[T], _: &cpal::InputCallbackInfo| {
        // Downmix interleaved frames to mono f32.
        mono.clear();
        mono.reserve(data.len() / channels + 1);
        for frame in data.chunks(channels) {
            let mut sum = 0.0f32;
            for &s in frame {
                sum += s.to_sample::<f32>();
            }
            mono.push(sum / channels as f32);
        }

        let mut out = Vec::with_capacity(mono.len());
        resampler.process(&mono, &mut out);
        if out.is_empty() {
            return;
        }
        if tx.try_send(out).is_err() {
            dropped += 1;
            if dropped % 100 == 1 {
                tracing::warn!(dropped, "ASR worker behind; dropping audio buffers");
            }
        }
    };

    let err_cb = move |err: cpal::Error| {
        let fatal = is_fatal(err.kind());
        tracing::error!(%err, fatal, "audio input stream error");
        if fatal {
            let _ = lost_tx.try_send(());
        }
    };

    let stream = device
        .build_input_stream(*config, data_cb, err_cb, None)
        .context("failed to build input stream")?;
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::{is_fatal, should_follow_default, LinearResampler};
    use cpal::ErrorKind;

    /// The rebuild trigger. `DeviceNotAvailable` is what a disconnecting
    /// Bluetooth mic reports; `DeviceChanged` documents a still-live rerouted
    /// stream, so treating it as fatal would tear down a working capture every
    /// time the system switches devices.
    #[test]
    fn only_dead_streams_trigger_a_rebuild() {
        assert!(is_fatal(ErrorKind::DeviceNotAvailable));
        assert!(is_fatal(ErrorKind::StreamInvalidated));

        assert!(!is_fatal(ErrorKind::DeviceChanged));
        assert!(!is_fatal(ErrorKind::Xrun));
        assert!(!is_fatal(ErrorKind::RealtimeDenied));
    }

    /// Feeding a signal in two arbitrary buffers must match feeding it as one
    /// (i.e. state carries across buffer boundaries with no gaps/dupes).
    #[test]
    fn streaming_matches_one_shot() {
        let input: Vec<f32> = (0..300).map(|i| (i as f32 * 0.1).sin()).collect();

        let mut whole = LinearResampler::new(48_000, 16_000);
        let mut a = Vec::new();
        whole.process(&input, &mut a);

        let mut split = LinearResampler::new(48_000, 16_000);
        let mut b = Vec::new();
        split.process(&input[..137], &mut b);
        split.process(&input[137..], &mut b);

        assert_eq!(a.len(), b.len(), "output length differs across split");
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6, "sample mismatch: {x} vs {y}");
        }
    }

    /// Following the system default. A named device must never follow — meeting
    /// mode pins an Aggregate Device and drifting off it would silently capture
    /// one side of the call.
    #[test]
    fn only_an_unnamed_capture_follows_the_default_input() {
        assert!(should_follow_default(None, "AirPods Pro", Some("MacBook Air Microphone")));

        assert!(!should_follow_default(None, "AirPods Pro", Some("AirPods Pro")));
        // No default at all: nothing better to move to, so keep what works.
        assert!(!should_follow_default(None, "AirPods Pro", None));
        assert!(!should_follow_default(
            Some("Aggregate Device"),
            "Aggregate Device",
            Some("AirPods Pro")
        ));
    }

    /// 3:1 downsample of N input frames yields ~N/3 output frames.
    #[test]
    fn downsample_ratio_is_correct() {
        let input = vec![0.5f32; 4800]; // 100 ms @ 48 kHz
        let mut r = LinearResampler::new(48_000, 16_000);
        let mut out = Vec::new();
        r.process(&input, &mut out);
        // ~1600 output frames (100 ms @ 16 kHz), within a couple of the boundary.
        assert!((out.len() as i64 - 1600).abs() <= 2, "got {} frames", out.len());
    }
}

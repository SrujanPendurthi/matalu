//! Microphone capture (cpal) + streaming downmix/resample to 16 kHz mono f32.
//!
//! The cpal callback runs on a realtime audio thread, so it must not block or
//! allocate unboundedly. It downmixes to mono, resamples to 16 kHz with a
//! stateful linear resampler, and `try_send`s the result to the ASR worker,
//! dropping data (rather than blocking) if the worker ever falls behind.

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{ErrorKind, FromSample, SampleFormat, SizedSample, Stream};
use std::sync::{Arc, Mutex};

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

/// How often capture re-checks which device it *should* be on. Neither a
/// default-device change nor a new request pushes anything at us, so it has to
/// be polled; 2 s is imperceptible against a human plugging in a headset and
/// costs one cheap CoreAudio query off the realtime thread.
const DEVICE_POLL: std::time::Duration = std::time::Duration::from_secs(2);

/// Which input device capture should be on, shared so it can change while the
/// app runs: meeting mode repoints it at an Aggregate Device (mic + system
/// audio) and clears it again on stop. `None` means follow the system default.
pub type DeviceRequest = Arc<Mutex<Option<String>>>;

/// A [`DeviceRequest`] pinned to `name`, or following the default if `None`.
pub fn device_request(name: Option<String>) -> DeviceRequest {
    Arc::new(Mutex::new(name))
}

/// Display names of every input device, for populating a device picker.
pub fn list_input_devices() -> Vec<String> {
    match cpal::default_host().input_devices() {
        // cpal 0.18 exposes the device name via `Display`, not a `name()` method.
        Ok(devices) => devices.map(|d| d.to_string()).collect(),
        Err(e) => {
            tracing::warn!(error = %e, "failed to enumerate input devices");
            Vec::new()
        }
    }
}

/// Whether an input device with this display name exists right now.
///
/// Public because the caller wants to warn *before* acting: meeting mode asks
/// for an Aggregate Device so it can hear both sides of a call, and capture
/// silently falls back to the bare mic when it is missing — which still records,
/// but only the user's half, and a one-sided transcript looks exactly like a
/// working one.
pub fn input_device_exists(name: &str) -> bool {
    list_input_devices().iter().any(|d| d == name)
}

/// Start microphone capture on a dedicated OS thread.
///
/// cpal's `Stream` is `!Send` and must be created and kept alive on one thread,
/// and on macOS `build_input_stream` blocks in CoreAudio until microphone
/// permission is resolved. Doing this on its own thread keeps that latency (and
/// any permission stall) off the server's startup path — `/health` and `/ws`
/// come up immediately regardless of mic state.
pub fn spawn_capture(target_rate: u32, request: DeviceRequest, tx: Sender<Vec<f32>>) {
    std::thread::Builder::new()
        .name("audio-capture".into())
        .spawn(move || capture_loop(target_rate, request, tx))
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
///
/// The same loop serves runtime device switching: `request` is re-read on every
/// rebuild, so pointing it at another device is enough to move capture there.
fn capture_loop(target_rate: u32, request: DeviceRequest, tx: Sender<Vec<f32>>) {
    let mut failures: u64 = 0;
    loop {
        // Capacity 1: one signal is all a rebuild needs, and a full channel
        // means one is already pending.
        let (lost_tx, lost_rx) = crossbeam_channel::bounded::<()>(1);
        let requested = requested_name(&request);
        match open_stream(target_rate, requested.as_deref(), tx.clone(), lost_tx) {
            Ok((stream, opened)) => {
                failures = 0;
                tracing::info!(device = %opened, "microphone capture running");
                // Block here for the life of the stream. Callbacks fire on
                // CoreAudio's own thread meanwhile and `stream` must stay alive
                // for them.
                match wait_for_rebuild(&lost_rx, &request, &opened) {
                    Rebuild::DeviceLost => tracing::warn!("input device lost; reopening"),
                    Rebuild::Moved(to) => {
                        tracing::info!(from = %opened, %to, "capture device changed; reopening")
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
    /// The stream is healthy but capture belongs on a different device now.
    Moved(String),
}

/// Block until something warrants rebuilding the stream.
///
/// Two reasons, and the second is why this polls instead of just blocking on the
/// channel: neither a default-device change nor a new [`DeviceRequest`] disturbs
/// the open stream at all, so cpal reports nothing. Without the poll we would
/// keep capturing the old device forever — audio, from the wrong microphone,
/// with no error and no log line to say so. Same failure shape as the disconnect
/// this function also handles, just quieter.
fn wait_for_rebuild(lost_rx: &Receiver<()>, request: &DeviceRequest, opened: &str) -> Rebuild {
    loop {
        match lost_rx.recv_timeout(DEVICE_POLL) {
            // Disconnected means the stream dropped its sender — also a rebuild.
            Ok(()) | Err(RecvTimeoutError::Disconnected) => return Rebuild::DeviceLost,
            Err(RecvTimeoutError::Timeout) => {
                let desired = desired_name(request);
                if should_reopen(desired.as_deref(), opened) {
                    return Rebuild::Moved(desired.unwrap_or_default());
                }
            }
        }
    }
}

fn requested_name(request: &DeviceRequest) -> Option<String> {
    request.lock().unwrap().clone()
}

/// The device capture should be on right now.
///
/// A request for a device that is not currently present resolves to the system
/// default — the same fallback [`open_stream`] performs. That keeps the poll
/// comparing against what is actually attainable: otherwise a request for an
/// absent Aggregate Device would differ from the fallback we opened on *every*
/// poll and rebuild the stream forever. It also self-heals, since the device
/// gets picked up on the first poll after it appears.
fn desired_name(request: &DeviceRequest) -> Option<String> {
    match requested_name(request) {
        Some(name) if input_device_exists(&name) => Some(name),
        _ => default_input_name(),
    }
}

/// Display name of the system default input, or `None` if there isn't one.
fn default_input_name() -> Option<String> {
    // cpal 0.18 exposes the device name via `Display`, not a `name()` method.
    cpal::default_host().default_input_device().map(|d| d.to_string())
}

/// Whether an open stream should be torn down and reopened elsewhere.
///
/// `desired` of `None` means there is nothing to move to — no request, and the
/// system reports no default input either. Keep whatever is still working rather
/// than rebuilding into a retry loop.
fn should_reopen(desired: Option<&str>, opened: &str) -> bool {
    matches!(desired, Some(d) if d != opened)
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
    let default = || {
        host.default_input_device()
            .ok_or_else(|| anyhow!("no default input device (microphone) found"))
    };
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
            match found {
                Some(d) => d,
                None => {
                    // Fall back rather than fail: erroring here would retry the
                    // same missing device once a second forever and capture
                    // nothing at all, which is strictly worse than the bare mic.
                    // Loud because a one-sided meeting transcript looks exactly
                    // like a working one — see `input_device_exists`.
                    tracing::warn!(
                        device = name,
                        "requested input device not found; falling back to the system default"
                    );
                    default()?
                }
            }
        }
        None => default()?,
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
    use super::{is_fatal, should_reopen, LinearResampler};
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

    /// When to move capture to another device. `desired` has already collapsed
    /// "explicit request" and "system default" into one answer, so this is the
    /// whole decision — and the `None` case is the one that matters: nothing to
    /// move to must mean stay put, or a machine with no default input rebuilds
    /// the stream every poll forever.
    #[test]
    fn capture_reopens_only_when_it_is_on_the_wrong_device() {
        // Default drifted away, or a meeting pinned an Aggregate Device.
        assert!(should_reopen(Some("MacBook Air Microphone"), "AirPods Pro"));
        assert!(should_reopen(Some("Matalu Aggregate"), "MacBook Air Microphone"));

        assert!(!should_reopen(Some("AirPods Pro"), "AirPods Pro"));
        assert!(!should_reopen(None, "AirPods Pro"));
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

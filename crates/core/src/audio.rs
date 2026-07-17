//! Microphone capture (cpal) + streaming downmix/resample to 16 kHz mono f32.
//!
//! The cpal callback runs on a realtime audio thread, so it must not block or
//! allocate unboundedly. It downmixes to mono, resamples to 16 kHz with a
//! stateful linear resampler, and `try_send`s the result to the ASR worker,
//! dropping data (rather than blocking) if the worker ever falls behind.

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, Stream};
use crossbeam_channel::Sender;

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

/// Start microphone capture on a dedicated OS thread.
///
/// cpal's `Stream` is `!Send` and must be created and kept alive on one thread,
/// and on macOS `build_input_stream` blocks in CoreAudio until microphone
/// permission is resolved. Doing this on its own thread keeps that latency (and
/// any permission stall) off the server's startup path — `/health` and `/ws`
/// come up immediately regardless of mic state.
pub fn spawn_capture(target_rate: u32, tx: Sender<Vec<f32>>) {
    std::thread::Builder::new()
        .name("audio-capture".into())
        .spawn(move || match open_stream(target_rate, tx) {
            Ok(_stream) => {
                tracing::info!("microphone capture running");
                // Keep `_stream` alive (dropping it stops capture); callbacks
                // fire on CoreAudio's own thread meanwhile.
                loop {
                    std::thread::park();
                }
            }
            Err(e) => tracing::error!(error = %e, "failed to start microphone capture"),
        })
        .expect("failed to spawn audio-capture thread");
}

/// Build + start the input stream (blocking on macOS mic permission).
fn open_stream(target_rate: u32, tx: Sender<Vec<f32>>) -> Result<Stream> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("no default input device (microphone) found"))?;
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
        SampleFormat::F32 => build_stream::<f32>(&device, &config, channels, in_rate, target_rate, tx),
        SampleFormat::I16 => build_stream::<i16>(&device, &config, channels, in_rate, target_rate, tx),
        SampleFormat::U16 => build_stream::<u16>(&device, &config, channels, in_rate, target_rate, tx),
        other => Err(anyhow!("unsupported sample format: {other:?}")),
    }?;

    stream.play().context("failed to start input stream")?;
    Ok(stream)
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    in_rate: u32,
    target_rate: u32,
    tx: Sender<Vec<f32>>,
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

    let err_cb = |err| tracing::error!(%err, "audio input stream error");

    let stream = device
        .build_input_stream(config.clone(), data_cb, err_cb, None)
        .context("failed to build input stream")?;
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::LinearResampler;

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

"""Analyze a MATALU_DUMP_WAV capture to tell a mic/capture problem from a
streaming one.

The dump is *exactly* the audio the model received live. So:
  - level/clipping/noise stats say whether the captured audio is even usable;
  - an OFFLINE transcription of the same audio (full, non-realtime) says what the
    model *can* get from it. If offline is clean but your live transcript was
    garbled -> the realtime streaming/chunking/context path is the suspect. If
    offline is ALSO garbled -> the captured audio itself is bad (mic gain, noise,
    distance), which the level stats will corroborate.

Usage: python3 sidecar/analyze_dump.py <dump.wav>
"""
import sys
import warnings

import numpy as np
import scipy.io.wavfile as wav

warnings.filterwarnings("ignore")  # quiet scipy's WavFileWarning on odd headers

path = sys.argv[1] if len(sys.argv) > 1 else "matalu_heard.wav"

try:
    sr, x = wav.read(path)
except Exception as e:
    print(f"could not read {path}: {e}")
    sys.exit(1)

if x.ndim > 1:
    x = x.mean(1)
x = x.astype(np.float32)
peak_i16 = 32768.0
xf = x / peak_i16  # normalize int16 -> [-1, 1]
n = len(xf)
dur = n / sr

if n == 0:
    print(f"{path} is empty — did you dictate before quitting? (idle feeds no audio)")
    sys.exit(1)

# --- level / quality stats ------------------------------------------------
peak = float(np.max(np.abs(xf)))
rms = float(np.sqrt(np.mean(xf**2)))
def dbfs(v):
    return -np.inf if v <= 0 else 20 * np.log10(v)

# Frame-wise RMS (20 ms) to estimate speech vs noise floor.
fr = max(1, int(0.02 * sr))
frames = xf[: n - n % fr].reshape(-1, fr)
frms = np.sqrt((frames**2).mean(1)) if len(frames) else np.array([rms])
noise_floor = float(np.percentile(frms, 10))   # quietest 10% ~ background
speech_lvl = float(np.percentile(frms, 90))     # loudest 10% ~ speech
snr_db = dbfs(speech_lvl) - dbfs(noise_floor)
clip = int(np.sum(np.abs(x) >= 32767))
clip_pct = 100 * clip / n
silence_frac = float(np.mean(frms < max(noise_floor * 2, 0.005)))

print(f"file:        {path}")
print(f"duration:    {dur:.1f}s @ {sr} Hz, {n} samples")
print(f"peak:        {peak:.3f}  ({dbfs(peak):+.1f} dBFS)")
print(f"rms:         {rms:.4f} ({dbfs(rms):+.1f} dBFS)")
print(f"noise floor: {dbfs(noise_floor):+.1f} dBFS   speech ~{dbfs(speech_lvl):+.1f} dBFS   SNR ~{snr_db:.0f} dB")
print(f"clipping:    {clip} samples ({clip_pct:.2f}%)")
print(f"silence:     {silence_frac*100:.0f}% of frames near noise floor")

# --- verdict heuristics ---------------------------------------------------
flags = []
if peak < 0.05:
    flags.append("VERY QUIET — mic gain too low or mic far away (model gets almost no signal)")
elif peak > 0.99 and clip_pct > 0.1:
    flags.append("CLIPPING — input too hot; distortion will garble ASR")
if snr_db < 15:
    flags.append(f"LOW SNR (~{snr_db:.0f} dB) — noisy environment / distant mic")
if dur < 0.5:
    flags.append("very short capture — dictate a full sentence for a fair read")
if flags:
    print("\n⚠️  audio issues detected:")
    for f in flags:
        print("   - " + f)
else:
    print("\n✅ captured audio levels look healthy (good gain, low clipping, decent SNR)")

# --- offline transcription of the exact captured audio --------------------
print("\ntranscribing the captured audio offline (full context)...")
try:
    import mlx.core as mx
    from parakeet_mlx import from_pretrained
    model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")
    csr = model.preprocessor_config.sample_rate
    sig = xf
    if csr != sr:  # dump is 16k; model is 16k — but be safe
        import scipy.signal as sps
        sig = sps.resample_poly(xf, csr, sr).astype(np.float32)
    chunk = csr // 2
    for ctx in [(128, 128), (256, 256)]:
        with model.transcribe_stream(context_size=ctx) as tx:
            for i in range(0, len(sig), chunk):
                tx.add_audio(mx.array(np.ascontiguousarray(sig[i:i+chunk])))
            print(f"  offline @ {ctx}: {tx.result.text!r}")
    print("\nIf the offline text above matches what you SAID but your live transcript")
    print("was garbled -> the realtime streaming path is the suspect. If the offline")
    print("text is ALSO wrong -> it's the captured audio (see the level flags above).")
except Exception as e:
    print(f"  (skipped transcription: {e})")

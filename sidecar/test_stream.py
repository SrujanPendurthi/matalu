"""Standalone smoke test: stream test_16k.wav through parakeet-mlx v2.
Downloads mlx-community/parakeet-tdt-0.6b-v2 on first run (~1.2 GB).
Usage: python3 sidecar/test_stream.py [wav] [model_id]
"""
import sys
import time

from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import load_audio

wav = sys.argv[1] if len(sys.argv) > 1 else "test_16k.wav"
model_id = sys.argv[2] if len(sys.argv) > 2 else "mlx-community/parakeet-tdt-0.6b-v2"

print(f"loading {model_id} ...", flush=True)
t0 = time.time()
model = from_pretrained(model_id)
sr = model.preprocessor_config.sample_rate
print(f"loaded in {time.time()-t0:.1f}s, sample_rate={sr}", flush=True)

audio = load_audio(wav, sr)
dur = len(audio) / sr
print(f"audio {dur:.1f}s; streaming in 0.5s chunks:", flush=True)

chunk = sr // 2
t1 = time.time()
with model.transcribe_stream(context_size=(256, 256)) as tx:
    for i in range(0, len(audio), chunk):
        tx.add_audio(audio[i : i + chunk])
        print("  partial:", repr(tx.result.text), flush=True)
    final = tx.result.text
elapsed = time.time() - t1
print(f"\nFINAL: {final!r}")
print(f"decoded {dur:.1f}s audio in {elapsed:.1f}s (RTF {dur/elapsed:.1f}x)")

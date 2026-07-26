"""matalu post-hoc speaker diarization (sherpa-onnx, offline).

Batch tool — NOT the streaming ASR sidecar. Run once when a meeting stops:

    python3 diarize.py <meeting.wav>

Reads a 16 kHz mono WAV and prints, on stdout, a JSON array of speaker
segments sorted by start time:

    [{"start": 0.0, "end": 3.4, "speaker": 0}, ...]

Speaker ids are anonymous clusters (0,1,2,…); the app maps them to
"Speaker 1/2/3" and lets the user rename them. Models load on run and the
process exits after — nothing stays resident (no RAM cost between meetings).

Requires `pip install sherpa-onnx` and two ONNX models under
`sidecar/models/diarization/` (or via the MATALU_DIARIZE_* env overrides). On
any missing dependency/model this exits non-zero with a message on stderr, and
the app falls back to an unlabeled transcript.
"""
import json
import os
import sys
import wave

import numpy as np

SR = 16000
_SIDECAR_DIR = os.path.dirname(os.path.abspath(__file__))
_DIA_DIR = os.path.join(_SIDECAR_DIR, "models", "diarization")
SEG_MODEL = os.environ.get("MATALU_DIARIZE_SEG_MODEL", os.path.join(_DIA_DIR, "segmentation.onnx"))
EMB_MODEL = os.environ.get("MATALU_DIARIZE_EMB_MODEL", os.path.join(_DIA_DIR, "embedding.onnx"))


def load_wav(path: str) -> np.ndarray:
    """Load a 16 kHz mono int16 WAV as float32 samples in [-1, 1]."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        if sr != SR:
            raise ValueError(f"expected {SR} Hz WAV, got {sr}")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def build_diarizer():
    import sherpa_onnx  # imported lazily so --selftest can run without it

    for label, path in (("segmentation", SEG_MODEL), ("embedding", EMB_MODEL)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"diarization {label} model not found: {path}")

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG_MODEL),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB_MODEL),
        # num_clusters=-1 → auto-detect the number of speakers via the threshold.
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    return sherpa_onnx.OfflineSpeakerDiarization(config)


def diarize(wav_path: str) -> list:
    sd = build_diarizer()
    samples = load_wav(wav_path)
    result = sd.process(samples).sort_by_start_time()
    return [
        {"start": float(s.start), "end": float(s.end), "speaker": int(s.speaker)}
        for s in result
    ]


def _selftest() -> None:
    """Runnable check: WAV round-trip always; full diarize only if models exist."""
    import tempfile

    t = np.linspace(0, 1, SR, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    path = os.path.join(tempfile.gettempdir(), "matalu-diarize-selftest.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(tone.tobytes())

    samples = load_wav(path)
    assert samples.dtype == np.float32 and len(samples) == SR, "load_wav round-trip failed"

    if os.path.isfile(SEG_MODEL) and os.path.isfile(EMB_MODEL):
        segs = diarize(path)
        assert isinstance(segs, list), "diarize did not return a list"
        print(f"selftest OK: load_wav + diarize ({len(segs)} segments)", file=sys.stderr)
    else:
        print("selftest OK: load_wav (models absent; skipped diarize)", file=sys.stderr)


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        _selftest()
        return
    if len(sys.argv) < 2:
        print("usage: diarize.py <wav> | --selftest", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(diarize(sys.argv[1])))


if __name__ == "__main__":
    main()

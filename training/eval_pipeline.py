"""Stage 0 — baseline the shipping pipeline against DisfluencySpeech ground truth.

The cleanup stage has never been measured for quality. DisfluencySpeech gives
ground truth for *both* halves of the pipeline (`transcript_a` = verbatim speech,
`transcript_c` = disfluencies removed), so we can score each stage separately.

Three measurements, because a single end-to-end score cannot tell you which
stage failed:

    audio --Parakeet--> raw       vs transcript_a   (1) ASR error alone
    transcript_a --Qwen--> ideal  vs transcript_c   (2) cleanup error alone
    raw --Qwen--> cleaned         vs transcript_c   (3) end-to-end

(2) is the number that has never existed: cleanup quality with ASR error removed
entirely. (3) - (2) is roughly what ASR error costs the cleanup stage.

Also reports the **fallback rate** — how often the guards reject the model's
output and the app pastes raw text instead. That decides whether a fine-tune is
worth anything: if cleanup already applies 95% of the time, there is little to
win. The guards are reimplemented here to mirror `src-tauri/src/cleaner.rs`;
keep them in sync.

    python3 training/eval_pipeline.py            # 200 utterances
    N_UTTS=50 python3 training/eval_pipeline.py  # quick pass
"""
import collections
import io
import json
import os
import re
import subprocess
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import soundfile as sf
from parakeet_mlx.audio import get_logmel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "sidecar"))

SR = 16000
CHUNK = SR // 2
CONTEXT = (128, 128)
ASR_BITS = int(os.environ.get("MATALU_ASR_BITS", "8"))
N_UTTS = int(os.environ.get("N_UTTS", "200"))

# Mirrors cleaner.rs — keep in sync.
MIN_OVERLAP = 0.90
MIN_LEN_RATIO = 0.50
MAX_LEN_RATIO = 1.30

_PUNCT = re.compile(r"[^a-z0-9' ]+")


def words(s):
    return _PUNCT.sub(" ", s.lower()).split()


def wer(ref, hyp):
    """Levenshtein edit distance over words -> (edits, ref_len)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1], len(ref)


def vet(raw, candidate):
    """The cleaner.rs guards. Returns the accepted text, or None to fall back."""
    candidate = candidate.strip()
    if not candidate or not raw.strip():
        return None
    ratio = len(candidate) / len(raw.strip())
    if not (MIN_LEN_RATIO <= ratio <= MAX_LEN_RATIO):
        return None
    out_w = words(candidate)
    if not out_w:
        return None
    available = collections.Counter(words(raw))
    hits = 0
    for w in out_w:
        if available[w] > 0:
            available[w] -= 1
            hits += 1
    if hits / len(out_w) < MIN_OVERLAP:
        return None
    return candidate


class Cleaner:
    """The real clean_sidecar.py over its real protocol, prompt cache and all."""

    def __init__(self):
        script = os.path.join(REPO_ROOT, "sidecar", "clean_sidecar.py")
        self.p = subprocess.Popen(
            [sys.executable, script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        while '"ready"' not in self.p.stdout.readline():
            pass

    def clean(self, text):
        """Returns (accepted_text_or_None, raw_model_output)."""
        self.p.stdin.write(json.dumps({"text": text}) + "\n")
        self.p.stdin.flush()
        reply = json.loads(self.p.stdout.readline())
        if "text" not in reply:
            return None, ""
        return vet(text, reply["text"]), reply["text"]

    def close(self):
        self.p.stdin.close()
        self.p.kill()


def load_utterances(limit):
    from datasets import Audio, load_dataset
    import librosa

    ds = load_dataset("amaai-lab/DisfluencySpeech", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    out = []
    for row in ds.select(range(limit)):
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        out.append((audio, row["transcript_a"], row["transcript_c"]))
    return out


def load_asr():
    from parakeet_mlx import from_pretrained

    model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")
    if ASR_BITS:
        nn.quantize(
            model,
            group_size=64,
            bits=ASR_BITS,
            class_predicate=lambda p, m: isinstance(m, nn.Linear) and "self_attn" not in p,
        )
    mx.eval(model.parameters())
    return model


def transcribe(model, audio):
    """Transcribe as the sidecar does: full-context decode of the utterance.

    Mirrors `matalu_sidecar.py::finalize` — partials stream for the UI, but the
    `final` (the only text the app injects) is one full-context pass over the
    buffered utterance. Streaming finals measured 6.12% WER vs 1.49% here.

    Set `MATALU_FULL_CONTEXT_FINAL=0` to score the old streaming path instead,
    for A/B against this one.
    """
    if os.environ.get("MATALU_FULL_CONTEXT_FINAL", "1") != "0":
        return model.generate(get_logmel(mx.array(audio), model.preprocessor_config))[0].text

    # Legacy streaming path. The tail must be zero-padded, not passed short: a
    # ragged final chunk makes parakeet-mlx's mel front-end compute a negative
    # frame count, surfacing as a 2**64-4096 metal::malloc. The app never hits
    # this because the sidecar reads fixed CHUNK*4 byte blocks off a pipe.
    with model.transcribe_stream(context_size=CONTEXT) as tx:
        for i in range(0, len(audio), CHUNK):
            block = audio[i : i + CHUNK]
            if len(block) < CHUNK:
                block = np.pad(block, (0, CHUNK - len(block)))
            tx.add_audio(mx.array(block))
        for _ in range(3):  # drain, as the app did
            tx.add_audio(mx.array(np.zeros(CHUNK, dtype=np.float32)))
        return tx.result.text


class Score:
    """WER plus the two metrics that catch the failure WER hides."""

    def __init__(self, name):
        self.name = name
        self.edits = self.length = 0
        self.removed = self.should_remove = 0
        self.kept = self.should_keep = 0
        self.fallbacks = self.n = 0

    def add(self, hyp, verbatim, clean, fell_back=False):
        self.n += 1
        self.fallbacks += fell_back
        e, n = wer(words(clean), words(hyp))
        self.edits += e
        self.length += n

        hyp_c = collections.Counter(words(hyp))
        # Words the reference deletes going verbatim -> clean are disfluencies.
        to_remove = collections.Counter(words(verbatim)) - collections.Counter(words(clean))
        self.should_remove += sum(to_remove.values())
        self.removed += sum((to_remove - hyp_c).values())
        # Content words that must survive.
        keep = collections.Counter(words(clean))
        self.should_keep += sum(keep.values())
        self.kept += sum((keep & hyp_c).values())

    def row(self):
        def pct(a, b, w=6, dp=1):
            # b == 0 means the metric is undefined for this row, not zero — the
            # ASR row is scored against the verbatim text, so it has nothing to
            # remove. Show a dash rather than a misleading 0.0% or nan.
            return f"{100.0*a/b:{w}.{dp}f}%" if b else f"{'—':>{w}} "

        return (
            f"{self.name:<26} {pct(self.edits, self.length, 6, 2)}  "
            f"{pct(self.removed, self.should_remove)}  "
            f"{pct(self.kept, self.should_keep)}  "
            f"{pct(self.fallbacks, self.n, 5)}"
        )


def main():
    print(f"loading {N_UTTS} utterances ...", flush=True)
    utts = load_utterances(N_UTTS)
    secs = sum(len(a) for a, _, _ in utts) / SR
    print(f"{len(utts)} utterances, {secs/60:.1f} min of audio", flush=True)

    print("loading ASR ...", flush=True)
    asr = load_asr()
    cleaner = Cleaner()

    asr_only = Score("1. ASR (vs verbatim a)")
    cleanup_only = Score("2. cleanup on truth (vs c)")
    end_to_end = Score("3. end-to-end (vs c)")
    raw_vs_c = Score("0. raw ASR, no cleanup")

    scores = (raw_vs_c, asr_only, cleanup_only, end_to_end)

    def report():
        print()
        print(f"{'':<26} {'WER':>7}  {'remove':>7} {'keep':>7} {'fallb':>6}")
        print("-" * 60)
        for s in scores:
            print(s.row())

    t0 = time.perf_counter()
    skipped = 0
    for i, (audio, a, c) in enumerate(utts, 1):
        # One bad utterance must not cost the whole run — this is a ~30 minute
        # sweep and an earlier version lost all of it to a crash at 185/200.
        try:
            raw = transcribe(asr, audio)
            # (1) ASR alone: scored against the verbatim reference.
            asr_only.add(raw, a, a)
            # (0) What you get with no cleanup at all — the bar cleanup must beat.
            raw_vs_c.add(raw, a, c)
            # (2) Cleanup with perfect ASR input.
            ideal, _ = cleaner.clean(a)
            cleanup_only.add(ideal or a, a, c, fell_back=ideal is None)
            # (3) The real pipeline.
            cleaned, _ = cleaner.clean(raw)
            end_to_end.add(cleaned or raw, a, c, fell_back=cleaned is None)
        except Exception as e:  # noqa: BLE001
            skipped += 1
            print(f"  !! utterance {i} skipped: {type(e).__name__}: {e}"[:160], flush=True)
        if i % 25 == 0:
            print(f"  {i}/{len(utts)}  ({time.perf_counter()-t0:.0f}s)", flush=True)
            report()  # partial results, so a late crash still leaves numbers

    cleaner.close()
    if skipped:
        print(f"\n{skipped} utterance(s) skipped due to errors")
    report()
    print()
    print("WER    = word error rate vs the row's reference (lower better)")
    print("remove = disfluencies correctly deleted (higher better)")
    print("keep   = content words preserved (higher better)")
    print("fallb  = guards rejected the model, raw text used instead")


if __name__ == "__main__":
    main()

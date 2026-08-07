"""Score a trained cleanup adapter against the base model, on held-out data.

Runs the **real `clean_sidecar.py`** both ways over `training/data/test.jsonl`
(250 DisfluencySpeech pairs never seen in training), applying the same
`cleaner.rs` guards production applies. So this measures the shipping path, not
an approximation of it.

Mirrors row 2 of `eval_pipeline.py` ("cleanup on truth"): ground-truth verbatim
text in, so ASR error is out of the picture and the number is pure cleanup
quality. Baselines from Stage 0:

    doing nothing   14.72% WER   <- the disfluency content
    base model      12.47% WER   <- captures only 15% of the available gain
    perfect          0.00% WER

Two metrics, not one, because either alone hides a failure: a model can score
well on removal by rewriting everything (caught by preservation), or well on
preservation by doing nothing (caught by removal).

    python3 training/eval_adapter.py [--adapter PATH] [-n 250]
"""
import argparse
import collections
import importlib.util
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_spec = importlib.util.spec_from_file_location(
    "eval_pipeline", os.path.join(REPO_ROOT, "training", "eval_pipeline.py")
)
ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep)

# A path that cannot exist, so clean_sidecar's resolve_adapter() falls through
# to the base model even when sidecar/adapters/cleanup/ is present on disk.
NO_ADAPTER = "/nonexistent-adapter-disable"


def load_pairs(limit):
    path = os.path.join(REPO_ROOT, "training", "data", "test.jsonl")
    if not os.path.exists(path):
        sys.exit(f"missing {path} — run: python3 training/build_dataset.py")
    pairs = []
    for line in open(path):
        msgs = json.loads(line)["messages"]
        verbatim = next(m["content"] for m in msgs if m["role"] == "user")
        clean = next(m["content"] for m in msgs if m["role"] == "assistant")
        pairs.append((verbatim, clean))
    return pairs[:limit]


def score(adapter, pairs, label):
    env = dict(os.environ)
    env["MATALU_CLEAN_ADAPTER"] = adapter
    old = os.environ.get("MATALU_CLEAN_ADAPTER")
    os.environ["MATALU_CLEAN_ADAPTER"] = adapter
    try:
        cleaner = ep.Cleaner()
    finally:
        if old is None:
            os.environ.pop("MATALU_CLEAN_ADAPTER", None)
        else:
            os.environ["MATALU_CLEAN_ADAPTER"] = old

    s = ep.Score(label)
    t0 = time.perf_counter()
    for i, (verbatim, clean) in enumerate(pairs, 1):
        accepted, _raw = cleaner.clean(verbatim)
        # Rejected by the guards => production pastes the input unchanged.
        s.add(accepted or verbatim, verbatim, clean, fell_back=accepted is None)
        if i % 50 == 0:
            print(f"    {i}/{len(pairs)}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    cleaner.close()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=os.path.join(REPO_ROOT, "sidecar/adapters/cleanup"))
    ap.add_argument("-n", type=int, default=250)
    args = ap.parse_args()

    if not os.path.isdir(args.adapter):
        sys.exit(f"no adapter at {args.adapter} — is training finished?")

    pairs = load_pairs(args.n)
    print(f"{len(pairs)} held-out pairs from test.jsonl\n")

    # "Doing nothing" is the bar: it is what the app produces when the guards
    # reject, so any model scoring worse than this is actively harmful.
    untouched = ep.Score("0. no cleanup at all")
    for verbatim, clean in pairs:
        untouched.add(verbatim, verbatim, clean)

    print("scoring base model ...", flush=True)
    base = score(NO_ADAPTER, pairs, "1. base model")
    print("scoring adapter ...", flush=True)
    tuned = score(args.adapter, pairs, "2. + QLoRA adapter")

    print()
    print(f"{'':<26} {'WER':>7}  {'remove':>7} {'keep':>7} {'fallb':>6}")
    print("-" * 60)
    for s in (untouched, base, tuned):
        print(s.row())

    ceiling = 100.0 * untouched.edits / untouched.length
    for s, name in ((base, "base"), (tuned, "tuned")):
        got = 100.0 * s.edits / s.length
        print(f"\n{name}: captures {100*(ceiling-got)/ceiling:5.1f}% of the available improvement")


if __name__ == "__main__":
    main()

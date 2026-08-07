"""Build the QLoRA training set for the cleanup model, from DisfluencySpeech.

`transcript_a` is verbatim speech (fillers, repeats, false starts) and
`transcript_c` is the same utterance with all of that deleted and nothing else
changed — exactly the deletion-only behavior the cleanup stage is supposed to
have, and already punctuated and true-cased like Parakeet's output.

Only **a → c** is used. Emitting a → b as well (as an earlier plan said) would
put two different targets on the same input: `b` keeps false starts and `c`
removes them. The model would learn the average of two contradictory labels.
`c` is what the shipping system prompt asks for, so `c` is the only target.

Rows where `a == c` are kept deliberately: they are naturally-occurring identity
pairs, and they are what teaches the model to leave already-clean text alone.
Stage 0 measured the base model *under*-editing, so the balance matters.

The system prompt is copied verbatim from `sidecar/clean_sidecar.py` so training
matches inference exactly.

    python3 training/build_dataset.py        # -> training/data/{train,valid,test}.jsonl
"""
import importlib.util
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "training", "data")


def system_prompt() -> str:
    """The live prompt from the sidecar — never a copy that can drift.

    `TUNED_SYSTEM_PROMPT`, not `SYSTEM_PROMPT`: the sidecar switches to the short
    prompt whenever an adapter is loaded, so that is the prompt this adapter will
    actually see at inference. Training on the long one would teach the model a
    context it never gets.
    """
    path = os.path.join(REPO_ROOT, "sidecar", "clean_sidecar.py")
    spec = importlib.util.spec_from_file_location("clean_sidecar", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TUNED_SYSTEM_PROMPT


def build(split, prompt, seen):
    rows, identity, dupes = [], 0, 0
    for r in split:
        a = (r["transcript_a"] or "").strip()
        c = (r["transcript_c"] or "").strip()
        if not a or not c:
            continue
        if (a, c) in seen:
            dupes += 1
            continue
        seen.add((a, c))
        if a == c:
            identity += 1
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": a},
                    {"role": "assistant", "content": c},
                ]
            }
        )
    return rows, identity, dupes


def main():
    from datasets import load_dataset

    prompt = system_prompt()
    print(f"system prompt: {len(prompt)} chars (from sidecar/clean_sidecar.py)")

    ds = load_dataset("amaai-lab/DisfluencySpeech")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Dedupe across splits too: a pair leaking from train into test would make
    # the eval report memorisation as skill.
    seen = set()
    total = 0
    for split_name, out_name in (("train", "train"), ("validation", "valid"), ("test", "test")):
        rows, identity, dupes = build(
            ds[split_name].remove_columns("audio"), prompt, seen
        )
        path = os.path.join(OUT_DIR, f"{out_name}.jsonl")
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        pct = 100.0 * identity / len(rows) if rows else 0.0
        print(
            f"  {out_name:5s} {len(rows):5d} pairs  "
            f"({identity} identity, {pct:.0f}%)  {dupes} dupes dropped  -> {path}"
        )
        total += len(rows)

    print(f"\n{total} pairs total. Train with:")
    print("  mlx_lm.lora --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \\")
    print("    --train --data training/data -c training/lora_config.yaml")


if __name__ == "__main__":
    sys.exit(main())

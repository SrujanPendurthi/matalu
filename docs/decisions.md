# Measurement archive

Evidence behind the verdicts in `CLAUDE.md`. The verdicts there are the
load-bearing part; this file is the numbers they came from. Read it before
re-opening one of those decisions — that is the only reason it exists.

Absolute numbers are only comparable **within** a table: the runs differ in
sample size, sampling order, and corpus. Never quote a row against a row from
another table.

## Cleanup adapter vs. base model

250 held-out DisfluencySpeech pairs, through the real sidecar with production
guards (`training/eval_adapter.py`).

| | WER | disfluencies removed | content kept | guard fallback |
|---|---|---|---|---|
| no cleanup | 13.67% | 0% | 100% | — |
| base model | 11.95% | 21.8% | 96.7% | 4.8% |
| **+ adapter** | **3.27%** | **54.4%** | **99.3%** | **0.0%** |

Removal more than doubled *while* preservation rose — those normally trade
off, so it learned the deletion-only transform rather than "edit harder". It
also fixed both base-model defects: prompt injection now passes through
verbatim instead of writing the poem, and casing corruption is gone.

Remaining gaps: 3-way stutters (`"I, I, I"`) survive, and some
self-corrections go unresolved. Both are it erring conservative, which is the
requested direction.

**Caveat:** the split is held-out but same-corpus (single-speaker studio
DisfluencySpeech). Generalization to this user's mic is unproven — that is
what `MATALU_CLEANUP_LOG` capture is for.

## Adapter fusion — four variants, all lose

100-pair subset (not the 250-pair split above; that is why the shipping row
reads 2.84% here and 3.27% there).

| | WER | removed | kept | fallback | mem | latency |
|---|---|---|---|---|---|---|
| base model | 11.27% | 21.8% | 96.7% | 4.0% | 860 MB | 485 ms |
| **base + adapter** | **2.84%** | **54.0%** | **99.0%** | **0.0%** | **860 MB** | 595 ms |
| fused, 4-bit requant | 9.67% | 34.2% | 96.4% | 5.2% | 839 MB | 508 ms |
| fused → mixed 4/8-bit | 3.15% | 53.1% | 99.0% | 0.0% | 1200 MB | 723 ms |
| fused → 8-bit | 3.62% | 54.4% | 99.0% | 0.4% | 1500 MB | 853 ms |
| fused `--dequantize` | 3.29% | 54.4% | 99.3% | 0.0% | 3087 MB | 1381 ms |

- **4-bit requant destroys the tune.** The delta is full-precision and small
  relative to the 4-bit step, so it rounds away.
- **`--dequantize` preserves it exactly**, which proves requantization is the
  culprit — but bf16 reads 3087 MB per token, making it 2x *slower* than the
  adapter.
- **Mixed 4/8-bit is the best fusion**, built with a custom `quant_predicate`
  protecting the 112 LoRA-touched modules (layers 12–27, all seven
  projections) at 8-bit and the rest at 4-bit — 6.441 bits/weight. It recovers
  nearly all the quality (9.67% → 3.15%) and beats AWQ's premise, since we
  *know* which weights changed rather than inferring salience from calibration
  data. It still loses on all three axes.

## ASR: streaming vs. full context

Random 60-utterance LibriSpeech test-clean sample, 8-bit.

| config | WER |
|---|---|
| full context (`model.transcribe`, non-streaming) | **1.49%** |
| streaming `context_size=(128,128)` — what the app runs | **6.12%** |
| nvidia published, test-clean | 1.69% |

At full context we match published, so the model and the quantization are
fine — the loss is entirely the streaming configuration. `(256,256)` measured
identical to `(128,128)`, so context size within streaming is not the knob.
For scale: streaming costs ~4.6 WER points while the entire cleanup LLM buys
1.33 (see `training/eval_pipeline.py`).

## Two-stage error decomposition

`training/eval_pipeline.py`, 200 DisfluencySpeech utts, current config.

| | WER |
|---|---|
| ASR alone (vs verbatim) | 4.37% |
| cleanup alone (perfect input) | 4.36% |
| **end-to-end** | **9.00%** |

4.37 + 4.36 ≈ 9.00 — the two error sources compound almost additively, and
content preservation drops 98.8% → 95.0% between clean input and ASR input
purely from the cleanup stage mishandling ASR errors. That cascade is
structural to any two-stage design and no amount of tuning either stage
removes it. A single audio → cleaned-text model has one error source; see the
end-to-end note in `CLAUDE.md` for why it stays unbuilt.

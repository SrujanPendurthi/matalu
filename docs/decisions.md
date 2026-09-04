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

## ASR: WER vs. utterance length

276 LibriSpeech test-clean utterances, 8-bit, **full-context** decode — the path
`finalize()` takes, so this describes the text the app injects, not streaming
partials. Sampled 40 per duration bucket (0–2 s had only 36 available) so no
bucket dominates the aggregate. `x RT` is decode throughput: audio seconds per
wall second.

| bucket | n | mean dur | WER agg | WER mean | x RT |
|---|---|---|---|---|---|
| 0–2 s | 36 | 1.8 s | **4.93%** | **7.04%** | 10.9x |
| 2–4 s | 40 | 3.0 s | 0.29% | 0.31% | 14.9x |
| 4–6 s | 40 | 4.8 s | 1.91% | 1.78% | 17.4x |
| 6–8 s | 40 | 6.9 s | 1.61% | 1.66% | 19.0x |
| 8–12 s | 40 | 9.8 s | 1.54% | 1.69% | 23.0x |
| 12–20 s | 40 | 14.7 s | 1.90% | 1.86% | 25.7x |
| 20 s+ | 40 | 23.4 s | 1.78% | 1.80% | 25.4x |

`WER agg` is total edits / total reference words. `WER mean` is the mean of
per-utterance WER, which weighs short utterances equally.

**There is no length at which the model gets worse — only a penalty for being
too short.** Above ~4 s it is flat out to 23 s (1.91, 1.61, 1.54, 1.90, 1.78),
scattering around nvidia's published 1.69% with no trend.

**Do not quote the 2–4 s row.** At 0.29% it is roughly one word error in the
whole bucket, an easy draw — and it is the reason the raw per-bucket minimum is
meaningless here. Between-bucket scatter is ±0.3 points and is driven by speaker
and content difficulty, which sampling balanced by *length* does not control.

**The trustworthy signal is the mean-vs-aggregate gap, not the aggregate.**
It is **+2.11 points at 0–2 s** (4.93 → 7.04) and ±0.13 in every other bucket.
That comparison is within-bucket — same utterances, same content — so speaker
and content difficulty cancel instead of dominating. It is the same effect that
inflated the bit-width sweep ~2.5x when that sampled shortest-first, isolated.

Consequences:

- **`MATALU_MAX_UTTERANCE_MS` (12000) needs no change.** Raising it buys nothing
  on accuracy (flat past 12 s) and nothing on speed (throughput saturates at
  ~25x by 12–20 s and does not improve at 23 s). It happens to sit in the right
  place for reasons unrelated to why it was chosen — it was picked to bound
  streaming-context growth.
- **Short dictations are the worst ASR case.** "yes", "send it" land in the 3x
  bucket. Anything that fragments speech into shorter utterances — a more
  aggressive VAD, a lower `MATALU_SILENCE_MS` — costs real WER, so the 700 ms
  default is load-bearing.

Caveats: read speech, single channel, and 40 per bucket means ±0.3 points is
noise — enough to rule out a trend, not to rank the flat buckets against each
other.

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

## Meeting mode: sidecar back-pressure (no defect found)

Investigated 2026-09-04, **hypothesis disproven**, no code change. Recorded so
it is not re-derived.

The claim: the sidecar is single-threaded, so while `finalize()` decodes it is
not reading stdin. In meeting mode capture runs continuously at `gate::REAL`
(it never passes through `SILENCE`), so speech arriving during a 12 s
`MATALU_MAX_UTTERANCE_MS` finalize would overflow and be dropped at
`pipeline.rs`'s gate `try_send` — silently, since the WAV is written *before*
the forward and would keep every buffer the transcript lost.

Headroom before any drop: the bounded(64) sidecar queue plus the macOS stdin
pipe buffer, measured at **65536 bytes = 16384 f32 = 1.024 s** @16 kHz.

`finalize()` cost, real speech through the sidecar's own `resolve_model` /
`quantize_model` / `finalize`:

| audio | finalize | x realtime |
|---|---|---|
| 1 s | 53 ms | 18.8x |
| 3 s | 153 ms | 19.7x |
| 6 s | 229 ms | 26.2x |
| **12 s** (the cap) | **431 ms** | **27.9x** |

Cost is **sublinear** — 8.09x for 12x the audio, so the cap is not a cliff.

End to end, 60 s of speech fed to the real sidecar at realtime cadence with
`MATALU_DUMP_WAV` capturing exactly what it received:

| | |
|---|---|
| audio received | **60.00 s of 60.0 s, gap +0.00 s** |
| write lateness max / p95 / mean | **−495 / −495 / −496 ms** |
| finals emitted | 5 in 60 s = one per 12 s (the cap fired repeatedly) |

Lateness is **negative**: every write completed ~half a chunk before the next
was due. The sidecar is never the bottleneck, back-pressure never reaches the
queue, and the gate never drops.

The reasoning was wrong because it framed `finalize()` against the ~1.1 s of
slack that streaming at ~1.1x RTF leaves per 12 s utterance. `finalize()` runs
at ~28x realtime — the pipe buffer alone covers it with 2.4x margin, before the
64-slot queue counts at all.

What did change: the gate's forwards now count dropped buffers and warn, the
same idiom `matalu::audio`'s capture callback already used. Justification is not
this bug but that the margin above is hardware- and model-dependent and nothing
else would report it shrinking. Re-measure if the ASR model, bit width, or
streaming context changes.

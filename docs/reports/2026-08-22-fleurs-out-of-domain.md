> **WITHDRAWN 2026-08-22. Every number below is void.** The FLEURS `id` is the
> *sentence* id, not the recording id: the 650-row test split carries only 333 distinct
> values because several speakers read the same sentence. The fetcher named wav files by
> that id, so ~317 recordings were overwritten by a later row, and the evaluator keyed
> per-utterance results by it, collapsing 650 rows to 333 records while the aggregate
> counters still summed all 650 — many of them scoring a reference against the wrong
> audio. Found by external review after publication, not by the design. Both scripts now
> key by row index and refuse to emit a number unless every row survives. The rerun is
> `exp-2026-08-22-fleurs-out-of-domain-rerun`.
>
> The one conclusion that survives, because it never depended on the measurement: the
> claim that these adapters are "the best free Greek model" was never supported by
> anything, and must not appear.

# Outside our domain, the fine-tuning does nothing — in either direction

2026-08-22 · `exp-2026-08-22-fleurs-out-of-domain` · CLOSED

Every number this project holds is Greek council speech, scored against references we
wrote ourselves. This is the one measurement that steps outside that: other speakers,
other material, references we did not write.

## What was run

FLEURS `el_gr` test split — 650 utterances, 1.91 hours, mean 10.6 s, 15,175 reference
tokens. Same machine, same CT2 float16 build, same frozen `CONTROL` decode config, same
`ftoks` normalisation and `sdi` counter as the validation scorer, same per-item seeding
discipline. Script: [`eval/controlled_eval/fleurs_el_eval.py`](../../eval/controlled_eval/fleurs_el_eval.py).

## Result

| system | WER | del | ins | sub |
|---|---:|---:|---:|---:|
| base whisper-large-v3 | 0.10478 | 0.00929 | 0.00521 | 0.09028 |
| `artifact-adapter-fixed` (published) | 0.10201 | 0.00804 | 0.00659 | 0.08738 |
| `cont_s13` (clean-pack) | 0.10122 | 0.00613 | 0.00540 | 0.08969 |

Paired bootstrap over utterances, 10,000 resamples:

| contrast | Δ WER | 95% CI |
|---|---:|---|
| published vs base | −0.00288 | [−0.00876, +0.00297] |
| `cont_s13` vs base | −0.00376 | [−0.00851, +0.00113] |
| `cont_s13` vs published | −0.00088 | [−0.00640, +0.00445] |

**Every interval contains zero.**

## What it means

The fine-tuning is a domain adaptation, not a general improvement to Greek ASR. It does
not degrade general Greek — no catastrophic forgetting, which was a real risk worth
ruling out — and it does not improve it either. The gains measured on council speech are
gains on council speech.

The practical consequence: **the claim that these adapters are "probably the best free
Greek model" is unsupported and must not appear.** Claims are restricted to Greek
council speech, which is what was measured.

## Limits

FLEURS is *read* speech. Both adapters were trained for spontaneous, multi-speaker,
room-acoustic council recordings. A null result on read sentences does not predict
behaviour on other spontaneous Greek, and it is the easier material: base whisper scores
0.105 here against 0.199 on the post-june council benchmark.

`CONTROL` was frozen for ~25 s council windows and is very likely not the best decode
config for 10–20 s read sentences. It was kept deliberately, because choosing a config
per benchmark is how this project has produced false wins before, and a config that
flatters no arm still ranks them fairly. The consequence is that the absolute WERs here
should not be compared to published FLEURS numbers — only to each other.

VoxPopuli would have been the better benchmark: European Parliament speech, the closest
public match to council proceedings. Its transcribed ASR subset covers 16 languages and
Greek is not among them, so it cannot produce a WER at all.

# HParl (`ddamianos/hparl`): does the text match the audio?

2026-08-14. Probe, not an experiment. Code: [`eval/hparl_probe.py`](../../eval/hparl_probe.py).

## Question

The standing warning about HParl
([2026-08-11-improvement-research.md](2026-08-11-improvement-research.md) §«Το πολιτικό
dataset») was that its targets are the Βουλή's **official minutes** — edited text,
force-aligned — so a share of rows would be text with no audio behind it, the same
defect this project measured in its own data (17.5% disagreement). Before spending
anything on 120 hours, check the pairs against an independent ASR.

## What was measured

100 random utterances, 16 speakers, two shards (`test-00000`, seed 20260814, n=40;
`train-00023`, seed 7, n=60). Each row's float32 audio was encoded to 16 kHz mono
32 kbps MP3 and transcribed by Soniox (`file_transcribe.py --lang el`, the same tool
used for the gap2/gap3 verification). Soniox output was aligned against the dataset
`sentence` on `eval.scoring.greek_normalize` tokens, and the errors were split by
direction. Soniox is a second opinion, not ground truth; nothing here says which side
is right when they disagree.

| slice | n | ref tokens | WER | sub | del (text w/o audio) | ins (speech not in text) | exact |
|---|---|---|---|---|---|---|---|
| all rows | 100 | 1201 | 0.132 | 0.046 | 0.077 | 0.009 | 26% |
| rows without `[UNK]` | 49 | 586 | **0.061** | 0.032 | 0.019 | 0.010 | 53% |
| rows with `[UNK]`, placeholder dropped | 51 | 553 | 0.114 | 0.061 | 0.040 | 0.013 | 27% |

## Findings

**1. The faithfulness worry is not what disqualifies HParl.** On rows whose text has
no placeholder, Soniox and the minutes disagree at 6.1% WER with a 1.9% deletion rate
— that is inside the range of Soniox's own error on hard chamber audio, and the median
row is an exact match. These are not rewritten minutes; they read like transcripts.

**2. `[UNK]` is pervasive.** 55.5% / 59.5% / 51.8% of rows in the three shards read
(`test-00000`, `train-00000`, `train-00023`) contain at least one `[UNK]` placeholder;
62 placeholders across the 100 probed rows. This is what drives the pooled 7.7%
deletion rate — the token is in the target, and there is real speech under it. Training
on these targets teaches the model to emit a literal placeholder or to delete.

**3. The targets are stripped text, and this is the decisive problem.** Zero rows in
any shard carry an accent or a punctuation mark: the references are lowercase,
unaccented, unpunctuated. Our labels are accented, punctuated, clean-verbatim Greek.
Training whisper-large-v3 against HParl targets as they ship would attack the
orthography the corrected adapter already gets right — this is not a mixture-ratio
question, it is a different output alphabet.

**4. Segments are alignment windows, not utterances.** Median 9 reference tokens,
median duration 4.7 s, cut on ~3 s frame boundaries (`utt_id` carries the frame span).
Words are chopped at the edges, which is a second, smaller source of disagreement.

## What this costs to store

MP3 at 32 kbps mono is 4.1 kB/s of audio: the full ~120 h is ≈ **1.8 GB** (vs 12.2 GB
of parquet download, 25.5 GB decoded). Conversion is cheap and the probe does it
per-row, so a full pass is affordable if the corpus ever earns one.

## Verdict

HParl is **usable audio with unusable targets**. Two honest routes, neither free:

- **Audio only, pseudo-labelled.** Ignore the minutes, transcribe with an ASR we
  trust, train on that. This is distillation of that vendor, and it inherits the
  vendor's deletions — the exact failure mode `exp-2026-08-13-targeted-deletion-training`
  is trying to remove. Not recommended as-is.
- **Repaired targets.** Keep only `[UNK]`-free rows (≈45%, ≈54 h), restore accents and
  punctuation, and accept that the restoration model now sets the label standard.

Both leave the earlier ranking intact: the 199 in-domain hours already identified beat
this. HParl stays where the 2026-08-11 review put it — «αν μείνει χρόνος».

Licence remains unverified at source (HF mirror says `cc-by-4.0`, CLARIN 1602 is
reported as CC BY-NC 4.0). Nothing above required accepting either.

## Addendum, same day: the second mirror is the usable one

`Elormiden/Hellenic-greek-parliamentary-speech` is the same CLARIN corpus processed
for ML (92,133 clips, 13.8 GB, train/validation/test). It fixes two of the three
defects above and keeps one:

- **accented**: 99.9% of rows carry tones. No `[UNK]` anywhere.
- **still lowercase and unpunctuated** (0% of rows have a punctuation mark).
- `<spoken_noise>` markers on **59%** of rows — untranscribed audio events, the
  structural equivalent of the other mirror's `[UNK]`.

The `ddamianos/hparl` local cache was deleted. Filter and page:
[`eval/hparl2_filter.py`](../../eval/hparl2_filter.py),
[`scripts/hparl2_review_page.py`](../../scripts/hparl2_review_page.py).

**Filter run**: 150 random rows of `test-00000-of-00003` (seed 20260814), MP3 at
4.0 kB/s, Soniox, `<...>` tags stripped, alignment = 1 − WER on `greek_normalize`
tokens.

| gate | rows kept | share of audio |
|---|---|---|
| align ≥ 0.80 | 121/150 (81%) | 87% |
| align ≥ 0.90 | 88/150 (59%) | 65% |
| **align ≥ 0.95** | **74/150 (49%)** | **50%** |
| align = 1.00 | 67/150 (45%) | 42% |

Pooled WER 0.093 (sub 0.047 / del 0.028 / ins 0.018). Extrapolated, the 0.95 gate
keeps ≈ **60 h of the 120 h**.

Two things to hold onto before treating that as 60 usable hours:

- **The gate is nearly an exact-match test.** Median row is 9 reference tokens, so
  ≥0.95 alignment means zero token errors on almost every row. It selects rows where
  Soniox and the transcript agree perfectly — which also selects *easy audio*, and
  makes the surviving set a Soniox-agreement sample, not a random one.
- ~~`<spoken_noise>` is not the discriminator. Keep rate is 48% on tagged rows and 52%
  on untagged — statistically the same.~~ **Wrong at n=150; corrected below.**

### Pilot pass: 10,133 rows, 28 shards

The 150-row sample held up on the headline number and broke on the side claim.

| | 150 rows, one shard | 10,133 rows, 28 shards |
|---|---|---|
| keep @0.95 | 49% | **46.3%** |
| pooled WER | 0.093 | 0.100 |
| del / ins | 0.028 / 0.018 | 0.030 / 0.017 |
| tagged rows keep | 48% | **40.8%** |
| untagged rows keep | 52% | **54.1%** |

Per-shard keep rate is 42–52% (median 46%), so the estimate is stable across the
corpus: **6.18 h kept out of 13.57 h processed**, extrapolating to ≈ **55 h of the
120 h**.

**Correction: `<spoken_noise>` *is* a discriminator.** At n=150 the 48%/52% gap was
noise and was reported as "not the discriminator". At n=10,133 it is 40.8% vs 54.1% —
a tagged row is materially less likely to survive, which is what should have been
expected: the tag marks audio the transcript does not spell out, and the ASR hears it.
It is still not a *sufficient* filter — 41% of tagged rows do pass — so admission
stays with the alignment score, but the tag is real signal and the earlier sentence
was an artifact of a small sample.

The unpunctuated, lowercase target problem from the main probe is **unchanged** and
still applies to whatever survives the filter.

**Throughput — the 3-worker default was inherited, not measured.** The gap2 verifier
used 3 concurrent Soniox calls and this pipeline copied it. Measured on 20–30 clip
batches, all with zero errors:

| workers | rows/min | vs realtime |
|---|---|---|
| 3 | 43.6 | 2.8× |
| 12 | 126.4 | 9.0× |
| 24 | 127.0 | 9.0× |
| 40 | 176.5 | 13.0× |

So the realtime path was never the wall it looked like: at 16 workers a full 92,133-clip
pass is on the order of **~12 hours of ASR**, not 41, and the bottleneck moves to the
~460 MB parquet range-read per shard. The async batch API (`SONIOX_API_KEY`, unset)
would still be the right tool for a whole-corpus pass, but it is no longer required.

## Punctuation restoration, same day

The kept rows still have the corpus's lowercase, unpunctuated targets. Each side
carries half a good label: the dataset has the **word sequence** (accented, human
derived), Soniox has the **punctuation, casing and sentence structure**. So the
label to build is *dataset words + Soniox punctuation*.
[`eval/hparl2_punctuate.py`](../../eval/hparl2_punctuate.py) runs two routes over the
74 kept rows:

- **`transfer()`** — deterministic token alignment; matched tokens inherit the
  Soniox token's leading capital and trailing punctuation. Puts a mark on 96% of
  rows. It cannot know whether a segment ends a sentence, so it copies Soniox's
  terminal period onto fragments: 64/74 rows end in a final mark.
- **`gpt-5.6-luna`** via the codex bridge, batches of 25, told explicitly that most
  segments are cut mid-sentence and must stay open.

Every model output is checked against a hard word guard — the `greek_normalize`
token sequence must be identical to the dataset transcript, or the row falls back to
`transfer()`. **74/74 passed; the model changed no words.** It differs from
`transfer()` on 61/74 rows, almost entirely by removing spurious terminal periods
and de-capitalising segments that begin mid-sentence: it marks only **19/74 (26%)**
as complete sentences, and its terminal punctuation matches that judgement on
19/19 — no fragment ends in a period.

That 26% is the number to hold onto. Three quarters of this corpus's segments are
sentence fragments, and a naive punctuation pass would have taught the model to end
a sentence every ~5 seconds.

Review page phase 2: `scripts/hparl2_review_page.py --only-kept` shows both
punctuation routes under each clip.

## The human calibration: the 0.95 gate is too strict

The user's exported verdicts arrived 2026-08-14 (43 judged rows, recovered from the
laptop). The two review rounds asked different questions, so they were separated by
regenerating both page item lists deterministically from their seeds: 29 rows were
gate judgements only, 13 punctuation judgements only, 1 appeared on both pages and is
excluded as ambiguous.

**Round 1 — is this row good?** (n=29, stratified across the alignment range)

| alignment band | n | judged good | |
|---|---|---|---|
| 1.00 | 6 | 6 | 100% |
| 0.85–0.95 | 8 | 7 | **88%** — all rejected by the gate |
| 0.50–0.85 | 8 | 6 | **75%** — all rejected by the gate |
| < 0.50 | 7 | 0 | 0% — correctly rejected |

The gate's **precision is perfect**: every row it keeps, the human accepts, 6/6.
Its **recall is bad**: of the rows it threw away, **57% were judged good**. The real
cliff in this data is at ~0.50, not 0.95. The 0.95 threshold was my choice, never
validated, and it has been discarding usable audio at roughly a 1:1 rate with what it
keeps.

What that costs, on the 10,133-row pilot:

| gate | rows kept | hours kept | extrapolated to 120 h |
|---|---|---|---|
| 0.95 | 46% | 6.18 | 55 h |
| 0.90 | 61% | 8.81 | 78 h |
| **0.85** | **72%** | **10.50** | **93 h** |
| 0.80 | 82% | 11.84 | 105 h |

Moving to 0.85 nearly doubles the usable hours and, on this evidence, admits rows the
human accepts 88% of the time. It also weakens the standing caveat that the gate
selects only easy audio — a looser gate is a less biased sample.

**Not adopted silently.** n is 6–8 rows per band; that is a signal, not a calibration.
Before the gate moves, it needs a second round with ~30 rows in each of the 0.80–0.95
and 0.50–0.80 bands, judged blind to the alignment score, which the current page shows.
The `align ≥ 0.95` pack stays as built; a `0.85` pack is a separate build, and the
choice is preregistered before training, not after seeing a WER.

**Round 2 — is the punctuation right?** (n=13, all kept rows) 11/13 accepted, 85%,
which matches the verbal report of occasional comma-where-a-period-belongs. Both
rejections are at align 1.00, so they are punctuation complaints, not row-quality ones.

## Blind round: the gate is measuring Soniox, not the labels

Second calibration, 2026-08-14, all 62 clips judged. Audio and candidate text only —
no score, no verdict, no diff, key map kept off the served directory, order shuffled,
the 43 earlier-judged rows excluded. Question: *does the audio say this text?*

| band | n | ναι | δεν ξέρω | όχι |
|---|---|---|---|---|
| control >0.99 (hidden) | 6 | **6** | 0 | 0 |
| 0.80–0.95 | 25 | **25** | 0 | 0 |
| 0.50–0.80 | 25 | 22 | 3 | 0 |
| control <0.40 (hidden) | 6 | 3 | 1 | **2** |

The good control passes 6/6. **The bad control fails**: half of the rows the gate
scores below 0.40 were judged to say their text. Those are 1–2 s clips where Soniox
returned something unrelated and a human had no trouble — so a low alignment score
mostly means *Soniox failed*, not *the label is wrong*. The gate has been measuring
the ASR's confidence and charging the corpus for it.

Against the 62 human judgements:

| gate | admits | ναι | δ.ξ | όχι | good rows it throws away |
|---|---|---|---|---|---|
| **≥0.95 (as built)** | 6 | 6 | 0 | 0 | **50** |
| ≥0.85 | 26 | 26 | 0 | 0 | 30 |
| ≥0.50 | 56 | 53 | 3 | **0** | 3 |
| ≥0.50 + dur ≥1.5 s | 53 | 51 | 2 | **0** | 5 |

Both rejections sit at align 0.00 **and** under 1.5 s — the failure mode is short
clips, and a duration floor removes it without touching the alignment score.

**The «δεν ξέρω» answers were the most informative result.** The user's description —
"it did say the text, but the start was cut, I heard half the first word" — is the
boundary-clipping failure, and the edge flags catch it exactly: all 4 have a missing
first or last reference token. But those flags also fire on 28 rows the human
approved, so they are high-recall and low-precision: a **down-weighting signal, not an
admission rule**.

Yield of the candidate gates on the 10,133-row pilot:

| gate | rows | hours | extrapolated |
|---|---|---|---|
| ≥0.95 (as built) | 46% | 6.18 | 55 h |
| ≥0.85 | 72% | 10.50 | 93 h |
| **≥0.50 + dur ≥1.5 s** | **93%** | **13.22** | **117 h** |

**What must stay attached to this.** 56 of 62 rows were accepted in a sample
deliberately enriched with low-alignment rows, which is a suspiciously good result.
Part of it is real — Soniox is being blamed for its own errors. Part of it is the
question: judges were told to ignore truncation, so a clipped-but-correct row counts
as ναι, and clipped audio is *not* harmless as a training label — it can teach the
model to emit a word that is only half present. That is why the edge flags survive as
a weighting signal even though they cannot gate.

n is still 62, with 6 rows in each control. This is enough to say **0.95 is wrong**;
it is not enough to fix the exact floor.

## From probe to a training pack

The output of all of the above is now a **pack**: the reusable unit an outside corpus
enters a fine-tune as, defined in
[`docs/reference/external-source-packs.md`](../reference/external-source-packs.md) and
built by [`scripts/build_training_pack.py`](../../scripts/build_training_pack.py).

```
~/.cache/oc-public/training-sets/hparl2-v1/
  audio/  train.jsonl  meta.json  README.md
```

`train.jsonl` rows carry `audio` (absolute), `text` (the restored target), `dur`,
`align`, `complete_sentence`, and both inputs (`text_dataset`, `text_asr`) so any
label can be re-derived. It is consumed directly by `notebooks/train_runpod.py` with
`PACK_MANIFEST=<...>/train.jsonl PACK_ARM=pn` — `pn` because this source has no
word-level timings, so the timestamped arm does not apply. `meta.json` carries the
gate parameters, the hours, the `sha256` of `train.jsonl`, the licence dispute and the
caveats, so they travel with the data instead of with somebody's memory.

Two packs exist, both from the same 10,133-row pilot audio:

| pack | gate | rows | hours | sha256 |
|---|---|---|---|---|
| `hparl2-v1` | align ≥ 0.95 | 4,502 | 6.01 | `a14890e693402c03` |
| **`hparl2-v2`** | **align ≥ 0.50 ∧ dur ≥ 1.5 s** | **9,334** | **13.16** | `29f4fe806b88e80a` |

v2 is the arm; v1 is kept so the gate itself can be tested if v2's result is ambiguous.
v2 carries `edge_clipped` and `weight` per row: 24.2% of it is clipped-edge audio at
weight 0.5, giving 11.66 weighted hours. **A sampler that ignores `weight` trains on
that quarter at full strength** — that is the one way this pack can be used wrong.

Punctuation over the wider gate cost one thing worth recording: the word guard rejected
~7% of LLM outputs among the newly admitted lower-alignment rows, against 0.6% at
align ≥ 0.95. Where Soniox is more wrong, the model is more tempted to reword the
target — and the guard is what stops it. Those rows fall back to the deterministic
transfer.

The point of the format is the next source: adding one is a filter script, an entry in
`SOURCES`, and a ledger artifact — the trainer does not change.

## Provenance

Audio, clips and per-row output live under `~/.cache/oc-public/hparl/`
(`probe.jsonl`, `probe_train23.jsonl`) — never in git. Sampling is seeded and the
row-group sampler is deterministic, so both runs reproduce.

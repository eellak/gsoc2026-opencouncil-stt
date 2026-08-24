# Why whole passages disappear (2026-08-24)

Open question 3 of [`FINAL_REPORT.md`](../../FINAL_REPORT.md): *36% of what our adapter
deletes vanishes in runs of five or more consecutive reference words, against 19% for
ElevenLabs Scribe. I did not find the cause.*

This screen verifies the number, characterises the 128 runs behind it, and tests every
explanation reachable from cached and public data. **No GPU, no ASR API call, nothing
paid.** The substrate is the public `report.json` of benchmark run
`2026-08-22-post-june-held-out-test-clean-pack-cont-` and OpenCouncil's public meeting
API.

Agreement-with-OpenCouncil throughout: the reference is our own published transcript, so
a "deletion" is a published word a system did not write. That distinction turns out to
carry a fifth of the effect.

Code: [`eval/controlled_eval/deletion_runs.py`](../../eval/controlled_eval/deletion_runs.py),
[`deletion_run_context.py`](../../eval/controlled_eval/deletion_run_context.py),
[`deletion_runs_explain.py`](../../eval/controlled_eval/deletion_runs_explain.py).
Results: `eval/results_deletion_runs*.json`.

## 0. The number is right

The 36% was computed by an uncommitted scratch script. Re-derived here from a freshly
downloaded copy of the same `report.json` — byte-identical to the cached one — with an
independently written aligner:

| | runs | tokens deleted | runs of 1 | 2–4 | 5+ | tokens in 5+ |
|---|---|---|---|---|---|---|
| `oc-cleanpack-cont-s47-b` | 1,759 | 3,507 | 1,239 | 392 | 128 | 1,274 (**36.3%**) |
| Scribe v2 | 731 | 1,058 | 604 | 107 | 20 | 197 (**18.6%**) |

Exactly the published table. Two robustness checks the original did not carry:

- **Alignment tie-break.** Flipping the backtrace to prefer deletions over substitutions
  moves the share to 36.0% (ours) and 19.3% (Scribe). The ledger's caveat that bucket
  boundaries are not exact holds, and does not matter here.
- **Meeting-clustered bootstrap**, 10,000 resamples over 116 meetings: ours **0.363,
  95% CI [0.302, 0.418]**; Scribe **0.186, [0.106, 0.268]**. The intervals do not
  overlap.

**Nothing dominates.** The 1,274 tokens are spread over 128 runs in 94 windows, 62
meetings and 11 cities. The largest single window carries 62 tokens (4.9%); the top five
carry 19.4%. Run length: median 7, mean 10, longest 32. This is not one broken window,
and it is not one broken meeting.

## 1. The finding that reframes the question

The property is not ours. It is whisper's, and we inherited it:

| system | deleted tokens | in 5+ runs | share |
|---|---|---|---|
| gpt-4o-transcribe | 5,646 | 2,766 | 49.0% |
| gladia-prod | 4,519 | 2,005 | 44.4% |
| `artifact-adapter-fixed` (restaged) | 5,859 | 2,481 | 42.3% |
| **`oc-cleanpack-cont-s47-b`** | **3,507** | **1,274** | **36.3%** |
| `hf-openai-whisper-large-v3` (base) | 3,746 | 1,200 | 32.0% |
| Soniox | 1,369 | 282 | 20.6% |
| Scribe v2 | 1,058 | 197 | 18.6% |

Base whisper-large-v3, with none of our training, already loses 32% of its deletions in
long runs. The incumbent adapter is worse than base at 42.3%. Our clean-pack adapter sits
between them and deletes 42% fewer tokens in long runs than the incumbent in absolute
terms (1,274 against 2,481).

So the question "what did our fine-tuning do to cause this" has a short answer: **it did
not cause it.** Fine-tuning moved the volume, not the shape. Whatever produces long runs
is a property of a whisper-style autoregressive decoder that the two low-latency
commercial engines do not share.

## 2. What the runs actually are

Reconstructing each window from the public meeting API's timestamped utterances succeeds
exactly for 375 of 391 windows (the other 16 assemble their `referenceText` differently
and are dropped rather than approximated). That puts 119 of our 128 long runs on the
meeting timeline. Every "baseline" below is the same run lengths in the same windows at
random start positions, one draw, fixed seed.

| | observed | matched baseline |
|---|---|---|
| run begins and ends exactly on utterance boundaries | **24.4%** | 2.5% |
| a speaker change sits at the run's leading edge | **41.2%** | 21.0% |
| the run is one complete speaker segment | 11.8% | — |
| both ends fall on sentence punctuation | **25.8%** | 3.1% |
| run touches only one utterance | 48.7% | 46.2% |

A tenfold enrichment on exact utterance alignment. **What disappears is not a random
stretch of words: it is a published utterance, or a whole speaker turn, dropped whole.**
Base whisper shows this more strongly than we do (30.6% against a 0.9% baseline), the
incumbent more strongly still (33.6% vs 2.3%), Scribe barely at all (5.3% vs 0%).

Median duration of a run: **2.3 seconds**. Median silence before it: 0.5 s. This is not a
collapse; it is a hiccup, repeated 128 times.

## 3. Hypotheses tested

### Refuted

**It is not truncation or window edges.** Only 6 of 128 runs reach the last reference
token and 2 begin at the first; 1,211 of 1,274 tokens are interior. Windows carrying a
long run have a median hypothesis/reference length ratio of 0.977 against 1.012 for the
rest — a 3.5% shortfall, nowhere near enough to be the model stopping early. Consistent
with `exp-2026-08-18-chunking-aware-decoding`'s flat position deciles, and extends it:
the deciles were flat for the *gap to Scribe*, and they are flat for *deletion runs* too.

**It is not the no-speech gate.** `exp-2026-08-12-decode-ablation` already established
that faster-whisper's `no_speech_threshold` fired **zero** times across 39 windows, so
arms with the gate disabled were byte-identical to the control. Whole-frame skipping is
not the mechanism. The VAD closure screen agrees from the other side: 91.3% of deletions
sit inside spans the decoder did emit a segment for.

**It is not a repetition loop.** 4-gram repetition in hypotheses for windows carrying a
long run is 0.013 against 0.008 elsewhere — and the reference itself scores 0.010, and
Scribe shows the same lift (0.022 vs 0.010). No degenerate-decode signature.

**It is not window-level speech density.** 2.23 reference words/s in windows with a long
run against 2.10 without — and the identical split appears for Scribe and for base
whisper, so it does not distinguish the systems that have the problem from those that
do not.

**It is not our training-data granularity.** The "v1 trained on 3.55 s single utterances,
so it learned to emit utterance-sized units" story predicts exactly the utterance
alignment of §2 — but base whisper-large-v3, which saw none of that data, shows the
alignment *more* strongly. The training recipe cannot be the cause of a pattern that
predates it.

### Supported, and partly explains it

**A fifth of the effect is the reference, not the model.** OpenCouncil's API flags which
utterances a human edited. 32.4% of all reference tokens sit in a human-edited utterance;
56.9% of the tokens in our long runs do, against 40.2% for the matched baseline. Crossing
that with what the other six systems wrote in the same span:

| | tokens | share |
|---|---|---|
| no other system wrote it, and the reference is human-edited | 227 | **19.1%** |
| 4+ other systems wrote it, and the reference is ASR-derived | 334 | **28.2%** |
| everything else | 625 | 52.7% |

21 of the 22 runs that **no** other system wrote sit in human-edited text. Reading them
confirms it: legal citations, protocol numbers, ΦΕΚ references, decision codes — text a
human put into the published transcript. Seven independent decoders did not
simultaneously go deaf; the words were not in the audio in that form.

This explains Scribe's 19% almost entirely: **90.6%** of Scribe's long-run tokens sit in
human-edited utterances, and 18 of its 20 runs were deleted by every system. Scribe's
long runs are mostly reference artifacts. Ours are mostly not — 28.2% of our tokens are
spans four or more other systems heard and wrote, which is genuine, unexplained deafness
on ordinary conversational Greek.

**Long runs sit on fast speech, but so does everyone's.** Local rate inside a run is
3.30 tokens/s against 2.83 for matched random spans — but base whisper is at 3.46 vs 2.73
and Scribe at 3.91 vs 2.84. Fast speech is a property of *any* long deletion run, not the
thing that separates us from Scribe.

### Suggestive, unconfirmed, and I do not trust it yet

**27% of our long runs start in the last 5 s of a 30 s encoder frame**, measured from the
start of the decoded window, against 15.1% expected from where the reference tokens
actually are. Meeting-clustered CI [0.193, 0.350], excluding the expectation. Base whisper
(0.180 [0.102, 0.255]), the incumbent (0.172) and Scribe (0.158) show nothing.

Three reasons not to build on it. It is post-hoc, and it is one of roughly a dozen
comparisons in this screen, none preregistered. The 30 s grid is only guaranteed for the
*first* frame — faster-whisper advances its seek to the end of the last emitted segment —
yet the runs are spread evenly across chunk indices, which the fixed-grid reading does not
predict. And it is the one result here that does *not* replicate on base whisper, which
makes it the most likely of all of them to be noise.

## 4. Conclusion: cause not found, but the question has changed

I did not find a single cause, and I do not think there is one. What the evidence supports:

1. **The 36% is real, robust to the aligner, and broadly distributed.** No window,
   meeting or city carries it.
2. **It is a whisper-family property we inherited, not damage our training did.** Base
   whisper is at 32%, the previous adapter at 42%, gpt-4o-transcribe at 49%. Our current
   adapter improved the absolute volume by 42% over the incumbent without changing the
   shape. Any fix aimed at our training recipe is aimed at the wrong layer.
3. **The unit that disappears is a published utterance or speaker turn**, dropped at its
   boundaries, lasting about 2 seconds — ten times more often than chance placement.
4. **About 19% of it is our own published transcript**, where a human wrote text no ASR
   system heard. Scribe's whole 19% is very largely this. Comparing our long-run rate to
   Scribe's therefore compares two different things, and the honest ours-versus-Scribe gap
   on real dropped speech is smaller than 36 against 19.
5. **About 28% is genuine, unexplained deafness** — spans four or more other systems
   transcribed and we did not.

What I could not test without spending:

- **Whether the dropped 2-second spans are acoustically distinct** (overlap, crosstalk,
  off-mic, low SNR). The held-out window audio is not in `~/.cache/oc-public/` — only the
  261 windows of the older benchmark are — and diarizing these 375 windows needs pyannote
  credits, which ran out on 2026-08-19.
- **Whether the decoder's own segment boundaries fall on the run edges.** The served
  `report.json` keeps only text. Deciding this needs one word-timestamped re-decode of the
  391 windows, which is a GPU run. That is the single measurement I would buy: it turns
  §2's "aligned with published utterances" into "aligned with the decoder's own emitted
  segments" or refutes it, and it is the only way to confirm or kill the chunk-phase
  signal in §3.
- **The Samothraki concentration** (24.5% of the gap to Scribe) is a separate open item;
  Samothraki holds only 10% of the long-run tokens, so it is not the same phenomenon.

The next free thing worth doing is not another screen of this substrate. It is deciding
whether "we delete more passages than Scribe" survives at all once human-edited reference
text is excluded from both sides.

## Ledger

`exp-2026-08-23-gap-to-scribe` carries the `next_action` this screen answers ("Screen the
128 long deletion runs and the Samothraki concentration"). The deletion-run half is done;
Samothraki is not. The record was left untouched here by instruction and needs updating
in the change that lands this report.

# Does Soniox per-word confidence predict the errors a human found?

2026-08-16. `exp-2026-08-16-soniox-confidence`. Go/no-go probe. Zero GPU, zero paid API.

**Answer: yes, on the free realtime model, well above chance — and the decision is GO
for one ~$0.82 paid run, as a hypothesis test, not as a validated capability.**

The number that carries the decision, frozen with its threshold before any outcome was
computed: the equal-weight mean of the six **within-meeting AUROCs** of
`1 − min-confidence` for predicting that an emitted Soniox word is an error is
**0.8167**, against a GO threshold of 0.60 and an AUROC null of 0.5. Meeting-cluster
95% interval [0.790, 0.852] over 6 clusters — descriptive, not significance.

What it does **not** say is at least as important, and is in
[Limits](#limits-that-must-travel-with-every-number-here).

## Why this could be measured at all

Soniox returns per-token `confidence`, `start_ms`, `end_ms` and `is_final`. Our client
threw all four away: `soniox_client.stream()` read only `tok["text"]`,
`file_transcribe.py` printed text, and every caller in this repo parses that stdout.
None of the ~79k cached Soniox responses in this project contains a confidence value.
A re-run was mandatory.

Two things made a **free** re-run possible without disturbing anything frozen:

- The gold-set hypotheses (`scripts/gold_set/run_hyps.py`) were produced through the
  realtime path, model `stt-rt-v4`, on a free Perplexity temp key. Re-running those 30
  cells reproduces the same path.
- The 247 benchmark windows were **not** touched. Those were produced with the paid
  `stt-async-v5`, they are one third of the fusion input W, and re-running them on v4
  would change the text underneath every frozen number in this project. They stay as
  they are.

Silence trimming was **not** enabled. The user's production Soniox client has an opt-in
`?trim=1` path measured at 61–72% word similarity against a 93.2% run-to-run control,
with ~11% of words disappearing — this project's exact failure mode. It stayed off.

## Tooling change

`/home/harold/projects/soniox-tools` — a separate directory, **not a git repository**,
so there is nothing to commit there and no revision to cite. Two strictly additive
edits:

- `soniox_client.py`: `stream()` gains an optional `on_token(tok)` callback, invoked
  with the raw token dict exactly as Soniox sent it (`<end>` excluded). Nothing else
  changes; the returned transcript is byte-identical whether or not it is supplied.
- `file_transcribe.py`: a `--json` flag. Without it, stdout is unchanged, including the
  `===== TRANSCRIPT =====` marker every existing caller parses. With it, stdout carries
  one JSON document (`model`, `audio`, `language_hints`, `text`, `tokens`,
  `residual_nonfinal_tokens`) and the live partial tail goes to stderr.

Re-run driver: [`scripts/gold_set/run_soniox_tokens.py`](../../scripts/gold_set/run_soniox_tokens.py).
Scorer: [`eval/soniox_confidence_probe.py`](../../eval/soniox_confidence_probe.py).
Raw token JSON is cached at `~/.cache/oc-public/gold-set-2026-08/hyp/soniox-tokens/`
and never enters git.

All 30 cells transcribed on the first attempt, 0 failures, 0 key re-mints (the run took
~16 minutes against a roughly hourly key lifetime; the retry path exists but never
fired). 6,263 final tokens, **zero** missing a timestamp or a confidence.

## Step 1 — does the text reproduce?

This check decided how much the rest was worth.

| | value |
|---|---|
| cells with identical normalized token sequence | 14 / 27 (51.9%) |
| cells with identical raw string | 12 / 27 (44.4%) |
| **pooled WER(cached, new) over all 27 cells** | **0.0216** |
| worst cell | 0.167 (samothraki jul6 610000) |

**Word-level reproduction is strong: 97.8% agreement.** The 51.9% exact-cell figure is
reported for completeness and should not be read as failure — exact equality of a 15 s
string is brittle, one changed word fails the whole cell.

Separately, the user's production Soniox client measured **run-to-run word similarity at
93.2%** on identical input. That is a *different* repeatability estimate — different
metric, audio distribution and client configuration — so it is **not** a ceiling this
run can be scored against. It is context: realtime Soniox is known not to be
deterministic, and 97.8% is comfortably in the range where that non-determinism is the
whole story.

Consequence: the confidences measured here can be attached to *this* re-run. They are
not retroactively attachable to the cached `hyp/soniox/*.json` text token by token,
because 13 of 27 cells differ somewhere.

## Step 2 — what was frozen, before any outcome was looked at

Written into the scorer docstring before the first metric ran:

- **Primary statistic**: equal-weight mean of the six within-meeting AUROCs of
  `1 − conf_min`, ties worth 0.5. Chosen over the pooled AUROC on Codex's advice (job
  `4759402624f643efacc4fb2e9cd63beb`): a pooled AUROC can look good merely because one
  meeting has both lower confidence and higher WER.
- **GO threshold**: 0.60.
- **Word confidence**: `conf_min` = min over all runes of the word (**preregistered
  here**); `conf_mean` secondary.
- **Region**: `core_envelope`, with the gold set's uncertainty time-masking.
  `core_strict` and `clip` as sensitivity.
- **Buckets**: equal-width 0.0–1.0 step 0.1, and sample deciles.
- **Alignment**: `gold_set_score.align_ops` (tie-break S > D > I) primary; all 6 op
  priorities × forward/reversed as a sensitivity envelope.
- **Nulls**: AUROC null 0.5; average-precision null = prevalence; bottom-decile lift
  measured against prevalence.

A third aggregate, `conf_min_lex` (min over **lexical** runes only, punctuation
excluded), was added **after** freezing, when the production algorithm in the user's
`soniox-core` became known. It is reported alongside `conf_min`, and `conf_min` remains
the preregistered primary. Nothing was retrofitted.

Word construction follows that production algorithm: finals only, subtokens exploded to
runes carrying their token's confidence, words cut on whitespace. Zero words lacked a
timestamp, a confidence, or a lexical rune; 4 words split into more than one normalized
token (each inherits the word's confidence).

## Step 3 — the ceiling, stated before the result

In `core_envelope`: **M = 859, S = 97, D = 43, I = 49.**

Deletions are **22.8% of all edit operations**, and a confidence score can only attach
to a word the system emitted. Everything below is about the other 77.2%: substitutions
and insertions among **emitted** words. It is not a statistic about all the errors the
human found. (A neighbouring-word confidence signal for deletions is conceivable and
untested — deletions are invisible to *this* rule, not proven undetectable in general.)

## Step 4 — the result

n = 1,005 emitted words, 146 errors, prevalence 0.1453.

| aggregate | mean within-meeting AUROC | pooled AUROC | average precision (null 0.145) |
|---|---|---|---|
| `conf_min` (preregistered) | **0.8167** | 0.8185 | 0.453 |
| `conf_min_lex` (production def.) | 0.8191 | 0.8288 | 0.489 |
| `conf_mean` | 0.8357 | 0.8334 | 0.455 |

Per meeting (`conf_min`), with n and error count:

| meeting | AUROC | n | errors |
|---|---|---|---|
| sparta jul1 | 0.806 | 123 | 10 |
| chalandri jul29 | 0.782 | 142 | 8 |
| samothraki jul6 | 0.783 | 221 | 69 |
| zografou jun18 | 0.900 | 31 | 1 |
| chania jun24 | 0.804 | 298 | 29 |
| xylokastro jun29 | 0.826 | 190 | 29 |

Leave-one-meeting-out mean stays in 0.800–0.824. Meeting-cluster bootstrap 95%
[0.790, 0.852] — 6 clusters, descriptive.

**Within-meeting permutation null** (confidences shuffled inside each meeting, 2,000
permutations): null mean 0.4986, 95% [0.403, 0.602], observed 0.8167, one-sided
p = 0.0005. The observed value sits outside the entire null band.

**Single-meeting domination check**: samothraki jul6 supplies 69 of 146 errors (47%).
Its own AUROC is 0.783 — at the low end — and removing it moves the mean to 0.823. The
headline is not one meeting's effect.

### Calibration (`conf_min`, equal-width)

| bucket | n | errors | error rate | meetings |
|---|---|---|---|---|
| [0.1, 0.2) | 1 | 1 | 1.000 | 1 |
| [0.2, 0.3) | 9 | 7 | 0.778 | 3 |
| [0.3, 0.4) | 16 | 8 | 0.500 | 3 |
| [0.4, 0.5) | 26 | 13 | 0.500 | 4 |
| [0.5, 0.6) | 43 | 19 | 0.442 | 6 |
| [0.6, 0.7) | 41 | 12 | 0.293 | 5 |
| [0.7, 0.8) | 43 | 12 | 0.279 | 6 |
| [0.8, 0.9) | 88 | 29 | 0.330 | 6 |
| [0.9, 1.0) | 738 | 45 | 0.061 | 6 |

Sample deciles: bottom 100 words 0.510 error rate (**3.51× lift** over prevalence); top
bucket, confidence ≥ 0.999, n = 347, 0.026.

The ordering is broad but **not strictly monotone**: [0.8, 0.9) sits above [0.7, 0.8)
by 5.1 points, with an approximate standard error of ~8.5 points. That is
noise-compatible and nothing is claimed from it. **No calibrated error probability is
claimed** — 738 of 1,005 words sit in a single wide bucket.

### The inherited operating point

The user's production client (`soniox-core`, `internal/soniox/soniox.go:36`,
`internal/soniox/stream.go:25-27`) flags a word when its min lexical-rune confidence is
strictly below **0.5**. That threshold has been in production unchanged since
2026-06-14 and was chosen as a UX judgement — "conservative so the orange 'unsure' marks
stay trustworthy rather than noisy". **It had never been calibrated against ground
truth. This is the first measurement of it.**

| rule | flag rate | precision | recall | lift |
|---|---|---|---|---|
| `conf_min_lex < 0.5` (production) | 3.4% | **0.706** | 0.164 | 4.86× |
| `conf_min < 0.5` | 5.2% | 0.558 | 0.199 | 3.84× |

The production threshold is a **high-specificity triage rule, not broad error
detection**: 7 flags in 10 land on a real error, but it catches only 1 error in 6. The
UX judgement holds up on its own terms.

### Sensitivity checks (not independent evidence)

These vary one analysis choice at a time on the same 1,005 words. They show the headline
is not an artefact of a particular choice; they do not replicate it.

- **Alignment tie-break**, 6 op priorities × forward/reversed: mean within-meeting AUROC
  0.8084–0.8178, pooled 0.8055–0.8201, S ∈ [95, 97], I ∈ [49, 50]. The alignment
  ambiguity is negligible here.
- **Region**: `core_strict` 0.780 (n = 473, deletion share 31.2%), `core_envelope`
  0.8167, `clip` 0.811.
- **Overlap split**: overlap (n = 218, prevalence 0.193) mean WM 0.799, 5 informative
  meetings; non-overlap (n = 787, prevalence 0.132) mean WM 0.820. Confidence works
  about equally well inside the region where the gold set says speech is lost — it does
  not degrade there, and it also does not solve it (the 0.714 overlap speaker recall is
  a *recall* problem, i.e. deletions, which confidence cannot see).
- **Insertions vs substitutions**: insertions-vs-match (n = 908, prevalence 0.054) mean
  WM **0.773**, AP 0.226, 5 informative meetings; substitutions-vs-match (n = 956,
  prevalence 0.102) mean WM 0.828, AP 0.386. **Insertion detection is the weakest of the
  three** — and insertions are exactly what kills the occupancy fusion arms
  (`exp-2026-08-16-char-vote-homophones`: occupancy columns hold 14.2% of the remaining
  oracle gap and fail the insertion gate). Confidence is a real but not decisive handle
  on that gate.
- **Length confound**: subtoken count alone predicts error at AUROC 0.555, so `min` is
  mildly length-confounded. Within fixed subtoken counts 1–6 the confidence AUROCs are
  0.79 / 0.81 / 0.86 / 0.88 / 0.85 / 0.78 — the signal survives the control. Counts ≥ 7
  have n ≤ 22 and are uninformative.

## Decision

**GO for one paid `stt-async-v5` run over the 247 windows (~8 audio hours, ~$0.82).**

The justifying number is **0.8167** against a preregistered 0.60 threshold, with an
AUROC null of 0.5, a within-meeting permutation null whose entire 95% band tops out at
0.602, all six meetings between 0.78 and 0.90, and leave-one-meeting-out stable at
0.800–0.824.

What GO means: on this frozen rt-v4 sample, confidence strongly discriminates
substitution and insertion errors among emitted words, consistently across meetings and
analysis choices. It **does not directly validate `stt-async-v5` confidence**, does not
cover deletions, and does not establish calibrated error probabilities or broad recall.
The $0.82 exists precisely to test the transport.

One logistics finding for whoever runs it, from the same production client and **not
verified here**: file mode there is *N* parallel realtime sessions over 60 s segments
(`maxJobs = 18`), so ~8 audio hours is roughly 27 minutes of wall clock, not 8 hours.
The 18 traces to an initial commit with no surviving measurement artefact — treat it as
folklore with a good track record. Soniox also checks the key only at the WebSocket
handshake, so a long session survives key expiry; only a reconnect after rotation fails.

If it had come back uninformative, that would have been a valid result and the
confidence idea would have been dropped here. It did not, and no variant hunting was
done: one aggregate was preregistered, three were reported, and the preregistered one
carries the decision.

## Limits that must travel with every number here

- **30 cells, 27 scored, 6 meetings, 6 cities — one meeting per city, so meeting and
  city are fully confounded.** The cell is the independent unit and clustering is by
  meeting. **This cannot support a population claim or any promotion gate.** A
  6-cluster interval excluding a null is not significance.
- **`stt-rt-v4` is not `stt-async-v5`.** Every conclusion transfers to the paid model
  only as a hypothesis. This is indirect evidence motivating an experiment, not
  evidence about async-v5's confidence.
- **The AUROC covers substitutions and insertions among emitted words only.** 22.8% of
  edit operations in this region are deletions, invisible to a per-word confidence by
  construction.
- **Nothing was tuned on the gold set.** The primary statistic, its threshold, the
  region, the bucket edges, the alignment and the nulls were all written down before
  any outcome was computed. `conf_min_lex` was added afterwards and is labelled as such;
  it is not the primary.
- **The secondary analyses are sensitivity checks on one dataset**, not independent
  replications. zografou jun18 contributes a single error and its 0.900 means almost
  nothing.
- The gold set's own caveats all still apply: one listener, one pass, no inter-annotator
  number, 19% of core tokens excluded as text-uncertain with their time masked out, and
  a reference that knowingly under-covers (17 spans the human judged as lost speech were
  never written into it).

## Further work, noted and not chased

The Soniox SDK exposes `Context{General, Text, Terms, TranslationTerms}` — a
hotword / vocabulary biasing hook. Neither `soniox-core` nor this repo wires it. Given
the frozen term lists under `research/ds_wer/terms/` and that names are the one
mechanical category enriched 6.8×/4.4× in the new deletions
(`exp-2026-08-16-deletion-hard-coverage`), that hook is a candidate nobody here knew
existed. Recording it; not pursuing it.

## Cost

**Zero.** Free Perplexity temp key on the realtime path, local CPU, no GPU, no paid API
call. The ~$0.82 is the *proposed* next step, not spend incurred.

## Reviews

Two Codex reviews at high effort, both before the thing they governed:

- Job `4759402624f643efacc4fb2e9cd63beb`, **before any scoring code**, replaced a pooled
  AUROC with the equal-weight within-meeting mean, corrected the AUROC null from 0 to
  0.5, forced the deletion ceiling to be stated up front, required a *within-meeting*
  permutation null rather than a global shuffle, and required the alignment tie-break
  envelope and the subtoken-length control.
- Job `69c1a01b05a54aa08deb776df756af8a`, **on the findings, before any claim was
  written**, killed three sentences: "nothing here is evidence about async-v5" (softened
  to *does not directly validate*), "robust across all six meetings" (jun18 has one
  error and the sensitivity checks are correlated, not independent), and the framing of
  93.2% as a reproduction *ceiling* rather than a separate repeatability estimate. It
  also ruled the [0.8, 0.9) calibration bump noise-compatible.

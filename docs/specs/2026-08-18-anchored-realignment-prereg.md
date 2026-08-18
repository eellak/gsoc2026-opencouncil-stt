# Anchored re-alignment and the drift-zone occupancy guard

Preregistration. 2026-08-18. `exp-2026-08-18-anchored-realignment`.

Written **before** either arm was implemented and before any WER, deletion rate or
`(S+D)/N` of any arm was computed. Zero GPU, zero paid API: the only new computation
is CPU forced alignment of hypothesis text that already exists in cache.

Predecessors: [`exp-2026-08-16-composition-over-selection`](../reports/2026-08-16-char-vote-homophones.md)
(W, the exact 3-way MSA plus the hierarchical per-column vote),
[`exp-2026-08-17-majority-error-taxonomy`](../reports/2026-08-17-majority-error-taxonomy.md)
(the column census this arm's failure mode was found in).

## 1. The defect

W's alignment is **text-only**. `msa.align3` minimises sum-of-pairs edit cost over
three token streams and has no notion of time. When one system's stream drifts by a
few words — a burst of insertions, a dropped clause — a very common Greek token («το»,
«και», «να») matches an identical token from a **different moment in the audio**,
because a spurious match is free and the correct gap costs 1. The mis-pairing then
chains until a distinctive word re-synchronises the streams.

The measured instance, on a 299 s slice: columns 84–94, where the adapter's words were
paired against tokens **2.2 to 2.8 seconds away**. The columns that resulted were
singletons, and the occupancy stage of `msa.vote_column` deleted every one of them —
a column with one occupant is `epsilon` by construction. **Four of W's sixteen
deletions in five minutes came from that one alignment artifact.**

Two independent things follow, and this experiment measures them separately.

- **The alignment should not have been allowed to drift.** → arm **A**, anchored
  re-alignment.
- **The vote should not treat a mis-alignment as a majority.** A single occupant
  inside a drift zone is not evidence that two systems heard silence; it is evidence
  that the three streams are out of register. → arm **G**, the drift-zone occupancy
  guard.

## 2. What must not move

- **`eval/controlled_eval/msa.py` is not edited.** Its sha256 keys an 18 MB alignment
  cache (`3751fe5a13320e2b…`). Both arms *call* `align3` and `compose` unchanged. The
  hash is recorded before and after the work.
- `scoring.py`, `fusion_lab.py`, `column_classes.py`, `bench_data.py` are not edited.
- The substrate is the frozen one: benchmark run
  `2026-08-10-corrected-adapter-label-prefix-fix-vs-ju`, trio
  `scribe-v2-clean` / `soniox` / `oc-runpod-fixed-2026-08-10`, **247 windows** after
  removing the 6 sealed temporal-holdout windows of `eval-freeze-2026-08`. The 7
  sealed holdout windows stay sealed and are never aligned, timed or scored.
- No transcript text and no audio in git. Everything new lands under
  `~/.cache/oc-public/anchored-2026-08/`.
- **W itself is not re-decoded and not re-voted.** The baseline in every contrast is
  the existing W token stream from the existing alignment cache.

## 3. Where the times come from, and why not from the systems

The anchor definition needs a clock. On this substrate **none of the three voters
carries usable per-word times**:

- `scribe-v2-clean` — the benchmark report stores `hypothesisText` only. No timing
  field exists for any provider, and there is no ElevenLabs credential in this
  environment.
- `soniox` — the cached text is the paid `stt-async-v5`; the client discarded the
  per-token times. The free `stt-rt-v4` has them but is a **different model with
  different text**, and grafting one model's times onto another model's tokens is the
  error `exp-2026-08-16-w-rt-confidence` refused to make.
- `oc-runpod-fixed-2026-08-10` — decoded with `word_timestamps: false`. The local
  re-decode that has them **changes the transcript in 101 of 102 paired windows at a
  pooled 7.7%** (`exp-2026-08-16-adapter-confidence`), so its times do not belong to
  the frozen text either.

So the times are **derived, not native**: each of the three frozen hypothesis texts is
force-aligned to the same window audio (`$SC/bench_windows/<item_id>.wav`) with one
CTC forced aligner, and every token gets an interval on **one common clock**.

This is the right instrument for this question and it is also the honest one:

- It **does not change a single character of any hypothesis**, so W is untouched and
  the baseline is the frozen baseline. Every alternative (re-decode, model swap)
  changes the thing being measured.
- All three streams are timed by the *same* tool against the *same* audio, so "these
  two systems put this word at the same moment" is a statement about the audio, not
  about two vendors' differing timestamping conventions. Comparing a Soniox clock with
  a Whisper clock would have been the weaker test even if both had existed.

**Recorded limitation, up front.** A forced aligner must place every hypothesis word
somewhere. A word that was never spoken (a hallucination, or a substitution) is
assigned an interval anyway, usually squeezed against its neighbours. Therefore:
derived times are trustworthy for words that were really said, and arbitrary for words
that were not. This weakens anchors (an anchor requires the *same* token in all three
streams, which is already strong evidence the word was said) less than it weakens
drift detection (arm G), and that asymmetry is stated again in the report.

**Aligner, frozen:** PyPI `ctc-forced-aligner` (deskpai, ONNX, MMS CTC), the same
package and the same call sequence the repo already uses in
`eval/hf_export/build.py`: `generate_emissions(window_length=30, batch_size=8)`,
`preprocess_text(romanize=True, language="ell", split_size="word",
star_frequency="edges")`, `get_alignments` / `get_spans` / `postprocess_results`.
Emissions are computed **once per window** and reused for all three texts, so the three
streams are timed against a bit-identical acoustic posterior.

**Mapping to the scorer's token space.** The aligner returns whitespace words with
punctuation; `msa` operates on `scoring.wtoks` tokens. Each aligner word is expanded
with `wtoks` and its interval copied to each resulting token, in order. The
concatenation **must reproduce `wtoks(full_text)` exactly**; a window where any of the
three streams fails this check is marked `timing_unavailable` and **both arms fall back
to plain W on it**, with the count reported. Times travel by occurrence index, never by
token string, so repeated words are never confused.

## 4. Arm A — anchored segmentation

### 4.1 Anchor definition (frozen)

Let `a`, `b`, `c` be the three normalized token streams with per-token start times
`t_a`, `t_b`, `t_c` on the common clock.

A **candidate triple** is a position triple `(i, j, k)` with

1. **Identity.** `a[i] == b[j] == c[k]` (the frozen `wtoks` normalisation: NFD, combining
   marks stripped, lowercased; final sigma *not* folded).
2. **Time agreement.** `max` over the three pairs of `|start_x − start_y| ≤ TOL`, with
   **`TOL = 0.5 s`**.

A candidate triple is **admissible** as an anchor if additionally

3. **Distinctiveness.** Either `a[i] ∉ STOPLIST`, or the triple is part of a **run of
   ≥ 2 consecutive candidate triples** — that is, `(i−1, j−1, k−1)` or `(i+1, j+1, k+1)`
   is also a candidate triple.

The **anchor set** is a subset of admissible triples that is

4. **Strictly increasing** in all three indices, and
5. **Separated**: consecutive chosen anchors differ by **`MIN_GAP = 10`** tokens or more
   in *every* stream.

Among all such subsets the chosen one **maximises the number of anchors**; ties are
broken by minimum total time spread `Σ (max start − min start)` over the chosen
triples, then by earliest first index. Computed by an exact O(n²) chain DP over
candidates, so it is deterministic and does not depend on the order candidates were
generated.

### 4.2 Constants, and why they are these numbers

Chosen from reasoning about the mechanism, before any arm output existed. None is
tuned on a result, and none may be retuned after a number is seen.

- **`TOL = 0.5 s`.** The defect being separated is a **2.2–2.8 s** mis-pairing. The
  noise being tolerated is CTC forced-alignment jitter on a word all three systems
  agree on, which lives at the scale of the 20 ms emission stride plus word-boundary
  ambiguity — under 0.2 s in normal speech. 0.5 s is roughly 3× the expected jitter
  and roughly 5× smaller than the observed drift, which is the widest separation the
  two scales allow. It is also the value the prompt named, so it is not a free
  parameter.
- **`MIN_GAP = 10` tokens.** At the ~2.5 words/s of council speech, 10 tokens is ~4 s.
  Below that a "piece" is too short for a 3-way MSA to have any freedom, and anchoring
  degenerates into imposing a fixed alignment token by token — which would be a
  different (and much stronger) intervention than the one being tested. 10 also keeps
  the piece count per two-minute window in the tens rather than the hundreds.
- **`STOPLIST`.** The high-frequency Greek function words, written from the language,
  not from these transcripts, in accent-stripped lowercase to match `wtoks`:
  `και, κι, ο, η, το, οι, τα, του, της, των, τον, την, τη, τους, τις, να, θα, δεν, δε,
  μην, μη, σε, στο, στη, στην, στον, στους, στις, στα, με, για, απο, που, ειναι, ως, η,
  αλλα, ομως, ετσι, αυτο, αυτη, αυτος, ενα, ενας, μια, μας, σας, τι, οτι, ναι, οχι`.
  A token in this list can still anchor, but only inside a run of ≥ 2 — two consecutive
  agreeing tokens at agreeing times is a far rarer coincidence than one.

### 4.3 Re-alignment

Streams are cut at the anchors. Each inter-anchor **piece** `(a[p_i:i], b[p_j:j],
c[p_k:k])` is aligned with the **unmodified frozen `msa.align3`** using the same band
rule `fusion_lab._band` applies (`max(40, maxlen − minlen + 20)`, computed per piece).
The resulting column lists are concatenated in order, with each anchor contributing one
**unanimous column** `(v, v, v)` between the pieces. The tail after the last anchor is
one final piece.

The concatenated columns are voted by the **unmodified frozen `msa.compose`**, with the
same `pivot` (the window's `consensus_pick` index) and the same `priority` W uses.
Arm A's output is that token stream.

A window with **zero** anchors produces exactly W's columns by construction (one piece =
the whole window), so it contributes zero change. This is checked as an invariant, not
assumed.

## 5. Arm G — the drift-zone occupancy guard

Independent switch. Operates on **W's own existing columns**, so it can be measured
with the alignment untouched.

On a column list with per-occupant times:

1. A column is **disagreeing** iff it has **≥ 2 timed occupants** whose start times
   span **more than `TOL = 0.5 s`** (the same tolerance; it is the same "same moment"
   question).
2. A column is **thin** iff it has exactly **1** occupant — i.e. the columns the
   occupancy stage deletes.
3. A **drift zone** is a maximal run of consecutive columns in which every column is
   disagreeing or thin, **and which contains at least 2 disagreeing columns.** The
   two-disagreeing-column floor is what distinguishes a drift (which chains) from one
   noisy timestamp.
4. **The guard:** inside a drift zone, a thin column **emits its single token** instead
   of being deleted. Everywhere else the vote is untouched.

Nothing else about the vote changes: identity voting, tie-breaks and the pivot are
`msa.compose`'s, unmodified.

**Expected direction, stated in advance so it cannot be reinterpreted afterwards:** the
guard can only *add* tokens. It therefore lowers the deletion rate and raises the
insertion rate mechanically. It passes only if the words it restores were really
missing — which is exactly what `(S+D)/N` and the insertion guard together decide.

## 6. Arms

| arm | alignment | vote |
|---|---|---|
| **W** | frozen MSA | frozen hierarchical vote | (baseline) |
| **A** | anchored re-alignment | frozen hierarchical vote |
| **G** | frozen MSA | drift-zone occupancy guard |
| **AG** | anchored re-alignment | drift-zone occupancy guard |

All four are scored on the same 247 windows against the same references with the same
frozen scorer (`scoring.wtoks` + `exp_fusion_deletions.sdi`).

**No arm has a fitted parameter.** Every constant above is frozen a priori, so
leave-one-city-out is vacuous by construction and its out-of-fold number is identical
to its in-fold number. `fusion_lab` states this itself (`fitted = False`); the report
must repeat it rather than presenting LOCO as if it priced anything. What LOCO and the
leave-one-out sweeps *do* price here is **stability**, not overfitting.

## 7. Endpoints

**Co-primary, both directional (lower is better), both required:**

- **P1 — WER** against the human-corrected reference. `(S + D + I) / N`.
- **P2 — `(S+D)/N`.** Substitutions plus deletions per reference token. Reported
  alongside WER at the user's request, because on this material a scored insertion is
  frequently the hypothesis being right where the reference omitted real speech:
  `exp-2026-08-17-insertion-fidelity` found 23.7% of the adapter's and 40.8% of
  Soniox's scored insertions matched to a certain gold occurrence the published text
  cannot be reading.

**Safety endpoints:**

- **S1 — deletion rate `D/N`. This is the binding one.** It may not increase.
  Standing project rule, restated so it cannot be traded away here: **a lower WER
  bought with a higher deletion rate is a FAIL, not a trade.**
- **S2 — insertion rate `I/N`, as a guard.** It may not increase. `(S+D)/N` is not
  allowed to travel alone: a previous experiment on this project produced a fake
  deletion gain precisely by padding the hypothesis, and `(S+D)/N` is what exposed it.
  Reporting `(S+D)/N` without the insertion rate beside it would rebuild that trap.

**Descriptive, reported for every arm, deciding nothing:** substitution rate, anchors
found per window, drift zones found per window, windows changed, `timing_unavailable`
count, and the share of the alignment-conditional column oracle recovered.

## 8. Statistics

- **Paired meeting-clustered bootstrap**, `scoring.cluster_bootstrap`, 10,000
  replicates, seed 7, resampling **meetings** (144 of them). Each arm is contrasted
  against W on the identical window set with the identical reference, and the pairing
  assertion in `cluster_bootstrap` enforces it.
- CIs are reported for WER, `(S+D)/N`, deletion rate, insertion rate and substitution
  rate.
- **Single-item domination.** For every contrast, the share of the net edit-count
  change contributed by the single largest window and by the single largest meeting is
  computed and reported **before** any delta is quoted. Leave-one-out over window,
  meeting and city (`fusion_lab._loo`) is reported with sign-flip counts.

## 9. The decision rule, frozen

An arm **PASSES** if and only if **all** of the following hold:

1. P1: WER strictly lower than W's, and its 95% CI excludes zero.
2. P2: `(S+D)/N` strictly lower than W's, and its 95% CI excludes zero.
3. S1: deletion rate `≤` W's.
4. S2: insertion rate `≤` W's.
5. No sign flip in leave-one-out over windows, meetings or cities.
6. No single **meeting** contributes more than 50% of the net WER edit change.

Anything else is a **FAIL**, and is reported as a failure in those words.

Specifically resolved in advance, so that no result can be argued into a pass:

- Lower WER with a higher deletion rate → **FAIL** (rule 3).
- Lower `(S+D)/N` with a higher insertion rate → **FAIL** (rule 4). This is the exact
  shape of the earlier fake gain.
- A CI that includes zero → **FAIL**, however favourable the point estimate.
- An arm that changes fewer than 10 of the 247 windows is additionally reported as
  **NO-OP / inconclusive** whatever its numbers, because a delta carried by a handful
  of windows on a substrate this size is not a measurement of the mechanism.

**Nothing ships on this run either way.** This substrate has been read many times by a
human. A passing arm earns a confirmation obligation, not deployment. A failing arm is
a clean negative and is written down as one.

## 10. What is recorded regardless of outcome

Whatever the verdict: the number of anchors found (total, per window, distribution),
the number of drift zones found, how many of W's deletions sit inside a drift zone, the
`timing_unavailable` count, and the sha256 of `msa.py` before and after the work.

A negative here is informative and will be reported as such: it would mean the
mis-alignment mechanism is real but rare enough not to move a 247-window aggregate,
which is itself a finding about how much of the remaining 5.3-point gap to the column
oracle alignment error can possibly hold.

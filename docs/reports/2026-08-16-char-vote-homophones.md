# Below 0.1005? The columns are not there

2026-08-16 · wayfinder [#24](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/24) ·
prereg [`2026-08-16-char-vote-homophones-prereg.md`](../specs/2026-08-16-char-vote-homophones-prereg.md) ·
fifth pass over the same 247 windows

**The census answered the ticket before either arm was built.** Of 80,659 alignment
columns, 34 are homophone columns an arm H could touch — 0.042%, seven times below
the user's own "if it is 0.3%, do not build a machine for nothing" line. Arm H was
therefore **not built**: no KenLM, no LLM. Arm C, which is text-only and cheap, was
built and measured, and out-of-fold it shows **no detected benefit**: 0.10046 →
0.10038, CI [−0.00026, +0.00009], which includes zero and fails the primary gate.

The finding worth keeping is the one behind those two numbers. The 5.30-point gap
between W (0.1005) and the alignment-conditional column oracle (0.0475) is **not
mostly in the columns where the three systems disagree about which word was said**.
Only 1,396 unresolved columns are ones where W's vote differs from the oracle's
choice, so even one full edit gained per column would close at most **35.2%** of that
gap, and a replay of perfect hindsight over *every* unresolved column closes **12.7%**
of it. Word-choice arbitration between three transcripts,
which is what both arms of this ticket are, is working in a small room.

## The census, which was the deliverable

Run before any arm existed, on the exact 3-way MSA of
`exp-2026-08-16-composition-over-selection`. Classes are decided by occupancy and
identity only; the partition is Codex job `55293f6b`'s, which replaced a first draft
that would have let an arm overturn a real 2-of-3 token majority.

| class | columns | share | of which W ≠ column oracle |
|---|---:|---:|---:|
| `agree` `[x,x,x]` | 62,919 | 78.01% | 0 |
| `exact_2_of_3` `[x,x,y]` | 6,645 | 8.24% | 1,245 |
| `two_present_same` `[x,x,ε]` | 4,569 | 5.66% | 1,532 |
| `singleton` `[x,ε,ε]` | 4,460 | 5.53% | 715 |
| `unresolved_three` `[x,y,z]` | 1,104 | 1.37% | 601 |
| `unresolved_two` `[x,y,ε]` | 962 | 1.19% | 795 |

The two unresolved classes are 2,066 columns, 2.56% — the same tie set the LLM
arbiter of `exp-2026-08-16-composition-over-selection` was handed. Inside them:
**34** columns are strict homophones, 35 under the loose map, 130 carry a partial
homophone relation, and 291 columns anywhere in the substrate are quarantined as
token-boundary split/merge disagreements.

After the frozen eligibility rules — token majorities protected, split/merge
quarantined, epsilon never offered, C restricted to three-string columns:

| arm | eligible columns | share of all | of which W ≠ oracle |
|---|---:|---:|---:|
| C | 136 | 0.17% | 77 |
| H | 34 | 0.042% | 23 |
| H (loose variant) | 35 | 0.043% | 24 |

A count bound follows directly: if arm H were right every time W is wrong on its
eligible columns, it would fix 23 tokens out of 74,917 — **0.031 WER points**, 0.58%
of the gap to the column oracle. For C the same bound is 77 tokens, **0.103 points**,
1.9%. That is the whole room these two ideas have.

## Arm C, measured

Leave-one-city-out over 10 cities; only out-of-fold outputs scored; paired bootstrap
clustered by meeting, 10,000 replicates, no refitting inside replicates.

| arm | WER | ΔWER vs W | CI95 | del | ins | sub | gates |
|---|---:|---:|---|---:|---:|---:|---|
| V (whole-window vote) | 0.12012 | | | 0.0247 | 0.0443 | 0.0512 | |
| **W** (per-column vote) | **0.10046** | — | — | 0.02032 | 0.03743 | 0.04270 | — |
| **C** | 0.10038 | −0.00008 | [−0.00026, +0.00009] | 0.02032 | 0.03743 | 0.04263 | **FAIL** (CI includes zero) |
| C, common words only | 0.10039 | −0.00007 | [−0.00024, +0.00010] | 0.02032 | 0.03743 | 0.04265 | FAIL |
| column oracle | 0.04748 | | | | | | |

Both rate gates pass, and pass **identically to the digit** — but empirically, not by
construction: C cannot change the token count, yet a substituted token can still move
the scorer's optimal alignment, so equal length does not guarantee equal D and I.

What C actually did: it fired on all 136 eligible columns. In 57 the character
composite reproduced exactly what W had already chosen. In 58 it produced a different
candidate. In 2 it produced a string **no system proposed** which the closed lexicon
admitted. In 19 the composite was off-lexicon and was rejected. Across 247 windows, 49
changed, 16 improved, 13 worsened, 218 tied; 5 cities improved, 4 worsened, 1
unchanged. Leave-one-out over windows, meetings and cities produces zero sign flips —
of an effect indistinguishable from zero.

Dropping the term lists (`C_common_only`) changes nothing, which is the expected
result when the lexicon gate fires twice in the whole substrate.

## Where the gap actually is

All rows below are **hindsight**: the column oracle's own entry is replayed into the
named columns and the result is scored by the frozen scorer. They are not arms, and
they are not ceilings — the replay is token-only, so it can substitute where the
oracle chose a token but it never deletes what W kept. Each is therefore a LOWER bound
on that class's unrestricted hindsight value and an optimistic target for anything
learnable. Do not add them up: edit-distance effects make such interventions
non-additive. The per-column attribution is read off the oracle DP's own backtrace
(`msa.oracle_select`) — an earlier version matched the oracle's token list back onto
the columns by membership and mis-attributed repeated words, which is what made
`agree` columns appear to carry 530 oracle disagreements they do not have.

| replay | WER | Δ vs W | share of the 5.30-point W→oracle gap |
|---|---:|---:|---:|
| C-eligible columns | 0.09970 | −0.00076 | 1.4% |
| H-eligible columns | 0.10028 | −0.00017 | 0.33% |
| H-eligible, loose map | 0.10027 | −0.00019 | 0.36% |
| C+H eligible | 0.09952 | −0.00093 | 1.8% |
| **every unresolved column** | **0.09374** | **−0.00671** | **12.7%** |
| every `exact_2_of_3` majority | 0.08723 | −0.01323 | 25.0% |
| occupancy columns (`singleton` + `two_present_same`) | 0.09294 | −0.00751 | 14.2% |

The occupancy row is the interesting failure: restoring text at occupancy columns
takes the deletion rate from 0.0203 down to **0.0124** and pushes insertions from
0.0374 up to **0.0391**, so even with hindsight it **fails the frozen insertion
gate**. That is a statement about this intervention, not a proof that every better
occupancy model must violate the gate.

The `exact_2_of_3` row says the opposite of comfort: overriding token majorities with
hindsight is worth 1.32 points, materially more than every unresolved column put
together. Nothing here shows how a text-only rule could know *when* to override a
majority — the three systems agreeing 2-to-1 and being wrong together is precisely
the case where their text carries no signal. But it is where the mass sits.

## What this does and does not license

Adopted verbatim from Codex job `46892fd0`:

- **Supported.** C shows no detected benefit in a partially cross-fitted evaluation
  and fails its primary gate. H's prevalence is 0.042%, far below the pre-specified
  0.3% threshold, so not building it followed the rule. Unresolved word-choice
  arbitration accounts for a minority of the W-to-oracle gap.
- **Not supported.** That the remaining 87.3% has been cleanly decomposed. That
  text-only per-column arbitration is exhausted *in general* — what is shown is that
  arbitration restricted to *these* sparse eligible sets has little demonstrated
  value. That 12.7% is a maximum of anything.
- **Not a clean out-of-sample estimate.** The common-word set is properly
  cross-fitted, but it changed 2 decisions, so leave-one-city-out is technically real
  and practically inert here. The term lists and rosters were mined from material
  overlapping this benchmark, and the class definitions were chosen by a human after
  four previous passes over these same 247 windows. This is a **partially
  cross-fitted, benchmark-adapted** result, not independent confirmation.
- **The check that would change it**, per Codex: one locked-box run of the frozen
  pipeline on cities and meetings that contributed nothing to any lexicon, roster or
  rule — and with enough audio for C to fire more than 136 times.

## Deliverable beyond the result

`eval/controlled_eval/fusion_lab.py` is the reusable half: `load_substrate()` builds
and caches the 247-window aligned substrate, and `evaluate(idea, sub)` returns
out-of-fold WER, both rate gates, the clustered CI, leave-one-out over
window/meeting/city, per-city deltas and oracle recovery for any object with `fit` and
`apply`. Both arms of this ticket and all seven hindsight replays above are the same
four lines of caller code. `column_classes.py` holds the frozen partition and
eligibility, `greek_phonetics.py` the two phonemic maps, with tests in
`test_column_classes.py` and `test_greek_phonetics.py`.

## Caveats

- The `agree` row of the census carries **zero** oracle disagreements: with correct
  per-column attribution, the oracle never drops or replaces a token all three systems
  produced. That also means the W→oracle gap is carried entirely by the five other
  classes plus the alignment's own freedom to recombine.
- **Fifth pass over the same 247 windows.** Freezing a design does not remove
  adaptive-overfitting pressure across passes.
- The 6 sealed temporal-holdout windows of `eval-freeze-2026-08` were removed by the
  same explicit filter before anything was computed. They were never scored.
- Scored with `eval/controlled_eval/scoring.py`, not the benchmark app's scorer.
  Comparable to `exp-2026-08-16-composition-over-selection`, not to the leaderboard.
- Agreement-with-OpenCouncil, not fidelity-to-audio.
- The phonemic maps are *pronunciation keys under the scorer's normalization*. The
  scorer strips diacritics, so stress and diaeresis are gone before the map sees a
  token, and `αϋ` is indistinguishable from `αυ`. STRICT keeps `αυ`/`ευ` opaque, which
  deliberately under-merges the common `αυτό`/`αφτό` class; the loose variant catches
  it and moved one single column.
- The meeting-clustered CI is conditional on the ten fitted folds and these ten
  cities. Ten cities are too few for a city-level bootstrap; per-city deltas are
  reported instead.
- Arm H was not implemented, so nothing here is evidence about what a KenLM or an LLM
  would decide on a homophone column. The evidence is that there are 34 of them.
- Two Codex passes shaped this. Job `55293f6b`, before any number: added the
  `exact_2_of_3` class (the first draft would have let an arm overturn a token
  majority), the split/merge quarantine, the rule that epsilon is never a candidate
  for H, the exclusion of two-string columns from C, and found the vowel-collapse bug
  that made `ποιητης` collide with `πιτης`. Job `46892fd0`, before this conclusion:
  cut back the structural claim to what the counts support and supplied the
  count-based bounds used above.

# Preregistration — per-character vote and homophone arbitration on top of W

Wayfinder [#24](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/24).
Frozen 2026-08-16, before any arm WER was computed. Substrate and gates are the
ticket's; the classification and the eligibility rules below were fixed after Codex
job `55293f6b` and before any arm was scored.

## Question

Does WER fall below W's 0.1005 using **the same three systems**, with a per-character
vote inside disagreement columns (arm C) and a homophone arbitration (arm H),
measured leave-one-city-out so the number is not born on the set it was fitted on?

## Substrate

247 two-minute windows, 144 meetings, 10 cities, 74,917 reference tokens, from
`2026-08-10-corrected-adapter-label-prefix-fix-vs-ju`. The 6 sealed temporal-holdout
windows of `eval-freeze-2026-08` are removed by the same explicit filter
`exp_fusion_deletions.py` carries, before anything is computed. They stay sealed.

Baselines: V (whole-window consensus vote) 0.1201; **W** (exact 3-way MSA +
hierarchical per-column vote, `exp-2026-08-16-composition-over-selection`) 0.1005;
alignment-conditional column oracle 0.0475.

Trio: `scribe-v2-clean`, `soniox`, `oc-runpod-fixed-2026-08-10`
(`artifact-adapter-fixed`). **No fourth voter** — rejected by the user as costly in
production. **No confidences exist**: the benchmark stores only `hypothesisText` per
provider, so every design needing them is out.

## Step 0, and it is a stop rule

Count the columns per class **before building anything**. If the class an arm targets
is a rounding error, the arm is not built and the count is the answer. #18's LLM
arbiter looked neutral partly because it only ever saw 2.6% of the columns.

## Column partition (frozen)

Decided by occupancy and identity only, per MSA column:

| class | shape |
|---|---|
| `invalid` | no token (cannot occur) |
| `singleton` | `[x, ε, ε]` |
| `two_present_same` | `[x, x, ε]` |
| `agree` | `[x, x, x]` |
| `exact_2_of_3` | `[x, x, y]` — a token majority, **settled, no arm may touch it** |
| `unresolved_two` | `[x, y, ε]` |
| `unresolved_three` | `[x, y, z]` |

Flags on the unresolved classes: `strict_homophone`, `loose_homophone`,
`partial_homophone`, `max_char_dist`, and `split_merge` — two systems spelling the
same string across an adjacent pair of columns but cutting it in different places.
Split/merge columns are **quarantined from both arms**: a per-column edit cannot see
the neighbouring column it would have to agree with.

## "Phonetically equivalent" in Greek (frozen)

`eval/controlled_eval/greek_phonetics.py`. Input is already lowercase and
diacritic-free — the frozen scorer strips combining marks, so **stress and diaeresis
are gone before this module sees a token**. What is produced is therefore a
*pronunciation key under the scorer's normalization*, not a phonetic transcription.

**STRICT** (primary): `ου`→u, `αι`→e, `ει`/`οι`/`υι`→i, `η`/`ι`/`υ`→i, `ω`/`ο`→o,
final `ς`=`σ`, doubled **consonants** collapse in the source string, consonants
transliterated one-for-one. `αυ`/`ευ`/`ηυ` are kept **opaque** so the `υ` never folds
to /i/ and `ευα` is not declared a homophone of `εια`.

**LOOSE** (declared secondary variant, never a competing primary): STRICT plus
`μπ`→b, `ντ`→d, `γκ`/`γγ`→g, `τσ`, `τζ`, `ξ`→ks, `ψ`→ps, and `αυ`/`ευ`/`ηυ` →
af/av, ef/ev, if/iv by the voicing of the next character.

Runs are **never** collapsed in the produced key: `ποιητης` maps to three vowel
slots, and collapsing them would make it collide with `πιτης` (Codex job `55293f6b`).

Tests: `eval/controlled_eval/test_greek_phonetics.py`,
`eval/controlled_eval/test_column_classes.py`.

## Arms

**C** — on `unresolved_three`, not strict-homophone, `max_char_dist ≤ 2`, not
split/merge: align the three candidate strings character-wise with the same exact
3-way DP, vote with the same hierarchical rule, and **accept the composite only if it
equals one of the candidates or appears in the closed lexicon** (common-word table +
the city's frozen term list + this meeting's roster surnames). Otherwise keep W's
token. `unresolved_two` is excluded: two strings have no character majority, so a
"vote" there is candidate selection wearing a vote's clothes. C never emits epsilon
and never revives a column W dropped, so it cannot change the token count.

**H** — on `unresolved_two` or `unresolved_three` with a strict-homophone flag, not
split/merge. Epsilon is **never** offered as a candidate: occupancy was settled by
the vote, so H can substitute but never delete. KenLM first; an LLM
(`gpt-5.6-luna`, closed candidate indices, enforced in code) only on the columns
where KenLM abstains — a mechanism-level criterion fixed here so that "KenLM was
insufficient" can never be decided after seeing a WER.

**C+H** — both, H winning any overlap.

**Primary arm: C.** H and C+H are secondary; Holm across the three if more than one
is claimed.

## Cross-fitting

Fold = **city**, 10 folds. Everything an arm learns is estimated from the other nine
cities' reference text only, applied once to the held-out city, and **only the
concatenated out-of-fold outputs are scored**. The harness is
`eval/controlled_eval/fusion_lab.py`; an arm with no fitted parameter is reported
with `fitted: false` and its out-of-fold number is identical to its in-fold one by
construction.

## Gates (frozen, from the ticket)

1. Reject if out-of-fold `del_rate` rises above W's.
2. Reject if out-of-fold `ins_rate` rises above W's.
3. Primary: out-of-fold WER, paired bootstrap **clustered by meeting**, 10,000
   replicates, no refitting inside replicates. The CI must exclude zero.
4. Leave-one-out over window, meeting **and city**: zero sign flips.
5. Report the share of the column oracle (0.0475, range 0.0461–0.0479) recovered.

## What this design cannot buy

Codex job `55293f6b`, adopted verbatim:

- The meeting-clustered CI is **conditional** on the ten fitted folds and on these
  ten cities. It is not an interval for a new city, and ten cities are too few for a
  reassuring city-level bootstrap. Per-city deltas and the city-unweighted mean are
  reported beside it.
- Leave-one-city-out does **not** remove the contamination that matters here: the
  city term lists were mined from data overlapping this benchmark, and the class
  definitions were chosen by a human who has already seen four passes over these same
  247 windows. `frozen now` prevents further leakage; it does not undo past leakage.
- Therefore no result here is independent held-out confirmation, a valid confirmatory
  p-value, or evidence about unseen Greek cities. It is development-benchmark
  evidence.
- "Oracle recovery" is the fraction of the observed gap to an
  **alignment-conditional** hindsight quantity, not a fraction of recoverable errors.

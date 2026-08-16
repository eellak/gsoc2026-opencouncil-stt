# Composition over selection — preregistration

Frozen 2026-08-16, before any number on the full sample was computed. Design lives on
[wayfinder #22](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/22); this
file records the parts the issue left to the implementation, and the two deviations
from the issue's wording that a Codex review forced.

Experiment record: `exp-2026-08-16-composition-over-selection`.
Code: [`eval/controlled_eval/msa.py`](../../eval/controlled_eval/msa.py),
[`eval/controlled_eval/exp_composition.py`](../../eval/controlled_eval/exp_composition.py),
tests [`eval/controlled_eval/test_msa.py`](../../eval/controlled_eval/test_msa.py).

## Substrate

The 253 windows of benchmark run `2026-08-10-corrected-adapter-label-prefix-fix-vs-ju`
common to all 9 providers, **minus the 6 sealed temporal-holdout windows of
`eval-freeze-2026-08` that sit inside them** — 247 windows, 144 meetings, 74,917
reference tokens. The seal filter is copied verbatim from `exp_fusion_deletions.py`
and is asserted, not assumed.

Trio, inherited and **not re-selected here**: `scribe-v2-clean`, `soniox`,
`oc-runpod-fixed-2026-08-10`.

Baseline **V** = the existing whole-window consensus vote. Recomputing it here must
reproduce WER 0.1201 / del 0.0247 / ins 0.0443 / sub 0.0512, or the substrate is not
the one this experiment claims.

## Alignment (frozen)

Exact three-way dynamic programming, unit-cost sum-of-pairs, banded, with the band
checked against the recovered path and widened and re-run if the path ever pressed
against it. That check is a heuristic guard, not a certificate of the unbanded
optimum; the band is additionally floored per window at the largest pairwise length
difference, which is the constraint that actually binds on this data.

- state `(i, j, k)`; a transition advances any non-empty subset of the three token
  streams; entries are the advanced tokens, epsilon elsewhere;
- column cost = sum over the three pairs, 0 for equal or both-epsilon, else 1;
- among equal-cost alignments: fewest columns, then the frozen transition order
  `ABC, AB, AC, BC, A, B, C`;
- tokenisation and normalisation are the frozen `scoring.wtoks`; the composed output
  is the space-joined normalised tokens.

**Deviation 1, on Codex job `8112dc72`.** The issue's design was progressive
pivot-anchored alignment. That is not a profile alignment — the third hypothesis is
compared against a single representative token and the other token already in the
column is ignored — and it biases column boundaries towards the pivot, i.e. towards
the baseline W is meant to beat. Exact DP is the primary result. Progressive
alignment survives as a **declared sensitivity**: all six pivot/second orderings are
run and their full range is reported, never the best of them.

## Voting (frozen)

Hierarchical, per column:

1. **occupancy** — token vs epsilon, majority of three;
2. **identity** inside the winning class, majority;
3. still tied — the pivot's entry if it is in the winning class, else the first of
   the frozen priority order `scribe > soniox > ours` that is.

The pivot is the hypothesis V itself chose, so W reduces to V's token wherever the
three systems give no majority.

**Deviation 2, on the same Codex job.** The issue implied a flat vote in which
epsilon is one candidate among three. A flat vote deletes the column
`(epsilon, x, y)` — two systems assert a word, one asserts silence, and `x != y`
makes silence win. That is exactly the deletion failure this whole line of work
exists to avoid, so occupancy is voted first. `test_msa.py` locks it.

## Arms

| arm | definition |
|---|---|
| **V** | the whole-window consensus vote (baseline) |
| **W** | exact MSA + hierarchical per-column vote. No LLM. The critical arm. |
| **W+len** | W plus the length fail-safe |
| **W+L** | W plus an LLM arbitrating only the tie-broken columns |
| **W+D** / **W+L+D** | plus speaker-grounded restoration of dropped runs |

### W+len — the length guard, frozen threshold

**Deviation 3, on the same Codex job.** The issue asks for "never a candidate
noticeably shorter than the median". The 0.95-of-median threshold drafted for it was
rejected as arbitrary and coarse in both directions (a dozen missing words pass; a
window that barely crosses gets rewritten wholesale), and the epsilon-suppressed
recomposition it triggered was rejected as union-like and insertion-raising. Frozen
instead:

```
if len(tokens(W)) < len(tokens(V)):  emit V   else:  emit W
```

Per-column occupancy voting is the real guard; this is a window-level fail-safe.

### W+L — what the LLM may do

It sees only the columns where the vote had to tie-break, each with ±8 tokens of
already-decided context, that column's own candidate list, and the meeting's closed
term list (`roster_lexicon.py`, unchanged). It returns **an index**. It cannot write
a token no system proposed — enforced in code, not in the prompt — and it is never
shown, and never chooses, a whole hypothesis. An invalid or missing answer is a
no-op. Model `gpt-5.6-luna`, reasoning effort low, one run, prompt hash recorded.

### D — the speaker-grounded restoration rule, frozen

A **dropped run** is a maximal run of columns W emitted as epsilon in which at least
one system did propose a token. Because occupancy is voted first, such a run has
exactly one system carrying text.

The run's time interval is the gap between the interpolated times of the surviving
tokens on either side. Times come from anchoring token streams onto pyannote
precision-2's `wordLevelTranscription` (exact matches are anchors, linear
interpolation between, mean-rate extrapolation outside) — the `token_times` function
already used for #17.

The run is **restored** when the gap lasts at least 0.30 s and pyannote's
**non-exclusive** `diarization` over the gap either

- carries ≥ 2 simultaneous speakers, or
- carries a speaker set disjoint from the speakers active at both flanking tokens.

Otherwise it stays dropped. Non-exclusive is primary by user decision: the exclusive
timeline fails drop-safety, 10 utterances lost per 1 recovered
(`docs/reports/2026-08-08-exclusive-diarization.md`).

## The alignment-conditional column oracle

Exact DP over columns × reference, choosing one entry per column; the chosen text is
**reconstructed and re-scored by the frozen scorer**, never read off the DP's own
backtrace, because several minimum-cost outputs exist.

It is named *alignment-conditional* on Codex's instruction and must be reported as
such: the columns come from one particular alignment, and gap placement changes what
recombinations are reachable. The six progressive orderings are reported beside it
as the price of that conditioning. There is no reference leakage — the columns are
built from hypotheses only.

## Gates, frozen

- **REJECT** any arm whose `del_rate` rises above V's.
- **REJECT** any arm whose `ins_rate` rises above V's.
- Paired clustered bootstrap by **meeting**, 10,000 replicates, on WER and on all
  three rate components.
- Leave-one-out by **window, meeting and city**; sign flips are reported.
- Report the share of the oracle gap recovered — against the whole-window trio
  oracle, against the 9-system whole-window oracle, and against the column oracle —
  not only ΔWER.

## What may not be claimed

- **Nothing about speaker-attribution accuracy.** There is no ground truth; it waits
  for #21. Only WER / del / ins / sub of the composed text are measurable today.
- Agreement-with-OpenCouncil, not fidelity-to-audio.
- This is the **third** pass over the same 247 windows. Every pass makes the number
  more optimistic than it deserves even with a frozen design. The windows disjoint
  from **both** the glossary mining fold and the fine-tune training manifest are
  reported separately, and are underpowered.
- The alignment is the declared technical risk. Its quality is measured directly —
  anchor fraction per system, and the distribution of the absolute time discrepancy
  between the two independent times a matched token receives through two different
  hypotheses. If an arm fails, the report must say whether it failed there or in the
  idea.

# The vote/negation guard, priced on the arm we actually ship

2026-09-06. Wayfinder W4
([#61](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/61)).
Ledger: `exp-2026-08-26-fusion-integration`, and the caveat held open by
`exp-2026-08-26-chooser-deletion-safety`.
Substrate: the 391 post-June held-out windows, 117 (city, meeting) clusters,
report `bench_2026-08-22-post-june-held-out-test-clean-pack-cont-.json`
(sha256 `e33d5617…`, the one pinned in `MANIFEST.json`).

## The question

The guard was priced once, in
[`2026-08-26-category-chooser.md`](2026-08-26-category-chooser.md): "it rescues 26
and breaks 31, net −5 islands, i.e. slightly worse WER". That measurement was made
on the **policy** arm — deterministic rules *plus* an LLM chooser — on the 169
confirmation-city windows.

What ships is **rules-only (R1+R2) plus the guard**, and `guard: true` is hardcoded
in `FusionTranscriber`. Nobody had recomputed the trade for that arm.

## What the frozen guard actually is

Read from `eval/controlled_eval/chooser/make_fixtures.py`, which is the frozen
definition (the pricing script from 2026-08-26 was never rescued out of `/tmp`).
An island is guarded **iff** all three hold:

- removing every token in `CRIT` from the three candidate spans leaves three
  identical spans;
- at least one span contains a `CRIT` token;
- the arm's own output for that island contains none.

Then the output is replaced by the span with the most `CRIT` tokens (ties: longest,
then scribe > soniox > ours).

**`CRIT = ("ναι", "δεν")` — two tokens.** Not «οχι», not «μη», not «μην». The
ticket's working assumption of a five-word lexicon is wrong about the artefact that
is in the build. This matters for the name as much as the numbers: it is a *yes/not*
guard, not a general negation guard. (Both frozen tokens are already in the scorer's
normalized form — `scoring.norm` strips accents and lowercases — so they match the
substrate as written.)

## Method

`eval/controlled_eval/chooser/guard_audit.py`, run with the cached atlas alignment.
It rebuilds every window from `window_parts()` under the rules arm, applies the
imported `apply_guard`, and prices each fired island by a **counterfactual**: rebuild
the same window with the guard applied at that island *only*, and score with the
frozen `sdi`. "Right or wrong" uses the alignment-conditional island oracle
(`msa.oracle_select`), the same object the 2026-08-26 pricing's `oracle` field came
from.

Reproduction check before any new number was read: guard-off
`[5030, 1865, 5452, 110694]` = 0.11154 and guard-on `[5036, 1848, 5519, 110694]` =
0.11205, both identical to the frozen totals in `MANIFEST.json`. The per-island
counterfactuals sum to **+56** edit operations, exactly the measured all-on minus
all-off difference — the 75 islands do not interact, so per-island attribution is
exact rather than approximate.

## 1. The 75 islands and what each rescues

| | |
|---|---:|
| islands in the arm | 13,414 |
| islands where some span holds «ναι» or «δεν» | 821 |
| … and the rules output holds none | 355 |
| … and the rules output is **empty** | 119 |
| … guard fires (core-equality condition) | **75** |

Rescued token occurrences over those 75 islands: **«ναι» 72, «δεν» 12** (84 tokens;
67 islands gain one token, 7 gain two, 1 gains three). The lopsidedness is the first
thing to notice — this is overwhelmingly a *yes* guard.

Every one of the 75 islands had an **empty** rules output before the guard, and in
zero of them did the guard change a non-critical token. So the guard is doing exactly
one thing here: turning a whole-island deletion into a one-to-three-word emission,
never rewriting a word the arm had already chosen.

Adopted span: 55× soniox, 16× scribe, 4× ours.
Island category: `SOLO_soniox` 48, `DUP` 14, `SOLO_scribe` 10, `SOLO_ours` 3 — i.e.
almost always a single system emitting a word the other two do not have at all.

## 2. Correct or wrong, the rules-only equivalent of 26/31

By the island oracle, over the 391 windows:

| | islands |
|---|---:|
| rescued (was wrong, now right) | **12** |
| broken (was right, now wrong) | **63** |
| wrong both ways | 0 |
| right both ways | 0 |
| **net** | **−51** |

The verdict is binary in every one of the 75 cases: either the oracle keeps nothing
at that island (63 cases — the guard emits a word the reference does not have there)
or it keeps exactly the span the guard adopted (12 cases). Rescued tokens by verdict:
correct «ναι» 9, «δεν» 3; wrong «ναι» 63, «δεν» 9.

By WER, the per-island counterfactual: 12 islands lower the edit count (−12 total),
60 raise it (+68 total), 3 are neutral. Net **+56 edits**, which is the whole of the
0.11154 → 0.11205 move: +6 S, −17 D, +67 I.

**On the same 169 confirmation windows the policy pricing used**, so the arms can be
compared without changing the substrate:

| arm | rescues | breaks | net islands |
|---|---:|---:|---:|
| policy (published 2026-08-26) | 26 | 31 | −5 |
| rules-only (this measurement) | **10** | **31** | **−21** |

The breakage is identical and the rescue count collapses. The guard is a much worse
trade on the arm that ships than on the arm it was priced on, and the "net −5,
slightly worse" framing does not transport. The reason is visible in §1: on the rules
arm the guard only ever fires where R1 has already emptied an island, and R1 empties
islands that are usually right to empty.

## 3. What the rules-only arm deletes at islands

Whole-island deletions by the rules arm, guard off, over the 391 windows: **1,757**
islands emptied, of which **181 (10.3%)** are wrong by the island oracle. Word types
in what those 181 destroy (oracle target, 200 tokens, 89 distinct types):

| type | n | type | n | type | n |
|---|---:|---|---:|---|---:|
| η | 25 | ο | 8 | της | 5 |
| ναι | 9 | σε | 6 | ειναι | 5 |
| και | 9 | παρων | 6 | το | 5 |
| οι | 8 | να | 6 | του | 5 |
| | | δεν | 5 | τα | 4 |

then στις 3, θα 3, απο 3, την 3, μας 2, γι 2, and a 70-type tail at 1.

Against the policy arm's published table (το 18, να 18, και 14, για 11, **ναι 15,
δεν 11** over 114 wrong deletions on 169 windows) two differences are real:

- The shape is the same — short function words dominate, with the vote/negation pair
  present but not leading.
- The **rate** is very different. On the identical 169 confirmation windows the rules
  arm empties 758 islands and is wrong in 76 of them (**10.0%**), against the policy's
  449 emptied and 114 wrong (**25.4%**). The rules arm deletes *more often* and is
  *wrong less than half as often* when it does. «ναι» and «δεν» account for 14 of the
  200 lost tokens (7%).

One entry is worth flagging for the product rather than for WER: `παρων` 6. Losing a
roll-call presence marker is not a function word being dropped.

## 4. Domination

Per the repo protocol, before quoting the +56:

| grouping | carriers | net | gross of per-carrier nets | largest share |
|---|---:|---:|---:|---:|
| meeting | 38 | +56 | 68 | 6/68 = **8.8%** |
| window | 55 | +56 | 74 | 6/74 = **8.1%** |

Top five absolute per-meeting contributions: 6, 6, 5, 5, 4. **No single meeting or
window dominates**, and removing any one carrier leaves the net strictly positive —
there is no carrier whose deletion flips the sign. The guard being a loss on this arm
is a broad effect, not one bad meeting.

## Side measurement: the wider lexicon

Because the ticket assumed «ναι, δεν, όχι, μη, μην», the same audit ran the wider
lexicon `{ναι, δεν, οχι, μη, μην}` as an exploratory arm — accent-free, because the
substrate is (the literal «όχι» was also in the set and can never match). It fires on
**84**
islands instead of 75 and scores **0.11211** (`[5037, 1845, 5528, 110694]`) — worse
than both the frozen guard (0.11205) and guard-off (0.11154). Widening the lexicon
does not rescue the guard. This is a post-hoc arm and is reported as description
only; nothing was frozen on it.

## What this does and does not settle

It settles the accounting: on the shipped arm the guard costs 56 edit operations,
buys 12 correct rescues, causes 63 wrong ones, and is not carried by any one meeting.

It settles nothing about whether the guard is *right*. Every number here is
**agreement-with-OpenCouncil**, decided by the alignment-conditional island oracle
against the published reference. The argument for keeping the guard was never that it
lowers WER — it was that WER counts a spurious «ναι» the same as a lost vote and a
minutes reader does not. That asymmetry is still not in the metric, and the 36-clip
vote-and-negation listening test under
`exp-2026-08-26-chooser-deletion-safety` is still **unrun**. What has changed is the
price of the asymmetry argument: it now has to justify 63 wrong emissions per 12
right ones on 391 windows, not 31 per 26.

Two limits on the reference side, both inherited: the oracle is conditional on one
particular three-way alignment, and the published reference is not a listener, so
"the reference does not have «ναι» here" is not "«ναι» was not said".

## Could not be computed

- **Whether the policy's 26/31 used this same unit.** The pricing script was never
  rescued from `/tmp`; only its result survives, in prose. The 10/31 above is
  computed the way the surviving artefacts define the object (island oracle, rule-
  emptied islands), on the same 169 windows, but the like-for-like claim rests on
  that reconstruction, not on the original code.
- **Whether any of the 75 words was actually spoken.** Needs the audio. The
  instrument exists (`listen_server_small.py`, clips under `$SC/deletion_clips`) and
  has never been run.
- **A full attribution of the arm's 1,865 deletions to islands versus agree
  columns.** §3 counts *whole-island* deletions, which is the comparable object to
  the published table; it is not an alignment-level accounting of every deletion
  operation, and the two must not be added together.

## Artefacts

- Script: [`eval/controlled_eval/chooser/guard_audit.py`](../../eval/controlled_eval/chooser/guard_audit.py)
- Counts: `$SC/chooser-2026-08-25/results_guard_rules_only.json` (never git — it
  carries reference word types)
- Frozen guard definition: `eval/controlled_eval/chooser/make_fixtures.py`

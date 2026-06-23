# Meeting trust cutoff: edit-fraction distribution instead of the bare flag

Status: **plan + first real numbers** (2026-06-23). Answers the mentor's point that
`taskStatus.humanReview` is unreliable on its own. Computed locally by
`eval/meeting_edit_fraction.py` over the 327 cached public meeting JSONs; full
output in [`data/reports/meeting-edit-fraction/`](../../data/reports/meeting-edit-fraction/).

> **TL;DR (GR):** Ο humanReview flag **είναι σωστός όταν λέει TRUE** (0 false
> positives), αλλά **χάνει 95 meetings** που στην πραγματικότητα είναι reviewed
> (frac ≥15%) και απλά δεν έχουν το flag — ακριβώς ό,τι είπε ο Χρήστος. Λύση:
> trust = `humanReview=true` **Ή** `frac_user ≥ κατώφλι`. Υπάρχει καθαρό κενό:
> 13 «νεκρά» meetings με frac <3% κόβονται, τα υπόλοιπα κρατιούνται.

## What we measure

For every meeting, from the meeting JSON utterances (`lastModifiedBy ∈
{none, task, user}`):

- `frac_user = (utterances whose last editor is a human) / (total utterances)`

This is the per-meeting **human-intervention fraction** — the same quantity the
old HIR metric used, but here it's a *data-quality signal*, not a model metric:
a meeting nobody finished reviewing has `frac_user ≈ 0`, because its `no-edit`
utterances mean "nobody looked", not "ASR was right".

## What the data shows (327 meetings)

- Micro human-edit fraction across all utterances: **25.1%** (143,283 / 571,124).
- Per-meeting `frac_user`: median **27.3%**, p10 16.2%, p90 42.0%.
- The distribution has a **dead tail then a single broad hump**: 13 meetings sit at
  0–5%, a gap, then the mass climbs from 15% and peaks at **25–30% (76 meetings)**.

### The flag is unreliable in exactly one direction

| | finding |
|---|---|
| `humanReview=TRUE` (212) | `frac_user` min **9.6%**, median 28.2% — **no false positives** (none are near-zero). |
| `humanReview=FALSE` (115) | `frac_user` min 0% … max **99.2%** — median 26.2%. |
| FALSE yet `frac_user ≥15%` | **95 meetings** — look fully reviewed, flag just never set (old / 2025). |
| TRUE yet `frac_user <5%` | **0 meetings** — the flag never lies upward. |

So **trusting only `humanReview=true` throws away ~95 genuinely-reviewed meetings**
(most of the "115 not reviewed" we reported earlier are mislabeled, not unreviewed).
That is a lot of data to lose for a flag that's simply missing on older meetings.

### There's a clean cut for the junk

The lowest meetings are unambiguous: `frac_user` 0.0–2.5% (meganisi, nea-smyrni,
peristeri, thessaloniki/apr1 at 0.4%, rhodes at 2.5%), then it **jumps to 8.7%**
(chania/jan27). Any cutoff in the 3–8% band drops exactly those **13 dead
meetings** and keeps everything else.

| cutoff | kept | dropped |
|---|---|---|
| ≥3% | 314 | 13 |
| ≥10% | 311 | 16 |
| ≥15% | 300 | 27 |

## Recommended rule

**Two signals, used for two different jobs** — this is the important nuance:

1. **Dropping junk meetings** (from review queue, curation, and the corrections
   training set): cut on the fraction. **Drop `frac_user < 5%`** (the 13 dead
   meetings). This is strictly better than the existing `≥20 human-edited *count*`
   gate — e.g. `thessaloniki/apr1` has 37 human edits (passes the count gate) but
   `frac_user` 0.4% (fails, correctly: 37 stray edits over 9,282 utterances is not
   a reviewed meeting).

2. **Trusting the no-edit backbone** (treating `no-edit` utterances as ground
   truth): keep the **stronger** gate. Trust a meeting's no-edit utterances if
   `humanReview=true` **OR** `frac_user ≥ 15%`. The `OR` rescues the 95
   flag-missing meetings. Caveat: a high fraction proves people *edited a lot*, not
   that they *checked everything* — so a meeting that's only partially reviewed can
   still have high `frac_user`. For the no-edit backbone specifically, prefer
   `humanReview=true` where available and treat the rescued 95 as a second tier
   (good for the *correction* pairs regardless; for *no-edit* ground truth, sample
   a residual-WER audit before fully trusting — see [data.md open Qs](../decisions/data.md#open)).

## Why a fraction and not the raw count

A fixed count (≥10, ≥20) scales wrong: a 9,000-utterance meeting with 30 edits is
clearly unreviewed but passes any small count; a 300-utterance meeting fully
reviewed with 30 edits is fine but looks the same by count. The fraction
normalizes by meeting size, which is what "was this reviewed?" actually depends on.

## To produce / decide

- [x] Compute `frac_user` per meeting + the flag cross-tab (`eval/meeting_edit_fraction.py`).
- [x] Confirm the flag is reliable upward, lossy downward (95 rescued).
- [x] **Junk cut decided 2026-06-23: drop `frac_user < 5%`** → the 13 meetings are
      denylisted in `data/exclusions/unreviewed_meetings.json` and enforced
      everywhere (see [data.md](../decisions/data.md#2026-06-23---exclude-13-unreviewed-meetings-5-human-edit-fraction-from-our-set)).
- [ ] **Pick the backbone-trust threshold at the Thursday meeting** (proposed 15%
      OR `humanReview=true`). Numbers above support it.
- [ ] Wire the chosen cut into the dataset build (`eval/build_split.py` /
      `prepare_asr.py`) and, if we keep the review filter, into
      `meeting-eligibility.ts` as a fraction option alongside the count.
- [ ] For the rescued 95: residual-WER spot-audit before using their no-edit
      utterances as ground truth.

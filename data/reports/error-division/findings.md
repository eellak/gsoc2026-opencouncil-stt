# Who fixes what — first numbers (Design A, 1 seed)

_2026-06-24. whisper-base fine-tune (CPU) vs the on-box LLM fix-task, scored on a
held-out, categorized val set: 114 corrected utterances from 28 meetings in two
cities (Orestiada, Argos) the model never trained on. The raw table is in
[matrix_categorized.md](matrix_categorized.md); the plan is in
[docs/specs/error-division.md](../../../docs/specs/error-division.md)._

## The setup in one line

Four transcripts of the same held-out audio, scored the same way: **A** raw ASR,
**B** raw ASR then the LLM fix-task, **C** fine-tuned ASR, **D** fine-tuned ASR then
the LLM. The point is to see, per error type, which stage actually removes the
error — so we know what the fine-tune should learn and what to leave to the LLM.

## Headline: the two stages add up, they don't overlap

| stage | WER (normalized) |
|---|---|
| A — raw ASR | 0.715 |
| B — ASR + LLM | 0.578 |
| C — fine-tuned ASR | 0.613 |
| D — fine-tuned + LLM | **0.478** |

The fine-tune takes about 0.10 off the error rate. The LLM takes about 0.14 off.
Run both and you get roughly the sum, down to 0.478. The paired numbers say it
plainly: the LLM removes ~0.135 whether or not you fine-tuned first (D−C ≈ B−A),
and the fine-tune removes ~0.10 whether or not the LLM runs after (C−A ≈ D−B). So
they're mostly fixing **different** errors. Neither makes the other redundant —
keep both.

(The fine-tune effect on its own, C−A, has a 95% CI of [−0.29, −0.005] — real but
wide on one seed. The two LLM effects are tight: [−0.16, −0.11].)

## The split shows up exactly where we predicted

Only `substitution_phonetic` has enough data (59 clips, 25 meetings) to stand on
its own; everything else is directional. But the two biggest categories are also
the two we most wanted to contrast, and they behave like the routing table said
they would:

| category | n | A (raw) | C (fine-tuned) | B (LLM) | D (both) | who fixes it |
|---|---|---|---|---|---|---|
| `substitution_phonetic` | 59 | 0.784 | **0.642** | 0.640 | 0.515 | **fine-tune** (and it stacks) |
| `punctuation_capitalization` | 27 | 0.556 | 0.536 | **0.388** | 0.393 | **LLM** (fine-tune adds ~nothing) |

- **Phonetic substitutions — the fine-tune earns its keep.** Raw ASR is at 0.784;
  fine-tuning alone drops it to 0.642, a cut as big as the LLM gets on its own. And
  fine-tuned+LLM (0.515) beats LLM-alone (0.640), so the acoustic model is catching
  things the LLM can't — exactly the errors that need the audio.
- **Punctuation and casing — leave it to the LLM.** Fine-tuning barely moves it
  (0.556 → 0.536). The LLM does all the work (→ 0.388), and fine-tuned+LLM is the
  same as LLM-alone. Spending acoustic-model capacity here buys nothing.

## Directional hunches (small n — treat as hints)

- `homophone` (14) patterns like phonetic: fine-tune helps (0.874 → 0.659), and
  both stacked go to 0.504. Acoustic — fine-tune territory.
- `number_date` (5) and `word_boundary` (11): fine-tune is neutral-or-worse, the
  LLM recovers it. Leave to the LLM.
- `place_name` (10) / `person_name` (4): the LLM does most of the work here; the
  fine-tune is roughly neutral. Consistent with "shared, LLM lands the exact
  spelling."

## What this means for the dataset

Bias the fine-tuning data toward the errors that need the audio — phonetic
substitutions and homophones first, word boundaries and names for getting close
acoustically. Don't spend the fine-tune's capacity on punctuation, casing, numbers,
dates, accents, or grammar: the LLM fix-task already handles those, and the numbers
say fine-tuning them adds nothing to the final transcript. Keep the ~50% no-edit
backbone (it's why ordinary speech didn't regress). And keep both stages in the
pipeline — they're additive.

## Caveats

One seed. Only the aggregate and `substitution_phonetic` have CIs you can lean on;
every other category is directional (n < 30 or fewer than 5 meetings) — read those
as a steer, not a result. Five clips were dropped because the LLM returned an
output we couldn't parse to a single clean line; they were dropped from all four
stages so the paired comparison stays aligned, and the count is logged, not hidden.
The fine-tune is the round-1 composition keeper (corrections + 1× no-edit backbone),
40 CPU steps — small, so the magnitudes are a floor, not a ceiling.

## Next

- Seeds 1–2 to tighten the fine-tune CI (mainly helps the aggregate and phonetic;
  ~quota-heavy, optional).
- Design B (composition sweep: natural / acoustic-focus / balanced) — now that the
  val set can resolve the fine-tune effect, this can actually rank mixes.
- Fold the routing into the dataset build's category weights and the
  [error taxonomy](../../../docs/reference/error-taxonomy.md) buckets.

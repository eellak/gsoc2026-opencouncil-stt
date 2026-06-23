# Metric: Human Intervention Rate (HIR)

Status: **likely NOT adopted** — mentor pushback at the 2026-06-23 sync. WER (+ CER)
stays the standard metric; HIR is under a final critical review by Angelos this week
with a current recommendation **not to implement**. See the "Meeting outcome" box
below before building anything on this. Computed on real data:
`ui/static/coverage.json` (published snapshot), `eval/fetch_speakers.py`.

> **Meeting outcome (2026-06-23 mentor sync) — see [mentor-sync](../meetings/2026-06-23-mentor-sync.md).**
> Christos argued against adopting HIR, on several grounds:
> 1. It also measures the **LLM fix-task pass** (model + prompt dependent), not the
>    ASR alone.
> 2. **Utterances are not intrinsic to speech** — Whisper emits them arbitrarily
>    (we feed it 30-min/hour audio and it self-segments; we do *not* cut per
>    utterance then transcribe). A different model → different utterance boundaries
>    → different HIR, with no change in transcript quality. Switching to
>    speaker-segments doesn't fix it (a meeting with few long speakers almost
>    always has ≥1 error → ~100%; many short speakers → better — the metric tracks
>    meeting structure, not quality).
> 3. It needs an **extra LLM task** to run (cost + time) for a metric that looks
>    fundamentally flawed.
> 4. It is **not in the literature** → not comparable; HF/paper readers expect
>    standard metrics. WER reflects what we want well.
>
> **The valid concern to keep (without HIR):** a fine-tune could introduce *new*
> errors that the fix-task hides, leaving us with a model dependent on the LLM.
> Track this as a **diagnostic** — compare raw fine-tuned transcript WER vs
> post-fix-task — not as a headline metric.

## What it measures

The project's real-world cost is **human reviewer time**. The single number that
tracks it: *what fraction of utterances did a human have to correct* after the
automatic pipeline (ASR → LLM fix-task) already ran.

> **HIR = (# utterances edited by a human) / (# total utterances)**, computed
> over a **fully human-reviewed meeting** (`taskStatus.humanReview = true`).

Per-utterance provenance comes from the meeting JSON's `lastModifiedBy`:
- `none` — neither the fix-task nor a human changed the raw ASR (accepted as-is).
- `task` — the LLM fix-task made the final edit; the human accepted it.
- `user` — **a human made the final edit → an intervention.**

So HIR counts `user` utterances. Lower is better; the goal is to **drive HIR down**
by improving the ASR (fine-tuning) and the fix-task.

### Positive-framed twin (optional, "sexier")
**First-Pass Yield (FPY) = 1 − HIR** — the share of utterances the pipeline gets
right with *zero* human touch. Borrowed from manufacturing QA (fraction passing
without rework); rises as quality improves. Same metric, glass-half-full. Pick
HIR for "reduce the cost" framing, FPY for "raise the autonomy" framing.

## Why it is well-defined (and the gate that makes it so)

HIR is only meaningful on **fully-reviewed** meetings. In a meeting nobody
finished reviewing, a `none` utterance doesn't mean "ASR was right" — it means
"nobody looked." Mixing those in corrupts the denominator. The
`taskStatus.humanReview` flag is the gate: **compute HIR only where it is true.**
(This is exactly why `thessaloniki/apr1_2026` — `humanReview=false`, 37 stray
edits over 9,282 utterances — must be excluded.)

## Aggregation + uncertainty (math-grounded)

- **Micro HIR** — pool all utterances across meetings, one ratio. Headline number.
- **Macro HIR** — mean of per-meeting HIR. Equal weight per meeting; surfaces
  variance across meetings/cities.
- **Confidence:** it's a binomial proportion → **Wilson 95% CI**. Utterances in a
  meeting are *not* independent, so for cross-meeting comparisons use a
  **meeting-clustered bootstrap**, not the naive per-utterance CI.
- Report micro + macro + CI together; a gain is real only if it survives the
  clustered CI.

## Current baseline (212 reviewed meetings, 394,742 utterances)

- **Micro HIR = 28.1%** (Wilson 95% CI [28.0%, 28.3%] — *approximate: assumes
  utterance independence, so it is artificially narrow; the true meeting-clustered
  bootstrap CI is wider*) → FPY ≈ 71.9%.
- **Macro HIR = 30.1%** (median 28.0%, p10 18.8%, p90 45.0%).
- By city (micro): chalandri 21.2%, chania 22.5%, zografou 23.7% (low) …
  athens 33.1%, samothraki 37.9% (high).

Interpretation: today, **~28 of every 100 utterances need a human fix** after the
automatic pipeline. That is the number we want to shrink, and the headline for
"how is the project doing."

## Useful variants (secondary)

- **Word-weighted HIR** — fraction of *words* changed by humans (finer; closer to
  WER, less sensitive to one-token fixes).
- **Duration-weighted HIR** — weight utterances by audio seconds (a 2-minute
  utterance ≠ a 2-second one).
- **Category-specific HIR** — restrict to named-entity / number / acronym
  utterances to track the hard, high-value errors separately.

## Relationship to WER

WER is the acoustic-quality proxy (word-level edit distance). **HIR is the human-
cost metric the project actually optimizes** (did a human have to step in). Track
both: WER explains *why* HIR moves; HIR is the goal. A fine-tune that lowers WER
should lower HIR — verify on held-out reviewed meetings with clustered CIs.

## Open questions for the meeting

- `user` = final editor. Should we instead count *any* human touch in the edit
  chain (task→user counts even if a later pass reverted)? Current def = final state.
- Floor for "reviewed": is `humanReview=true` enough, or also require a minimum
  reviewed fraction / reviewer sign-off?
- Headline = micro or macro? (Proposed: micro headline, macro + CI alongside.)

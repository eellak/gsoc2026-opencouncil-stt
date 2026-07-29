# Where the fine-tune's gain actually comes from (2026-07-29)

Scripts: `eval/diagnose/decompose_training_gap.py`, `eval/diagnose/full_meeting_context.py`
Raw: `eval/diagnose/results.json`, `eval/diagnose/ctx_results.json`

Follow-up to [the stuck-state handoff](../handoff/2026-07-29-where-we-are-stuck.md). Purpose:
explain why the training eval, the n=50 controlled A/B and the n=59 name-focused A/B
disagreed, and find out what — if anything — the LoRA actually learned.

## Setup

n=300 held-out (argos + orestiada) human-corrected utterances. Reference =
`final_after_text` (human). Span filter 0.3–30 s, ±0.2 s pad — the training script's own
`ok_span`. Scored with the training script's `gnorm` (lowercase, accent-strip,
final-sigma fold, punctuation→space). Both models served on one identical stack.

## Headline

| config | WER raw | WER norm |
|---|---|---|
| scribe_before (see caveat) | 27.06 | 22.39 |
| base, faster-whisper beam=2 | 35.67 | 27.08 |
| **ours, faster-whisper beam=2** | **32.22** | **22.66** |
| base, faster-whisper greedy | 37.05 | 28.78 |
| ours, faster-whisper greedy | 32.57 | 23.00 |
| base, HF generate greedy | 36.62 | 28.25 |
| ours, HF generate greedy | 35.82 | 28.18 |

Bootstrap (2000 resamples) on ours−base, faster-whisper beam=2: **−4.35 pp,
95% CI [−6.86, −2.28]**. Head-to-head 107 better / 63 worse / 130 tie.

**`clean_up_tokenization_spaces` produced byte-identical WER** (True vs False). Ruled out
as a factor in the June numbers.

## 1. The n=50 "regression" was a sampling artifact

The 2026-07-24 A/B filtered to `4–30 s AND ≥6 words` and concluded the fine-tune regressed
(base 0.158 vs ours 0.176). Applying that exact filter to this run:

| subset | n | base | ours | delta (95% CI) |
|---|---|---|---|---|
| all | 300 | 27.08 | 22.66 | **−4.35 [−6.86, −2.28]** |
| the old filter (≥4 s, ≥6 words) | 84 | 16.99 | 16.43 | −0.41 [−3.44, +1.73] |
| everything it excluded | 216 | 38.96 | 29.98 | **−8.98 [−12.80, −5.46]** |

On the old filter the two models tie. The old n=50 result was noise, and the filter removed
exactly the utterances where the model helps. The "do not migrate" conclusion rested on it.

## 2. The gain is entirely a short-utterance effect

WER by utterance duration (normalized, faster-whisper beam=2):

| duration | n | base | ours | delta | scribe |
|---|---|---|---|---|---|
| 0–2 s | 144 | 48.9 | 36.7 | **−12.2** | 34.3 |
| 2–4 s | 70 | 28.1 | 23.0 | −5.1 | 26.4 |
| 4–8 s | 61 | 19.9 | 17.7 | −2.2 | 17.2 |
| 8–15 s | 21 | 15.4 | 16.1 | +0.7 | 14.7 |
| 15–31 s | 4 | 9.8 | 11.6 | +1.8 | 10.4 |

Monotonic. The advantage is large on very short clips, gone by 8 s, mildly negative beyond.

## 3. What it actually fixed: base whisper's short-clip hallucination

The wins are not subtle rescoring — base **invents** text on short isolated clips:

| | text |
|---|---|
| REF | με ισόποση μείωση του κεφαλαίου. |
| base | Γεια σας, είμαι ο Κωνσταντίνος και θα σας πω τι θα λέω. |
| ours | Εξ όπους η μείωση του κεφαλαίου. |

| | text |
|---|---|
| REF | Όσον αφορά το κατεπείγον, |
| base | Σας ευχαριστώ. |
| ours | Όσον αφορά το κατεπείγον... |

This is the well-known Whisper failure mode on short segments (it falls back to high-prior
filler phrases). **The LoRA's main achievement is suppressing it** — which is a genuine and
useful behaviour, but it is a pathology *of isolated short clips*, and our eval creates
isolated short clips by construction. See the open question in §6.

Losses look different — name/term drift, the failure mode the earlier postmortem described:

| | text |
|---|---|
| REF | να πάτε να καταλάβετε από τη Διεύθυνση Γεωργίας, |
| base | Να πάτε να καταλάβετε από τη Δένση Γεωργίας. |
| ours | Να πάτε να καταλάβατε από τη ΔΕΝ της Αγιοργίας, |

## 4. Per-category effect

Pooled WER delta by the reviewer's error labels (categories with ≥150 reference words):

| category | ref words | base | ours | delta |
|---|---|---|---|---|
| number_date | 156 | 42.3 | 31.4 | **−10.9** |
| word_boundary | 258 | 40.3 | 32.6 | **−7.8** |
| insertion | 231 | 31.2 | 24.2 | **−6.9** |
| substitution_phonetic | 1409 | 29.2 | 23.5 | −5.7 |
| punctuation_capitalization | 599 | 19.5 | 17.7 | −1.8 |
| homophone | 355 | 24.8 | 23.1 | −1.7 |
| place_name | 182 | 25.8 | 25.3 | −0.5 |
| noun_case | 317 | 19.9 | 19.9 | ±0.0 |
| article_pronoun | 186 | 17.2 | 18.3 | +1.1 |
| verb_inflection | 245 | 22.9 | 24.1 | +1.2 |

Gains concentrate in categories that co-occur with short fragments and formatting
(`number_date`, `word_boundary`, `insertion`). Morphology categories (`noun_case`,
`verb_inflection`, `article_pronoun`) are flat or slightly worse — the adapter did **not**
learn Greek grammar, and the small regressions there are consistent with the drift seen in §3.

## 5. Onset drop is real but small

First-word mismatch rate against the reference:

| model | mismatch |
|---|---|
| scribe_before | 21.3% |
| base | 34.3% |
| **ours** | **38.7%** |

The fine-tune loses the first word slightly more often than base (+4.4 pp), and produces
2.1% fewer words overall (hyp/ref word ratio 0.979 vs base 1.000). Consistent with training
on cut clips. It is a real regression, just much smaller than the short-clip gain.

## 6. Caveats that limit what this proves

**`scribe_before` is not a fair competitor.** The reference was produced *by editing
Scribe's output*, so Scribe's "WER" is simply the size of the human edit, not a measure of
ASR quality on fresh audio. It is a useful floor ("how much did humans change") and nothing
more. Do not report it as "Scribe beats our model" — the comparison is structurally rigged
in Scribe's favour. (Only 8.0% of these utterances normalize to identical text, so it is not
degenerate, but the bias is unavoidable.)

**The gain may be an artifact of isolated-clip evaluation.** Everything above cuts each
utterance out and decodes it alone — the exact condition the model was trained on, and the
exact condition that triggers the short-clip hallucination the model fixes. Production
decodes continuous meeting audio, where no utterance is presented in isolation and the
pathology may never arise. **This is the load-bearing open question**, and
`eval/diagnose/full_meeting_context.py` is running to answer it: same utterances, scored both
as isolated clips and as extracted-by-word-timestamp spans of a continuous 25-minute decode.

**The June training numbers are still not explained.** The candidates in the handoff
(checkpoint selected on the val set; `val_reg` scored against another ASR's output) remain
untested. What *is* now known: the decoder stack matters enormously for the fine-tune
(22.66 under faster-whisper vs 28.18 under HF generate) while barely affecting base
(27.08 vs 28.25). That asymmetry is unexplained and deserves attention before any number
from either harness is trusted.

**n=300 from 2 cities.** Meeting-clustered CIs would be wider than the bootstrap above,
which resamples utterances independently.

## Bottom line

The fine-tune is **not** the failure the 2026-07-25 postmortem described — that conclusion
came from a subset that excluded the effect. It gives a real −4.35 pp on held-out corrected
utterances, driven almost entirely by suppressing base whisper's short-clip hallucination,
at the cost of a small onset regression and mild name drift.

Whether that survives continuous decoding decides whether it is worth anything in
production. Until the context run reports, **no migration decision should be made.**

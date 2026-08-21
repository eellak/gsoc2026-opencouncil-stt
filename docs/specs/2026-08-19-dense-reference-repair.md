# Frozen protocol: audio-faithful repair of two validation references

Frozen 2026-08-19, before any repaired text exists and before any model is rescored
against a repaired reference.

## Why

The blind listening audit
([report](../reports/2026-08-19-training-listening-audit.md)) judged both
insertion-heavy validation windows `material_omission`: the published OpenCouncil
reference omits words that are clearly spoken. Until those references are repaired,
extra model words on these windows cannot be separated into genuine insertions and
recovered speech.

## Scope, frozen by id

Exactly two windows, named here so that nothing is re-selected by outcome later:

| window_id | duration | published tokens | audio sha256 (first 16) |
|---|---|---|---|
| `win_argos_oct31__2_2025_2353650` | 149.675 s | 244 | `1955f6e2e3a5f161` |
| `win_argos_sep24_2025_371824` | 149.263 s | 313 | `0172711b5455c4b6` |

The builder must assert that the validation manifest holds exactly one row per id,
that the prior audit's hidden key maps exactly these two validation items, and that
both prior answers are `material_omission`. Any mismatch is a hard failure; the queue
is never recomputed from insertion deltas again.

## Two references, two metrics, never merged

After repair each window carries both:

- `published` — the original OpenCouncil reference tokens, unchanged and preserved.
- `audio_faithful` — the human repaired reference, reference-assisted (see below).

and the project's two metrics stay separate:

- **agreement-with-OpenCouncil** stays computed against the original 39 published
  references. Nothing in this repair changes that substrate.
- **fidelity-to-audio** may be computed only on these two repaired windows.

**A hybrid 39-window WER — 37 published references plus 2 repaired ones — is
forbidden.** It would merge the two metrics into a number that means neither.

## Reference-assisted, not independent gold

**Revised 2026-08-19 after the first reviewer attempt.** The original design hid the
published text during an audio-first pass and asked which words were missing. That
question is unanswerable: without seeing what was written down, a listener cannot say
what is absent from it. The hidden pass is withdrawn.

The window is cut into fixed 20 s intervals. Each interval shows its own play control
and, in an editable field, the published words whose forced-aligned start falls inside
it. The reviewer listens and edits that text into what is audible — adding what is
missing, deleting what was not said, fixing wrong words. The window's repaired
reference is the concatenation of its intervals in order.

Word times come from `eval/controlled_eval/align_published_reference.py`, which
force-aligns the published reference text — and only that text — onto the window audio
with the same CTC aligner `anchor_timings.py` uses. No model transcript is involved and
no character of the published reference changes. An interval boundary can still fall
mid-word, so a word may appear one interval early or late; that affects only where the
text is displayed, never the concatenated result. The builder asserts that joining the
intervals reproduces the published reference verbatim.

The artifact is therefore a *reference-assisted repair*, not an independently
transcribed gold reference, and must be labelled that way wherever it is used. Because
the published text is visible throughout, the repair can inherit its omissions; that is
the accepted cost of making the task performable at all.

## Transcription policy

Two careful reviewers must produce the same tokens, so:

- Write what is audible, verbatim, including repetitions and false starts.
- Punctuation, casing and accents are irrelevant; the scoring tokenizer discards them.
- Overlapping speakers: transcribe every word that is separately intelligible, in the
  order it starts. Do not attribute speakers; the reference is speaker-free. Where
  simultaneous speech cannot be separated, leave it out or mark it `[?]` — a tangle
  nobody can transcribe is a limit of the artifact, not a defect to resolve.
- Numerals stay as spoken (words if spoken as words, digits if read as digits); the
  tokenizer treats both as tokens and the published reference already mixes them.
- Unintelligible span: mark `[?]`. It is not a token for scoring and it does **not**
  block finalization; the aggregate reports `uncertain_spans` per window so that a
  window carrying many of them is read with that limit attached.
- Non-speech (applause, microphone noise, silence) is not transcribed.
- Names whose spelling is acoustically ambiguous: write the most plausible spelling
  and leave a note; spelling variants are a known limit of this artifact.

## Scoring, frozen

- Tokenizer: `eval.controlled_eval.eval_freeze.ftoks` (the frozen scoring tokenizer
  minus hesitation fillers). No new normalizer.
- Alignment: `eval.controlled_eval.exp_same_stack.sdi`, reused unmodified.
- Reported per window: `published_tokens`, `repaired_tokens`,
  `sdi(published, repaired)` as substitutions / deletions / insertions, and
  `net_token_delta = repaired_tokens - published_tokens`, with the arithmetic
  identity `repaired_tokens == published_tokens - deletions + insertions` asserted.
  Here deletions are published tokens the repair removed and insertions are tokens
  the repair added.
- A window is `complete` only when its answer carries `repaired_text`,
  `listened_to_full_audio: true`, `finalized: true`, a matching `source_fingerprint`
  and a numeric client timestamp, and `ftoks(repaired_text)` is non-empty.
  `repaired_text` is written by the page as the ordered join of the interval fields.
- A finalized window whose normalized repair equals the published tokens is reported
  as `needs_adjudication`, not `complete`: the prior audit called it
  `material_omission`, so an unchanged repair is a contradiction to resolve, not a
  result.

## Privacy and artifacts

- Council audio, published tokens, repaired text and free-text notes never enter git.
  They live under `~/.cache/oc-public/dense-reference-repair-2026-08/`.
- The served page contains no model hypothesis, arm, seed, WER, S/D/I, selection
  reason or real window id — served items carry opaque review ids only.
- Git receives one content-free aggregate, `eval/results_dense_reference_repair.json`:
  counts, S/D/I, and sha256 of each repaired text. No transcript, no notes.
- On finalization the repaired text is copied into an immutable private
  `repaired-references.json`; the mutable `answers.json` is never the canonical
  artifact.

## Completion criterion

Both windows finalized, the immutable private artifact written, the content-free
aggregate committed, and this protocol frozen before any model output is read against
a repaired reference.

## What this does not do

- It does not build the strict validation set. Two outcome-selected windows are a
  diagnostic, not a validation substrate; the strict set is frozen separately.
- It does not revise the dense screen. `SCREEN — STOP` stands regardless of outcome.
- It does not authorize a rescore. Rescoring the frozen dense hypotheses against the
  repaired references consumes candidate output and needs its own preregistration
  naming the hypothesis artifact hashes, all three paired seeds, both reference
  targets and their denominators, with every repaired-reference number labelled
  post-hoc diagnostic.

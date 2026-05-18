# Matching decisions

Meeting JSON usage, `utterance.text` semantics, and open matching/taxonomy questions.

## Accepted

### 2026-05-12 - Use the large meeting JSON for the first prototype

The first exploration UI should use the existing meeting transcript JSON rather than requiring a new API endpoint.

Reason: the endpoint already contains meeting metadata, city, transcript segments, utterances, people, parties, subjects, and speaker tags. That is enough for local exploration if we can match CSV corrections to utterances.

### 2026-05-12 - `utterance.text` is the corrected text, not the original

In the OpenCouncil meeting JSON, `utterance.text` reflects the **after-edit** state. The original pre-edit text exists only in the corrections export (`before_text` column), not in the live transcript JSON.

Implication: we cannot recover `before_text` by reading the meeting JSON; it must come from the corrections export. Matching CSV rows to utterances should compare `after_text` to `utterance.text`, not `before_text`.

## Open

### Correction-to-utterance matching strategy

How should a CSV row be matched to a specific utterance in the large meeting JSON?

Likely matching fields:

- `audio_url`
- `youtube_url`
- `meeting_name`
- `meeting_date`
- `utterance_start`
- `utterance_end`
- `after_text` compared with `utterance.text`

Need to define confidence levels: exact match, timestamp-near match, text-near match, ambiguous, unmatched.

### Error taxonomy

We need a practical taxonomy that supports both LLM pre-classification and human correction in the UI.

The taxonomy should separate:

- ASR/domain errors useful for training;
- punctuation/formatting-only edits;
- semantic/contextual edits;
- timestamp/alignment problems;
- unclear or useless rows.

### Include/exclude semantics

Define exactly what `include` means.

Likely meaning: include this correction in the candidate dataset for future training/evaluation, not necessarily final training approval.

### Training/evaluation pair

For the first UI, do not force a final decision. Show each CSV correction as an edit pair.

Later, for training/evaluation, decide whether the unit is:

- individual edit pair;
- first `beforeText` to last `afterText` per utterance;
- only human edits;
- only selected included rows.

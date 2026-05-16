# Error Taxonomy and Routing

This note refines the proposal appendix into an operational taxonomy: each correction pair should eventually answer "where is this error best fixed?"

Primary reference: [GSoC proposal Appendix A](gsoc-proposal.md#appendix-a-corrections-sample-analysis-and-error-taxonomy).

Data reference: [Correction data quality](../../data/reports/data_quality.md).

## Routing Buckets

| Route | Use for | Typical examples | Implementation target |
| --- | --- | --- | --- |
| `asr_finetune` | Errors likely caused by acoustic/phonetic recognition or Whisper language-model bias | iotacism, consonant confusion, root-morpheme errors, stable domain terms | Whisper/faster-whisper fine-tuning dataset |
| `llm_post_correction` | Errors requiring text context, grammar, or dynamic vocabulary | agreement, acronyms, names when a municipality glossary exists | LLM prompt examples and vocabulary injection |
| `rule_based` | Deterministic cleanup cheaper than model calls | whitespace, ASCII `?` to Greek `;`, punctuation-only cases | pre/post-processing normalizer |
| `review` | Rows needing audio inspection or alignment checks | missing speech, hallucination, long realignment, malformed rows | manual review before training |

## Proposed Taxonomy

### A. ASR-Fine-Tuning Candidates

- Phonetic confusions: words sound similar but differ textually.
- Greek iotacism and vowel spelling confusions.
- Consonant cluster errors.
- Morphological/root errors where the spoken form was likely correct.
- Repeated domain terms that should be learned by the acoustic model.

Use these for audio -> corrected text examples when:

- `audio_url` is valid.
- `utterance_start` and `utterance_end` are valid.
- `after_text` is complete and clean enough to be the reference.
- The correction is not just punctuation/capitalization.

### B. LLM Post-Correction Candidates

- Grammar and agreement corrections.
- Contextual substitutions where the correct answer depends on sentence meaning.
- Acronyms and administrative codes when a valid vocabulary list exists.
- Proper nouns that change per municipality or meeting.

These are useful as few-shot prompt examples and as tests for dynamic vocabulary injection. They should not automatically be pushed into Whisper fine-tuning, because local officials, place names, and meeting-specific entities change over time.

### C. Rule-Based Normalization

- Whitespace normalization.
- Greek question mark normalization: `?` -> `;` when the utterance is Greek.
- Pure capitalization/punctuation differences when the words are otherwise identical.

These should be removed from expensive training and LLM correction unless there is a specific evaluation reason to keep them.

### D. Review / Alignment Risk

- Empty before/after text.
- Invalid or missing timestamps.
- Missing/invalid audio URL.
- Very short or very long durations.
- Rows where the CSV parser detected broken field alignment.
- Large text length changes that may reflect truncation or hallucination.

These can still be valuable, but only after audio listening or source re-extraction.

## Bootstrap Labels in the Clean CSV

[`data/clean/corrections_clean.csv`](../../data/clean/corrections_clean.csv) includes:

- `heuristic_route`
- `heuristic_error_family`
- `text_similarity`

These are triage labels, not ground truth. The current heuristic relies on text similarity and simple punctuation/case rules. It cannot know whether the speaker misspoke or the ASR misheard without audio.

## Manual Review Plan

For each route, sample at least 50 rows:

1. Confirm whether the route is plausible.
2. Mark whether the row is suitable for ASR fine-tuning, LLM prompt examples, rule-based cleanup, or exclusion.
3. Listen to the audio for ambiguous rows.
4. Revise the heuristic thresholds after review.

# Correction CSV Data Quality

Source: [`archive/old-data/corrections-sample.csv`](../../archive/old-data/corrections-sample.csv)

Outputs:
- Clean CSV: [`data/clean/corrections_clean.csv`](../clean/corrections_clean.csv)
- Rejected rows: [`data/reports/corrections_rejected.csv`](corrections_rejected.csv)
- Machine-readable summary: [`data/reports/corrections_summary.json`](corrections_summary.json)

## Record Counts

- Parsed source records: 4865
- Clean analysis-ready records: 4629
- Rejected records: 236

## Rejection Reasons

- `no_text_change`: 104
- `missing_after_text`: 79
- `invalid_edited_by`: 42
- `invalid_timestamps`: 42
- `invalid_audio_url`: 40
- `missing_before_text`: 19
- `non_positive_duration`: 1

Rows can have more than one rejection reason. The most important class is `invalid_edited_by`, which usually indicates broken CSV alignment rather than a real editor value.

## Editor Split

- `task`: 2566
- `user`: 2063

Use `task` as the current LLM/post-correction stage and `user` as human review intervention. This split is central for HIR and for comparing what the LLM already fixes against what humans still correct.

## Initial Heuristic Routing

- `asr_finetune`: 3233
- `llm_post_correction`: 766
- `rule_based`: 394
- `review`: 236

This is a rough bootstrap label, not ground truth. Use it to prioritize manual review and mentor discussion:

- `asr_finetune`: likely useful for Whisper/STT fine-tuning.
- `llm_post_correction`: likely useful for prompt examples, grammar/context correction, or dynamic vocabulary.
- `rule_based`: likely cheap normalization before either model.
- `review`: likely needs audio/listening or alignment checks before use.

Error-family counts:

- `morphological_or_phonetic`: 2800
- `capitalization_or_punctuation`: 678
- `likely_phonetic_confusion`: 433
- `punctuation_or_spacing`: 388
- `missing_hallucinated_or_realigned_speech`: 236
- `semantic_or_grammar_context`: 88
- `greek_question_mark`: 6

## Audio Coverage

- Unique audio URLs: 270
- Unique meeting names: 234
- Duration min/median/mean/max: 0.010s / 2.262s / 3.395s / 72.200s

Top audio files:

- 99 rows: `https://data.opencouncil.gr/audio/59nocub1zsi_v1.mp3`
- 73 rows: `https://data.opencouncil.gr/audio/n5atnqw9bpk_v1.mp3`
- 72 rows: `https://data.opencouncil.gr/audio/oeszmqy2fs_v1.mp3`
- 63 rows: `https://data.opencouncil.gr/audio/5zu7sv48lmq_v1.mp3`
- 59 rows: `https://data.opencouncil.gr/audio/0dsgkisqz9e6_v1.mp3`
- 57 rows: `https://data.opencouncil.gr/audio/u8jskxcf0zs_v1.mp3`
- 53 rows: `https://data.opencouncil.gr/audio/69sp0o4cjvk_v1.mp3`
- 53 rows: `https://data.opencouncil.gr/audio/52i7uk8da4r_v1.mp3`
- 53 rows: `https://data.opencouncil.gr/audio/iaz2mi7b6c7_v1.mp3`
- 52 rows: `https://data.opencouncil.gr/audio/zb4ozpbxzy_v1.mp3`

Top meeting labels:

- 138 rows: Δημοτικό Συμβούλιο 11/02/26
- 113 rows: Δημοτικό Συμβούλιο 10/12/25
- 112 rows: Συνεδρίαση 09/02/26
- 100 rows: Δημοτικό Συμβούλιο
- 76 rows: Δημοτικό Συμβούλιο 19/11/25
- 74 rows: Δημοτικό Συμβούλιο 22/12/25
- 74 rows: Δημοτικό Συμβούλιο 28/01/26
- 69 rows: Δημοτικό Συμβούλιο 11/12/25
- 67 rows: Δημοτικό Συμβούλιο 19/02/26
- 63 rows: Δημοτικό Συμβούλιο 15/12/25

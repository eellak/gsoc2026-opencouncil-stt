# Error-division — Design A pipeline matrix

_Generated 2026-06-24 01:45._

Stages: **A** baseline ASR · **B** baseline+LLM · **C** fine-tuned ASR · **D** fine-tuned+LLM. Metric: micro-averaged WER (summed edits / summed reference words); CI = 95% meeting-clustered bootstrap.

- scored clips: **114** across **28** meetings
- fine-tune seeds (C/D): [0]
- dropped (unparseable in some stage): ['cmm55sc2e07xapgwxzwpvl5fn', 'cmhvrno4d0470w5rkfn8jt8u0', 'cmhks0e4b02is13npgnyev9ix', 'cmlii445r06231pbrlrq1vb5c', 'cmmdaixva0b2711y02y09ezlm']
- LLM parse failures: ['cmhks0e4b02is13npgnyev9ix', 'cmhvrno4d0470w5rkfn8jt8u0', 'cmlii445r06231pbrlrq1vb5c', 'cmm55sc2e07xapgwxzwpvl5fn', 'cmmdaixva0b2711y02y09ezlm']

## Aggregate (all categorized corrections)

| stage | WER_norm | 95% CI | WER_raw | CER_norm |
|---|---|---|---|---|
| A | 0.7151 | [0.5926, 0.9156] | 0.7842 | 0.3550 |
| B | 0.5777 | [0.4490, 0.7929] | 0.6640 | 0.3220 |
| C | 0.6125 | [0.5626, 0.6626] | 0.6807 | 0.2850 |
| D | 0.4775 | [0.4201, 0.5372] | 0.5701 | 0.2523 |

## Paired deltas (WER_norm, meeting-clustered)

Negative = the second stage is better. C-A = fine-tune effect (no LLM); B-A = LLM effect on baseline; D-C = LLM effect after fine-tune; D-B = fine-tune effect after LLM.

| delta | meaning | point | 95% CI |
|---|---|---|---|
| C-A | fine-tune, no LLM | -0.1026 | [-0.2941, -0.0051] |
| B-A | LLM on baseline | -0.1373 | [-0.1640, -0.1064] |
| D-C | LLM after fine-tune | -0.1350 | [-0.1643, -0.1077] |
| D-B | fine-tune after LLM | -0.1002 | [-0.3039, 0.0055] |

## Per error category (WER_norm)

`directional` = n_clips < 30 or n_meetings < 5 — read as a hunch, not a result.

| category | n_clips | n_meetings | A | B | C | D | flag |
|---|---|---|---|---|---|---|---|
| substitution_phonetic | 59 | 25 | 0.7840 | 0.6405 | 0.6420 | 0.5151 | ok |
| punctuation_capitalization | 27 | 16 | 0.5556 | 0.3875 | 0.5356 | 0.3932 | directional |
| homophone | 14 | 12 | 0.8741 | 0.7333 | 0.6593 | 0.5037 | directional |
| noun_case | 12 | 8 | 0.6207 | 0.4089 | 0.6108 | 0.4335 | directional |
| word_boundary | 11 | 7 | 0.7207 | 0.5045 | 0.7928 | 0.5495 | directional |
| place_name | 10 | 7 | 0.5878 | 0.3893 | 0.5954 | 0.3664 | directional |
| verb_inflection | 10 | 8 | 0.6090 | 0.5338 | 0.6241 | 0.5564 | directional |
| insertion | 7 | 4 | 0.5890 | 0.5342 | 0.5479 | 0.4521 | directional |
| article_pronoun | 6 | 6 | 0.5729 | 0.4375 | 0.5208 | 0.3958 | directional |
| number_date | 5 | 5 | 0.6316 | 0.4386 | 0.7719 | 0.4737 | directional |
| deletion | 4 | 3 | 0.9643 | 0.8929 | 1.0357 | 1.0000 | directional |
| person_name | 4 | 3 | 0.8333 | 0.8333 | 0.7500 | 0.7500 | directional |
| semantic_rewrite | 4 | 3 | 0.6842 | 0.6053 | 0.6842 | 0.6842 | directional |
| accent_tonos | 1 | 1 | 0.3636 | 0.2727 | 0.3636 | 0.2727 | directional |
| disfluency_cleanup | 1 | 1 | 0.9231 | 0.8846 | 0.8846 | 0.8462 | directional |
| morphology | 1 | 1 | 0.2727 | 0.1818 | 0.1818 | 0.0909 | directional |

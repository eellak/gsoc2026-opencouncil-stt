# Verify sample TIER-1 (original 7,364) — audio faithfulness

- transcribed: 999 (of 1000 sample)
- **faithful (local-align cer ≤0.20): 84.0%**  (≤0.15: 72.9%, ≤0.25: 89.9%)
- genuinely suspect (>0.45): 2.2%  — of which non-empty (real bad-label candidates): 17

## faithful (≤0.20) by category

- acronym_abbreviation: 88% (n=58)
- homophone: 99% (n=67)
- insertion_deletion: 86% (n=88)
- morph_grammar: 86% (n=69)
- named_entity: 80% (n=352)
- number_date: 85% (n=55)
- other_lexical: 81% (n=280)
- word_boundary: 100% (n=30)

Note: full-clip cer_soniox is inflated by loose utterance spans; local alignment (fuzz.partial_ratio) is the right measure. See conversation for examples.

# Step 7 — final fine-tune edit list

- shortlist: 15,000
- judged so far: 15,000
- shortlist ∩ judged: 15,000
- verdicts: {'keep': 7364, 'reject': 5967, 'unsure': 1669}
- eligible (keep): 7,364
  of which acoustic=true: 7,350

**Final selected: 7,364 edits -> `data/next-batch/selected_edits.jsonl`**
- raw correction-span audio: 5.4 h (concatenation to 15-30s speaker-turn segments expands this toward the ~30h target)
- distinct meetings: 177, cities: 11, distinct sig3: 7,173

## Category mix (final)

- named_entity: 2,597 (35.3%)
- other_lexical: 2,072 (28.1%)
- insertion_deletion: 650 (8.8%)
- morph_grammar: 500 (6.8%)
- homophone: 492 (6.7%)
- acronym_abbreviation: 439 (6.0%)
- number_date: 398 (5.4%)
- word_boundary: 216 (2.9%)

- acoustic=true: 7,350 (99.8%)

## Duration buckets

- <1.5s: 1,828
- 1.5-3s: 3,501
- 3-5s: 1,348
- 5-10s: 600
- 10-15s: 63
- 15-25s: 24
- 25-30s: 0

## ebclass

- task_then_user: 5,206
- user_only: 2,149
- task_plus_user: 9

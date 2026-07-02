# Step 7 — final fine-tune edit list

- shortlist: 25,000
- judged so far: 25,000
- shortlist ∩ judged: 24,966 (dropped 34 audio-confirmed bad)
- verdicts: {'keep': 13377, 'reject': 8953, 'unsure': 2636}
- eligible (keep): 13,377
  of which acoustic=true: 13,350

**Final selected: 13,377 edits -> `data/next-batch/selected_edits.jsonl`**
- raw correction-span audio: 10.0 h (concatenation to 15-30s speaker-turn segments expands this toward the ~30h target)
- distinct meetings: 177, cities: 11, distinct sig3: 12,773

## Category mix (final)

- named_entity: 4,907 (36.7%)
- other_lexical: 4,294 (32.1%)
- insertion_deletion: 949 (7.1%)
- morph_grammar: 840 (6.3%)
- homophone: 733 (5.5%)
- acronym_abbreviation: 709 (5.3%)
- number_date: 676 (5.1%)
- word_boundary: 269 (2.0%)

- acoustic=true: 13,350 (99.8%)

## Duration buckets

- <1.5s: 4,001
- 1.5-3s: 5,501
- 3-5s: 2,420
- 5-10s: 1,268
- 10-15s: 143
- 15-25s: 44
- 25-30s: 0

## ebclass

- task_then_user: 9,842
- user_only: 3,517
- task_plus_user: 18

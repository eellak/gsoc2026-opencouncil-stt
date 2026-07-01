# Step 5 — free interestingness ranking + diverse select

- exclude already-curated/built: 93,584 -> 92,735  (dropped 849)
- hard prefilter (dur/word-diff/chain/cer_proxy bounds): 92,735 -> 61,730  (dropped 31,005)

**Survivors after prefilter: 61,730** -> greedy-diverse shortlist: 15,000

## Shortlist category mix

- named_entity: 5,219 (34.8%)
- other_lexical: 4,064 (27.1%)
- insertion_deletion: 2,371 (15.8%)
- number_date: 1,236 (8.2%)
- acronym_abbreviation: 682 (4.5%)
- morph_grammar: 660 (4.4%)
- homophone: 533 (3.6%)
- word_boundary: 235 (1.6%)

## Shortlist duration buckets

- <1.5s: 3,453
- 1.5-3s: 6,875
- 3-5s: 2,887
- 5-10s: 1,501
- 10-15s: 215
- 15-25s: 68
- 25-30s: 1

## Coverage
- distinct cities: 11
- distinct meetings: 178
- distinct sig3 (specific edits): 14,370

- base_score range: 0.026..1.485

## ebclass

- task_then_user: 9,193
- user_only: 5,795
- task_plus_user: 12

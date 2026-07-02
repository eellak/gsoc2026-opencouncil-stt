# Step 5 — free interestingness ranking + diverse select

- exclude already-curated/built: 93,584 -> 92,735  (dropped 849)
- hard prefilter (dur/word-diff/chain/cer_proxy bounds): 92,735 -> 61,730  (dropped 31,005)

**Survivors after prefilter: 61,730** -> greedy-diverse shortlist: 25,000

## Shortlist category mix

- named_entity: 8,761 (35.0%)
- other_lexical: 7,750 (31.0%)
- insertion_deletion: 3,278 (13.1%)
- number_date: 1,978 (7.9%)
- morph_grammar: 1,085 (4.3%)
- acronym_abbreviation: 1,052 (4.2%)
- homophone: 799 (3.2%)
- word_boundary: 297 (1.2%)

## Shortlist duration buckets

- <1.5s: 7,163
- 1.5-3s: 10,030
- 3-5s: 4,657
- 5-10s: 2,629
- 10-15s: 401
- 15-25s: 118
- 25-30s: 2

## Coverage
- distinct cities: 11
- distinct meetings: 178
- distinct sig3 (specific edits): 23,198

- base_score range: 0.012..1.485

## ebclass

- task_then_user: 16,179
- user_only: 8,797
- task_plus_user: 24

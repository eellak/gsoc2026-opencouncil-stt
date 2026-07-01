# Step 1 — candidate pool

Start: 287,605 chains (one per utterance_id)

- (a) keep has_user==True (drop LLM-only chains): 287,605 -> 156,863  (dropped 130,742)
- (a2) drop broken edit chains (links_ok==False): 156,863 -> 156,771  (dropped 92)
- (b1) drop normalisation-only edits (greek_normalize-equal): 156,771 -> 121,143  (dropped 35,628)
    normalisation-only breakdown (dropped): accent_tonos=1,269, final_sigma=9, no_change=3,697, punctuation_capitalization=30,653
- (b2) drop empty normalised gold_final: 121,143 -> 115,859  (dropped 5,284)
- (c1) drop unreviewed-meeting denylist (13 meetings): 115,859 -> 115,614  (dropped 245)
- (c2) drop held-out eval meetings (57 meetings): 115,614 -> 93,584  (dropped 22,030)

**Final candidate pool: 93,584 utterances** -> `data/next-batch/candidates.parquet`

- rows missing audio span after join: 0
- rows with invalid/zero/NaN duration: 123

## Category distribution (kept)

- other_lexical: 32,974
- named_entity: 32,771
- insertion_deletion: 8,556
- morph_grammar: 6,197
- number_date: 5,786
- homophone: 3,616
- acronym_abbreviation: 2,862
- word_boundary: 822

## ebclass distribution (kept)

- task_then_user: 53,348
- user_only: 40,168
- task_plus_user: 68

## chain length (n_edits) distribution

- 1 edit(s): 33,662
- 2 edit(s): 47,972
- 3 edit(s): 9,344
- 4 edit(s): 1,859
- 5 edit(s): 473
- 6 edit(s): 161
- 7 edit(s): 58
- 8 edit(s): 27
- 9 edit(s): 6
- 10 edit(s): 5
- 11 edit(s): 6
- 12 edit(s): 3
- 14 edit(s): 1
- 16 edit(s): 2
- 17 edit(s): 3
- 23 edit(s): 1
- 27 edit(s): 1

## duration buckets (valid only, seconds)

- <1.5s: 35,050
- 1.5-3s: 23,790
- 3-5s: 15,828
- 5-10s: 14,092
- 10-15s: 3,487
- 15-20s: 1,099
- 20-30s: 111
- >30s: 4

## char_diff (raw) describe
```
count    93584.000000
mean        12.077214
std         27.200794
min          1.000000
25%          4.000000
50%          7.000000
75%         12.000000
max       1529.000000
```

## norm_word_diff describe
```
count    93584.000000
mean         0.442856
std          0.472475
min          0.013699
25%          0.166667
50%          0.333333
75%          0.571429
max         37.000000
```

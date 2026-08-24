# Can the per-column vote see a near-miss? (2026-08-24)

The three-way composition vote (`msa.vote_column`) settles a column by **exact** token
identity. Where all three systems disagree it has no majority to find, and falls back
to the pivot or to a frozen system priority. The proposal screened here: let two
candidates that are the *same word off by a character or two* count as a majority.

Zero GPU, zero API. Substrate, scorer and baseline are the ones already in the repo.

## Substrate and what "the same substrate" means here

247 two-minute windows, 144 meetings, 10 cities, 74,917 reference tokens — the
`2026-08-10-corrected-adapter-label-prefix-fix-vs-ju` benchmark run with the 6 sealed
`eval-freeze-2026-08` windows removed, i.e. `fusion_lab.load_substrate()`, trio
`scribe-v2-clean` + `soniox` + `oc-runpod-fixed-2026-08-10`, exact 3-way MSA cached at
`align_65b1c4d64618a429`. The 391-window post-June set of
`exp_composition_postjune.py` was **not** used: its `report.json` is not in
`~/.cache/oc-public/`, and downloading it is an API call this screen was told not to
make. The 247-window substrate is the one the existing column census, arm C and arm H
were measured on, so the comparison is like-for-like.

Baseline is **W**, the hierarchical per-column vote, scored by
`eval/controlled_eval/scoring.py` (NFD, strip marks, lowercase, `\w+`; final sigma is
NOT folded). Both arms are scored against the same references with
`fusion_lab.evaluate`.

## 1. The disagreement positions, measured before any rule was written

`eval/controlled_eval/exp_near_miss_vote.py census` →
`results_near_miss_census.json`. 80,659 columns.

| class | columns | share |
|---|---|---|
| `agree` [x,x,x] | 62,919 | 78.0% |
| `exact_2_of_3` [x,x,y] | 6,645 | 8.2% |
| `two_present_same` [x,x,ε] | 4,569 | 5.7% |
| `singleton` [x,ε,ε] | 4,460 | 5.5% |
| **`unresolved_three` [x,y,z]** | **1,104** | **1.37%** |
| `unresolved_two` [x,y,ε] | 962 | 1.19% |

The "decides cleanly on 91%" figure checks out: the first three classes are **91.91%**
of columns. The three-way-disagreement class this idea targets is 1,104 columns.

**Minimum** pairwise character distance inside `unresolved_three` (this is the property
the idea needs; the existing census only recorded the **maximum**, which is why arm C
was scoped to 136 columns and this is scoped to 667):

| min pairwise distance | 1 | 2 | 3 | 4 | 5 | 6+ |
|---|---|---|---|---|---|---|
| columns | 415 | 252 | 167 | 119 | 64 | 87 |

So **37.6% of three-way disagreements have a pair at distance 1, and 60.4% at distance
≤ 2.** That is more near-miss mass than the 44% the brief assumed. A strict Greek
homophone pair (`greek_phonetics.phon`) exists in 144 of the 1,104 (13.0%).

### The number that decides it

Reference attribution is alignment-conditional: each column's reference token is the
one the column oracle's DP path matched against it (`oracle_align`). 609 of the 667
`d ≤ 2` columns get a reference token attributed.

| | d = 1 (386 attributed) | d ≤ 2 (609) | folded (141) |
|---|---|---|---|
| near pair **contains** the reference word | 242 (62.7%) | 358 (58.8%) | 95 (67.4%) |
| priority-chosen member **is** the reference word | 135 (35.0%) | 206 (33.8%) | 64 (45.4%) |
| **W already emits the reference word** | 127 (32.9%) | 186 (30.5%) | 51 (36.2%) |

**This is the whole result.** A near-miss majority exists often, and it contains the
right word about 60% of the time — but the vote cannot know *which* member. Picking by
frozen priority is right 33.8% of the time against W's 30.5% on the identical columns:
a net gain of **20 columns out of 74,917 reference tokens ≈ 0.00027 WER**, from 357
tokens rewritten. The folded arm is the better discriminator (45.4% vs 36.2%) but sits
on 141 columns: net **13 columns ≈ 0.00017 WER**.

Even the oracle that always picks the right member of the pair when it is there gains
358 − 186 = 172 columns, **≈ 0.0023 WER** — the entire ceiling of this idea.

## 2. The frozen rule

Written into `exp_near_miss_vote.py` and into this file **before any arm WER was
computed**. Both thresholds (d = 1 and d = 2) are reported; neither is selected on its
result.

**Arm N\<d\> — near-miss majority.** For each MSA column, in this order:

1. the column must be `unresolved_three`: all three systems present, all three tokens
   distinct;
2. the column must not be flagged `split_merge` by `column_classes.split_merge_columns`;
3. take the closest pair of candidates by character Levenshtein (`scoring.edist`); ties
   broken by system-index order (0,1) → (0,2) → (1,2); it must be within **d**
   characters;
4. the third candidate must be **strictly further** from both members of that pair than
   they are from each other — a real cluster of two, not three mutually near strings;
5. emit that pair's member from the earliest system in the frozen priority order
   `(scribe-v2-clean, soniox, ours)`.

Otherwise W's token stands. The arm never emits epsilon and never consumes a column W
dropped, so **the token count and the deletion rate cannot move by construction**.
Arms scored: **N1** (d = 1) and **N2** (d = 2).

**Arm F — folded-identity majority.** Identical scope and representative rule, except
the pair is defined by equality of the STRICT Greek phonemic key (`phon`: ω/ο→o,
η/ι/υ/ει/οι/υι→i, αι→e, ου→u, final sigma, doubled consonants) instead of character
distance, and the third candidate must **not** share that key (a three-way homophone
tie belongs to arm H of `exp_char_homophone.py`). This is a separate hypothesis, not a
variant of N: "the two systems wrote the same Greek word" is not "the two strings are
close".

**Arm N2+F** — F first, N2 on any column F did not claim.

`unresolved_two` ([x, y, ε]) is **excluded from every arm**: two present tokens have no
majority to find, near or not, and choosing between them is candidate selection wearing
a vote's clothes. Its numbers are in the census for description only.

Success criteria, also frozen, are `fusion_lab`'s existing gates: WER improves, the
paired meeting-clustered 95% CI excludes zero, the deletion rate does not rise, the
insertion rate does not rise, and no leave-one-out sign flip over window / meeting /
city.

## 3. Scored result

`exp_near_miss_vote.py score` → `results_near_miss_vote.json`. Out-of-fold WER against
W on all 247 windows, 74,917 reference tokens, same scorer, same alignment cache.

| arm | WER | sub | del | ins | ΔWER vs W | 95% CI | gates |
|---|---|---|---|---|---|---|---|
| **W (baseline)** | 0.100458 | 0.042714 | 0.020316 | 0.037428 | — | — | — |
| N1 | 0.100365 | 0.042674 | 0.020289 | 0.037401 | −0.000093 | [−0.000325, +0.000141] | FAIL |
| N2 | 0.100351 | 0.042661 | 0.020289 | 0.037401 | −0.000107 | [−0.000410, +0.000193] | FAIL |
| F (folded) | 0.100325 | 0.042581 | 0.020316 | 0.037428 | −0.000133 | [−0.000318, +0.000052] | FAIL |
| N2+F | 0.100271 | 0.042581 | 0.020289 | 0.037401 | −0.000187 | [−0.000522, +0.000145] | FAIL |

CIs are the paired bootstrap, 10,000 resamples, **clustered on (cityId, meetingId)** —
193 clusters (`results_near_miss_vote_citymeeting.json`). Note that `fusion_lab`'s own
CI clusters on `meetingId` alone, which merges same-named meeting days across cities
into 144 clusters; both are reported and they agree to the fourth decimal.

Raw counts, N2+F against W: S 3200 → 3190, D 1522 → 1520, I 2804 → 2802. **272 tokens
rewritten, 14 net errors removed.** Every arm leaves the output token count at exactly
76,199, identical to W — the arms only substitute, so no arm can and no arm does buy
WER by deleting. Deletion and insertion rates move by −0.000027 at most, and that
movement is the reference alignment re-routing around a changed substitution, not
dropped speech.

**No arm passes.** Every CI crosses zero. Head-to-head per window for the best arm
(N2+F) is 45 windows better, 35 worse, 167 tied — a coin flip with a thumb on it.

## 4. Domination and leave-one-out

No leave-one-out sign flip for any arm over window, meeting or city — the sign is
stable, the size is not.

| arm | full Δ | Δ with the most favourable meeting removed | share from one meeting |
|---|---|---|---|
| N1 | −0.000093 | −0.000054 (`may19_2025`) | 42% |
| N2 | −0.000107 | −0.000067 (`may19_2025`) | 37% |
| F | −0.000133 | −0.000107 (`may19_2025`) | 20% |
| N2+F | −0.000187 | −0.000147 (`may19_2025`) | 21% |

One Chania meeting supplies 20–42% of an effect that is already inside the noise. Per
city, N2+F is better in 7 cities, worse in 2, unchanged in 1; dropping Xylokastro takes
the delta to −0.000130, another 30%.

## 5. Is 0.013629 the right ceiling?

**No.** That number is
`results_fusion_headroom.json → all.subsets["gpt-4o-transcribe+greek-whisper-v3-turbo+scribe-v2-clean"].headroom`,
and it measures: on the **2026-06-10** benchmark run (250 windows, a different
substrate from this one), the gap between the best single system in that particular
trio (`scribe-v2-clean`, 0.13189) and a **whole-window oracle selector** over that
trio (0.13024). It is wrong for this idea on four counts:

1. wrong substrate — different run, different window set, sealed-window filter not applied;
2. wrong systems — that trio contains `greek-whisper-v3-turbo` at 0.484 WER and neither
   `soniox` nor our adapter;
3. wrong granularity — it bounds a method that picks a whole hypothesis per window; a
   per-column vote is explicitly *not* bounded by whole-window selection, which is the
   founding argument of `exp-2026-08-16-composition-over-selection`;
4. it is not even the largest number in that file — `headroom_all` over all seven
   providers is 0.0303, and `gladia-prod+soniox` alone is 0.0257.

The right ceiling for a per-column rule on this substrate is the alignment-conditional
column oracle: W 0.100458 → 0.047479, i.e. **0.0530 of column-level headroom**. The
near-miss idea's own ceiling — always picking the correct member of the near pair
whenever it is present — is **≈0.0023**, about 4% of that, and the frozen rules capture
0.18–0.35% of the column-oracle gap (`oracle_recovery_vs_W`).

## 6. Verdict

**The idea does not work, and the census says why before the scoring does.** Near
misses are common — 60% of three-way disagreements have a pair within two characters —
and the pair contains the right word about 59% of the time. But an exact-match vote
already gets 30.5% of those columns right, and no reference-free rule for choosing
*which* member of the near pair to emit does much better than that: frozen priority
gets 33.8%. The near-miss signal identifies where the answer is; it carries almost no
information about which candidate it is.

The folded-identity arm is the more interesting half and should be recorded as such: it
is a genuinely better discriminator (45.4% correct vs W's 36.2% on the same columns,
the largest margin in the census) and it is the only arm whose deletion and insertion
rates are bit-identical to W's. It fails only on mass: 144 columns in 80,659.

### Caveats, all of them

- **Substrate.** 247 windows of the 2026-08-10 run, adapter row
  `oc-runpod-fixed-2026-08-10`, not the 391-window post-June held-out set and not the
  clean-pack adapter. The post-June `report.json` is not cached locally and fetching it
  is an API call. A different adapter row changes the disagreement geometry; nothing
  here transfers to the post-June set without rerunning the census there.
- **Reference attribution is alignment-conditional.** "Is the reference word" is decided
  by the column oracle's DP path over one fixed MSA lattice. A different valid alignment
  attributes some columns differently. 966 of 1,104 three-way columns get a reference
  token attributed at all; the other 138 are aligned as insertions and are excluded from
  every ratio in section 1.
- **The representative rule is a design choice, not a fitted one.** Frozen system
  priority `(scribe, soniox, ours)` was taken from `msa.compose` unchanged. A different
  tie-break — longest string, lexicon membership, per-system reliability — could beat
  33.8%, and the 58.8% pair-contains-reference figure is the ceiling any such rule is
  playing for. That ceiling is 0.0023 WER, so it is not worth building.
- **This does not price a confidence-weighted rule.** All four arms are text-only. If
  per-word confidence from the same decode pass ever exists for all three systems
  (`exp-2026-08-21-fusion-production` is where that lives), the member-selection problem
  this screen failed at is exactly what confidence would attack.
- **The effect, such as it is, is meeting-concentrated.** One Chania meeting carries a
  fifth to two fifths of it.
- **Overlap with existing arms is partial, not total.** Arm C of
  `exp_char_homophone.py` scopes on *maximum* pairwise distance ≤ 2 (136 columns); these
  arms scope on *minimum* (667 at d ≤ 2). Neither set contains the other, and arm C's
  own result was −0.00008 with a CI crossing zero — the same shape of answer, reached
  from the other side.
- **Nothing here says composition is finished.** 0.0530 of column-level headroom remains
  on this substrate. It is simply not in the near-miss columns.

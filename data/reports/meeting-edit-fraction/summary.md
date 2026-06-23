# Per-meeting human-edit fraction — distribution & trust cutoff

_Computed from 327 cached public meeting JSONs (`ui/.cache/meetings/`). frac_user = user-edited utterances / total utterances._

## Headline

- meetings: **327**  ·  humanReview=true: **212**  ·  false: **115**
- micro human-edit fraction (all utts): **25.1%** (143,283/571,124)
- per-meeting frac_user: median **27.3%**, p10 16.2%, p90 42.0%

## Histogram (frac_user, 5-pt bins)

| bin | meetings |
|---|---|
| 0-5% |  13 █████████████ |
| 5-10% |   3 ███ |
| 10-15% |  11 ███████████ |
| 15-20% |  31 ███████████████████████████████ |
| 20-25% |  69 █████████████████████████████████████████████████████████████████████ |
| 25-30% |  76 ████████████████████████████████████████████████████████████████████████████ |
| 30-35% |  47 ███████████████████████████████████████████████ |
| 35-40% |  29 █████████████████████████████ |
| 40-45% |  23 ███████████████████████ |
| 45-50% |  11 ███████████ |
| 50-55% |   6 ██████ |
| 55-60% |   3 ███ |
| 60-65% |   2 ██ |
| 65-70% |   1 █ |
| 70-75% |   1 █ |
| 95-100% |   1 █ |

## humanReview flag vs fraction (is the flag reliable?)

- **humanReview=TRUE** (212): frac_user min 9.6%, median 28.2%, max 70.5%
- **humanReview=FALSE** (115): frac_user min 0.0%, median 26.2%, max 99.2%
- humanReview=FALSE yet frac_user ≥15% (flag likely wrong, looks reviewed): **95**
- humanReview=TRUE yet frac_user <5% (suspicious / barely touched): **0**

## Candidate cutoffs (drop meetings below the fraction)

| cutoff | meetings kept | dropped | kept that are humanReview=false |
|---|---|---|---|
| ≥3% | 314 | 13 | 102 |
| ≥5% | 314 | 13 | 102 |
| ≥8% | 314 | 13 | 102 |
| ≥10% | 311 | 16 | 100 |
| ≥15% | 300 | 27 | 95 |

## Lowest-fraction meetings (cut candidates)

| city | meeting | humanReview | n_utt | n_user | frac_user |
|---|---|---|---|---|---|
| meganisi | mar23_2026 | False | 1642 | 0 | 0.0% |
| nea-smyrni | jan13_2025 | False | 4264 | 0 | 0.0% |
| peristeri | dec23_2024 | False | 1448 | 0 | 0.0% |
| plastiras | apr21_2026 | False | 901 | 0 | 0.0% |
| vari-voula-vouliagmeni | jan20_2025 | False | 2336 | 0 | 0.0% |
| vari-voula-vouliagmeni | feb24_2025 | False | 5259 | 2 | 0.0% |
| karditsa | apr20_2026 | False | 2014 | 1 | 0.0% |
| argos | jan21_2025 | False | 2319 | 2 | 0.1% |
| argithea | mar26_2026 | False | 1410 | 2 | 0.1% |
| thessaloniki | apr1_2026 | False | 9282 | 37 | 0.4% |
| kalamata | feb5_2025 | False | 2352 | 14 | 0.6% |
| dorida | jun6_2025 | False | 2422 | 50 | 2.1% |
| rhodes | jul17_2025 | False | 5439 | 136 | 2.5% |
| chania | jan27_2025_2 | False | 6298 | 551 | 8.7% |
| chania | feb5_2025 | False | 1594 | 145 | 9.1% |

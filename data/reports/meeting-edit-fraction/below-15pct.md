# Meetings below the 15% human-edit threshold

Από [`distribution.tsv`](distribution.tsv) (327 public meetings, `eval/meeting_edit_fraction.py`). `frac_user` = utterances που διόρθωσε άνθρωπος / σύνολο utterances. Εδώ τα **<15%** — δηλ. όσα ΔΕΝ θα περνούσαν το προτεινόμενο backbone-trust cutoff.

- Σύνολο κάτω από 15%: **27** / 327 meetings (humanReview=true: 7, false: 20).
- Τα **<5%** (τα πρώτα) είναι τα «νεκρά»/ουσιαστικά μη-reviewed → cut.
- Τα **5–15%** είναι γκρίζα ζώνη: μερικώς reviewed· καλά για correction pairs, επισφαλή ως no-edit ground truth.

| # | city | meeting | humanReview | n_utt | n_user | n_task | frac_user |
|---|---|---|---|---|---|---|---|
| 1 | meganisi | mar23_2026 | False | 1642 | 0 | 387 | 0.0% |
| 2 | nea-smyrni | jan13_2025 | False | 4264 | 0 | 1491 | 0.0% |
| 3 | peristeri | dec23_2024 | False | 1448 | 0 | 587 | 0.0% |
| 4 | plastiras | apr21_2026 | False | 901 | 0 | 220 | 0.0% |
| 5 | vari-voula-vouliagmeni | jan20_2025 | False | 2336 | 0 | 706 | 0.0% |
| 6 | vari-voula-vouliagmeni | feb24_2025 | False | 5259 | 2 | 1541 | 0.0% |
| 7 | karditsa | apr20_2026 | False | 2014 | 1 | 583 | 0.0% |
| 8 | argos | jan21_2025 | False | 2319 | 2 | 813 | 0.1% |
| 9 | argithea | mar26_2026 | False | 1410 | 2 | 441 | 0.1% |
| 10 | thessaloniki | apr1_2026 | False | 9282 | 37 | 1997 | 0.4% |
| 11 | kalamata | feb5_2025 | False | 2352 | 14 | 963 | 0.6% |
| 12 | dorida | jun6_2025 | False | 2422 | 50 | 810 | 2.1% |
| 13 | rhodes | jul17_2025 | False | 5439 | 136 | 1962 | 2.5% |
| 14 | chania | jan27_2025_2 | False | 6298 | 551 | 1865 | 8.7% |
| 15 | chania | feb5_2025 | False | 1594 | 145 | 564 | 9.1% |
| 16 | chalandri | feb25_2_2026 | True | 228 | 22 | 35 | 9.6% |
| 17 | chania | apr2_2025 | False | 3803 | 442 | 976 | 11.6% |
| 18 | chania | jan15_2025 | True | 2921 | 341 | 894 | 11.7% |
| 19 | chania | feb3_2025 | False | 654 | 80 | 196 | 12.2% |
| 20 | sparta | feb24_2_2026 | True | 16 | 2 | 5 | 12.5% |
| 21 | chania | feb26_2025 | False | 3497 | 464 | 1025 | 13.3% |
| 22 | sparta | feb3_2026 | True | 118 | 16 | 35 | 13.6% |
| 23 | chania | feb2_2026 | True | 319 | 44 | 65 | 13.8% |
| 24 | argos | oct10_2025 | False | 441 | 61 | 130 | 13.8% |
| 25 | chalandri | oct15_2_2025 | False | 2439 | 354 | 517 | 14.5% |
| 26 | zografou | jan8_2026 | True | 107 | 16 | 30 | 15.0% |
| 27 | chalandri | feb11_2026 | True | 1269 | 190 | 213 | 15.0% |

> Σημείωση: ο `humanReview=true` με χαμηλό frac δεν εμφανίζεται εδώ ως false positive — το ελάχιστο true είναι 9.6%, άρα τα true εδώ είναι οριακά (9.6–15%), όχι λάθος-flag.
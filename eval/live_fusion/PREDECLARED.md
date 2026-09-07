# Pre-declaration, live fusion pilot 2026-09-07

Written 03:10 EEST, **before any aggregate number from this run existed**. The
only live number seen before this file was written is the single preflight
window `win_orestiada_apr28_2026_1872686`, whose totals were read at 03:02 while
proving the pipeline runs at all. That window is disclosed here rather than
excluded, because dropping it after seeing it would be the worse choice.

## The sample

- Pool: the 260 cached benchmark windows, restricted to those whose
  `(city, meeting)` appears in none of `clean-pack-arm-2026-08`,
  `spliced-pack-arm-2026-08`, `spliced-pack-arm-jitter-2026-08`. That is **206
  windows in 162 meetings across 10 cities**.
- Seed `live-fusion-2026-09-07/v1`, shuffle, then take in order under a per-city
  cap.
- **Stage A: cap 4 per city → 40 windows.** Running since 03:06.
- **Stage B: cap 8 per city → 80 windows**, the same seed and the same shuffle,
  so stage A is a prefix of stage B. Declared here, now, so that extending the
  sample cannot be a reaction to stage A's result. Stage B runs if the clock
  allows; if it does not finish, the report quotes stage A and says so.

Nothing is re-drawn. A window that fails stays in the ledger as a failure.

## The arms

`scribe`, `soniox`, `ours`, `W` (majority vote), `rules_on` (R1+R2 + the
vote/negation guard) — all five decoded live from the same audio bytes and
scored with `fusion.normalize.wtoks` / `sdi`, corpus-aggregated
(`ΣS+ΣD+ΣI` over `ΣN`).

## The reading rule

- Primary contrast: **`rules_on` against `W`** — does the rule layer add
  anything over a plain vote.
- Practical contrast: **`rules_on` against `scribe`** — the incumbent.
- The headline may be stated unqualified **only if** it survives removing any
  single window (`reverses_on_removing_one_window` false). Otherwise it is
  reported as concentrated in one window.
- The metric is **agreement-with-OpenCouncil**. It records product
  compatibility. It says nothing about which system is closer to the audio, and
  no sentence in the report may claim that it does.
- The live numbers and the frozen 0.11205 are **separately labelled descriptive
  results on different cohorts**. Their difference is not a replication, an
  improvement, or a regression.

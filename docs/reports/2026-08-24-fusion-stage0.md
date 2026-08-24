# Stage 0 of the two-system fusion screen: 3 pairs x 2 bases x 3 arms, all 18 cells

Ran 2026-08-24 against [`docs/specs/2026-08-21-fusion-production.md`](../specs/2026-08-21-fusion-production.md) §5.1,
for `exp-2026-08-21-fusion-production` (OPEN). Zero GPU, zero API. Code:
[`eval/controlled_eval/exp_fusion_pairs.py`](../../eval/controlled_eval/exp_fusion_pairs.py),
numbers: `eval/controlled_eval/results_fusion_pairs.json`.

Substrate: the cached 247 windows of `2026-08-10-corrected-adapter-label-prefix-fix-vs-ju`,
74,917 reference tokens, the 6 sealed `eval-freeze-2026-08` holdout windows removed by the
same explicit filter `exp_fusion_deletions.py` carries. Scorer is
`eval/controlled_eval/scoring.py` + `exp_fusion_deletions.sdi`, so every number here is
comparable with `exp-2026-08-16-composition-over-selection` and **not** with the benchmark
app's leaderboard.

**Clusters are `(cityId, meetingId)`, which gives 193 clusters, not the 144 that
`meetingId` alone gives on this run.** Meeting ids repeat across cities here; earlier
fusion experiments on this substrate clustered on `meetingId` alone and therefore merged
windows from different cities into one cluster. Every interval below is a paired
meeting-clustered bootstrap, 10,000 resamples, on the 193-cluster key.

## What could not be run, and why

This is the honest half of the result and it comes first.

* **There are no per-word timestamps and no per-word confidences for this run's
  hypothesis text.** The benchmark report carries text only. The project's cached Soniox
  word tokens are `stt-rt-v4` and the cached adapter confidences are a separate decode
  pass; joining either to this text is forbidden by `exp-2026-08-18-conf-substrate`
  (0 of 133 windows reproduce) and by the caveat on `artifact-soniox-rt-tokens-2026-08-16`.
* Therefore the **anchored alignment of §2.2 is not exercised**. Alignment is plain
  pairwise text DP — exactly the drift-prone thing §2.2 exists to replace.
* **R2's 0.30 s clause degrades** to a token-count clause, fitted out of fold over {1,2,3}.
* **R3 is fitted without calibrated confidence**, on span shape only (producer, length).
* **R4 (diarization) is not implemented.** It never restores alone, so its absence removes
  a flag, not a restore path.
* **Arm P2 as specified is NOT RUNNABLE.** A calibrated cross-vendor confidence comparison
  needs confidences that do not exist here. What ran under that slot is **P2\***, a
  context-calibrated surrogate that picks the identity path from span shape, fitted
  leave-one-city-out. It is labelled P2\* everywhere and it is not evidence about P2.
* **R1 is implemented as a suppressor, not a restorer.** The spec's own Codex correction
  says an echo means the alignment failed locally and the fix is realign-and-collapse,
  never a licence to emit a second copy. "Restore if R1" would emit that duplicate, so
  restore here is `not R1 and R2 and R3`.
* KEEP−DROP costs are **local**: read off each span's own hypothesis-to-reference
  alignment, not off a re-scored merged window.

## Single systems, this scorer

| system | WER | del | ins | sub |
|---|---:|---:|---:|---:|
| `scr` — ElevenLabs Scribe v2 (clean) | 0.13220 | 0.03799 | 0.03740 | 0.05681 |
| `snx` — Soniox (this run's provider row) | 0.14089 | **0.01416** | 0.07498 | 0.05175 |
| `adp` — `oc-runpod-fixed-2026-08-10` | 0.13858 | 0.04589 | **0.02271** | 0.06998 |

## Diagnostics, per pair

| pair | union coverage of correct ref tokens | both-wrong | both-correct | excl. correct (x) | excl. correct (y) | pairwise alignment-conditional oracle |
|---|---:|---:|---:|---:|---:|---:|
| `adp+snx` | **0.9595** | **0.0405** | 0.8588 | 0.0254 | 0.0753 | **0.0630** |
| `scr+snx` | 0.9580 | 0.0420 | 0.8813 | 0.0239 | 0.0528 | 0.0774 |
| `scr+adp` | 0.9537 | 0.0463 | 0.8356 | 0.0696 | 0.0485 | 0.0654 |

For scale, the trio's alignment-conditional column oracle on this same substrate is
0.0475 and its voted arm W is 0.10046.

**Conditional accuracy on identity disagreements** — the quantity §3.3 says defines the
frozen priority, and it does not agree with the marginal substitution rates:

| pair | disagreement columns | x correct | y correct | x-only | y-only | neither | x share of decisive |
|---|---:|---:|---:|---:|---:|---:|---:|
| `adp+snx` | 6130 | 0.2799 | 0.5080 | 1648 | 3046 | 1368 | 0.351 |
| `scr+snx` | 4796 | 0.3413 | 0.4016 | 1549 | 1838 | 1321 | 0.457 |
| `scr+adp` | 6708 | 0.4940 | 0.3399 | 3058 | 2024 | 1370 | 0.602 |

Soniox wins identity disagreements against our adapter almost 2:1 (3046 vs 1648), even
though its *marginal* substitution rate (0.0518) is barely better than Scribe's and its
overall WER is the worst of the three. Against Scribe it wins narrowly. This is the
measurement §3.3 asked for and it flatly contradicts reasoning from marginal rates.
Between 22% and 28% of disagreement columns have **neither** side right — the structural
ceiling, visible directly.

**KEEP−DROP per singleton direction** (positive = DROP is cheaper):

| pair | direction | spans | tokens | mean KEEP−DROP | total | share where KEEP is better |
|---|---|---:|---:|---:|---:|---:|
| `adp+snx` | adp-only | 228 | 254 | +0.013 | +3 | 0.443 |
| `adp+snx` | **snx-only** | 1622 | 4579 | **+0.721** | **+1169** | 0.385 |
| `scr+snx` | scr-only | 217 | 253 | +0.327 | +71 | 0.327 |
| `scr+snx` | snx-only | 1922 | 4156 | +0.298 | +572 | 0.449 |
| `scr+adp` | **scr-only** | 743 | 1945 | **−0.427** | **−317** | 0.603 |
| `scr+adp` | **adp-only** | 862 | 1517 | **−0.920** | **−793** | 0.727 |

The asymmetry §3.2 predicted is real and larger than expected. Soniox-only spans are
overwhelmingly worth dropping against agreement-WER, and the cost of keeping them grows
sharply with span length (mean KEEP−DROP by length 1/2/3/4+ against our adapter:
+0.13, +0.41, +0.87, **+2.49**). Between Scribe and our adapter the sign reverses: both
directions of singleton are worth **keeping**, and again more so for long spans.

Stability: union coverage by city runs 0.918–0.977 (worst `argos`, best `chalandri`) for
every pair — the ordering of pairs is the same in every city. By meeting, p05 ≈ 0.86–0.90,
p50 ≈ 0.970, p95 ≈ 0.99.

## The 18 cells

Every threshold, priority and bucket fitted **leave-one-city-out**; every number below is
computed only from out-of-fold predictions. `Δ vs best single` compares against the
lower-WER member of that pair.

| pair | base | arm | WER | del | ins | sub | Δ vs best single | 95% CI |
|---|---|---|---:|---:|---:|---:|---:|---|
| adp+snx | adp | P0 | 0.12315 | 0.03441 | 0.03749 | 0.05124 | −0.01543 | [−0.02094, −0.01003] |
| adp+snx | adp | P1 | **0.12238** | 0.03441 | 0.03749 | 0.05047 | **−0.01620** | [−0.02175, −0.01083] |
| adp+snx | adp | P2\* | 0.12238 | 0.03441 | 0.03749 | 0.05047 | −0.01620 | [−0.02175, −0.01083] |
| adp+snx | snx | P0 | 0.14089 | 0.01416 | 0.07498 | 0.05175 | +0.00231 | [−0.00706, +0.01208] |
| adp+snx | snx | P1 | 0.14036 | 0.01362 | 0.07564 | 0.05110 | +0.00178 | [−0.00760, +0.01153] |
| adp+snx | snx | P2\* | 0.14036 | 0.01362 | 0.07564 | 0.05110 | +0.00178 | [−0.00760, +0.01153] |
| scr+snx | scr | P0 | 0.13127 | 0.03499 | 0.04370 | 0.05258 | −0.00093 | [−0.00474, +0.00277] |
| scr+snx | scr | P1 | 0.13060 | 0.03200 | 0.04699 | 0.05162 | −0.00160 | [−0.00561, +0.00229] |
| scr+snx | scr | P2\* | 0.12962 | 0.03202 | 0.04701 | 0.05059 | −0.00258 | [−0.00622, +0.00089] |
| scr+snx | snx | P0 | 0.14089 | 0.01416 | 0.07498 | 0.05175 | +0.00869 | [+0.00227, +0.01530] |
| scr+snx | snx | P1 | 0.13991 | 0.01416 | 0.07498 | 0.05078 | +0.00772 | [+0.00131, +0.01434] |
| scr+snx | snx | P2\* | 0.13891 | 0.01410 | 0.07491 | 0.04991 | +0.00671 | [+0.00059, +0.01321] |
| scr+adp | scr | P0 | 0.13220 | 0.03799 | 0.03740 | 0.05681 | 0.00000 | [0, 0] |
| scr+adp | scr | P1 | 0.12396 | 0.02631 | 0.04107 | 0.05658 | −0.00824 | [−0.00992, −0.00663] |
| scr+adp | scr | P2\* | 0.12527 | 0.02403 | 0.04361 | 0.05764 | −0.00693 | [−0.00906, −0.00487] |
| scr+adp | adp | P0 | 0.12505 | 0.03756 | 0.03126 | 0.05622 | −0.00715 | [−0.01097, −0.00366] |
| scr+adp | adp | P1 | **0.12112** | 0.02328 | 0.04126 | 0.05658 | −0.01108 | [−0.01307, −0.00916] |
| scr+adp | adp | P2\* | 0.12242 | 0.02096 | 0.04376 | 0.05770 | −0.00978 | [−0.01226, −0.00743] |

Three structural facts explain most of this table.

1. **When the fitted priority equals the base, P0 degenerates to the base system exactly.**
   That is why `adp+snx|base=snx|P0` reproduces Soniox to the digit and
   `scr+adp|base=scr|P0` reproduces Scribe to the digit. It is a correctness check, not a
   result.
2. **The fitted priority was identical in all ten folds of every cell**: Soniox over our
   adapter, Soniox over Scribe, Scribe over our adapter. No fold disagreed.
3. **The restore gate is where the two pairs part company.** On `adp+snx` with our adapter
   as base the gate fired **0 times out of 1622** candidates — the out-of-fold R3 buckets
   were positive at every length, so no Soniox-only span was ever restored. On `scr+adp`
   it fired 634 of 743 and 635 of 862. That is the KEEP−DROP table acting, not a tuning
   accident.

## What wins, and on what

**`adp+snx`, base = our adapter, arm P1: 0.12238, −0.01620 [−0.02175, −0.01083] against
the better half of its own pair.** It is the largest two-system gain in the table and its
interval is comfortably clear of zero.

Its mechanism, stated plainly, is **not** rich composition. With priority fitted to Soniox
in all folds, the arm is: take our adapter's occupancy, take Soniox's word identities
wherever both systems spoke, drop all 1622 Soniox-only spans (4579 tokens), keep our
adapter's 228 solo spans, then apply the frozen phonetic roster repair (89 changes). It is
an **occupancy filter on Soniox with our adapter as the second opinion**. The whole of P0's
gain is that filter; P1 adds only the phonetic repair (−0.00077, the same order as arm E's
measured −0.00083 on V).

Two independent estimates agree: the local KEEP−DROP table says dropping Soniox-only spans
should buy 1169 edits ≈ −0.0156 WER off Soniox's 0.14089, giving ≈0.1253; the assembled arm
scores 0.12315. The 0.002 gap is the local approximation, as declared.

**Second: `scr+adp`, base = our adapter, arm P1: 0.12112, −0.01108 [−0.01307, −0.00916]
against Scribe.** Lower absolute WER than the winner, but measured against a stronger
comparator, and it needs a credential this environment does not have. Its gain is the
opposite mechanism — restoring singletons rather than dropping them.

## What fails, and it matters

**The component guard of the spec's §5.2 gate 3 fails for every cell that wins on WER.**

| cell | Δ del | Δ ins |
|---|---:|---:|
| `adp+snx / adp / P1` vs adp | −0.01148 [−0.01400, −0.00924] | **+0.01479 [+0.01059, +0.01937]** |
| `scr+adp / adp / P1` vs scr | −0.01471 [−0.01662, −0.01292] | **+0.00386 [+0.00300, +0.00478]** |
| `scr+adp / scr / P1` vs scr | −0.01168 [−0.01320, −0.01025] | **+0.00367 [+0.00288, +0.00454]** |

Every winning cell buys WER by trading our adapter's deletions for insertions, and every
insertion interval is above the +0.002 ceiling the spec sets. On the spec's own frozen
rule, **no cell here would ship**. Stage 0 was never allowed to ship anything, but the
gate that would stop it is already visible at Stage 0.

## Domination

No leave-one-out sign flips in any winning cell, on windows, on `(city, meeting)` or on
cities. The largest single-group share of the effect:

* `adp+snx / adp / P1`: worst window 4.9% (`win_athens_may20_2_2026_1191014`), worst
  meeting 5.9% (`sparta/apr15_2026`), **worst city 27.0% (`athens`)**.
* `scr+adp / adp / P1`: worst window 2.3%, worst meeting 2.4%, worst city 6.3% (`chania`).

Neither is single-window dominated. The `adp+snx` winner does lean on one city more than
the `scr+adp` winner does; at 27% it is well below the 67% precedent this project has, but
it is the one number in the winning cell that deserves a second look on new audio.

The `adp+snx|base=snx` cells flip sign when `athens` is dropped, but they are null cells
whose intervals include zero anyway.

## What this does and does not settle

* Two-system per-column composition has now been measured. It beats the better half of its
  pair, out of fold, with a clustered interval clear of zero. It does **not** reach the
  trio's W (0.10046) and nothing here suggests it would.
* The pair the spec chose on cost grounds, `artifact-ct2-fixed` + Soniox, is also the pair
  with the best union coverage (0.9595), the lowest both-wrong rate (0.0405), the lowest
  pairwise oracle (0.0630) and the largest realised gain. The cost-constrained choice and
  the diagnostic choice happen to agree. That was an open question in the spec and it is
  now answered on measurement.
* The identity-priority question is answered and it is counter-intuitive: **Soniox wins
  identity disagreements against both other systems**, despite having the worst overall WER.
  Marginal substitution rates would have given the opposite answer.
* Nothing here is fidelity-to-audio. Every number is agreement-with-OpenCouncil. The
  winning arm's mechanism is *discarding 4579 tokens Soniox emitted*, and
  `exp-2026-08-17-insertion-fidelity` measured that 40.8% of Soniox insertions are real
  speech the published reference omits. On the metric that decides, this arm may be
  actively worse. The two metrics are not reconciled here and must not be merged.
* Stage 0 is the seventh pass over these 247 windows. Cross-fitting prices parameter
  overfitting only; the arm list and the class definitions were written by someone who has
  already seen this substrate six times.

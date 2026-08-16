# Preregistration: search / confirmation partition for the autoresearch loop

2026-08-16 · protocol `autoresearch-2026-08-16b` ·
implemented in [`eval/controlled_eval/autoresearch.py`](../../eval/controlled_eval/autoresearch.py) ·
reviewed by Codex jobs `362e2a7b` (design) and `59c9564` (implementation)

This freezes the protocol under which an automated loop may propose and score ideas on
top of W, and under which any survivor may be promoted. It is written before the first
idea was registered. The numbers it governs are in
[`docs/reports/2026-08-16-autoresearch-harness.md`](../reports/2026-08-16-autoresearch-harness.md).

## Why a preregistration is required here at all

`fusion_lab.evaluate` makes one idea cost about ten seconds of CPU. The promotion gate
it carries is a paired clustered bootstrap CI excluding zero. Those two facts together
are a machine for manufacturing a false result: at a nominal 5%, a loop that evaluates
40 independent null ideas produces "at least one significant idea" **87% of the time**
(measured, not asserted — see the simulation in the report). Every expensive mistake
on this project has been a measurement mistake. This document is the defence.

## 1. The substrate

247 two-minute windows, 144 meetings, 10 cities, 74,917 reference tokens, from run
`2026-08-10-corrected-adapter-label-prefix-fix-vs-ju`. The baseline is **W**, the
hierarchical per-column vote of `exp-2026-08-16-composition-over-selection`
(WER 0.10046).

**The 16 locked evaluation windows stay sealed and are not available to this harness.**
6 of the 7 sealed temporal-holdout windows of `eval-freeze-2026-08` fall inside the 253
common windows of this run and are removed by `load_substrate`'s explicit filter before
anything is computed, which is why the substrate is 247 and not 253.

## 2. The partition

The ten cities are cut **once**, by a rule that reads only reference-token counts:
sort cities by reference tokens descending, give each to whichever side is currently
furthest below a 65 / 35 token target. No WER, no idea and no outcome enters the rule.
It is implemented in `plan_partition` and the realized split is pinned in code, with
`assert_partition` failing if the rule and the pin ever disagree.

| partition | cities | windows | meetings | ref tokens |
|---|---|---:|---:|---:|
| SEARCH | athens, chalandri, chania, orestiada, vrilissia, zografou | 153 | 103 | 47,252 |
| CONFIRM | argos, samothraki, sparta, xylokastro | 94 | 61 | 27,665 |

Each side contains a city that contributed nothing to fine-tuning — orestiada in
search, argos in confirm — so neither partition is purely seen or purely unseen.

The loop iterates **freely** on SEARCH. CONFIRM is not read, aggregated or glanced at
during search, and the API enforces that rather than trusting the caller: `run_search`
refuses a substrate that contains a confirmation city, and `run_confirmation` refuses
anything that is not **exactly** the confirmation partition — not the same substrate
passed twice, and not a hand-picked subset of it.

## 3. Search-stage protocol

- Fitting is leave-one-search-city-out over the 6 search cities; only held-out outputs
  are scored.
- Reported per idea: out-of-fold WER, deletion / insertion / substitution rates, the
  percentile clustered CI of `fusion_lab` (unchanged, for comparability with the four
  earlier passes), the wild-cluster p-value of §5, the meetings touched, the domination
  share, and the leave-one-out sign flips.
- **Nothing at the search stage is a result.** A search number may not be quoted as
  evidence that an idea works.

### The search screen (which ideas may be promoted)

All of these, jointly:

| gate | threshold |
|---|---|
| WER improves vs W | ΔWER < 0 |
| effect floor | ΔWER ≤ −0.0010 (0.10 WER points) |
| deletion-rate gate | del_rate ≤ W's |
| insertion-rate gate | ins_rate ≤ W's |
| support | ≥ 8 meetings with a **non-zero** error delta |
| no single-item domination | max_b \|d_b\| / Σ_b \|d_b\| < 0.50 |
| percentile CI excludes zero | as today |
| minimum-effect test | one-sided p ≤ 0.05 against H₀: ΔWER ≥ −0.0010 (§5) |

The screen is a **screen**, not the ship decision: it is uncorrected for multiplicity by
design, because its job is to shrink the confirmation family, not to decide anything.
The point-estimate effect floor is kept beside the minimum-effect test only because it
is cheap; the test is what carries the weight, on search and on confirmation alike.

The effect and support floors are not decoration. `docs/reports/2026-08-16-harness-coverage-mde.md`
measured that a **monotone** arm — one that never worsens anything — touching k of B
meeting clusters yields a percentile CI excluding zero whenever k ≥ 4, at *any* effect
size, because the CI is measuring how many blocks carry the sign, not how large the
effect is. An automated loop will find such arms. Significance is therefore never
sufficient here.

## 4. Confirmation

- Parameters are fitted **once on the entire search partition** and frozen. Applying
  them to the confirmation cities is a locked-box run with no refit. (Search
  cross-validation estimates a smaller-training-set procedure; confirmation estimates
  the shipped one. That difference is expected and is not bias.)
- **A cycle freezes EXACTLY ONE confirmation batch, before any confirmation number
  exists.** Codex `362e2a7b`: "at most five" is not a protection if the first result
  decides who gets the second slot — the later hypothesis would then depend on
  confirmation data. Codex `59c9564` sharpened it: five *sequential* singleton batches,
  each Holm-corrected inside itself, give a familywise error of 1 − 0.95⁵ = **22.6%**.
  So a second batch is refused outright and requires a new `PROTOCOL_VERSION`, which
  re-keys every idea and starts a new, separately reported cycle.
- **Budget: at most 5 ideas in that batch**, enforced by the journal inside a single
  lock so two concurrent processes cannot both see an unspent budget. Every attempt
  counts, including failures and inconvenient results.
- The same screen of §3 applies again on confirmation, with the support floor lowered
  to 6 meetings to reflect the smaller partition.

## 5. Inference

The p-value fed to the multiplicity correction is a **null-imposed, studentized wild
cluster bootstrap-t** over meetings, not the percentile bootstrap's tail mass. For
meeting contributions d_b (error delta vs W) and n_b (reference tokens):

- Δ = Σd_b / Σn_b, cluster-robust SE = sqrt(Σ_b (d_b − Δn_b)²) / Σn_b, T = (Δ − Δ₀)/SE.
- Bootstrap DGP imposes the null: d\*_b = Δ₀n_b + (d_b − Δ₀n_b)·w_b with Rademacher w_b.
- p = (1 + #{|T\*| ≥ |T|}) / (R + 1), R = 9,999. The one-sided form
  p = (1 + #{T\* ≤ T}) / (R + 1) is what the ship gate uses.
- **Weights are shared across every idea in a batch**, so the tests keep their joint
  dependence.

A zero cluster-robust SE is **not** automatically the A/A case: it also arises when
every meeting carries the same non-zero rate. The degenerate branch therefore returns
p = 1 only when the estimate equals the null.

Codex rejected the percentile-tail p-value that was in the first draft: resampling the
observed data centres the distribution on the observed effect, so its tail mass is not a
p-value under H₀, it double-counts atoms at zero, and it can exceed 1. Δ₀ = 0 is reported as the descriptive p and drives the
search-family BH diagnostic; the **ship gate uses the one-sided Δ₀ = −0.0010 test**,
because an idea should ship only when the data reject "smaller than useful", not when
they reject "exactly zero".

The percentile CI of `scoring.cluster_bootstrap` is **unchanged** and still reported, so
every number stays comparable with the four earlier passes over this substrate.

## 6. Multiplicity

- **Holm at familywise α = 0.05 over the frozen confirmation batch is the ship gate**,
  applied to the one-sided minimum-effect p-values of §5. Holm bounds the probability
  that *any* shipped idea is false, which is the decision being made. BH bounds the
  expected false *fraction* among discoveries, which is not — and BH's guarantee needs
  independence or PRDS, which sharing bootstrap weights does **not** establish for
  arbitrary ideas scored on the same meetings. BH here is descriptive.
- **Benjamini-Hochberg is reported alongside**, over the same confirmation batch and,
  separately, over the whole search family. The search-family BH is a **fishing
  diagnostic only** and never promotes anything.
- The confirmation family is every idea in the frozen batch, and only those. Search-only
  ideas do not enter it: their confirmation p-values were never computed and the
  selection used disjoint data. Charging confirmation for them would discard the point
  of the split.
- **Every report states the denominator**: ideas registered, ideas searched, duplicates
  refused, ideas screened through, confirmations spent, confirmations remaining.

## 7. Dedup

An idea is fingerprinted by what it *does*: the canonical edit events between its output
and W's, anchored to W's token positions (so one insertion does not renumber everything
after it), each keyed-hashed with HMAC-SHA-256. Enforcement uses the **exact** Jaccard
over those hash sets; MinHash exists only to shortlist. A new idea whose Jaccard against
any **already-evaluated** idea is ≥ 0.90 is refused as a cosmetic variant, the verdict is written **inside the
search-result record itself** (so a crash cannot leave a known duplicate with no
refusal beside it), and the idea can never reach confirmation. The empty firing set (an
arm that changes nothing) is exempt — it is nobody's variant.

The guard **fails closed**: if an earlier idea's hash set is not on disk, or was keyed
with a different secret, the threshold cannot be evaluated and the run aborts. Purging
`$SC` must not be a way to resubmit a refused variant.

Hash sets live under `$SC`, never in git: they are derived from council speech. The
journal keeps only the set size, a digest, and the dedup key's identity.

## 8. Journal

`research/autoresearch/journal.jsonl`, append-only, one JSON object per line, each
carrying a sequence number and the hash of the previous record, plus a
`journal.head` checkpoint holding the record count and the hash of the last record.
Editing, deleting or reordering any record makes the journal fail to load; the
checkpoint closes the hole the chain leaves, which is that **any prefix of a valid
journal is itself valid**, so deleting the tail would otherwise be invisible.
Read-decide-append runs inside one file lock, so a confirmation cannot be spent twice by
two concurrent processes. Counts only — no Greek text, ever.

This is tamper **evidence**, not a security boundary: the chain is unkeyed and both
files are writable, so anyone willing to recompute them can rewrite the history.

## 9. What this protocol does NOT buy

Stated up front so no later reader has to infer it:

- **It does not undo adaptive proposing.** The ideas are proposed by agents that have
  already seen five passes over these same 247 windows, confirmation cities included.
  Sample splitting controls the multiplicity of *testing*, not the adaptivity of
  *hypothesis generation*.
- **Four cities are four cities.** The estimand a confirmation licenses is performance
  over meetings drawn from argos, samothraki, sparta and xylokastro. It is not a claim
  about the next Greek municipality; four city clusters cannot support one. Meeting-level
  clustering is the inference unit and per-city deltas are reported raw beside it.
- **It is agreement-with-OpenCouncil, not fidelity-to-audio.** A shipped idea would be
  compatible with our published text, which decides nothing about what was said.
- **The dedup guard is behavioural, not semantic.** Two genuinely different ideas that
  happen to fire on the same columns are indistinguishable to it, and an idea can evade
  it by adding irrelevant firings.
- **The idea fingerprint has a stated hole.** It pins the factory, its whole MRO,
  `fusion_lab` and the partition, but not free helper *functions* in a caller's module;
  hashing the caller's module would re-key every idea in a library whenever one idea is
  added to it. Ideas must keep their decision logic inside the class. The harness's own
  bytes are represented by `PROTOCOL_VERSION`, which a behavioural change must bump.
- **A confirmation failure is ambiguous**: it can be overfitting, city heterogeneity, or
  both. A confirmation success establishes the effect for this fixed four-city mixture.

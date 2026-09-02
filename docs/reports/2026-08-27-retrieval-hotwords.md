# A retrieved prompt is gentler than a static one and still fails the gate

2026-08-27. CPU only, no GPU, no paid API, no sealed-holdout access.
Record: `exp-2026-08-27-retrieval-hotwords`. Spec:
[`docs/specs/2026-08-26-training-round-2.md`](../specs/2026-08-26-training-round-2.md) §3.4.

`exp-2026-07-25-hotwords` is CLOSED: a static 160-token roster prompt on every window
lifts correctly matched name mentions 126 → 161 of 246 and costs +133 deletions
(del_rate +0.0112 [+0.0020, +0.0207], CI excludes zero). This run asked a different
question, adapted from Fun-ASR (arXiv 2509.12508) §5.5: if the prompt is **retrieved
per window** from the model's own unbiased first pass instead of supplied statically,
does the name benefit survive with the deletion penalty gone?

**Verdict: `STOP`.** Not `PARTIAL`. The closed record's do-not-ship decision stands.

## What ran

`artifact-ct2-fixed`, model hash `8a1a3b257d0c1bdb`, verified before decoding. Pass 1
is read byte for byte from `~/.cache/oc-public/decode-ablation/eval-A.json` — the same
file the contrast is scored against — so R − A is paired by construction and the 20
windows that retrieve nothing enter as exact zero-delta observations.

Retrieval parameters were frozen in the spec before any decode: STRICT phoneme fold,
`max_distance` 1, `min_skeleton_len` 4, `k_max` 3, `token_budget` 24,
`include_exact` False, lexicon = the closed arm's roster-present `person_surname`
universe carrying its recorded inflections.

**19 of 39 windows** retrieved anything and ran a second pass: 8 windows with one
surname, 4 with two, 7 with three. Mean prompt **11.1 tokens**, maximum 21 — against
the closed arm's 160 on all 39.

## The eight frozen gates

| # | gate | measured | required | |
|---|---|---:|---|---|
| 1 | ΔWER point | **+0.00118** | < 0 | ✗ |
| 2 | ΔWER 95 % upper | **+0.00852** | < +0.0010 | ✗ |
| 3 | Δdeletion-rate 95 % upper | **+0.01175** | ≤ 0 | ✗ |
| 4 | Δinsertion rate | −0.00118 | < +0.0005 | ✓ |
| 5 | additional correct name mentions | **+15** of 246 | ≥ 18 | ✗ |
| 6 | name precision / out-of-roster | +0.0324 · oor 0 | neither worsens | ✓ |
| 7 | leave-one-window-out sign | — | no reversal | n/a |
| 8 | name-gain domination | — | no unit > 25 % | n/a |

Gates 7 and 8 are reported `n/a`, never passed: both are shares of the name gain, and
gate 5 failed, so the denominator is not established. (For the record they would also
have failed: one window reverses the ΔWER sign, and `win_argos_dec23_2025_581922`
supplies 10 of the 15 extra mentions, a 66.7 % share.)

Levels, meeting-clustered bootstrap over 32 clusters, 4,000 replicates:

| | WER | del | ins | sub |
|---|---:|---:|---:|---:|
| control (A) | 0.15893 | 0.06003 | 0.02015 | 0.07875 |
| arm R | 0.16010 | 0.06364 | 0.01897 | 0.07749 |

All three contrast CIs include zero. **WER could not have decided this arm** and was
never asked to: the 80 %-power MDE here is 1.37 points and the whole effect is 0.12.
WER enters only as the non-inferiority gate, which it fails on the upper endpoint.

## The prediction I wrote down before decoding was wrong

The ledger recorded, before the run: gate 6 is the most likely to fail, because 27 % of
retrievals fire on ordinary Greek words (`μέλη` → *Δελής*, `πάρει` → *Σαρρής*,
`ίδια` → *Ηλίας*) and should produce spurious name emissions.

It did not fail. Name precision **rose** 93.3 % → 96.6 % and out-of-roster name
insertions stayed at 1, unchanged. The decoder ignored the irrelevant candidates rather
than emitting them. Prompting a name the audio does not contain is, in this stack,
close to free on the precision side.

## What the decoy control found, which is the more useful result

The preregistered negative control re-ran the same 19 windows with the same retrieved
names **plus one extra surname drawn deterministically from a different city's roster**.
Mean prompt 18.4 tokens instead of 11.1. Same model, same seeds, same audio, same real
candidates present.

| | R | R + one decoy |
|---|---:|---:|
| additional correct mentions | +15 | **+10** |
| Δdeletion rate | +0.00361 [−0.00329, +0.01175] | **+0.00814 [+0.00040, +0.01693]** |
| ΔWER | +0.00118 | +0.00730 |
| out-of-roster name insertions | 1 (unchanged) | **1 (unchanged)** |

One irrelevant surname costs a third of the name gain and more than doubles the
deletion penalty, and it is the only contrast here whose deletion CI **excludes zero**.
The decoys themselves were never emitted.

**The damage is suppression, not hallucination.** A prompt entry the decoder does not
use is not inert: it still occupies the `sot_prev` slot, and what it displaces is
output the model would otherwise have produced. That is a controlled contrast — one
added token group, everything else held — which the R-versus-B comparison is not.

It does **not** isolate prompt length. The decoy changes length and relevance at once,
so "longer prompt" and "more irrelevant content" remain confounded. The four deletion
deltas do line up with prompt size (0 tokens → 0; 11.1 → +0.0036; 18.4 → +0.0081;
160 → +0.0112), but the 160-token point differs in treatment frequency and candidate
inventory too, so this is a consistency check and not a dose-response measurement.
Identifying length alone needs a matched padded-prompt arm, which was not built.

## What this closes and what it does not

Closed: **retrieval does not rescue decode-time hotword biasing on this stack.** Making
the prompt conditional, short and relevant shrinks the deletion penalty to a third of
the static arm's and keeps the WER effect inside noise, and it still buys only +15 of
the 18 mentions the gate required, from one dominating window. Two of the three
deciding statistics moved the right way; neither moved far enough to ship.

Not closed, and not touched here: post-hoc roster repair
(`exp-2026-08-11-name-repair`) remains the name mechanism that works — −30
substitutions, −0.25 WER, CI excluding zero, zero collateral, no decode cost.

**These endpoints are agreement-with-OpenCouncil, not fidelity-to-audio.** They can
stop an engineering screen, which is what they did. They could not have authorised
shipping without adjudicating the changed spans against the recordings.

Raw results: `~/.cache/oc-public/retrieval-hotwords-2026-08/`
(`results-eval-R-retrieval.json`, `results-eval-R-retrieval-decoy.json`, and the two
hypothesis files). Harness `scripts/serving_stack/retrieval_hotwords_arm.py`; policy
`eval/controlled_eval/retrieval_hotwords.py`; 26 tests across
`eval/tests/test_retrieval_hotwords_arm.py` and
`eval/tests/test_retrieval_hotwords_gates.py`.

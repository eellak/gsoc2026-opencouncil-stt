# Scribe's confidence knows where Scribe is struggling, not where the fusion helps

2026-09-09. `exp-2026-09-09-scribe-confidence-routing`, leg B. Pre-registered before any
outcome was computed; the gate it declared **fails**.

The idea was to stop paying for three recognisers on 100% of the audio. Scribe returns a
per-word logprob we throw away. If low confidence marked the places where the other two
systems are worth calling, we could call them only there.

## The correction that made this measurable

The first version of this experiment asked "does Scribe's confidence predict Scribe's
errors". A Codex Astra review (`a5b91215`) said that is the wrong target: routing depends on
predicting **where the fusion improves on Scribe**, and those are different rankings. A hard
window is often hard for all three systems, and the fusion sometimes makes a window worse.

So the quantity is, per window, in edit operations against the reference:

    g_i = E_scribe,i − E_fusion,i        (negative where the fusion hurts)

and for a routed set R, exactly:

    WER(R) = (Σ E_scribe,i − Σ_{i∈R} g_i) / Σ N_i

This is exact, not a simulation, because both transcripts already exist for all 391 windows
of the 2026-09-08 benchmark run. Routing here **selects between two frozen outputs**; it
re-runs nothing. The x-axis is routed **audio seconds**, because that is what the vendors bill.

Both endpoints reproduce under this pipeline before anything else was computed: Scribe
0.14027 and fusion 0.09913, matching the published run. Across 391 windows the fusion helps
in **339**, hurts in **43**, ties in 9, for 4,554 edit operations of total available gain.

## What the confidence buys

Share of the total 4,554-edit gain captured, by routed-audio budget:

| ranking | 10% | 25% | 50% | 75% |
|---|---:|---:|---:|---:|
| frac(p < 0.90) | 12.3% | 34.5% | 63.2% | 82.1% |
| 1 − mean(lowest decile) | 13.3% | **34.5%** | 61.1% | 81.2% |
| 1 − mean(p) | 12.8% | 33.0% | 62.5% | 82.3% |
| count(p < 0.50) per 100 words | 16.1% | 33.4% | 57.5% | 80.7% |
| baseline: random, mean of 1000 draws | 10.0% | 25.2% | 50.0% | 75.0% |
| baseline: random, 95th percentile | 13.0% | 29.8% | 55.3% | 79.7% |
| baseline: longest windows first | 7.7% | 22.6% | 51.0% | 71.5% |
| **oracle, ranked by gain per second** | 33.3% | **61.6%** | 90.8% | 104.8% |

**Pre-declared gate: at a 25% budget, capture ≥50% of the gain and beat random's 95th
percentile. Result: 34.5% against a 29.8% random p95 — beats random, fails the gate.**

The signal is real and it is weak. It beats random by 9.3 points of gain at the primary
budget and beats a confidence-free duration ranking by 11.9, so Scribe's logprob does carry
routing information. But the oracle captures 61.6% at the same budget, so nearly half of what
was there to be had is invisible to it — the failure is the signal's, not the setup's.

The sentence that decides the product question: **to keep 80% of the fusion's gain you must
route 75% of the audio anyway**, which is where the saving was supposed to come from.

## WER and deletions along the curve

| ranking | 25% budget | 50% budget | 75% budget |
|---|---|---|---|
| 1 − mean(lowest decile) | 0.12608, del 0.01368 | 0.11514, del 0.01635 | 0.10685, del 0.01800 |
| oracle | 0.11493, del 0.01184 | 0.10291, del 0.01405 | 0.09715, del 0.01545 |
| anchors | Scribe 0.14027, del 0.00955 | | fusion 0.09913, del 0.01866 |

Deletions rise monotonically with every edit of gain bought, on every ranking. A quarter of
the audio buys about a third of the WER gain and about 45% of the extra deletions. Nothing
here says whether those deletions are fillers or speech — that is still
`exp-2026-08-26-chooser-deletion-safety`, still unrun.

No single window carries the captured gain: the largest supplies 7.4% of it, and 7 of the
routed windows contribute negatively. On the two training-disjoint cities the picture is the
same, not better: argos captures 38.9% of its own gain at the 25% budget, orestiada 28.8%.

## What this does and does not settle

**Settled, for whole-window routing on this benchmark:** Scribe's per-word confidence ranks
windows better than chance and better than length, and far worse than needed. Routing at a
budget that would actually save money keeps only about a third of what the fusion is worth.

**Not settled, and deliberately not attempted here:**

- Whether Scribe's confidence predicts Scribe's *own* errors against a human who listened.
  That is leg A, ~$0.07 of Scribe over the 30 frozen gold cells, and the pre-registration
  says it runs only if leg B passes. It did not, so it was not run. The question survives for
  a different purpose — flagging passages for human review is not routing.
- Region-level splicing (leg C). Its weak point was named before any code was written: Scribe
  and the fusion have different word sequences and timestamps, a Scribe deletion has no word
  to anchor an island around, and a hybrid must be spliced by a reference-independent rule
  and then scored as an actual transcript. Note the ceiling first: even a perfect
  *window-level* router leaves 38% of the gain unreachable at a 25% budget.
- Anything about fidelity to the audio. Every number here is agreement with OpenCouncil's own
  published transcript, which records product compatibility and decides nothing about
  accuracy.

Any operating point read off this curve was chosen after seeing it, on the frozen benchmark.
It would need a confirmation set that is not these 391 windows.

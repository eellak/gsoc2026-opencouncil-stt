# precision-2 over the corpus: what is worth measuring, decided in advance

Frozen 2026-08-03, before the API pass finished. Codex reviewed the draft at high effort
and rejected the framing of three of the five measurements; the rewrites are marked
`[codex]` and the original wording is kept where it was wrong, so the record shows what
changed.

Companion to [the overlap screen](../reports/2026-08-03-overlap-screen.md) and the
[synthetic-overlap preregistration](synthetic-overlap-preregistration.md).
Script: `eval/controlled_eval/precision2_corpus.py`, analysis `precision2_analyze.py`.

## The thing to keep straight

Three questions were being conflated. `[codex]` They are separate and a positive answer
to one does not force a positive answer to another:

1. **Is overlap causally harmful at the tested dose?** — only the synthetic experiment
   answers this.
2. **How common is the deployment-relevant exposure?** — needs human-calibrated
   measurement. Two diarizers agreeing does not answer it.
3. **Is overlap the best observable routing marker?** — a prediction question, answerable
   out-of-sample against competing markers.

## M1. Alternative detector operationalization — NOT an independent estimate

The original framing was "independent re-estimate of overlap prevalence, which validates
community-1's recall". That is wrong and the wrongness matters. `[codex]` precision-2 sees
the same audio, is from the same model family, and the listening audit showed both
diarizers share a blind spot: they count **miked** speakers, while the human counted the
room. Two models with one blind spot agreeing about zeros is weak evidence about recall.

Reported as: the paired ratio `R = prevalence(precision-2) / prevalence(community-1)` and
the absolute difference, meeting-clustered bootstrap, over a **fixed denominator** so the
comparison is not driven by the two models segmenting speech differently.

Rule, three-way, no undefined region: `[codex]`

- **practically concordant** only if the whole 95% CI for `R` lies inside [0.70, 1.30];
- **materially higher** only if the whole CI lies above 2.0;
- **inconclusive** otherwise.

Both estimates enter the ceiling arithmetic as sensitivity analyses **regardless of which
branch fires**. Recomputing only after crossing a threshold is an avoidable
discontinuity. Even the concordant branch is worded "the two detector-defined measures
agree", never "prevalence is settled".

## M2. Paired robustness of the bucket association — not a replication

Same windows, same references, same ASR hypotheses, related diarizers. That is robustness
to measurement, not independent replication, and the report will say so. `[codex]`

"CIs clear of zero for a majority of the seven systems" is discarded. The seven systems
share 232 windows and are heavily dependent, failure to reject is not evidence of no
effect, and a vote over correlated measurements is arbitrary. `[codex]`

Instead, with `C(d,s)` the high-minus-zero WER contrast for diarizer `d` and system `s`:

```
Cbar(d) = mean over the 7 prespecified systems of C(d,s)
Δ       = Cbar(precision-2) − Cbar(community-1)
```

Both recomputed inside **every** bootstrap replicate on resampled meetings, so their
covariance is preserved and Δ gets a real CI. Noninferiority margin **δ = 1.0 WER point**,
frozen here, chosen to match the speech-specific margin in the synthetic gate.

- **supported**: lower bound of `Cbar(precision-2)` above zero *and* lower bound of Δ
  above −δ;
- **contradicted**: upper bound of `Cbar(precision-2)` at or below zero, *or* upper bound
  of Δ below −δ;
- **inconclusive**: anything else.

Because detector-specific tertiles select different windows, Δ mixes a measurement
difference with a membership difference. A continuous analysis on frozen overlap burden
is reported alongside so nothing rests on bucket edges. `[codex]`

The original conclusion — "failure weakens the DiCoW case regardless of the synthetic
result" — is replaced. `[codex]` Failure weakens the robustness and natural-generalization
claims. It does not override a positive causal result from the synthetic experiment; it
raises uncertainty about **how often that mechanism matters naturally**.

## M3. Turn density — reframed from causal to predictive

The original plan bucketed turn density and compared its contrast with overlap's. That
comparison is confounded and cannot identify which variable is the lever. `[codex]` Latent
"busyness" causes both; difficult acoustics cause diarizer fragmentation, apparent turns
*and* ASR errors; missed off-mic speakers depress both measures; overlap itself creates
turn boundaries. Turn density is not measured independently of overlap even with
exclusive diarization.

What it can answer is: **which marker identifies high-error windows, and does either add
anything once you know the other.** Four frozen models, nested leave-one-meeting-out,
leave-one-city-out as a fragility check:

```
U_overlap = L(base + turns) − L(base + turns + overlap)
U_turns   = L(base + overlap) − L(base + overlap + turns)
```

Out-of-sample loss, bootstrapped by meeting. Both materially positive means both carry
information; both negligible while each works alone means they are interchangeable
markers at this sample size; neither predicting out of sample means the original bucket
association was unstable. **None of these identifies causality.**

Turn-boundary conventions frozen before running: minimum turn duration 0.25 s, rapid
label flips under that merged, rate per **speech** minute, turns touching the window
boundary counted with the window censored flag set.

## M4. Event geometry — descriptive, does not move the dose

Durations, events per window, burden per window, event-weighted **and** window-weighted
summaries, boundary-censored events reported separately, and boundary agreement between
the two diarizers.

The preregistered synthetic dose stays uniform(1.5, 3.0) as primary. `[codex]` Empirical
quantiles enter as a prespecified sensitivity grid, not a replacement — precision-2's
durations are shortened by missed low-SIR tails and reshaped by its own smoothing, so
deriving the dose from them would import its blind spot into the causal experiment.

## M5. Exclusive diarization — stored, quarantined

Stored with model id, parameters, timestamps and request metadata for reproducibility.
Any analysis conceived after looking at it is exploratory by construction and is labelled
so, or preregistered against a holdout.

## The two additions Codex rated highest

**A1. Disagreement-stratified human audit.** After the pass, sample from four strata:
community-1 only, precision-2 only, both positive, both zero. Weight back to the corpus.
This is the one route by which the full pass can actually address measurement error —
auditing detected events alone estimates precision, which we already have at 40/40.

**A2. Detector-recall surface.** Inject a known off-mic competitor across a duration × SIR
grid and measure both diarizers' detection recall. This may matter more than ASR efficacy:
**if deployment triggers DiCoW on detected overlap, a treatment that works but never fires
at low SIR is worthless.** The synthetic-overlap builder already produces exactly these
mixtures, so the marginal cost is one diarization pass.

## What none of this licenses

That true prevalence is known; that either diarizer has good recall on off-mic speech;
that overlap rather than busyness is the causal variable; or that agreement between two
models of the same family resolves a shared blind spot.

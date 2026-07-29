# Contextual biasing with roster hotwords — first controlled run (2026-07-25)

Script: `eval/controlled_eval/ab_hotwords.py` · raw: `eval/controlled_eval/results_hotwords.json`

Follow-up to the [fine-tune eval postmortem](../handoff/2026-07-25-finetune-eval-postmortem.md),
idea #2: can inference-time biasing get the name accuracy the fine-tune was supposed to
buy, without retraining?

## Setup

Same 50 actually-corrected held-out utterances (argos+orestiada) as the capstone A/B,
same faster-whisper/CTranslate2 stack (float32, beam=2, `condition_on_previous_text=False`),
same normalization, scored against the human reference. Base/ours hypotheses were reused
from the capstone run, so only the two biased configs were re-transcribed.

Hotwords = the per-meeting speaker roster from `data/pii/rosters_full.json` (311 meetings;
`data/improve_loop/rosters.json` covers only 73 and left 40/50 utterances without a roster).
Terms are ordered full-names-first and truncated to a 180-token budget — median 13 terms per
utterance. The roster comes from the OpenCouncil API, never from the reference, so this is
not an oracle.

## Results (n=50)

| config | WER | vs base |
|---|---|---|
| scribe_before (original ASR text) | 0.1552 | — |
| base whisper | 0.1576 | — |
| ours (LoRA) | 0.1761 | +11.7% worse |
| **base + hotwords** | **0.1552** | −1.5% |
| ours + hotwords | 0.1638 | +3.9% worse |

Per-utterance head-to-head:

- `base+hotwords` vs `base`: better 7 | worse 6 | tie 37
- `ours+hotwords` vs `ours`: better 18 | worse 8 | tie 24
- `base+hotwords` vs `ours`: better 20 | worse 7 | tie 23

## Read

1. **On base, biasing is a wash.** −0.24pp WER with 7 wins against 6 losses is noise at
   n=50. It does not hurt, which is worth knowing (no hallucinated roster names), but there
   is no gain to claim here.
2. **On the fine-tune, biasing recovers about two thirds of its regression**
   (0.176 → 0.164, 18 better / 8 worse). Consistent with the postmortem's story: the
   adapter drifts on name spelling, and the prompt pulls it back. Still worse than plain
   base, so this changes nothing about the "do not migrate" conclusion.
3. **The run did not actually test the name hypothesis.** The roster-recall column is
   `1/1` — across all 50 utterances the references contain exactly one roster term. Widened
   to the full pool: of 98 corrected, clip-eligible held-out utterances, only **5** contain
   a roster name. The corrected-utterance subset cannot measure name accuracy; it is too
   name-poor. Any conclusion about biasing and names from this run would be unsupported.

## Run 2 — the name-focused subset (same day)

Script: `eval/controlled_eval/ab_hotwords_names.py` · raw: `results_hotwords_names.json`

Sampled from the general held-out pool (`data/asr/val_manifest.csv`, 9,875 utterances,
argos+orestiada) restricted to utterances whose reference contains a per-meeting roster
name: **59 utterances, 114 gold names**, 5–25 s, ≥10 words. Roster terms are
document-frequency filtered over the val set (a term in >1% of utterances is not treated
as a name); 114 of 260 held-out roster terms occur in val at all, none exceeded the DF cap.
Same stack, same normalization as above.

| config | WER | name recall |
|---|---|---|
| base | 0.3412 | 27.2% (31/114) |
| base + hotwords | 0.3460 | **36.0% (41/114)** |
| ours (LoRA) | 0.3336 | 33.3% (38/114) |
| **ours + hotwords** | **0.3072** | **37.7% (43/114)** |

Head-to-head: `base+hotwords` vs `base` 15/18/26 · `ours+hotwords` vs `ours` 30/8/21 ·
`base+hotwords` vs `ours` 29/20/10.

Significance (McNemar exact on per-name hits; paired bootstrap, 2000 resamples, on WER):

| comparison | name recall | WER delta (95% CI) |
|---|---|---|
| base+hotwords − base | +13/−3, **p = 0.021** | +0.0049 [−0.012, +0.024] |
| ours+hotwords − ours | +7/−2, p = 0.18 | **−0.0264 [−0.041, −0.014]** |
| ours − base | +12/−5, p = 0.14 | −0.0076 [−0.045, +0.026] |
| ours+hotwords − base | — | **−0.0340 [−0.068, −0.003]** |

## Read

4. **Biasing does what it was supposed to do: names.** On base it lifts name recall
   27.2% → 36.0% (p = 0.021) at no WER cost (CI on the WER delta straddles zero). That is
   the first real, measured gain in this line of work.
5. **Fine-tune + biasing is the best config on this subset** — WER 0.307, −0.034 vs base
   with a CI that excludes zero, and 30 wins to 8 losses over the unbiased fine-tune. The
   adapter alone is not significantly better than base (−0.008, CI straddles zero); the
   two together are.
6. **This does not overturn the postmortem.** On the *corrected* held-out utterances the
   fine-tune is still worse than base (0.176 vs 0.158) and biasing only narrows it. The
   two subsets disagree, which is exactly why the harness (idea #1) has to come before any
   migration decision. n = 59 with uncorrected references is one datapoint, not a verdict.

## Next

- Fold both runs into the single controlled harness; report the general, name-focused and
  corrected subsets side by side so a config cannot look good on one and hide on another.
- Widen the name subset (drop the ≥10-word floor, add more meetings) to get n into the
  low hundreds before treating the +8.8pp as sized rather than merely real.
- Cheap knobs still untried: `initial_prompt` instead of `hotwords`, a larger token budget,
  and per-city glossary terms (`data/glossary/glossary.json`) alongside the roster.
- Serving note: hotwords need the per-meeting roster at inference time. That is available
  in the OpenCouncil pipeline (the meeting endpoint), so this is deployable — but it is a
  change to how the ASR is *called*, not to the model.

# The post-editor needs a gate, and the gate is what makes it work (2026-08-01)

Follow-up to [the lexical-thesis experiments](2026-07-29-lexical-thesis-experiments.md),
which left the LLM post-editor as the project's best measured result (0.155 → 0.119 WER
on 50 corrected utterances) and two unanswered objections: it occasionally writes
commentary instead of a transcript, and it had only ever been tested on utterances that
were already known to be wrong.

Script: `eval/controlled_eval/exp_postedit_gate.py` · raw: `results_postedit_gate.json`
· 348 claude-sonnet calls, same frozen prompt, same scorer, no GPU.

## 1. The gate is not a safety net. It is the result.

98 held-out corrected utterances (a superset of the earlier 50), post-editing the
original Scribe text with the per-meeting roster:

| variant | WER | vs source, cluster CI over 29 meetings |
|---|---|---|
| Scribe (source) | 0.1529 | — |
| post-edit, ungated | 0.1324 | −0.0205 [−0.0446, +0.0135] — **not significant** |
| post-edit, "repair" (keep text before the first blank line) | 0.1318 | — |
| **post-edit, gated** | **0.1144** | **−0.0385 [−0.0487, −0.0279] — significant** |

The gate rejected 4 of 98 outputs and that alone moved WER by 1.8 points. Without it
the improvement does not survive a meeting-level confidence interval; with it the
interval is nowhere near zero. Head-to-head, gated post-editing beats the source on 49
utterances, ties on 41, and loses on 8.

The "repair" strategy the earlier script applied silently — keep everything before the
first blank line — recovers almost nothing (0.1318 vs 0.1324 ungated). It only catches
outputs whose first paragraph is already correct, which is not the failure mode.

Of the 4 rejections: 3 were outputs that had made the utterance worse (29 errors
avoided between them) and 1 threw away a genuine improvement. That is the trade the
thresholds encode, and at 3:1 it is worth taking.

## 2. What it costs on text that is already right

Every number published for this idea so far came from utterances a human had to
correct. In production the editor sees everything, and most of it is already fine.

To measure that without a contaminated reference: feed the editor the **human-corrected
text** and score its output against that same text. Input equals reference, so a
well-behaved editor scores 0.000 and anything above it is damage the editor caused. No
ASR, no audio, no old-system references.

| sample | ungated | gated | utterances made worse |
|---|---|---|---|
| 98 held-out corrected utterances | 0.0255 | **0.0124** | 15 / 98 |
| 150 training-city utterances | 0.0097 | **0.0097** | 24 / 150 |

So the editor damages roughly **one in six already-correct utterances**, costing about
1 WER point overall. The gate halves that on the first sample and does nothing on the
second, where the single rejection was a meta-marker rather than a rewrite.

## 3. The number that decides deployment

Gain on utterances that need correction: **0.0385 per reference word.**
Cost on utterances that do not: **0.0097 to 0.0124 per reference word.**

Break-even is therefore at **20% to 24%**: unless at least a fifth of utterances in a
real meeting actually contain an ASR error worth fixing, running the post-editor over
everything makes the transcript worse, not better.

That fraction turned out to be already measured, in a file built for something else.
`data/reports/meeting-edit-fraction/distribution.tsv` records, for every cached meeting
JSON, how many utterances a human edited out of the total — it exists because the same
quantity was used to set the unreviewed-meetings trust cutoff. Across the 314 meetings
that survive the denylist, that is **530,036 utterances, of which 143,039 (27.0%) were
touched by a human**. Discounting the 8.8% of corrections that experiment A found to be
pure formatting (invisible after WER normalization) leaves **24.6%** as the WER-relevant
share. Reproduce with `eval/controlled_eval/breakeven.py`.

| cost estimate | break-even | margin | meetings below break-even |
|---|---|---|---|
| 0.0097 (training-city sample) | 20.1% | +4.5% | 75 / 314 (24%) |
| 0.0124 (held-out sample) | 24.4% | **+0.2%** | 138 / 314 (44%) |

**A blanket post-editor is a coin flip.** On the pessimistic cost estimate the margin is
two tenths of a percentage point, and 44% of individual meetings sit on the wrong side
of the line. Averaged over the corpus it might come out slightly ahead; on any given
meeting it is as likely to hurt as help. That is not something to deploy.

What this does say is that the **selective** post-editor is the only version worth
building: run it only where an error is likely (low ASR confidence, a roster name in the
window, an out-of-vocabulary token) instead of everywhere. Selection does not need to be
clever to pay — the arithmetic above is dominated by the 75% of utterances that need no
help and can only lose. Halving how often the editor touches a correct utterance moves
break-even from 24% down to roughly 14%, which the real 24.6% clears comfortably.

Caveat on the 27.0%: "a human edited it" is a proxy for "it needed correction" that errs
in both directions. Reviewers miss errors, which pushes the true fraction up; they also
edit for reasons an ASR could never have got right, which pushes it down beyond the
formatting adjustment. It is the best available estimate, not a measurement of the thing
itself.

## Caveats

- Arm 2's input is human-polished text, which is not quite the same object as ASR
  output that happens to be correct. It may be cleaner and more uniform than the real
  thing, so the measured over-editing tax could be optimistic. The honest fix is to
  measure on real ASR output whose correctness a human has confirmed, which the current
  data does not contain.
- The 4 rejections in arm 1 are a small sample of a rare event. The rejection rate
  (4.1%, 3.1%, 0.7% across the three arms) is stable enough to plan around, but the
  3-saved-to-1-lost ratio is not a precise estimate.
- One model, one prompt, one temperature setting. Nothing here says the same gate
  thresholds transfer to a different editor.

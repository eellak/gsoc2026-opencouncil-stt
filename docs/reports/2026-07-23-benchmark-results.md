# Benchmark: fine-tune vs commercial ASR on the OpenCouncil bench (2026-07-23)

## What we ran

We put the fine-tuned `whisper-large-v3` council model through the OpenCouncil
transcription benchmark (`bench.opencouncil.gr`) against the three providers
already scored there: Gladia (prod config), ElevenLabs Scribe, and OpenAI
`gpt-4o-transcribe`. The model was served from the self-hosted CPU endpoint on the
mini-PC (`https://asr.haroldpoi.dev/v1`, faster-whisper, int8) and registered as an
`openai-compatible` provider. We cloned the reference sample so every provider is
scored on identical audio, and only our new provider actually cost anything to run.

The single number to compare is `wer-nofillers` (hesitations like "εεε" stripped
from both sides). Lower is better.

## The contamination caveat, up front

The reference sample has 65 clips. Of those, 51 (about 78%) come from meetings that
were in our training set. That makes the pooled 65-clip score for our model
misleading in our favor: the model has literally seen most of that audio. So the
honest comparison is the held-out slice.

Our two validation cities, **argos** and **orestiada**, were excluded from training
entirely (city hold-out split). Ten of the 65 clips come from those two cities, five
each. That subset is genuinely unseen, so it is the fair test.

## Results on the clean, held-out subset (n=10)

Word-weighted pooling over the argos + orestiada clips. All four providers scored
all 10.

| Provider | wer-nofillers | CER |
|---|---|---|
| ElevenLabs Scribe | 0.147 | 0.102 |
| OpenAI gpt-4o-transcribe | 0.149 | 0.075 |
| **OC fine-tune (ours)** | **0.151** | 0.100 |
| Gladia (prod config) | 0.154 | 0.086 |

On data it has never seen, our self-hosted CPU model lands in the middle of the
pack: 0.151 against a spread of 0.147 to 0.154. It is statistically indistinguishable
from Scribe, gpt-4o, and Gladia. For a LoRA fine-tune served on a CPU mini-PC, that
is a good result. It matches the commercial providers rather than trailing them.

An earlier read off the first five clips suggested we were clearly behind (~0.19).
That was small-sample noise. With the full ten the gap closes to nothing.

## Results on all 65 clips (contaminated, for context only)

| Provider | wer-nofillers | scored |
|---|---|---|
| **OC fine-tune (ours)** | **0.135** | 61/65 |
| Gladia (prod config) | 0.135 | 65/65 |
| OpenAI gpt-4o-transcribe | 0.139 | 65/65 |
| ElevenLabs Scribe | 0.153 | 65/65 |

Here we look tied for best, but this is the number inflated by the 78% of clips the
model trained on. Do not read it as a real ranking. It is here so the gap between
"contaminated" and "held-out" is visible: the held-out subset is the only one that
tells the truth.

## What this means for migration

The project's bar for swapping the production transcriber is roughly a 25% relative
WER improvement over Gladia. On the clean subset we are about 2% better than Gladia,
so we do not clear that bar. The takeaway is not "the model is bad." It is "the model
matches general commercial ASR but does not yet beat it." Matching is not a reason to
migrate on its own, but it means the fine-tune is a real, competitive artifact, and
the headroom to actually beat these providers is in the council-specific errors
(names, place names, acronyms) rather than in general transcription quality.

## Infrastructure notes

Two things fought us on the way to a completed run, both now fixed:

- **HTTP 524 on long clips.** Cloudflare's edge cuts a request at about 100 seconds.
  The CPU endpoint was slower than that on multi-minute clips. Dropping
  `cpu_threads` from 16 to 8 (hyperthreading was hurting, not helping: RTF went from
  0.85 to 0.53), turning off word timestamps on the `/v1` route, and lowering the
  beam brought long clips back under the ceiling.
- **A greedy-decode repetition loop.** With `beam_size=1` the decoder occasionally
  fell into a repeat loop and pinned the CPU for minutes. Setting `beam_size=2`,
  `condition_on_previous_text=False`, and a hard `MAX_INFER_SEC` guard fixed it. A
  110s clip went from timing out to 39s.

Four of the 65 clips still failed for our provider (61/65). These are the longest
clips, where even the faster CPU path runs past Cloudflare's 100s window before it
finishes. That is a limitation of serving on CPU behind a tunnel, not a model
problem, and it is the reason for the next step.

## Next step

The mini-PC endpoint proved the model works and is competitive, but CPU-behind-a-
tunnel is not a good fit for the benchmark's longer clips or for a fair re-run. The
plan is to serve the same model on a GPU (RunPod) with the same request shape, so
long clips finish well inside any timeout, then re-run the reference benchmark
cleanly.

Separately, the sample itself needs fixing: a benchmark that is 78% training
meetings cannot measure progress. The right sample is recent-only, meetings from
after the training cutoff (2026-05-19), drawn from all cities. That is a request to
the OpenCouncil side and is tracked for later.

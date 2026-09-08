# The decoder does not repeat itself

2026-09-07. `exp-2026-09-07-decode-nondeterminism`.

Found while timing the fusion route against the benchmark's per-request limit.
The latency question is in
[the live fusion pilot](2026-09-07-live-fusion-pilot.md); this is the separate
thing that turned up on the way.

**The same audio, decoded twice with the same configuration on the same machine,
does not produce the same words.** It is not a serving-stack quirk: the canonical
CPU int8 measurement stack does it too.

## What was measured

One window, `win_argos_apr24_2026_1015557.wav`, 140.85 s, 4.51 MB. Three
consecutive decodes per stack. No parameter changed between runs.

**CPU int8, `OURS_DECODE` from `eval/live_fusion/providers.py`, four threads,
faster-whisper 1.2.1:**

| run | words | segments | segments that fell back |
|---|---:|---:|---:|
| 1 | 247 | 28 | 4 (14%) |
| 2 | 239 | 31 | 10 (32%) |
| 3 | 240 | 29 | 7 (24%) |

| pair | edit distance | self-WER |
|---|---:|---:|
| run1 vs run2 | 24 | 0.0972 |
| run1 vs run3 | 30 | 0.1215 |
| run2 vs run3 | 23 | 0.0962 |

**GPU float16, the serverless endpoint (`artifact-ct2-cleanpack-cont-s47`,
provenance hash identical on every call):**

| pair | self-WER |
|---|---:|
| cold vs warm | 0.1883 |
| cold vs warm2 | 0.1172 |
| warm vs warm2 | 0.1778 |

Self-WER is Levenshtein over lowercased, diacritic-stripped, punctuation-stripped
word sequences. Raw outputs are under
`~/.cache/oc-public/cpu-determinism-2026-09-07/` and
`~/.cache/oc-public/idle-probe-2026-09-07/` — never in git.

## The mechanism, confirmed at three levels

1. **Configuration.** `OURS_DECODE` passes
   `temperature: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`. The serverless handler
   (`angelospk/oc-asr-serverless`, `handler.py:130`) passes no `temperature` at
   all, so faster-whisper's default — the same ladder — applies.
2. **Library source.** In faster-whisper 1.2.1 `generate_with_fallback`, a
   candidate temperature above zero switches decoding to **sampling**
   (`beam_size=1`, `sampling_topk=0`, `sampling_temperature=temperature`); only
   `temperature == 0` uses beam search. faster-whisper itself sets no random
   seed. **It does not follow that seeding is impossible**: CTranslate2 exposes
   `ctranslate2.set_random_seed`, which this project has not tested and which is
   not documented to give reproducibility under concurrent requests. A seeded
   ladder is an untested option, not a closed door.
3. **Observation.** Every falling-back segment above reported
   `temperature == 1.0`. Read that carefully: faster-whisper reports the **last
   attempted** temperature, and on exhausting the ladder it returns the candidate
   with the highest average log-probability among those under the
   compression-ratio threshold — or among all of them if none passed. So
   `1.0` means *the ladder was exhausted*. It does **not** establish that the
   returned text was the one sampled at 1.0, nor that the last attempt rescued
   anything. The count of such segments differs on every run.

`condition_on_previous_text=False` in both stacks removes previous-text
prompting, but it does **not** confine a divergence to the segment that produced
it: generated timestamps can still move the next audio seek position.

One further caution about the table above: an emitted segment is not a decoding
attempt — a single decode can emit several segments. The 4/28, 10/31 and 7/29
counts therefore describe output structure, and cannot be used to price how many
decode passes `temperature=0.0` would save.

## What this does and does not put in doubt

**Probably not the fusion result — but this part is extrapolation, not
measurement.** *If* per-window noise near 10% is independent across windows, it
would average down over 391 windows to order ±0.003–0.004 of pooled WER, against
a measured fusion gain over Scribe of 0.056 with CI [−0.0734, −0.0439]. That
independence has not been checked and the pooled noise has never been measured
directly. Treat this as the reason not to panic, not as a result.

**Not the comparison between arms of one run.** The component cache hands every
arm the *same* `ours` output, keyed by (audio sha, provider, params sha), so the
draw is shared and the noise cancels. The vote-versus-`rules_on` contrast
(−0.00666, CI [−0.0104, −0.0035]) is a paired comparison on one draw and is not
affected.

**Three things it does put in doubt:**

1. [The live fusion pilot](2026-09-07-live-fusion-pilot.md) calls the
   Scribe/Soniox residual of 0.00017–0.00038 "the noise floor of this
   comparison". That is the *vendors'* noise floor. The `ours` arm's is roughly
   ten times larger, so the −0.00131 measured between the CPU and GPU stacks sits
   inside the noise and is not a finding. The report's own stated conclusion —
   that the serving stack moves the fused number by no more than about 0.001 —
   survives as an upper bound.
2. **Every per-window claim.** "80 of 80 windows better than Scribe" and "the
   largest single window supplies 11.4% of the net gain" are per-window
   statistics measured against a decoder with ~10% per-window self-disagreement.
   They have not been re-checked against that noise. This is not a claim that
   they fail.
3. **Exact reproducibility.** No number this project has published with this
   decode configuration will come out the same way twice.

## The candidate fix, and why it is not yet the fix

Passing `temperature=0.0` leaves the fallback loop with a single iteration, so
decoding is deterministic beam search. It is the **first candidate to test, not
the change to ship**. Removing retries is the leading hypothesis for why a 141 s
window costs 126 s of warm GPU time, but it has not been measured: reaching the
benchmark's ~100 s budget from 126 s needs a **20.6%** cut in GPU execution
before any other overhead, and from the 175.8 s the route actually spends, 43.1%.

The cost is that it changes the output of both stacks, so every frozen number
would have to be measured again. And threshold failure is a *suspicion* signal,
not proof that a passage was hallucinated — turning the ladder off may keep a
degenerate segment that the ladder would have rescued, on exactly the hard
passages this project cares most about.

Other policies exist and were not evaluated: a shorter ladder, a sparse ladder
such as `[0.0, 1.0]`, a **seeded** full ladder, or deterministic rescue of
flagged passages through a different decoder. Test `temperature=0.0` against the
existing ladder first; reach for another policy only if that comparison exposes a
tradeoff that needs one.

Two items ship with whichever policy wins:

- The component cache key (`params_sha` in `eval/live_fusion/providers.py`)
  hashes endpoint, language, ct2 hash and transport. It does not include the
  decode parameters. **This is the trap**: a `temperature=0.0` run would read
  cached ladder transcripts and report the old, plausible WER while fresh calls
  supply the new latency — "same quality, faster, deterministic", a combination
  no system ever produced. Prove that changing temperature causes a cache miss
  before trusting any number.
- The serverless handler does not report per-segment `temperature`,
  `avg_logprob` or `compression_ratio`. Had it done so, this would have been one
  reading rather than three runs.

## Limits of this measurement

One window, dense continuous council speech, from one city. Three runs per
stack. The fallback rate on other material is unmeasured, and the pooled-noise
estimate above is an extrapolation from a single window, not a measurement of
pooled noise. Bounding it properly needs the same repetition over roughly twenty
windows.

## Postscript, 2026-09-08: the cause was confirmed by intervention

The section above diagnosed the ladder from configuration, library source and
observation. It was then tested the only way a cause can be tested — by removing
it. `temperature=0.0` was set in `handler.py` and the image rebuilt **on top of
the pinned digest**, so `model.bin` keeps sha256 `3aa6368f…` and the endpoint's
provenance is unchanged. The handler now also reports its decode policy, which
this report had listed as a missing diagnostic.

Same window, three consecutive decodes, nothing else altered.

**CPU int8:**

| | ladder (2026-09-07) | `temperature=0.0` |
|---|---|---|
| times | 181.4 / 181.4 / 178.4 s | **78.8 / 77.7 / 77.9 s** |
| words | 247 / 239 / 240 | **237 / 237 / 237** |
| segments that fell back | 4 / 10 / 7 | **0 / 0 / 0** |
| self-WER | 0.0972 / 0.1215 / 0.0962 | **0.0000 / 0.0000 / 0.0000** |

**GPU float16, same endpoint:** execution fell from 123.9–145.4 s to
**14.7 / 13.7 / 13.9 s**, all three returning byte-identical text (242 words,
hash `03b13cdfb4f6a925`).

So the ladder was costing 57% of CPU decode time and roughly nine tenths of GPU
decode time, on top of the reproducibility. Both symptoms disappeared together
when the single suspected variable was removed, which is what the diagnosis
above predicted.

End to end through the production route on a fresh window (136.8 s of audio):
**31.2 s total**, against 96.5 s before — soniox 21.9 s, ours 22.3 s, scribe
28.7 s. **Scribe is now the critical path**, so further decode optimisation of
`ours` buys nothing on this route.

### What this does not settle

The quality cost. The ladder exists to rescue chunks that trip the
compression-ratio or log-probability thresholds, and it has been removed. Word
counts moved — 247 to 237 on the CPU window — and nothing here establishes
whether what left was degenerate output or speech. That is the 60-window
experiment with predeclared bounds recorded in
`exp-2026-09-07-decode-nondeterminism`, and it has not been run.

Until it is, `temperature=0.0` is a measured improvement in speed and
reproducibility and an **unmeasured** change in transcription quality.

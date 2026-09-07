# The fusion runs live, and it wins on real audio

2026-09-07. `exp-2026-08-26-fusion-integration`.

Everything measured before today came from **recorded token streams**. Three ASR
systems had never been called on the same audio in the same run, and the
production route had never reached a vendor. This is the first live measurement,
and the first end-to-end pass through `POST /v1/audio/transcriptions` with three
real providers.

Two things came out of it: a number, and four defects that no offline gate could
have found.

## What was measured

80 windows of ~140 s, 10 cities, drawn from the 260 cached benchmark windows with
a seed and a per-city cap fixed in
[`eval/live_fusion/PREDECLARED.md`](../../eval/live_fusion/PREDECLARED.md) before
any aggregate existed. The pool is restricted to the **206 windows whose
`(city, meeting)` appears in none of the 2026-08 training packs**: a contaminated
`ours` inflates the fused arm too, because fusion reads it as one of three votes.
Stage A (40 windows, cap 4 per city) is a prefix of stage B (80, cap 8) under the
same shuffle, and stage B was declared before stage A's aggregate was read.

All five arms are decoded live from the same audio bytes and scored with the same
`fusion.normalize.wtoks` / `sdi` the frozen 0.11205 was measured with.

**80 windows, 0 failures, `ours` on the canonical CPU int8 stack:**

| arm | WER | deletion rate |
|---|---:|---:|
| Scribe v2 | 0.16654 | 0.01764 |
| Soniox `stt-async-v5` | 0.14058 | 0.01839 |
| ours (`artifact-ct2-cleanpack-cont-s47`, CPU int8) | 0.13795 | 0.03142 |
| majority vote (W) | 0.11693 | 0.01703 |
| **R1+R2 + guard (`rules_on`)** | **0.11027** | 0.02044 |

- **Against Scribe: −0.05627**, cluster bootstrap CI [−0.0734, −0.0439], excludes
  zero. **80 windows of 80 are better.** The largest single window supplies 11.4%
  of the net gain, and removing any one window never reverses it.
- **Against the plain vote: −0.00666**, CI [−0.0104, −0.0035], excludes zero.
  41 better, 23 tied, 16 worse. So the rule layer is doing work the vote does not
  do, and it is not one window doing it.
- **The price is visible.** `rules_on` deletes more than Scribe (0.02044 against
  0.01764) and more than the vote (0.01703). A model that lowers WER by omitting
  hard passages looks better and is worse, so this is the axis the listening test
  exists to settle, not something this table settles.

The 40-window stage A said the same thing on its own: Scribe 0.18348, Soniox
0.16124, ours 0.15562, vote 0.13480, `rules_on` 0.12641, with 40 of 40 windows
better than Scribe. The levels differ between the stages because the windows do;
only the contrasts are comparable, and those held.

**The metric is agreement-with-OpenCouncil.** The reference is OpenCouncil's own
published transcript, not a human who listened to this audio. It records product
compatibility. It says nothing about which system is closer to what was said.

These numbers and the frozen 0.11205 are separately labelled results on different
cohorts. Their difference is not a replication, an improvement, or a regression.

## Does the serving stack move the number?

The same 40 windows were run twice. The only difference is where `ours` was
decoded: CPU int8 on this machine, which is the canonical measurement stack, or
the RunPod serverless endpoint at float16, which is what production calls. The
endpoint's own runbook says the two "give different tokens from the same
weights", so they are never merged — they are two runs, compared on the 37
windows both scored.

On the 78 windows both runs scored:

| arm | local (CPU int8) | gpu (serverless float16) | delta |
|---|---:|---:|---:|
| Scribe | 0.16722 | 0.16705 | −0.00017 |
| Soniox | 0.14206 | 0.14244 | +0.00038 |
| ours | 0.13812 | 0.13579 | −0.00233 |
| vote | 0.11762 | 0.11665 | −0.00097 |
| rules_on | 0.11072 | 0.10940 | −0.00131 |

**Read the control first.** Scribe and Soniox should be identical in both runs —
they are the same cached responses — and they are not, by 0.00017 and 0.00038.
The two runs executed in parallel and raced on some windows, so for those the two
runs hold two different live responses from the same vendor for the same audio.
That residual is the noise floor of this comparison.

So the honest statement is not "the GPU is 0.00131 better". It is: **the serving
stack moves the fused number by no more than about 0.001 WER, which is a few
times the vendor noise on the same audio** — and roughly a fortieth of the 0.056
that fusion itself buys. The earlier 37-window comparison, where the control was
exactly zero, gave −0.00109 in the same direction.

Two of the 80 windows never completed on the GPU: the RunPod job stayed
`IN_QUEUE` past a 900 s wait, twice, so they were left as failures rather than
tried a third time. They stay in the ledger; the cohort is 78 of 80.

For the production decision that is the useful part: serving the fine-tune on the
GPU endpoint instead of the measurement stack does not meaningfully change what
the fusion returns.

## What the live path broke on

The route had a 391-window replay gate at 15/15, zero hard mismatches. It still
could not have worked, for two reasons a fake provider can never expose:

1. **`ocasr.ts` sent RunPod a field called `url`.** The endpoint answers
   `input needs either 'audioUrl' or 'audioBase64'`. Measured against the live
   endpoint, not read.
2. **The response parser looked in the wrong place.** It read `words` or
   `segments`; the endpoint returns the OpenCouncil `Transcript` schema. Every
   real call normalized to zero words, and the route correctly turned that into a
   Scribe fallback — a green route serving one provider.

Two more came from the environment rather than the code:

3. **`RUNPOD_API_KEY` was never missing.** It was in `~/.runpod/config.toml` and
   `~/.bashrc`, and two nights of planning treated the GPU endpoint as
   unreachable because only `.env` was read.
4. **Node's fetch gives up on response headers after ~5 minutes.** A cold RunPod
   worker plus two cloud vendors on a 140 s window is longer than that, so the
   first live call failed as `HeadersTimeout` while the route was still working.
   Any real caller of this route needs the same treatment.

## The bucket, and what replaced it

All three providers took a public URL, produced by uploading the audio to a
DigitalOcean Spaces bucket that does not exist on this machine. Each vendor
accepts the bytes directly, so each now has a byte path: Scribe multipart,
Soniox `file_id`, ours `audioBase64`.

The transport is resolved **once per segment**, before any provider is
dispatched. Three parallel providers each deciding for themselves — by inspecting
an artifact `ensurePublicUrl` mutates — would let timing pick the path.

RunPod documents 10 MB for `/run`. Base64 costs `4*ceil(n/3)`, so the raw ceiling
is about 6.7 MB: a 140 s window fits, **a 15-minute segment does not**. Oversized
audio is refused as `inline_payload_too_large` and takes the Scribe fallback.
It is not transcoded or split to fit — both change the audio the model hears.
**Full-length segments still need a bucket.**

## How often it falls back

The first live attempt fell back with `timing_invariant`; the second fused. One
call cannot tell a 50% fallback rate from a 2% one, so every recorded live
response was replayed through the real route offline: **39 of 39 fused, zero
fallbacks**. The one observed `timing_invariant` was not recorded and cannot be
reproduced; the trace now carries the fallback message, not only its category, so
the next one can be.

## What this does not establish

- Nothing about fidelity to the audio. See the metric note above.
- Nothing about full-length segments: every window here is ~140 s.
- The windows are disjoint from the **2026-08** training packs only. The adapter
  continued from an earlier one whose data was not audited here, so this is an
  upper bound on how clean the `ours` arm is.
- Deletions rose. `exp-2026-08-26-chooser-deletion-safety` is still open and
  still the thing that decides.

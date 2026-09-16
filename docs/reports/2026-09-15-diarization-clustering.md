# Slicing a meeting by topic: what breaks, what fixes it, and what is left

`exp-2026-09-15-diarization-clustering` and `exp-2026-09-15-voiceprint-diarization`.
Plan reviewed by Codex Astra (`6d9715190ad149778f958c62ddc0f8f0`) before any job was
submitted. Both run on the same 6 meetings, plus a seventh held out for confirmation
(§7): the clustering arms took **35 jobs and 12.25 audio-hours**, the voiceprint arms
**56 jobs and 20.97** (the threshold arm 7 jobs and 1.20 h, the held-out meeting 2 jobs
and 2.36 h), for **91 jobs and 33.22 audio-hours** of a 150-hour trial. Zero failures.
Two arms — `a0_local` and `c3` — submitted nothing and cost nothing.

## The question

Upstream measured that the exclusive timeline fixes only ~13% of speaker
misattribution, and that the rest is pyannote merging several real people into one
voice ([opencouncil-tasks#15](https://github.com/schemalabz/opencouncil-tasks/issues/15)).

Harold's hypothesis: a practical ceiling near 30 speakers per job, beaten by diarizing
shorter pieces cut per agenda topic.

## Headline

| arm | what it is | mean B³ F |
|---|---|---:|
| **a0** | whole meeting, no hints | **0.8964** |
| c3 | slice boundaries, identity from a0 | 0.8933 |
| b0 | whole meeting + enrolled voiceprints | 0.8842 |
| **b3** | ~5-min topic slices + voiceprints | **0.8277** |
| b3L | ~15-min topic slices + voiceprints | 0.7906 |
| a0_local | the same a0 output, merely cut | 0.5280 |
| a3 | ~5-min slices, re-diarized, no identity | 0.5210 |

Four meetings, the ones where topic slicing produces more than one slice.

## 1. The ceiling was never exercised

Automatic counting returned **13, 7, 9, 15, 15, 19** voices for 15.8 to 134.4 minutes.
The falsification threshold frozen before the run was 35 clusters in one job; nothing
came close. In four of six meetings pyannote returns *fewer* voices than there are
reviewed identities — 15 against 40 in the worst case.

**No observed output saturated near the suspected cap, so this experiment did not
determine whether a cap exists.** Returning 15 voices against 40 reference identities
establishes under-clustering *relative to that reference*; it is not evidence of a
hard-cap mechanism, and it is not a refutation of one either. pyannote's documentation
states automatic detection has no upper limit.

Degradation **tracks speaker count more strongly than duration in this sample**:
identity count against precision r = −0.935, duration against precision r = −0.282, and
the 134-minute meeting with 15 identities beats the 72-minute meeting with 24. The
pattern survives within meetings (r = −0.674 over 29 slices). That is not causal
isolation: the 29 slices are nested inside four meetings rather than independent
replications, and speaker count co-varies with turn length, noise, overlap and
enrolment exposure.

## 2. Cutting does not destroy recall — cutting *while resetting identity* does

Partitioning the *same* a0 output at topic boundaries **and giving each slice its own
label space** costs recall **0.9156 → 0.5705**. That penalty is a property of discarding
cross-slice identity, not an unavoidable acoustic cost of short audio: `c3`, which keeps
global ids through the same boundaries, sits at 0.8933 — **−0.0031 from the whole-meeting
arm**. Topic segmentation is compatible with meeting-wide identity.

Re-diarizing each slice on top of the reset buys nothing: **−0.0074, better in 0 of 4
meetings**, no leave-one-out reverses the sign.

The reason is a trade the arm cannot escape. Cutting reduces how many people a piece
contains, which helps — and reduces how much voice each of them has in it, which hurts,
because pyannote clusters by embedding. Measured: slices where each speaker has little
voice lose **−0.0304**; slices where each has more lose **−0.0061**. And where slicing
is easy the model was already right: 8 of 29 slices return Δ exactly 0.0000, and
chania's two partitions are **identical** (Jaccard 1.0000, 4951/4951 co-clustered
pairs).

## 3. Enrolled voiceprints are the missing piece, and they are large

Production Postgres holds **409 enrolled pyannote voiceprints**. `/identify` returns
the enrolled personId as the speaker label, and that label is stable across separate
calls — so a person recurring in ten slices keeps one identity and the sliced arm needs
no relinking at all.

**On sliced audio this is worth +0.3067, in 4 of 4 meetings**: 0.5210 → 0.8277.

| meeting | a3 (no identity) | b3 (+voiceprints) | gain |
|---|---:|---:|---:|
| chania/jul27 | 0.5673 | 0.9712 | **+0.4039** |
| athens/may18_2 | 0.6325 | 0.6444 | +0.0119 |
| athens/may28 | 0.4043 | 0.7789 | **+0.3746** |
| vrilissia/may27_2 | 0.4797 | 0.9162 | **+0.4365** |

And they deliver what the product actually needs — a *name*, which `a0` cannot emit at
all: **pooled named accuracy 0.9182 over 5952 reviewed turns**, 3.4% left unnamed.

## 4. It still does not beat leaving the meeting whole

b3 against a0 is **−0.0687, better in 1 of 4**. The one win is a tie
(chania +0.0006). Two findings explain the rest, and both are actionable.

**Bigger slices are worse, not better.** 15-minute slices score 0.7906 against
0.8277 for 5-minute ones. The hypothesis that slices were simply too small is
contradicted.

**The failure is a threshold, not the idea.** `matching.threshold` defaults to **0**,
so a slice hands every second of speech to one of the enrolled labels no matter how
poor the match. Whole meetings leave 37% of `athens/may18_2_2026` anonymous; its slices
leave **0.1%** — and name accuracy collapses from 0.913 to **0.524**. The damage tracks
exactly how much speech belongs to people the voiceprint set cannot legitimately claim:

| meeting | speech by someone not enrolled | b3 − a0 | named accuracy sliced |
|---|---:|---:|---:|
| vrilissia/may27_2 | 0.0% | −0.0218 | 0.955 |
| athens/may28 | 16.4% | −0.0494 | 0.925 |
| athens/may18_2 | 43.2% | **−0.2043** | **0.524** |

`chania`, at 2.9% exposed, ties the whole-meeting arm outright.

## Recommendation

> **Use whole-meeting processing as the acoustic and identity scope, and apply agenda
> segmentation afterwards while retaining global speaker ids.** `c3` measures exactly
> that architecture at −0.0031 from the best arm. Add voiceprints when names are
> required, accepting a −0.0122 clustering trade. The tested sliced-identification
> configuration underperformed and is not adopted.

## What this licenses

Licensed: in these four meetings, independent per-slice processing preserved identity
worse than whole-meeting processing; enrolled voiceprints recover most of what a slice
reset destroys and supply names directly; and the residual damage correlates with speech
the enrolment set cannot claim.

Not licensed, and three of these were overstated in an earlier draft:

- **That slicing intrinsically fails.** This is a *configuration* comparison. Only two
  rejection thresholds were exercised (§5 at 50 on slices, §7 at 70 on a whole meeting).
- **That high enrolment is sufficient for slicing to break even.** `vrilissia` had
  **0.0% exposed speech and still lost 0.0218**. `chania`'s +0.0006 is a tie with
  variability unmeasured.
- **That exposure predicts anything.** §7 falsifies it: a held-out meeting with 46%
  unenrolled speech misnames only 0.85% of turns. There is no rule for when a threshold
  helps.
- **That 5-minute slices beat 15-minute ones in general.** Two settings, one direction.

Also not licensed: any end-to-end production number, since this uses human utterance
boundaries while production uses Scribe's, and Scribe errors could reorder arms rather
than lower all scores equally; any claim about speaker-count hints, built and
deliberately not run; and any naming figure as unbiased, since some speakers are
enrolled from the meeting they are scored on (§7).

## 5. The threshold was run, and it confirms the mechanism without rescuing slicing

`matching.threshold = 50` — the value pyannote's own documentation uses in both of its
examples, declared before the arm ran — on the existing 5-minute slices of
`athens/may18_2_2026`, the worst case (43.2% of its speech from people with no
voiceprint). 7 identify jobs, 1.2 audio-hours.

| arm | B³ F | correct / **all** turns | wrong name on unenrolled | no name claimed |
|---|---:|---:|---:|---:|
| a0 whole, no voiceprints | 0.8487 | — | — | 1.0000 |
| b0 whole + voiceprints | 0.8207 | 0.5247 | 0.0239 | 0.4049 |
| b3 slices, threshold 0 | 0.6444 | 0.3440 | **0.3962** | 0.0029 |
| **b3t50 slices, threshold 50** | **0.7098** | 0.3353 | **0.0588** | 0.4057 |

**The causal story holds.** Giving the call the ability to abstain cut wrong names on
unenrolled speakers by **85%** (0.3962 → 0.0588) while correct names over all turns
barely moved (0.3440 → 0.3353). It did not buy silence at the price of accuracy; it
removed specifically the false claims. B³ F rose 0.0654.

**And it does not rescue slicing.** 0.7098 still loses to whole-meeting processing with
voiceprints (0.8207) and without (0.8487). The threshold was the leading explanation for
the residual gap and it explains only part of it: even unable to misname, sliced
identification stays behind.

Arms without voiceprints sit at 1.0000 "no name claimed" by construction — they emit
`SPEAKER_xx`, which is not a naming claim. Counting those as wrong names, as an earlier
draft of this scorer did, would have charged them with misnaming 40% of the meeting.

## 6. The threshold on whole meetings, simulated offline for nothing

`/identify` returns a `voiceprints[]` block carrying, for **every** diarization
speaker, its match and its confidence against each enrolled print. That is enough to
replay any threshold against the saved responses without a new call. Applied to all six
whole-meeting runs, as a share of every reviewed turn:

| threshold | correct names | wrong names | no name claimed |
|---:|---:|---:|---:|
| 0 (as run) | 0.739 | 0.070 | 0.191 |
| 50 | 0.739 | 0.066 | 0.195 |
| **70** | **0.739** | **0.052** | 0.210 |
| 85 | 0.731 | 0.049 | 0.220 |

**At 70, wrong names fall 26% and not one correct name is lost.** At 85 the trade
begins: correct names start dropping. So 70 is the last free point.

**The whole effect is one meeting.** `chalandri/may25_2026` goes from 0.520 wrong to
0.029; the other five are unchanged to three decimals. That meeting is also the one
where 64.2% of speech comes from people with no voiceprint — the same exposure that
predicted the sliced-arm damage. The rule this supports is narrow and conditional:

> A threshold buys nothing where nearly everyone speaking is enrolled, and buys a great
> deal where they are not. It never costs correct names below 85.

Since the enrolled speakers in these meetings match at confidence 77–92 and the
rejected ones at 18–72, a threshold anywhere in 70–75 separates them cleanly here. One
cohort, six meetings: that separation is an observation, not a calibrated operating
point.

## 7. The held-out meeting does not reproduce the threshold gain

`athens/mar16_2_2026` — 70.7 minutes, outside the cohort, outside
`exclusion-manifest-2026-08`, `humanReview: true`, 1534 reviewed turns, 8 enrolled
voiceprints, 2.36 audio-hours. **Threshold 70 was frozen from the previous cohort's
simulation before this meeting ran**, which makes it a genuine held-out test.

| arm | B³ F | correct / all turns | wrong on unenrolled | no name |
|---|---:|---:|---:|---:|
| threshold 0 | 0.8990 | 0.5750 | 0.0085 | 0.3950 |
| threshold 70 | 0.8983 | 0.5717 | 0.0072 | 0.4003 |

**Neutral.** The wrong-name reduction is 13 turns out of 1534, and it costs 5 correct
names. Neither the 26% cut the simulation predicted, nor any harm.

**And it falsifies the exposure rule.** This meeting takes **46% of its speech from
people with no voiceprint** — the condition that was supposed to predict damage — and
still misnames only **0.85%** of turns, against chalandri's 52%. Unenrolled-speech share
does not predict misnaming on unseen data. There is currently **no rule** for when a
threshold helps.

### Voiceprint leakage, found here and present in the original cohort

Checking `VoicePrint.sourceSegmentId` against the meeting being scored: **2 of the 8
enrolled speakers were enrolled from this very meeting**, 30 seconds each, covering
20.5% of its speech. In the original six-meeting cohort, **3 of 44 person-meeting pairs**
are likewise self-enrolled. Enrolment audio independence was never checked before those
runs.

Excluding the two leaked speakers changes nothing qualitatively — threshold 0 gives
0.9043 / 0.4904 / 0.0104 and threshold 70 gives 0.9041 / 0.4856 / 0.0088 — so the
threshold conclusion survives the contamination. The naming figures elsewhere in this
report carry it as an unquantified favourable bias.

## Still untested

**A live call at any threshold other than 50.** Section 6 replays 50, 70 and 85 against
saved confidence scores, which assumes the service's rejection rule is exactly "best
confidence below threshold". No live run has confirmed that, and a live call could also
change which speakers the exclusive matcher assigns in the first place — the simulation
holds the assignment fixed and only withdraws names.

**Whether 70 transports.** Its entire measured benefit is one meeting out of six. A
value that helps only where unenrolled speech is high needs a second high-exposure
meeting before it is an operating point rather than an observation.

**The sliced threshold beyond one value and one meeting.** `b3t` is 50, on
`athens/may18_2_2026` alone.

## Method notes that decide how these numbers read

**Identity is `personId`, not `speakerTagId`.** One person in this cohort carries up to
20 speaker tags; scoring by tag would fold the previous pipeline run's splitting into
the answer key. Utterances on unresolved tags are counted and reported separately.

**Merging and splitting are scored as a pair.** B-cubed precision falls when one voice
covers several people, recall when one person is spread across several voices. Purity
alone would rate one-voice-per-utterance as perfect; the test suite asserts that total
merge, total fragmentation, rare-speaker absorption and refusal-to-answer each fail.

**Two defects were found by measurement, not review.** `/identify` does not return null
for unmatched speech — it puts `SPEAKER_xx` in the same field as real personIds, and
chania returned 13 labels for 2 voiceprints sent; without namespacing those, two slices'
anonymous speakers merge into one person. And overlap was taken from the widest single
segment instead of summed per speaker, which mis-assigns a speaker holding the floor
across several short turns. Both are covered by regression tests.

**35 tests pass before any paid job is submitted.**

## Artifacts

Inputs, raw responses, voiceprints and scores:
`~/.cache/oc-public/diarization-clustering-2026-09{,-long}` (never git — transcript text
and biometric embeddings; the voiceprint directory is `chmod 700`). Reference snapshot
frozen 2026-09-15T15:53:52Z. Code and frozen parameters:
[`eval/diarization_clustering/`](../../eval/diarization_clustering/README.md).

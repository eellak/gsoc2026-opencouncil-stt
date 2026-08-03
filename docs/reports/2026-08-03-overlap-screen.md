# Overlapping speech is rare, expensive, and the vote does not fix it

2026-08-03. Script: `eval/controlled_eval/exp_overlap.py` · raw: `results_overlap.json` ·
232 benchmark windows, ~140 meetings, 10 cities, `pyannote/speaker-diarization-community-1`
on CPU, 4h04m, no GPU, no labels, no training.

## What this is allowed to claim

Codex reviewed the design at high effort before it ran, and the limits it imposed are the
first section rather than a footnote, because the temptation to overread these numbers is
the whole risk.

**What we measured:** associations between pyannote-*estimated* overlap and the error
rates of seven existing ASR systems, on the benchmark windows with local audio. These are
descriptive risk-marker associations.

**What we did not measure:** the effect of true overlap, or what an overlap-targeted
system would recover. Overlap is not randomly assigned. Windows with crosstalk are also
noisier, busier and faster, and pyannote has never been validated on Greek council audio,
so it may over-detect overlap exactly where the audio is bad. Both push the same
direction and neither is fixable by reanalysis.

So this report does not say that overlap causes X% of our errors, and it does not say the
diarization-conditioning line is justified or killed. **Error concentration is not
addressability.** What it does is decide whether the expensive path is worth a real
experiment.

## 1. Overlap is rare

| | |
|---|---|
| overlap as a share of speech time | mean 2.2%, median 0.3% |
| max over all 232 windows | 21.0% |
| windows with no detected overlap at all | 100 of 232 (43%) |
| detected speakers per two-minute window | mean 2.4, max 7 |

Council meetings are chaired. People mostly wait, and the chair cuts in rather than
talks over. Whatever intuition says about a room with forty people in it, the recordings
say two seconds per minute.

Buckets were frozen from the diarization output before any error was looked at: no
detected overlap, then tertiles of the positive values. The top bucket is 44 windows
holding 19.9% of all reference words.

## 2. But it is a strong marker, for every system

Micro-WER by bucket, with meeting-clustered bootstrap CIs on the high-minus-none
difference:

| system | none | low | mid | high | high − none | concentration |
|---|---|---|---|---|---|---|
| Scribe v2 | 0.102 | 0.126 | 0.124 | 0.202 | +0.0994 [+0.061, +0.143] | 1.55 |
| Soniox | 0.093 | 0.137 | 0.135 | 0.239 | +0.1452 [+0.098, +0.197] | 1.73 |
| Gladia | 0.115 | 0.134 | 0.132 | 0.203 | +0.0880 [+0.046, +0.132] | 1.46 |
| whisper-large-v3 | 0.111 | 0.146 | 0.136 | 0.213 | +0.1018 [+0.060, +0.146] | 1.49 |
| our fine-tune | 0.114 | 0.139 | 0.140 | 0.232 | +0.1174 [+0.078, +0.158] | 1.58 |
| gpt-4o-transcribe | 0.115 | 0.241 | 0.157 | 0.226 | +0.1109 [+0.065, +0.161] | 1.35 |

Seven systems out of seven, every CI clear of zero. In the high-overlap quartile error
rates roughly double. Concentration is errors-share divided by reference-words-share: the
top bucket carries 19.9% of the words and 26% to 34% of the mistakes.

A bound worth stating carefully. If high-overlap windows scored the way zero-overlap
windows do, Scribe would fall by about **2.0 WER points** and Soniox by 2.9. That is
arithmetic on this sample, not a forecast: it assumes everything that makes those windows
hard is removable, and the confounding above is exactly the reason to doubt that. Read it
as the largest number this category could possibly be worth, not as what it is worth.

For scale, the [consensus vote](2026-08-02-asr-fusion.md) delivered 1.1 points and it
already exists.

## 3. The vote does not fix overlap

This was the free question and the answer went the other way from what I expected.

| overlap bucket | consensus gain over Scribe |
|---|---|
| none | −0.0158 |
| low | −0.0096 |
| mid | −0.0088 |
| high | −0.0018 |

Interaction, high minus none: **+0.0141, CI [+0.0041, +0.0249]**. The vote's advantage
shrinks as overlap rises and is essentially gone in the top bucket.

The reasoning that made me expect the opposite was that independent systems disagree where
audio is ambiguous, so the majority should rescue the odd one out. What the data says is
that in overlap they fail *together*. A vote can only pick a hypothesis that some system
produced, so when all three mistranscribe the same buried voice, there is nothing to
select. It is a selector, not a recogniser.

The useful consequence: fusion and overlap are **separate problems, not substitutes**. The
1.1 points the vote already earns are mostly earned outside overlap, so whatever overlap
costs is still on the table.

(Subgroup contrast over four prespecified buckets, exploratory.)

## 4. Does the reference even contain the interjector?

The gating worry: if the human transcriber wrote only the main speaker, a system that
correctly recovers both is scored as worse, and this benchmark cannot evaluate an overlap
fix at all.

The test is whether insertions rise faster than substitutions between the low and high
buckets, as a difference-in-differences with meeting-clustered CIs. Comparing "insertions
significant, substitutions not" would not have been a test of a difference.

| system | ins/word none → high | DiD (ins − sub) |
|---|---|---|
| Soniox | 0.031 → 0.158 | **+0.0995 [+0.065, +0.142]** |
| Scribe v2 | 0.017 → 0.081 | **+0.0322 [+0.011, +0.058]** |
| whisper-large-v3 | 0.013 → 0.066 | **+0.0204 [+0.001, +0.042]** |
| greek-whisper-v3-turbo | 0.012 → 0.017 | +0.0192 [−0.007, +0.050] |
| gpt-4o-transcribe | 0.012 → 0.048 | −0.0038 [−0.025, +0.021] |
| Gladia | 0.013 → 0.055 | −0.0031 [−0.017, +0.013] |
| our fine-tune | 0.015 → 0.052 | −0.0052 [−0.025, +0.018] |

Three of seven positive and clear of zero, four not. Soniox is the outlier by a factor of
three: its insertion rate quintuples in the high bucket.

**This decides nothing.** A positive DiD is consistent with the reference omitting the
interjector, and equally consistent with hallucination in noisy audio or with transcribers
dropping disfluencies. What it is good for is booking the audit: the split across systems
is itself informative, since a reference-policy effect should hit everyone and a
hallucination effect should track how talkative a decoder is. Soniox transcribing five
times more unmatched words than it does in clean audio is the single most interesting
thing in this file, and it cannot be resolved without listening.

## 5. The listening audit: pyannote is right, and the second voice is off-mic

2026-08-03. 60 blinded 12-second clips, 40 centred on a detected overlap event and 20
drawn from windows with no detected overlap at all, shuffled, no ASR output visible, key
sealed until the answers came back.

| | |
|---|---|
| overlap clips where ≥2 speakers were audible | **40 of 40** (CI95 [1.00, 1.00]) |
| control clips with exactly one speaker audible | 13 of 20 (CI95 [0.45, 0.85]) |

**Not one false positive in 40.** Whatever else is true of pyannote on Greek council
audio, when it says two people are talking, two people are talking. That was the single
largest threat to section 2 — that the diarizer over-detects overlap exactly where the
audio is bad, manufacturing the association — and it is now the least likely explanation.

The control number is weaker than it looks and must not be read as a miss rate. The
question asked was "how many speakers do you hear", not "do they speak at the same time",
so a control clip with two audible speakers is consistent with ordinary turn-taking. What
it does establish is that a window with zero *detected overlap* is not a single-speaker
window: about a third of them contain more than one voice inside twelve seconds.

The free-text comments are the most useful thing here, and they were volunteered on only
four clips: *"ακούγονται και άλλοι στο background χωρίς μικρόφωνα"*, *"δεν ακούγονται στο
μικρόφωνο"*. The competing speech is largely **off-mic room noise**, not a second
amplified voice. That reframes the whole category. Off-mic speech is exactly what a human
transcriber is entitled to leave out, and exactly what a decoder is most likely to
hallucinate words from, so both explanations of Soniox's insertion spike survive.

**The Soniox question is still open.** Settling it needed the audible words written out so
they could be matched against the unmatched hypothesis tokens, and 56 of 60 clips have a
speaker count but no text. The count answers the pyannote question completely; it cannot
answer this one. What is needed is small and specific: the second speaker's words on the
~15 clips where they are intelligible at all.

## 6. The paid model agrees with the free one

`eval/controlled_eval/precision2_compare.py` · raw: `results_precision2.json`.
pyannoteAI's commercial `precision-2` on the same 60 clips, judged against the same human
answers.

| | community-1 | precision-2 |
|---|---|---|
| overlap found in the 40 flagged clips | 40/40 (human-confirmed) | 40/40 |
| overlap found in the 20 zero-overlap controls | 0/20 by construction | 1/20 |

Complete agreement on the events, and an independent confirmation that 19 of the 20
control windows really are overlap-free. **There is nothing to buy here.** The open-source
model is not the limiting factor in any measurement in this file, so the overlap numbers
stand as computed and precision-2 does not change a single one of them.

The speaker *counts* diverge in an informative way:

| | precision-2 says 1 | says 2 | says 3 |
|---|---|---|---|
| human heard 1 | **13** | 0 | 0 |
| human heard 2 | 5 | **17** | 0 |
| human heard 3+ | 1 | **23** | 1 |

Perfect agreement wherever the human heard one voice, and systematic under-counting
wherever they heard more. Read together with the free-text notes — *"ακούγονται και άλλοι
στο background χωρίς μικρόφωνα"* — the explanation is not model error. The diarizers count
**miked** speakers; the human counted everyone audible in the room. That is the same
finding as section 5 arriving from the other direction, and it is the most concrete thing
we now know about what overlap in this corpus actually is: mostly off-mic room speech.

## What would settle it

Two things, in this order, neither of which needs a GPU.

**A blinded listening audit of 50 to 100 overlap events.** Two Greek speakers, no ASR
output in front of them, transcribing main and secondary speaker separately, then marking
which audible words appear in the benchmark reference. The direct diagnostic: do the
alleged insertions become matches once the secondary speaker is added to the reference?
That answers the reference-policy question and validates pyannote's overlap detection at
the same time. This is the piece that needs human time.

**A paired synthetic-overlap intervention.** Take clean single-speaker council audio, add
a real interjector at prespecified levels, and send both the untouched and the mixed
version through the same systems. The main-speaker reference is identical in both arms, so
the paired WER difference is the effect of adding that overlap with the target speech held
constant. It is causal for the manipulation performed, needs no new labels, and it is the
only cheap thing here that estimates a burden rather than an association.

Only after those does a DiCoW pilot make sense, and it should be gated on an engineering
threshold agreed in advance: the minimum WER improvement that would justify the training,
decided against the confidence interval rather than against significance.

## Caveats

18 windows were dropped for missing local audio, and they are not a random subset: Scribe
scores 0.1496 on them against 0.1305 on the 232 kept. The screen describes the kept
subset.

pyannote's overlap estimate is a model output with no Greek ground truth behind it. Every
number here inherits its errors, and `overlap_frac_of_speech` inflates when speech is
under-detected, so absolute overlap seconds are reported alongside it in the raw results.

The high-overlap bucket is 44 windows. The bucket contrasts are the strongest thing here
and they still rest on that.

Levenshtein error types are not semantic labels. A secondary speaker's words can land as
substitutions, and tie-breaking between equally optimal alignments can move them again.
Section 4 is built on that machinery and inherits its slack.

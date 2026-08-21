# Arm P as a served decoder, and what a whole meeting does to it

2026-08-21. Experiment: `exp-2026-08-18-chunking-aware-decoding`. Artifacts:
`artifact-decode-policy-p` (`5ae98472227696e0`), `artifact-ct2-fixed`
(`8a1a3b257d0c1bdb`). Capability: `cap-decode-p-local`. Runbook:
[decode-p-policy](../runbooks/decode-p-policy.md).

## What was and was not established before today

Arm P is the best measured decode arm by WER on the 39 frozen evaluation
windows: 0.15893 → **0.14751**, delta −0.01142, meeting-clustered CI
[−0.02025, −0.00195], excluding zero.

**It failed the preregistered gate of that experiment.** The gate was on
*deletions*, and P's deletion delta is −0.00428 with a CI containing zero. The
WER win comes from *substitutions* (938 → 860). P is the best measured arm; it
is not an arm that passed a gate. Nothing below changes that.

The experiment's own `next_action` named the missing piece: P had never been run
on contiguous whole-meeting audio, only on pre-cut ~150 s windows. That is what
this report closes.

## The served decoder

`serve/decode_p.py` is arm P as a configuration rather than a script: audio in,
text out. It imports the experiment's own splitter and silence detector rather
than copying them, so the served segmentation cannot differ from the measured one
by construction; the frozen `POLICY` dict is immutable and hashes to
`5ae98472227696e0…`, and `DecodeP.load()` hashes `model.bin` **in the directory
it is about to load** and refuses anything that is not `artifact-ct2-fixed`.
Overriding `model_dir` does not get around that.

Two things it returns that the experiment harness did not:

* every internal boundary with its kind (`vad_silence` / `forced`), so a bad
  stretch of transcript can be traced back to the cut that produced it;
* a `plan` operation that segments without decoding, so the forced-cut rate of a
  meeting can be inspected before paying for the transcription.

The boundary kind is recomputed from the silence list, not read from the
splitter's own `cut_silence_duration` field. That field is a trap: when the
short-tail repair *moves* a boundary it carries the old value over unchanged.
The test file pins the case — duration 29.6 s with one silence at [25.0, 26.0)
gives a piece `(0.0, 14.8)` still claiming 1.0 s of silence at a boundary
11 seconds away from it.

### Anti-drift

`eval/tests/test_decode_p_policy.py`, 39 assertions, all passing. Three of them
carry the weight:

1. the served constants equal `chunking_decode`'s, and `POLICY["decode"]` equals
   `CONTROL` exactly;
2. the policy digest equals a **pinned** value, not merely "some value that
   changes when the policy changes";
3. the served policy reproduces the measured arm's boundaries, cut for cut, on
   real cached audio (`win_orestiada_mar23_2026_6383163`, pinned by ID so the
   coverage of the test does not depend on which windows a machine happens to
   have).

A frozen planner fixture covers six segmentation regimes — silence cuts, blind
continuous speech, sub-minimum silences, the short-tail merge, the voiced-ground
fallback, and the stale-provenance case. Reusing the experiment's splitter is
good for locality but is *not* by itself drift protection: the splitter could
change without the policy hash moving. The fixture is what catches that.

## Does the served decoder actually reproduce the measured arm

Three windows, pinned by ID, chosen as a minimum coverage set rather than a
sample: the largest single-window gain, a window with two **forced** boundaries,
and a window where **P is worse than the control**, so the check is not
cherry-picked. Both arms decoded fresh on this machine with the experiment's own
per-window seed (`eval/decode_p_conformance.py`).

| window | boundaries | served P | served A | delta | P == cache |
|---|---|---:|---:|---:|---|
| `win_orestiada_mar23_2026_6383163` | 5 × `vad_silence` | 0.0942 | 0.1688 | −0.0747 | yes |
| `win_orestiada_dec11_2025_600052` | 2 × `forced`, 3 × `vad_silence` | 0.1432 | 0.1857 | −0.0424 | yes |
| `win_argos_apr24_2026_1015557` | 5 × `vad_silence` | 0.2222 | 0.1646 | **+0.0576** | yes |

The served decoder is byte-identical to the measured arm P on all three windows.
The control was byte-identical on two of the three; on
`win_orestiada_dec11_2025_600052` it came out at 0.1857 against the cached
0.1910. That window reaches the temperature fallback ladder, and unlike the
arm P cache the arm A cache records **no environment at all**, so it cannot be
shown to have come from this stack. It is a caveat on the control cache, not on
the served policy.

Wall time, local CPU int8 at 16 threads: P took 77 s, 117 s and 68 s for ~148 s
of audio each; the control took 78 s, 206 s and 191 s. P is *faster* here,
because a piece that decodes cleanly never enters the temperature fallback the
way a whole 150 s window does.

## The 39-window picture, from cache

Not recomputed — read out of
`~/.cache/oc-public/chunking-decode-2026-08/results-eval.json` and
`~/.cache/oc-public/decode-ablation/eval-A.json`:

* **25 windows improve, 4 are unchanged, 10 get worse.** The worst regression is
  +0.0576 (`win_argos_apr24_2026_1015557`), the best gain −0.0747.
* **No single-window domination.** Dropping the single most favourable window
  leaves the pooled delta at −0.00974; the three most favourable, dropped one at
  a time, give −0.00974, −0.01012 and −0.01020 against the full −0.01142.

## Whole-meeting audio: it works, with three caveats

`argos__jun22_2026`, 12.69 minutes of contiguous audio, decoded end to end on
local CPU:

| | |
|---|---|
| pieces | 32 |
| pieces under 5 s | 0 |
| speech dropped | 0.0 s |
| boundaries | 31, all `vad_silence` |
| segments | 93, timestamps monotonic |
| reruns under the same seed | byte-identical |

The splitter's own counter says **2 forced cuts** while the post-hoc classifier
says **0 forced boundaries**. Both are right and they measure different things: a
cut candidate is a silence *midpoint*, so a silence that starts before the 29.5 s
ceiling but whose midpoint falls past it is not a candidate — and the blind cut
at the ceiling then lands inside that silence anyway. Two of this meeting's
"forced" cuts got lucky. Read `forced_boundaries` for where the cut actually
landed and `forced_cuts` for whether the splitter had a choice.

Planning scales fine. On `argithea__jun25_2_2026` (64.4 minutes) the whole plan
takes about 9 s: 3.4 s to decode the mp3, 5.6 s for the VAD, 0.01 s to split.
The splitter is not the quadratic hazard it looks like — 2 h of synthetic
silences plans in 0.11 s. Peak RSS is ~1.5 GB for an hour of audio without the
model, ~3.2 GB with it, and grows with file length because the whole waveform is
decoded into memory before planning.

The three caveats are real and none of them was visible on pre-cut windows.

### 1. The forced-cut rate is a property of the meeting, not of the policy

| audio | boundaries | forced |
|---|---:|---:|
| the 39 evaluation windows | 207 | 7.7% |
| `samothraki__jun18_2026` (81.9 min) | 193 | 5% |
| `argos__jun22_2026` (12.7 min) | 31 | 6% |
| `argithea__jun25_2_2026` (64.4 min) | 156 | **41%** |

One of three sampled meetings gets 41% of its cuts forced — that meeting reports
only 256 silences in 64 minutes, one every fifteen seconds. A forced cut is
precisely the case P's mechanism does not address, so the −0.01142 measured on
windows whose boundaries were 92% silence-justified should not be expected to
transfer to a meeting like that. Run `plan` and read `forced_boundaries` first.

### 2. The true start and end of a meeting hallucinate

The decoded meeting opens with `Υπότιτλοι AUTHORWAVE` and ends with it twice
more — the classic Whisper subtitle-credit hallucination in a non-speech region.
Three occurrences in one 12-minute meeting. Pre-cut evaluation windows are cut
out of the middle of a meeting and so never contain the opening and closing
silence, which is why 39 windows never showed this. `CONTROL` has
`vad_filter=False`, so those regions are decoded rather than skipped. Arm V
(`vad_filter=True`) is the arm whose measured gain was *entirely* less
hallucination in non-speech; on whole meetings, V and P are plausibly
complementary rather than competing. That is a hypothesis, not a result.

### 3. Segment timestamps are not trustworthy at the seams

31 of the 93 segments end **after** the boundary of the piece that produced
them; the worst overshoots by 29.98 s. `CONTROL` has `word_timestamps=False`, so
the smallest timed unit is a whole segment, and a segment decoded near the end
of a piece routinely claims time that belongs to the next piece. The *text* is
unaffected — pieces tile the audio exactly in samples and nothing is decoded
twice — but any downstream consumer that trusts these timestamps for alignment,
diarization, or clipping will be wrong at roughly a third of the segments.

## What this does not settle

* No fidelity measurement on whole-meeting audio. There is no reference
  transcript for these meetings in this run, so "it works" here means the
  segmentation and the plumbing hold, not that the WER holds.
* The 39-window result is local CPU int8. The 247-window GPU run has never been
  done for arm P, and the two stacks must not be mixed.
* Arm PI is still unscored.

## Housekeeping

`eval/chunking_decode.py::EXPERIMENT` is the string
`exp-2026-08-17-chunking-aware-decoding`, a day earlier than the ledger record
`exp-2026-08-18-chunking-aware-decoding`. It is written into every cached state
on disk, so it was left alone and the ledger id is carried in the policy instead,
as `ledger_experiment_id`. Both name the same experiment.

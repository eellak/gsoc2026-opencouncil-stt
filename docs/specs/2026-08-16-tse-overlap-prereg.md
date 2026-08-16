# Preregistration — target speaker extraction over simultaneous speech

`exp-2026-08-16-tse-overlap`. Written 2026-08-16, **before any arm was scored**.
Environment feasibility and the external API were verified first (see
"What was checked before this document"); no gold-set arm had been run.

**Revised once, on 2026-08-16, after a Codex review at high effort (job
`e44b33d3`) and still before any arm was scored.** The review's verdict was "do
not run this as written": Stage 2 was quoting a 217-token denominator its own
Section 1 had already contradicted, G1 compared two different processing stacks,
and G2's inequality was written with the sign reversed. Section 9 lists every
change. The design below is the frozen one.

The idea under test: run pyannote-style overlap detection, build a clean
enrollment sample per speaker from that speaker's non-overlapping audio, run
WeSep target speaker extraction (TSE) once per speaker over each overlap
region, re-run our ASR on each isolated track, and measure whether the words
that simultaneous speech currently costs us come back.

---

## 1. The substrate, measured before any arm

These numbers come from the frozen gold set and from `hyp/adapter` decodes that
already existed. They are counts, not results, and they are stated first because
they determine what this experiment is allowed to claim.

| quantity | value |
|---|---|
| scored gold audio | 945 s (27 cells × 35 s, 6 meetings) |
| human-marked simultaneous speech | **20.0 s** |
| pyannote precision-2 overlap over the same clips | 21.3 s (2.25%) — similar in aggregate; no per-interval IoU computed, so no agreement is claimed |
| overlap-participating gold blocks (primary region) | 38 |
| … of which text is **certain** (not `text_unc`) | **20 blocks, 217 tokens, 15 cells, 5 meetings** |
| simultaneous **and** certain **and** surviving uncertainty masking | **10.6 s, ≈23 tokens, 6 blocks, 3 cells, 2 meetings** |
| certain overlap blocks where `artifact-ct2-fixed` emits **zero** words in the span | **1 of 20** |

Primary region throughout is `core_envelope`, as in `eval/gold_set_score.py`.
The uncertainty masking is the one from that scorer and is mandatory: without it
every WER on this asset doubled.

Two consequences, both accepted in advance.

**There is no lost span for TSE to recover on our ASR.** The gold-set finding
that 12 of 38 overlap blocks have not one of their words published is a fact
about **PUB**, the published OpenCouncil pipeline, whose loss inside overlap is
diarization- and composition-shaped. Our own adapter already writes words into
19 of 20 certain overlap blocks. So TSE's target here is the **content** of what
the adapter writes in overlap, not a missing region.

**There is damage, and the denominator cannot price it.** Restricted to certain
overlap blocks the adapter's fidelity-to-audio WER is **0.3963** [0.2016, 0.5512],
against **0.2749** [0.1770, 0.4554] outside overlap; deletion rate 0.1982 against
0.1202. The point estimate is higher inside overlap and agrees directionally with
every prior overlap analysis in this project; no paired CI for the
overlap-minus-non-overlap contrast was computed, so "the direction is real" is
not claimed.

And 217 tokens is the wrong number for Stage 2 anyway. **The scored real-overlap
substrate, after the mandatory uncertainty masking, is 6 blocks, ≈23 reference
tokens, 3 cells and 2 meetings** — before enrollment eligibility cuts it further.
One word is ≈4.4 WER points. A two-cluster bootstrap is uninformative.
**Therefore no aggregate WER, no delta and no bootstrap interval will be quoted
from real gold overlap in this experiment**, and Stage 2 is registered as a
case-level failure-mode audit rather than a measurement (Section 5).

There is also a standing result that bounds the prize:
`exp-2026-08-03-synthetic-overlap` added a real interjector to clean windows by
additive mixing at natural dose and measured a burden of **0.0016 WER**
(0.0084 at equal loudness), against a preregistered gate of 0.020. That bounds
the damage *its own additive-mixing construction* produces; it is not a
measurement of natural reverberant overlap. It does mean that if real overlap
damage resembles that construction at all, the prize available to any repair is
small.

Given all of that, this is registered as a **two-stage mechanism screen with no
promotion gate**, not as a candidate arm.

---

## 2. What was checked before this document

Verified against the actual `wenet-e2e/wesep` source at commit `99eca54b`
(master, last pushed 2025-10-04), not from memory:

- There is **no PyPI package**. Source install only, `setup.py`, no `pyproject.toml`.
- `wespeaker` is a **hard, undeclared** dependency (`wesep/models/bsrnn.py` imports
  it at module load) and is also not on PyPI.
- The CLI is **broken in two ways** on master and is not used here: `main()`
  reads `args.normalize_output` while argparse defines `--output_norm`, so any
  successful extraction raises `AttributeError` before the file is written; and
  `--bsrnn` looks up a hub key that does not exist (`Hub.Assets` has only
  `"english"`) and calls `sys.exit(1)`. We call the Python API
  (`load_model_local` → `Extractor.extract_speech_from_pcm`) instead.
- **Exactly one pretrained model exists**: `bsrnn_ecapa_vox1.tar.gz`, 261,570,027
  bytes, hosted on **ModelScope**, auto-downloaded to `~/.wesep/english/`.
- Its shipped `config.yaml`, now read directly: BSRNN, `feature_dim: 128`,
  `num_repeat: 6`, speaker encoder `ECAPA_TDNN_GLOB_c512` initialised from
  VoxCeleb, `joint_training: true`, `spk_feat: true`, loss `SISDR`, trained with
  online dynamic mixing at 16 kHz on 3 s chunks (`chunk_len: 48000`), and —
  the load-bearing line — **`reverb_prob: 0`, `noise_prob: 0`**.
- Feasibility smoke test on CPU (not scored, not on any evaluation region):
  ~2.1× realtime for an 8 s mixture; extraction of one Greek council speaker from
  an additive two-speaker mix reproduced the enrolled source at |corr| 0.992 and
  the interferer at 0.011.

**Transfer risk, stated before measurement.** The checkpoint was trained on
anechoic, noise-free, dynamically mixed VoxCeleb utterances. Council audio is
far-field, reverberant, PA-amplified Greek. Parakeet-TDT's published FLEURS
figures did not transfer to this domain at all (WER 0.3567, zero wins in 247
windows). TSE is acoustic rather than linguistic and should transfer better, but
the training distribution has neither reverberation nor additive noise, and real
council overlap is not additive mixing of two close-mic'd sources. Stage 1 is
deliberately generous to TSE for exactly this reason, and its result must never
be quoted as a result about real overlap.

---

## 3. Frozen configuration

**ASR, identical in every arm of both stages.** `faster_whisper.WhisperModel`
over `artifact-ct2-fixed` at `/home/harold/oc-asr-serve/ct2-fixed`,
`device="cpu"`, `compute_type="int8"`, `cpu_threads=16`;
`transcribe(language="el", beam_size=5, word_timestamps=True,
condition_on_previous_text=False)`. This is the serving configuration in
`serve/oc-asr/oc_asr_server.py` and it is not tuned here.

**Separation.** `Extractor` from `~/.wesep/english`, `device="cpu"`,
`resample_rate=16000`, `set_vad(False)` (the VAD path applies only to the
enrollment, strips leading/trailing silence only, and returns `None` on a silent
enrollment — an extra failure mode with no benefit here), `output_norm` left at
its default `True`. Mono float32, 16 kHz, read with `soundfile`, never through
`torchaudio.load` (torchaudio 2.11 requires TorchCodec for that path).

**Only BSRNN is run.** It is the only published checkpoint. No second
architecture will be tried if it fails. Iterating through SpEx+, Conv-TasNet,
pDPCCN and TF-GridNet hunting for one that wins is the multiple-comparisons trap
the autoresearch harness exists to prevent, and none of them has a released
Greek-capable checkpoint anyway.

**Seed** 20260816 for every random choice. **Bootstrap** 10,000 draws, resampling
**meetings**, per `eval/gold_set_score.py`. Six clusters at most: descriptive,
never a significance claim.

**Cost.** CPU only. No GPU pod will be created. No paid API is called. If CPU
turns out to be infeasible the experiment stops and says so.

**Isolation.** WeSep lives in `/home/harold/wesep-build/.venv` (uv, Python 3.11),
outside the repo; `.venv-eval` is untouched. All audio, mixtures and extracted
tracks live under `~/.cache/oc-public/tse-2026-08/`. No audio and no transcript
text enters git. The 16 locked evaluation windows are not read.

---

## 4. Stage 1 — does TSE work at all on these voices

More reference tokens than Stage 2 and an exact reference, on artificial additive
mixtures of two real council speakers. This is **deliberately generous to TSE**:
additive mixing of two separately-recorded sources is the regime the checkpoint
was trained for. It is not "high power" — there are still only six meeting
clusters. A failure here kills the idea; a pass says nothing about real overlap.

**Items.** Every gold block that is certain (`text_unc == false`), does **not**
participate in overlap, lasts >= 2.0 s, carries >= 5 reference tokens, and whose
speaker has >= 3.0 s of *other* certain non-overlap audio in the same cell to
enrol from. Measured supply: **37 items** from 52 candidate blocks, 6 meetings.
Each item draws, from the seeded RNG, one **masker** block and one **absent third
speaker** block, both from meetings other than the target's. The complete
manifest — target/masker/enrollment offsets, gains, tiling flags — is written and
sha256'd before a single arm is decoded, and the conclusion is explicitly
conditional on that one frozen set of pairings.

**Level.** SIR is computed on **active speech only**: the RMS over samples whose
short-time (20 ms) energy is within 40 dB of the block's loudest frame. Whole-file
RMS would let silence in one block move the mixing gain. The masker is scaled;
the target is never rescaled. Maskers at least as long as the target are
random-cropped with the seeded RNG; shorter maskers are tiled, and the number of
tiled items is reported. Primary SIR **0 dB**; secondary **+5 dB**. **Gates are
evaluated at 0 dB only**; +5 dB is descriptive and cannot rescue a failed gate.

**Arms**, all decoded with the identical ASR config. `_NORM` arms carry the exact
peak-to-0.9 normalisation that WeSep applies to its own output, so that every
contrast used in a gate is same-stack:

| arm | audio | scored against |
|---|---|---|
| `CLEAN` | target alone | target text |
| `CLEAN_NORM` | target alone, peak-normalised like a TSE output | target text |
| `MIX` | the mixture, raw | target text |
| `MIX_NORM` | the mixture, peak-normalised like a TSE output — **the baseline every gate uses** | target text |
| `TSE` | extraction with the target's own enrollment | target text |
| `TSE_WRONG` | extraction with the **masker's** enrollment — the speaker actually present in the mixture | **masker** text, and target text |
| `TSE_ABSENT` | extraction with a **third** speaker's enrollment, absent from the mixture | target text |
| `TSE_CLEAN` | extraction applied to the **clean** target with its own enrollment | target text |

`TSE_WRONG` is the targetedness control the first draft got wrong: enrolling a
speaker who is not in the mixture tests out-of-set behaviour, not selection
between the two voices present. `TSE_ABSENT` is kept as a separate diagnostic.
All four enrollments are duration-matched to within 0.5 s where supply allows,
and the matched-vs-mismatched channel confound is disclosed in Section 8.

**Separation metrics, not just WER.** Stage 1 has the exact source waveforms, so
**SI-SDR of each extracted track against the target and against the masker** is
computed and reported beside every WER. WER alone can show that preprocessing
helped an ASR; it cannot establish that target-speaker separation happened.

### Preregistered stage gates

Evaluated at 0 dB SIR only. Deterministic robustness criteria, not significance
claims — with six clusters a bootstrap CI is not a credible inferential gate, so
it is reported for description and the gates below do the deciding. All four must
hold to proceed to Stage 2. `B = MIX_NORM`, `D(x) = WER(B) - WER(x)`.

- **G1 — it separates.** `D(TSE) > 0`, **and** every leave-one-meeting-out
  recomputation of `D(TSE)` is positive, **and** no single item supplies >= 50%
  of the total `MIX_NORM - TSE` error reduction.
- **G2 — it is targeting, not enhancing.** `D(TSE_WRONG) < 0.5 * D(TSE)`,
  evaluated as the point-estimate contrast `C = D(TSE_WRONG) - 0.5*D(TSE) < 0`.
  A ratio is not used; it is unstable when `D(TSE)` is near zero. If enrolling
  the *other* voice in the mixture recovers the target nearly as well, the model
  is doing generic enhancement, the enrollment framing is wrong, and the
  experiment **stops**.
- **G3 — it is safe off-target.** `WER(TSE_CLEAN) - WER(CLEAN_NORM) <= +0.05`,
  same-stack. This is an **engineering rejection threshold, not evidence of
  deployment safety**: +0.05 over Stage 1's ~700 tokens is ~35 extra errors and
  ~18% relative degradation, and its real cost depends on an overlap detector's
  false-positive exposure, which this experiment does not measure. Above it the
  technique cannot be run behind any imperfect detector and the experiment
  **stops**.
- **G4 — it does not buy WER with deletions.** `delrate(TSE) - delrate(MIX_NORM)
  <= +0.02`. An arm that lowers WER by dropping hard passages looks better and is
  worse; the threshold is frozen here, before any decode.

`MIX` vs `MIX_NORM` and `CLEAN` vs `CLEAN_NORM` are diagnostics: if they differ
beyond noise, the write/normalise path is itself moving the number.

**Reported for every arm, always:** raw S/D/I counts as well as rates; WER,
deletion rate and insertion rate separately; per-meeting leave-one-out; and the
single-item share of any headline delta. Bootstrap is 10,000 draws, percentile,
two-sided 95%, resampling **meetings**, ratio-of-sums recomputed from raw edit
counts in each resample.

## 5. Stage 2 — real gold overlap, case-level failure-mode audit

Runs **only** if G1-G4 all pass. **This is not a measurement.** With 6 blocks,
~23 post-mask reference tokens and 2 meetings, no aggregate WER, no delta and no
bootstrap interval will be computed or quoted. What it can support is exactly
this list, per overlap group:

- the reference text, the baseline hypothesis and each extracted track's
  hypothesis, with raw S/D/I counts per case;
- enrollment coverage: which participating speakers had >= 3.0 s of certain
  non-overlap audio and which did not;
- named failure modes: extraction returned silence, tracks duplicated each
  other's words, extraction removed the target, `None` returned;
- whether any word absent from the baseline appears in an extracted track and is
  present in the human reference — a **count**, reported as a count.

**Overlap regions** are the human `ov_with` annotations, not a detector — adding a
detector adds a second failure mode. An operational deployment would need one;
this experiment says nothing about detection.

**Overlap region** = the time hull of a maximal transitively-connected group of
`ov_with`-linked blocks, padded 0.5 s each side, clipped to the clip. Reference
text in the padding is **excluded**; only the group's own certain block text is
referenced.

**Enrollment**, per participating gold speaker: that speaker's certain,
non-overlap gold blocks **elsewhere in the same cell**, taken in time order,
first 10.0 s, minimum **3.0 s**. Speakers below 3.0 s are **excluded and
counted**. The frozen estimand is **all groups, retaining an unenrollable
speaker's reference words as unavoidable TSE deletions** — the alternative
(complete-case groups only) would silently select the easy cases. Baseline,
placebo and TSE are described over exactly the same reference material.

**Arms**: `BASELINE` (adapter on the raw region), `MIX_NORM` (same round-trip and
peak normalisation, no separation), `TSE` (per-speaker extraction, each track
decoded separately), `TSE_WRONG`. Track hypotheses are **reported per track**;
they are not merged into a single serialised string, because merging independent
Whisper timestamps against one linear reference imposes an arbitrary word order
and would let timestamp jitter move a number.

## 6. Stop rules

- Any Stage-1 gate fails → stop, write it up, do not try a second architecture.
- The environment cannot be built, or CPU is genuinely infeasible → stop and
  report. **Do not create a GPU pod.**
- The same failure occurs twice → stop, per the project's hard rules.
- No threshold, SIR, region or enrollment length is chosen after a WER is seen.

## 7. What this experiment cannot answer

- Whether TSE helps the **published pipeline**, whose overlap loss is
  diarization-shaped rather than acoustic. This measures our ASR only.
- Anything on the 247-window benchmark. That reference is
  agreement-with-OpenCouncil — our own published text, which itself drops
  overlapping speech — so a genuine recovery inside overlap scores there as an
  insertion. Measuring overlap recovery against it would systematically punish
  the thing we are looking for. It is not run, and the two metrics stay separate.
- Whether a TSE model **trained on reverberant Greek council audio** would work.
  Only the one released anechoic VoxCeleb checkpoint is tested.

---

## 8. Stage-1 construction biases, disclosed before measurement

Stage 1 is favourable to TSE in at least these ways, all of which are reasons a
pass there does not transfer to real overlap:

- The correct enrollment and the target come from the **same cell and channel**;
  the masker and the absent-speaker enrollment come from other meetings. The
  model may exploit channel and session match rather than voice identity. This is
  why `TSE_WRONG` enrols the masker *actually present in the mixture* rather than
  only an absent speaker.
- Target and masker come from different rooms, so they carry distinct room and
  microphone signatures — easier to separate than two voices in one room.
- Target identity is oracle, and the enrollment is oracle-clean.
- The "sources" are not isolated sources. Each block already contains its own room
  noise and ambience, so mixing duplicates the noise floor.
- Tiling a short masker produces repeated speech and artificial seams that real
  overlap does not have. The tiled-item count is reported.
- Equal-loudness mixing at 0 dB maximises both the impairment and the headroom for
  a rescue. It is not the corpus's natural dose.
- Eligibility selects blocks that are long, text-certain and belong to speakers
  with plenty of clean audio — the easier, better-represented speakers.
- One masker draw per target: the result is conditional on 37 arbitrary pairings
  from the frozen manifest, not marginal over pairings.

## 9. What the Codex review changed

Job `e44b33d3`, high effort, run on the first draft before any arm was scored.
Adopted:

1. Stage 2 demoted from a WER measurement to a case-level failure-mode audit, and
   its denominator corrected from 217 tokens / 5 meetings to ≈23 tokens /
   2 meetings. This was a real contradiction inside the first draft: Section 1
   already contained the smaller number.
2. Every gate now compares against `MIX_NORM` / `CLEAN_NORM`, which carry the same
   peak normalisation as a WeSep output. The first draft compared a normalised
   TSE output against un-normalised audio — a same-stack violation, and the
   placebo that would have caught it was labelled "diagnostic only".
3. G2's inequality was written with the sign reversed and would have passed when
   wrong enrollment helped *more* than correct enrollment. Rewritten as the
   contrast `D(TSE_WRONG) - 0.5*D(TSE) < 0`; no ratio, which is unstable near zero.
4. `TSE_WRONG` now enrols the **masker present in the mixture**. Enrolling an
   absent third speaker tests out-of-set behaviour, not selection between the two
   voices present; it is kept separately as `TSE_ABSENT`.
5. SI-SDR against both sources added. WER can show preprocessing helped an ASR; it
   cannot show that separation happened.
6. Gates converted from "CI excludes zero" to deterministic robustness criteria
   (sign, all leave-one-meeting-out, single-item domination), since the document
   elsewhere calls a six-cluster bootstrap descriptive. The two positions
   conflicted.
7. G4, an explicit frozen deletion-harm threshold, added. Reporting the deletion
   rate is not a gate.
8. Gates restricted to the primary 0 dB SIR; +5 dB cannot rescue a failure.
9. SIR defined on active speech rather than whole-block RMS; random-crop preferred
   over tiling; tiling counted.
10. G3 relabelled an engineering rejection threshold rather than evidence of
    deployment safety, since detector false-positive exposure is not measured.
11. Wording repairs: "high power" removed; "the direction is real" withdrawn to a
    point-estimate statement; the synthetic-overlap conclusion restated as
    construction-specific; pyannote "agrees" reduced to "similar in aggregate";
    WER units standardised to decimals throughout.

Not adopted, with reasons:

- **Multiple balanced masker draws per target.** Rejected on CPU cost. The
  alternative the review offered is taken instead: the conclusion is explicitly
  limited to the frozen mixture manifest (Section 8).
- **A permutation- or time-constrained multi-talker WER for Stage 2.** Moot —
  Stage 2 no longer produces a WER. Tracks are reported separately and never
  merged.

## 10. Frozen before decoding

Manifest sha256 (targets, maskers, enrollments, offsets, gains, tiling flags);
WeSep commit `99eca54b`; wespeaker pinned at `e9bbf739` (2024-10-22 — master
requires `s3prl`, which the shipped checkpoint does not need); checkpoint
`avg_model.pt` 282,633,800 bytes from `bsrnn_ecapa_vox1.tar.gz` 261,570,027
bytes; torch 2.13.0+cpu, torchaudio 2.11.0+cpu, faster-whisper 1.2.1,
ctranslate2 4.8.1; the ASR options listed in Section 3, with every other
`transcribe` argument left at the library default; primary SIR 0 dB for gates,
+5 dB descriptive; the enrollment order rule (time order, first 10.0 s, minimum
3.0 s); the domination threshold (50%); the deletion-harm threshold (+0.02).

**Infeasibility is defined**: if separation or decoding exceeds 6 h of wall clock
for Stage 1, the experiment stops and reports CPU infeasibility. It does not
create a GPU pod.

**"The same failure twice" is defined**: the same exception, or the same
degenerate output mode, on the same arm after **one** mechanical remediation
(a missing dependency, a path, a dtype). A second occurrence stops the run.

# External resources for the open ASR tasks

**Updated:** 2026-08-16  
**Map:** [Wayfinder map](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/3)  
**Research ticket:** [Έρευνα πηγών για name-repair, confidence, overlap και evaluation harness](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/26)

This is a resource index, not evidence that any external method transfers to
Greek council audio. Read it before proposing work on the open map tickets
`#6`, `#7`, `#9`, `#13`, or `#25`. The local ledger and frozen experiment reports
remain authoritative for this project.

## Acquisition status

The MacBook SSH path worked (`harolds-MacBook-Air.local`, user `harold`). The
MacBook `wayfinder` and `grok-research` skills were copied to:

- `~/.codex/skills/wayfinder/SKILL.md`
- `~/.codex/skills/grok-research/SKILL.md`

`grok doctor` passed after adding Homebrew Node and Bun to the non-interactive
SSH `PATH`. Four parallel Grok research calls and one tighter retry then timed
out waiting for the Grok composer textbox. No Grok answer is treated as a
source below. The fallback sources are direct papers, official documentation,
or source repositories and should be rechecked by the research ticket when the
Grok browser is healthy.

## Name repair and contextual biasing

### CB-Whisper — LREC-COLING 2024

- Source: [ACL Anthology paper](https://aclanthology.org/2024.lrec-main.262/)
- Supports: rare names, organizations and terminology are a distinct Whisper
  failure mode; the paper uses open-vocabulary keyword spotting before the
  decoder and prompt construction to inject entities.
- Relevant task: map ticket `#6`, and the unresolved question of whether a
  roster-aware repair should stay a post-hoc shadow arm or move into decoding.
- Reproduction: requires the KWS/TTS front end and its entity prompts; this is
  not a drop-in faster-whisper option. Evaluate with held-out meetings and a
  false-positive/hallucination guard.
- Does not establish: Greek transfer, far-field council transfer, or that
  decoder-time biasing beats the already measured roster repair.

### KWS-Whisper — multitask contextual biasing

- Source: [paper](https://arxiv.org/abs/2309.09552)
- Supports: an open-vocabulary KWS signal can be used to bias Whisper toward
  user-defined named entities rather than changing all model weights.
- Relevant task: `#6` only as a future alternative if the current shadow arm
  cannot meet its gates.
- Reproduction: requires training or obtaining the KWS component and a biasing
  list; the paper's data and language setting are not the council setting.
- Does not establish: a safe production threshold or an improvement on Greek
  domain terms.

### WhisperBiasing — open-source research implementation

- Source: [GitHub repository](https://github.com/BriansIDP/WhisperBiasing)
- Supports: a concrete implementation of neural contextual biasing with a
  tree-constrained pointer generator and biasing-list preparation, with tests
  and scoring scripts in the repository.
- Relevant task: `#6` as a prototype reference, not a serving dependency.
- Reproduction: repository targets Whisper experiments and LibriSpeech/SLURP/
  DSTC-style data; inspect the exact commit and dependencies before using it.
- Does not establish: maintenance, CTranslate2 compatibility, Greek support,
  or a no-regression guarantee on non-name speech.

### Official Whisper and faster-whisper controls

- Sources: [OpenAI Whisper](https://github.com/openai/whisper), [faster-whisper
  README](https://github.com/SYSTRAN/faster-whisper/blob/master/README.md),
  [faster-whisper transcription source](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)
- Supports: the actual controls that can be tested without inventing a new
  model: `initial_prompt`, `hotwords`, `condition_on_previous_text`, beam and
  fallback settings, and `word_timestamps`.
- Relevant tasks: `#6` and `#13`.
- Reproduction: pin the current serving stack, decode config, model artifact
  and normalizer. Treat every prompt/hotword variant as a separate preregistered
  arm. The faster-whisper source shows that `hotwords` are inserted into the
  decoder prompt and are length-limited; this is a mechanism, not a guarantee.
- Does not establish: that enabling a control is transcript-neutral. The
  project must measure transcript text and error rates under the same decoder
  stack before using it.

## Target-speaker extraction and overlap

### WeSep

- Sources: [WeSep GitHub](https://github.com/wenet-e2e/wesep), [WeSep paper,
  Interspeech 2024](https://arxiv.org/abs/2409.15799)
- Supports: target-speaker extraction with enrollment, dynamic mixture
  simulation, multiple front-end models, and deployment-oriented code.
- Relevant task: `#25` and the now-closed local TSE screen.
- Local conclusion: the English BSRNN/ECAPA checkpoint passed an additive-mixture
  mechanism screen but did not demonstrate recovery in the tiny natural-overlap
  audit. See `docs/reports/2026-08-16-tse-overlap.md` and ledger artifact
  `artifact-wesep-bsrnn-ecapa-vox1`.
- Does not establish: Greek, reverberant, far-field transfer; adequate speaker
  enrollment coverage; or a production detector-plus-extractor pipeline.

### pyannote.audio and overlap-aware diarization

- Sources: [pyannote.audio GitHub](https://github.com/pyannote/pyannote-audio),
  [speaker diarization pipeline source](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/pipelines/speaker_diarization.py),
  [pyannote.audio paper](https://arxiv.org/abs/1911.01255)
- Supports: open-source building blocks for speech activity, speaker change,
  speaker embeddings and overlapped-speech detection; the pipeline source makes
  overlap handling and embedding exclusion explicit parameters.
- Relevant tasks: `#25` and any future detector gate around TSE.
- Reproduction: pin the pipeline/model revision, verify the license and model
  access token requirements, and score overlap and non-overlap separately.
- Does not establish: that diarization correctness implies ASR fidelity or that
  an overlap detector's alerts are precise enough for a TSE front end.

### pyannote.metrics

- Sources: [principles and evaluation-map documentation](https://pyannote.github.io/pyannote-metrics/basics.html),
  [diarization implementation](https://pyannote.github.io/pyannote-metrics/_modules/pyannote/metrics/diarization.html),
  [Interspeech 2017 paper](https://www.isca-archive.org/interspeech_2017/bredin17_interspeech.html)
- Supports: overlap-aware diarization metrics and an evaluation map for scoring
  only annotated regions. The diarization implementation keeps overlap in the
  default scoring path unless `skip_overlap` is explicitly enabled.
- Relevant task: `#7` and `#25`.
- Reproduction: use region maps and frozen annotation hashes; report whether
  overlap is included or excluded rather than relying on a default.
- Does not establish: a single correct WER for simultaneous speech. Speaker
  metrics and fidelity-to-audio ASR metrics answer different questions.

### LibriCSS continuous overlapping speech

- Sources: [Microsoft Research paper/PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/04/ICASSP2020__Continuous_speech_separation__dataset_and_analysis.pdf),
  [LibriCSS overlap evaluation reference](https://www.isca-archive.org/interspeech_2022/kanda22_interspeech.pdf)
- Supports: a primary benchmark design for continuous speech separation and
  overlapped-speech recognition, including overlap ratios and separated-stream
  evaluation.
- Relevant task: `#25` as an external design reference for future natural-overlap
  evaluation.
- Reproduction: LibriCSS is English and derived from LibriSpeech; use its
  protocol concepts, not its WER as a Greek council expectation.
- Does not establish: performance on reverberant Greek council audio or the
  value of a target-speaker enrollment from a single same-cell segment.

## Confidence and selective decisions

### Word confidence is an implementation output, not calibrated truth

- Sources: [faster-whisper `Word` and transcription source](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py),
  [OpenAI Whisper timing source](https://github.com/openai/whisper/blob/main/whisper/timing.py),
  [OpenAI Whisper decoding source](https://github.com/openai/whisper/blob/main/whisper/decoding.py)
- Supports: faster-whisper exposes per-word `probability` when word timestamps
  are requested; Whisper's alignment uses cross-attention/timing machinery and
  token probabilities. Segment-level `avg_logprob` and `no_speech_prob` are
  separate signals.
- Relevant task: `#13`.
- Reproduction: compare `word_timestamps=False` and `True` under the exact same
  decoder settings. The alignment path can alter the returned segmentation and
  must not be assumed transcript-neutral; the local 247-window run is the right
  check.
- Does not establish: calibration, deletion detection, or transfer from emitted
  token confidence to words the decoder never emits.

### Confidence estimation papers

- Sources: [Learning word-level confidence for subword E2E ASR](https://arxiv.org/abs/2103.06716),
  [evaluation of word-level confidence for E2E ASR](https://arxiv.org/abs/2101.05525),
  [Adopting Whisper for Confidence Estimation](https://arxiv.org/abs/2502.13446),
  [calibrating overconfidence in noisy speech recognition](https://arxiv.org/abs/2509.07195)
- Supports: confidence estimation is a separate prediction/calibration problem;
  raw decoder scores can be overconfident in noise and post-hoc calibration is
  a documented research direction.
- Relevant task: `#13` and the evaluation-harness decision in `#7`.
- Reproduction: calibrate on a human-audited, non-sealed development partition;
  report reliability diagrams/ECE or equivalent, AUROC for emitted-word errors,
  and a separate deletion analysis.
- Does not establish: that a threshold selected on emitted-word errors can find
  silent omissions or that confidence can safely drive fusion without a full
  paired gate.

## Evaluation design and leakage controls

### Source-level controls

- [OpenAI Whisper source](https://github.com/openai/whisper) and
  [CTranslate2/faster-whisper integration](https://github.com/OpenNMT/CTranslate2)
  are the relevant serving references. Pin commits and expose every decoder
  option used by an arm.
- [pyannote.metrics evaluation maps](https://pyannote.github.io/pyannote-metrics/basics.html)
  provide a useful pattern for region-restricted scoring: bind the scored region
  to an immutable annotation/hash instead of slicing after seeing hypotheses.
- [LibriCSS continuous-speech evaluation](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/04/ICASSP2020__Continuous_speech_separation__dataset_and_analysis.pdf)
  provides a precedent for overlap-dose strata and stream-aware reporting.

### Local rules to preserve

1. Keep fidelity-to-audio and agreement-with-OpenCouncil as separate metrics.
2. Report S/D/I and deletion rate separately; never let a lower WER hide a
   deletion increase.
3. Freeze decode config, scored regions, and selection rules before reading an
   outcome. Report single-item and single-meeting domination.
4. Keep the seven sealed temporal holdout windows sealed and keep audio/transcript
   text under `~/.cache/oc-public/`.
5. Treat external benchmarks as mechanism or protocol references until a
   same-stack Greek council replication exists.

## Training decision boundary

Map ticket `#9` is a human decision about whether to reopen training, not an
unanswered literature lookup. This resource pass does not recommend a new GPU
run, a new dataset, or a new architecture. The negative single-seed screens,
the shared training freeze, artifact validity records, and the three-seed
preregistration remain the decision boundary. External papers can inform a
future preregistration only after the user explicitly reopens that branch.

## Three cheap next checks

These are proposals for the research ticket, not approvals to run them.

1. **Name-repair shadow contract (`#6`):** use the existing hash-frozen roster
   files and a small untouched-meeting smoke set to compare the current post-hoc
   repair against one explicitly preregistered `hotwords`/`initial_prompt` arm,
   with false-positive names, substitutions, insertions and deletions reported
   separately. No sealed holdout and no new model.
2. **Confidence contract (`#13`):** finish the already-running local paired
   decode, then calibrate only on the existing human-audited gold development
   material. Report emitted-word calibration and a separate omission result;
   do not promote a confidence threshold from AUROC alone.
3. **Overlap harness contract (`#7`/`#25`):** add an immutable region-map/resource
   schema and a small non-sealed regression fixture that reports overlap/non-
   overlap fidelity, S/D/I, deletion rate, enrollment coverage and detector
   exposure. Do not aggregate the current three-case TSE substrate.

No GPU, paid API, sealed-window decode, or new training run is implied by these
checks.

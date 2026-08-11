# Search prompt: πώς αλλιώς να ρίξουμε το WER

Δώσε το παρακάτω αυτούσιο σε έναν research agent. Είναι γραμμένο ώστε να μην μπορεί να
απαντήσει με γενικότητες: κάθε ισχυρισμός του πρέπει να προσγειώνεται σε νούμερα που
ήδη έχουμε.

---

I need concrete, literature-grounded options for improving a Greek ASR system. Do not
give me generic advice — I have specific measurements and I need you to reason against
them. Cite real papers, systems, or repos, and say plainly when something is untested
for Greek or for this setting.

## The system

`openai/whisper-large-v3` + LoRA (r=32, α=64, q_proj/v_proj only, encoder frozen,
lr 1e-4, 2 epochs), fine-tuned on Greek municipal-council speech. Served with
faster-whisper / CTranslate2 int8.

## The data

22.5 hours of training audio, 29k clips, **mean clip length 2.79 seconds**, 209
meetings, 291 speakers, 9 cities, Aug 2024 – May 2026.

Half of it (11.16h) is human-corrected utterances. **The other half (11.32h) is
"no_edit" rows: utterances a human reviewer chose not to touch, whose labels are
therefore the unverified output of an older ASR pipeline.** So half the supervision may
be teaching the model to reproduce another system's errors.

Because clips average 2.79s inside Whisper's fixed 30s encoder window, **~91% of the
encoder compute is padding**.

## What has already been measured — do not propose these again without a new argument

- **Packing short clips into ~30s windows made it worse**, and adding Whisper timestamp
  supervision made it worse still (deletions nearly doubled).
- **Mixture ratio (20/80 vs 50/50 corrections-to-backbone) is indistinguishable**:
  −0.24 points, 90% CI [−0.89, +0.36]. **Per-seed variance is 2.1 WER points**, ten
  times the effect being chased.
- **Prompt-based contextual biasing with the speaker roster is saturated**: giving the
  model an oracle list of exactly the names spoken buys **zero** extra name recall over
  dumping the whole roster (36.0% both). The decoder cannot produce 64% of these names
  even when told the answer.
- **An LLM post-editor damages ~1 in 6 already-correct utterances**; break-even needs
  20–24% of utterances to contain a real error, the actual rate is 27%, so the margin is
  +0.2 points and 44% of meetings fall the wrong side. Rejected as a blanket component.
- **A free 3-way consensus vote across ASR systems beats the best single system by 1.1
  points.** Not yet productised.
- Second training epoch bought 0.16 points over the first. Essentially nothing.

## Where the errors actually are

Against ElevenLabs Scribe v2 on 39 held-out windows (11,903 words), our error profile
versus theirs:

| category | ours | Scribe | diff |
|---|---|---|---|
| inflection / near-miss spelling | 355 | 209 | **+146** |
| **deletions** | 731 | 604 | **+127** |
| phonetic word substitution | 216 | 122 | +94 |
| homophones (η/ι/υ/ει/οι, ω/ο, αι/ε) | 89 | 51 | +38 |
| numbers / dates | 57 | 162 | −105 |
| insertions | 221 | 386 | −165 |

Our deletion-to-insertion ratio is **3.3:1**; Scribe's is 1.6:1. Qualitatively the
deletions are: truncated tails of roll-call lists, whole skipped sentences next to
decoder degeneration (foreign-token garbage), and dropped short speaker turns
("Ευχαριστώ", "Κύριε X").

Our advantage in the numbers/insertions rows is a **transcription convention**, not a
skill: the reference writes digits and so do we, while Scribe spells numbers out and
pays for it. Neutralise that and our real recognition deficit is **~3.5 points, not the
1.0 the headline shows**.

## The evaluation is itself suspect

The benchmark reference is the project's own published transcript, produced by an
earlier ASR pipeline and then human-corrected. An independent listening audit found
**17.5% disagreement with what a careful listener hears**, and **70.5% of words the
reference omits were actually spoken**. A system that transcribes more faithfully is
penalised.

## What I want from you

1. **The deletion problem is the biggest single lever (+127 errors, 3.3:1 ratio).** What
   is actually known about Whisper dropping speech — segment abandonment, timestamp-driven
   skipping, no-speech thresholds, long-form chunking strategies, attention collapse at
   chunk boundaries? Which specific decoding or architectural interventions have published
   evidence, and what did they measure?

2. **Half our labels are another system's output.** What does the literature say about
   training on pseudo-labels of this kind — noisy-student, confidence filtering, label
   smoothing, dropping unverified targets entirely? Given only 11.16h of genuinely
   human-verified audio, is training on the verified half alone likely to beat training on
   both? What evidence exists either way at this data scale?

3. **91% padding.** Are there Whisper fine-tuning approaches that avoid the fixed 30s
   window cost without the packing failure we already measured? Anything on variable-length
   encoders, attention masking of padding, or curriculum by clip length?

4. **Biasing is saturated at the prompt level.** What stronger mechanisms exist for
   injecting a known vocabulary into an ASR decode — shallow fusion, on-the-fly LM
   rescoring, TCPGen or similar copy mechanisms, keyword boosting in CTC/transducer
   architectures? Which of these are available for a Whisper-family encoder-decoder in
   practice, and which would require leaving Whisper?

5. **Greek specifically.** Homophone confusion (η/ι/υ/ει/οι) and rich inflection are
   visible in our errors. What Greek or morphologically-rich-language ASR work addresses
   this — subword vocabulary choices, morphological post-processing, Greek LM rescoring?
   What Greek speech corpora exist beyond CommonVoice that could add hours?

6. **Given 2.1 points of per-seed variance and ~12,000 words of clean evaluation, most
   effects are unresolvable.** What evaluation design would let us resolve a 0.5-point
   improvement without collecting enormous new labelled data? Paired designs, variance
   reduction, matched decoding, bootstrap over what unit?

For each answer: say what the expected effect size is, what it would cost in GPU or human
hours, and what would have to be true for it to fail. **Rank your recommendations, and
say explicitly which of my six questions is a dead end.** I would rather hear "this is
not worth doing" than a hedged list.

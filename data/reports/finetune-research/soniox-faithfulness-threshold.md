# Soniox faithfulness filter: metric, normalization, and threshold

_Research note for auto-selecting a next training batch from the ~287k Greek council-ASR corrections._
_Date: 2026-06-30. Sources via Grok live-search; URLs inline. Citation arrays came back empty on every call, so URLs below are the ones Grok embedded in its answers — treat the load-bearing ones as "verify before betting the pipeline on it" (flagged in Open Risks)._

## TL;DR — concrete recommendation

Re-transcribing with Soniox and measuring distance to the corrected text **is a sound idea, but a single absolute ~15% tolerance is the wrong knob** — it conflates bad labels with hard cases and is too loose to catch the bad ones reliably.

Do this instead:

1. **Metric: CER, not WER.** On short utterances WER is high-variance (one word error swings it wildly) and word-boundary sensitive. CER (normalized character Levenshtein) penalizes proportionally and is the better faithfulness gauge for morphologically rich Greek. Report WER too, but **gate on CER**.
2. **Normalize hard before scoring** (Greek-specific, see Q3): NFC, lowercase, strip punctuation, strip tonos, unify final sigma ς→σ, canonicalize numerals, collapse whitespace. Apply the *same* normalizer to `before_text`, `after_text`, and the Soniox output.
3. **Make the threshold RELATIVE, not absolute.** Compute both `CER(Soniox, after)` and `CER(before, after)`. The decision rule that actually separates "label drifted from audio" from "normal ASR noise + legit human edit" is the *comparison between the two ASRs vs. the human text*, not an absolute number. (Detailed rule in Q4/Q5.)
4. **Concrete starting gate:** keep if `CER(Soniox, after) ≤ ~0.10` **AND** `CER(Soniox, after) ≤ ~1.5–2× CER(before, after)`. Treat **low `CER(before, after)` + high `CER(Soniox, after)`** as the prime *bad-label* signal (the original ASR was already close to the human text, yet a second independent ASR is far → the human text likely drifted or the audio span is mismatched). Treat **both distances high** as the *hard-case keep* signal.
5. **Calibrate empirically.** Hand-audit 100–500 examples, plot the CER distributions, set the threshold where good/bad separate (ROC-style). Don't ship the 0.10 number untested.

On the user's specific question: **~15% is too loose for catching bad labels and is also the wrong shape (absolute).** At a ~13–15% WER / ~5–8% CER error floor on this corpus, a 15%-WER-equivalent tolerance mostly only flags egregious mismatches while admitting plenty of drift. Tighten to ~6–10% CER absolute *and* layer the relative-to-`before_text` rule on top.

---

## Q1 — Realistic Greek ASR error rates

Greek is a mid-resource language; strong systems land in the **low-to-mid teens WER** on read/formal benchmarks, higher on spontaneous/noisy real-world audio. CER runs roughly **0.4–0.6× WER**.

- **Common Voice Greek (el):** Whisper large-v3 zero-shot ≈ **13.7–14.0% WER**; fine-tuned Greek XLS-R ≈ **11.6% WER**; domain fine-tuning (Greek Podcast Corpus) pushes medium models to ~12.8% WER. (Greek Podcast Corpus paper, arXiv 2406.15284: https://arxiv.org/html/2406.15284v1)
- **FLEURS Greek (el):** Whisper large-v3 ≈ **11.0% WER** (large-v2 ≈ 12.95%). Speechmatics claims ~6.6% WER on FLEURS Greek. (GPC paper above; Speechmatics Greek page: https://www.speechmatics.com/speech-to-text/greek)
- **Dialectal / spontaneous Greek:** much worse — 28%+ WER even after fine-tuning, up to ~100% on distant varieties (Griko). Real-world conversational/meeting register sits **above** the clean-benchmark numbers. (Vakirtzian et al., Interspeech 2024: https://par.nsf.gov/servlets/purl/10561636)
- **Soniox:** vendor self-reports very low "semantic WER" on real-world Greek (~1.25% in their own comparison) vs. Whisper/OpenAI ~3%+. Treat vendor numbers as marketing, not benchmark. (https://soniox.com/compare/soniox-vs-openai/greek)
- **CER:** less often reported but typically **~5–8%** for good Greek models at these WERs.

**Implication for the filter:** the corpus's measured ~13–15% WER (Scribe v2 13.4%, Gladia 14.7%, whisper-large-v3 15.0%, Soniox between Gladia and Scribe) is exactly the expected SOTA-ish floor. So a *correct* human label will routinely sit ~5–8% CER away from a fresh Soniox transcript **purely from normal ASR error** — your threshold must be above that noise floor to avoid nuking good data, but a flat 15% sits too far above it to catch drift.

## Q2 — Is ASR-vs-reference distance an established data-selection / label-quality method?

Yes — it's mature, mostly under **pseudo-label filtering** and **noisy-label / data-curation** for semi-supervised ASR. The direct "run an ASR, compare to human reference, filter" framing is exactly label-quality auditing.

Established techniques and the thresholds papers actually use:

- **Confidence-based pseudo-label filtering** (self-training / IPL with Whisper, wav2vec2/XLSR, Zipformer): drop low-confidence utterances; common sequence-level cutoffs ~**0.8**, stricter ~**0.95**, often relaxed over iterations. (Iterative Pseudo-Labeling, Interspeech 2020: http://www.interspeech2020.org/uploadfile/pdf/Mon-2-11-9.pdf)
- **Inter-ASR agreement / consensus:** run multiple models, keep where pairwise **CER is low**; filter high-disagreement. Used to shrink huge pseudo-sets (e.g., 7500h → ~100h, keeping ~1–5%) with comparable fine-tuning gains. (Efficient Data Selection, arXiv 2506.03681: https://arxiv.org/html/2506.03681v1 ; Interspeech 2025: https://www.isca-archive.org/interspeech_2025/rangappa25_interspeech.pdf)
- **Predicted-WER selection:** train a classifier on acoustic+text embeddings to pick "low-WER" segments (threshold often **≤50% WER** for the low-WER class).
- **Heuristic hallucination filters:** word-rate (words/sec), perplexity, compression/length ratio (e.g., length ratio kept in **[0.95, 1.15]**). (Speech Data Selection w/ WhisperX, ICASSP 2025: https://publications.idiap.ch/attachments/papers/2025/Rangappa_ICASSP2025_2025.pdf)

So the method is well-grounded; the project's twist (human reference exists, second ASR used as the independent probe) is just label-quality auditing via ASR consistency.

## Q3 — Best metric + required Greek normalization

**Metric:** use **CER (normalized character Levenshtein)** as the gate, especially for short utterances; report WER alongside for diagnostics. WER is coarse and unstable on short spans and sensitive to tokenization; CER is proportional and robust for Greek morphology. (HF Audio Course eval chapter: https://huggingface.co/learn/audio-course/en/chapter5/evaluation)

**Greek normalization (apply identically to before/after/Soniox before scoring):**

- **Unicode NFC** (precomposed) — Greek tonos can live as a combining mark in NFD; NFC makes comparison stable.
- **Lowercase.**
- **Strip punctuation** (consider keeping apostrophe for elision).
- **Strip tonos / diacritics** (Unicode `Mn` category) — monotonic Greek uses tonos for stress; ASR vs. human output is inconsistent about it.
- **Final sigma:** normalize ς → σ (positional variant of the same letter).
- **Final-nu (τελικό ν):** standardize optional/dialectal final ν to one form.
- **Numerals:** canonicalize digits vs. spelled-out ("5" vs "πέντε") — inconsistent handling massively inflates error.
- **Collapse whitespace.**

**Tooling caveats:**
- HF `evaluate` / `jiwer` do **no normalization by default** — you must normalize yourself first.
- OpenAI's **`BasicTextNormalizer`** is *not* Greek-specific: it lowercases, removes punctuation/symbols, drops `\p{Mn}` combining marks, collapses whitespace. It's a reasonable base but can be over-aggressive on Greek; build a small custom Greek layer (sigma/final-nu/numerals) on top. The English normalizer (number expansion, contractions) doesn't apply to Greek. (Whisper paper: https://cdn.openai.com/papers/whisper.pdf ; "What is lost in Normalization?": https://arxiv.org/html/2409.02449)

## Q4 — Disambiguating BAD LABEL from HARD CASE (the core tension)

A large `CER(Soniox, after)` alone is ambiguous. Resolve it with **two-ASR triangulation against the human text** — the relationship between the systems is the signal, not the absolute distance.

**Decision logic:**

| `CER(before, after)` | `CER(Soniox, after)` | Reading | Action |
|---|---|---|---|
| high | high | Both independent ASRs are far from the human text → genuinely hard audio that the human fixed. The high-value training case. | **KEEP** |
| low | high | Original ASR was already close to the human text, yet a *second, independent* ASR lands far away → the human text likely drifted from the audio (or the audio span is mismatched/misaligned). | **DROP / flag for review** |
| low | low | Both ASRs agree with the human text → clean, easy, low training value but safe. | keep (or downweight as easy) |
| high | low | Original ASR was far but Soniox matches the human text → human corrected a single-system error; faithful and useful. | **KEEP** |

The decisive bad-label pattern is **two independent ASRs agreeing with each other but disagreeing with the human label** — strong evidence the label, not the audio, is the outlier. This is the ROVER / multi-system-consensus idea applied to label-error detection (ROVER, Fiscus 1997, still the canonical hypothesis-combination/voting method). Modern work uses multi-ASR fusion + LLM correction for exactly this ("Better Pseudo-labeling with Multi-ASR Fusion and Error Correction by SpeechLLM", Prakash et al. 2025; HTEC / human-transcription-error-correction lines of work).

**Extra signals to layer in (cheap wins first):**
- **Soniox word/utterance confidence:** high confidence + far from human text ⇒ lean bad-label; low confidence + far ⇒ lean hard-case.
- **Word-rate / duration sanity:** catches hallucinations and span-misalignment (the most common silent source of "bad labels" in clip-extracted corpora — wrong audio span, not wrong transcription).
- **Semantic / LLM-judge check** on the high-CER tail to separate meaning-preserving paraphrase from real drift (WER/CER over-penalize legit rephrasing).

## Q5 — Is ~15% sound? What the threshold should actually be

**Too loose, and wrong shape.** Given a ~13–15% WER / ~5–8% CER floor on this corpus:

- A 15%-WER-equivalent tolerance (~5–8%+ CER) sits *at or barely above the normal ASR noise floor*. As an absolute keep-gate it mostly only rejects egregious >20–30% mismatches, letting routine drift through.
- Tighten the **absolute** component to **~6–10% CER** for the second-ASR-vs-human check.
- **Prefer the relative rule** (independently surfaced across two research rounds): keep if `CER(Soniox, after) ≤ ~1.5–2× CER(before, after)` **and** absolute `CER(Soniox, after) ≤ ~0.10`. This auto-adapts to utterance difficulty — hard spans have a naturally higher baseline, so a flat absolute cut would wrongly discard exactly the hard cases you want.
- Calibrate the actual numbers per-corpus: sample 100–500, compute good-vs-suspected-bad CER distributions, pick the separating point (quantile / ROC), then freeze.

(Whisper Greek WER/CER context and threshold practice: arXiv 2406.15284; data-filtering CER cutoffs ~5–10% as the common sweet spot: arXiv 2506.03681.)

---

## Open risks / what to validate empirically

1. **Empty citation arrays.** Grok returned answers with inline URLs but no structured citations. Spot-check the load-bearing links before relying on them — especially the Greek WER numbers (arXiv 2406.15284), the data-selection thresholds (arXiv 2506.03681), and the ROVER/multi-ASR-consensus claim. The *method-level* conclusions are well-supported and internally consistent across independent queries; the *exact percentages* are the part to verify.
2. **Span misalignment is the real bad-label generator.** In a clip-extracted corpus, the most likely cause of a huge Soniox-vs-human distance is the audio clip not matching the utterance text (off-by-one utterance, padding, overlap), not a transcription error. Add a duration/word-rate guard and inspect the high-CER tail manually before trusting the CER gate.
3. **The relative rule degenerates when `CER(before, after) ≈ 0`.** If the human barely edited the original ASR, `1.5–2×` of near-zero is near-zero and will drop legit examples. Floor the denominator (e.g., use `max(CER(before,after), 0.03)`) or fall back to the absolute gate when `before≈after`.
4. **Soniox isn't fully independent of the corpus's original ASR.** If `before_text` was produced by a system sharing training data / architecture lineage with Soniox, "two independent ASRs agree" is weaker evidence. Confirm the original ASR provider differs enough; otherwise add a third, architecturally different system (e.g., whisper-large-v3 local) for triangulation.
5. **Normalization can hide or fabricate errors.** Over-aggressive tonos/numeral stripping can mask real errors (lower CER artificially) or, if applied inconsistently between texts, inflate it. Unit-test the Greek normalizer on a handful of known pairs.
6. **CER threshold ≠ training value.** The faithfulness gate only ensures the label matches the audio; it says nothing about whether the example is *informative*. Pair it with a difficulty/diversity selector (e.g., high `CER(before, after)` = high-edit = informative) so you keep faithful-AND-hard examples, not faithful-AND-trivial ones.

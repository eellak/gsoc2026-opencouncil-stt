# Full fine-tune run: whisper-large-v3 Greek council LoRA (2026-07-23)

## Goal

Run the full LoRA fine-tune of `whisper-large-v3` on the combined ~28.6h Greek
municipal-council dataset (human corrections plus an un-edited backbone of the
same meetings) on a cheap community RunPod pod. Keep enough safety scaffolding
(checkpoints, resume, off-pod backups, a watchdog) that a mid-run interruption
would not cost the run. Then measure WER against the zero-shot baseline and ship
the model.

## Configuration

- Base: `openai/whisper-large-v3`, encoder frozen.
- Adapter: LoRA `r=32`, `alpha=64`, `dropout=0.05`, targets `q_proj`, `v_proj`.
- 2 epochs, lr 1e-4, fp16, seed 13, gradient checkpointing.
- Data: `train=28,967` clips, `val_corr=3,157`, `val_reg=4,722`. Held-out
  validation cities: argos, orestiada.
- Checkpoints every 400 steps, per-epoch generate-eval, resume-from-checkpoint.

## The bug that kept killing the run

Before this run finished, training died twice at almost exactly Map 79% (index
~22,991/28,967) with no traceback. The earlier read was that the community pod had
frozen or been throttled. That was wrong.

The process was dead, not frozen: `ps` showed no python process at all. The clips
around that index all read and feature-extracted fine in under 8s. Disk had 75GB
free. The real tell was `cat /sys/fs/cgroup/memory.max`, which reported ~54GB, the
container's actual memory cap, not the 220GB that `free` showed. The `.map()` that
builds Whisper features was accumulating them in RAM. Large-v3 features are
128x3000 float32, about 1.5MB each, and roughly 29k of them cross 54GB right
around 79% of the map. The kernel OOM-killed the process silently.

The fix was to stream the feature cache to disk instead of holding it in memory:
`keep_in_memory=False`, an explicit on-disk `cache_file_name`, and a small
`writer_batch_size`. Confirmation that it worked: the on-disk feature cache grew
to 44GB while RAM stayed clean, and the map crossed 79% into training for the
first time.

## Results

Scored against held-out human-corrected utterances. `val_corr` is the hard set,
utterances a human actually corrected. `val_reg` is a regression check on general
speech, to confirm the model does not forget everyday Greek. Deltas are against
the un-adapted `whisper-large-v3` baseline scored the same way. Lower is better.

| Set | n | WER | WER (norm) | CER |
|---|---|---|---|---|
| val_corr (corrections) | 3,157 | 37.74 (−17.9) | 29.35 (−17.6) | 17.46 (−15.5) |
| val_reg (general) | 4,722 | 10.46 (−14.3) | 4.87 (−10.4) | 3.43 (−10.9) |

The hard corrections set dropped about 32% relative on WER. General speech
improved rather than regressed, so the risk we were watching for (the model
forgetting ordinary Greek while chasing corrections) did not happen.

Run stats: 7,242 steps, 2 epochs, 8h31m wall-clock on one RTX 3090. The final
adapter passed the acceptance gate and is backed up at
`/home/harold/oc-train-checkpoints/adapter/`. The pod is terminated.

One caveat on comparability: the older baseline in the vault (val_corr WER 33.4)
was measured on a smaller, differently-built val set, so it is not directly
comparable with these numbers. The deltas above use this run's own baseline pass.

## Getting it into the OpenCouncil pipeline

OpenCouncil transcription runs in `schemalabz/opencouncil-tasks`. Its transcribe
task (`src/tasks/transcribe.ts`) calls ElevenLabs Scribe directly through
`scribeTranscriber.transcribe({audioUrl, language})` and returns a `Transcript`
JSON object. There is no provider abstraction: Scribe is imported directly. So
slotting in our model means matching that call shape and output schema.

`eval/oc_inference_harness.py` does that. It loads base plus adapter, transcribes
an audio URL or clip, and emits the exact OpenCouncil `Transcript` / `Utterance` /
`Word` schema, so it is a drop-in alternative to the Scribe transcriber. Three
honest divergences, all flagged in the output:

- `speaker`, `channel`, and `drift` are set to 0. Scribe runs with
  `diarize:false` and speakers are assigned downstream by pyannote. A standalone
  ASR pass has no diarization, so it does not invent speakers.
- `confidence` is a placeholder. Whisper does not expose a calibrated per-word
  confidence the way Scribe returns logprobs, so downstream must not read these as
  Scribe-comparable.
- Utterance segmentation copies Scribe's rules (pause of at least 1s,
  sentence-final punctuation, 30s cap) so downstream sees the same granularity.

On cost: CPU works for a single short clip (free, about 30s of compute for 9s of
audio). For throughput, the same file runs as a RunPod serverless handler
(`handler(event)` at the bottom of the harness), billed per inference-second and
scaling to zero. Nothing starts a paid pod automatically.

End-to-end check on a held-out place-name clip:

| Source | Text |
|---|---|
| Scribe (raw) | ...στο Δήμο **Άγιους Μικυνών** |
| Our model | ...στο Δήμο **Άγριος Μυκεινών** |
| Human reference | ...στο Δήμο **Άργους Μυκηνών** |

The fine-tuned model lands much closer on the second word, one η/ει homophone away
from correct, where Scribe missed both words.

## Model publication

The LoRA adapter is published to HuggingFace as a public model repo,
[`opencouncil/whisper-large-v3-el-council-lora`](https://huggingface.co/opencouncil/whisper-large-v3-el-council-lora).
The model card carries the base model, LoRA config, the results above, an honest
training-data description (the dataset stays private under GDPR/DPO legal hold),
intended use, and a note on the low but nonzero PII-memorization risk of LoRA. The
dataset legal hold applies only to the data. The model is a separate artifact:
the repo holds the LoRA adapter weights plus tokenizer and processor config, and
no dataset or raw audio. The repo was pushed private first, then flipped to public
once confirmed it holds no dataset or raw personal data.

## Next steps

- Optionally wrap the harness as a serverless endpoint and A/B it against Scribe
  on the production bench (`bench.opencouncil.gr`).
- If per-word confidence parity with Scribe is needed later, add a generate-scores
  pass. Deferred, since the drop-in does not need it.

# Audio Segmentation Plan

The clean correction data can become supervised ASR examples if each corrected utterance is aligned to the right audio span.

Data source: [`data/clean/corrections_clean.csv`](../../data/clean/corrections_clean.csv).

## Available Fields

- `audio_url`: source audio file.
- `utterance_start`: start time in seconds.
- `utterance_end`: end time in seconds.
- `duration_seconds`: derived duration.
- `before_text`: current ASR or pre-correction transcript.
- `after_text`: corrected reference transcript.
- `edited_by`: `task` for the current LLM correction stage, `user` for human intervention.

## Fine-Tuning Example Shape

For Whisper-style fine-tuning, the candidate example is:

```json
{
  "audio": "local clipped audio segment",
  "text": "after_text",
  "metadata": {
    "edit_id": "...",
    "audio_url": "...",
    "start": 0.0,
    "end": 1.0,
    "edited_by": "user|task",
    "heuristic_route": "asr_finetune"
  }
}
```

## FFmpeg Extraction

The basic extraction command should be shaped like:

```bash
ffmpeg -ss START_SECONDS -to END_SECONDS -i SOURCE_AUDIO -ac 1 -ar 16000 output.wav
```

For many rows from the same `audio_url`, download/cache each source audio once, then extract all segments from the local file. Do not repeatedly fetch the same MP3 for every utterance.

## Quality Gates

Before training:

- Reject rows with missing/invalid `audio_url`, timestamps, or `after_text`.
- Flag durations below 0.5s unless the text is genuinely very short.
- Flag durations above 30s for manual alignment review.
- Prefer `heuristic_route == asr_finetune` for initial training candidates.
- Keep `rule_based` rows out of ASR training unless evaluating punctuation behavior explicitly.

## Longer Segment Strategy

The proposal notes that short utterance fine-tuning can hurt long-form transcription. The next dataset step should group adjacent utterances from the same audio file into synthetic 2-5 minute windows, preserving timestamps and using the corrected text sequence as reference.

Open decision: whether to train first on individual utterances for a smoke test, then move to concatenated windows for the real experiment.

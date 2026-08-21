# Dense-window 300-step screen

Frozen before creating the GPU pod or training any arm.

- A: current single-utterance control parquet.
- B: the exact same original rows in the already-built dense packs, `PACK_ARM=pn`.
- Seeds and order: A13, B13, A29, B29, A47, B47, sequentially on one pod.
- Same base, LoRA recipe, LR, effective batch 8 and 300 optimizer steps. The VFM
  16 GB A4000 uses micro-batch 1 × gradient accumulation 8 instead of 2 × 4; both
  arms use the same setting. Shared preprocessing caches change no weights.
- Because one dense example contains several original utterances, fixed optimizer steps
  deliberately give B more labelled speech per step. This screen tests packing under a
  fixed update/example budget; it does not isolate context shape from useful-token
  density and must not be described as doing so.
- Primary endpoint: paired WER on the frozen 39 validation windows, decoded for all six
  adapters on one GPU float16 faster-whisper stack with the frozen served config.
- Report training WER on the existing frozen 300-row sample as diagnostic only.
- Promotion requires every gate in `docs/decisions/training-evidence.md`: mean ΔWER<0,
  at least 2/3 negative, mean deletion delta<=0, insertion delta<+0.0005, no
  leave-one-window-out sign reversal, and no window above 25% of net gain.
- The seven temporal holdouts remain sealed. No medium/full stage follows automatically.

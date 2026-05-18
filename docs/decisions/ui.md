# UI decisions

Exploration-vs-training stance and prototype UI choices.

## Accepted

### 2026-05-12 - Exploration before training

We will not start fine-tuning yet. First we need to understand the correction dataset and build tooling to inspect corrections with audio and context.

Reason: training on noisy or poorly understood correction pairs can optimize for the wrong target.

### 2026-05-15 - Waveform bars + peaks-cache prefetch

Switched wavesurfer.js rendering to bar mode (`barWidth:2, barGap:1, barRadius:2, normalize:true`) to eliminate the "double shape / square wave" visual artifact caused by the filled polygon rendering two mirrored shapes.

Added a module-level peaks cache (`ui/src/lib/domain/peaks-cache.ts`) that pre-decodes neighbor audio via OfflineAudioContext. The next item's waveform renders without decode lag.

Added `data-sveltekit-preload-data="eager"` on the "next item" link so SvelteKit fetches the next page's server data as soon as the current page mounts (not on hover).

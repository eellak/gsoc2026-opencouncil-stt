# Self-hosted ASR endpoint

The fine-tuned Greek `whisper-large-v3` is served as an HTTP endpoint at
`https://asr.haroldpoi.dev`. It runs on the mini-PC (CPU only) via faster-whisper
and is exposed through a Cloudflare Tunnel. The response matches the OpenCouncil
tasks-server `Transcript` schema, so it can stand in for the ElevenLabs Scribe
transcriber.

Code and the full operational README live in [serve/oc-asr/](../../serve/oc-asr/).
This note is the short version: how to call it.

## Auth

Every request except `/health` needs the API key. Send it as `X-API-Key`, or as
`Authorization: Bearer <key>` on the OpenAI-compatible route. The key lives on
the mini-PC in `~/oc-asr-serve/api_key.txt` — it is a secret, not in this repo.

```bash
KEY=$(ssh minipc cat '~/oc-asr-serve/api_key.txt')
```

## Endpoints

| Route | Body | Returns |
| --- | --- | --- |
| `GET /health` | — (no key) | liveness |
| `POST /transcribe` | JSON `{"audioUrl": "...", "language": "el"}` | `Transcript` |
| `POST /transcribe/upload` | multipart `file` | `Transcript` |
| `POST /v1/audio/transcriptions` | multipart `file`, `model`, `language` | `{"text": ...}` |

By URL — this is how OpenCouncil would call it, with a pre-cut segment URL, not
a whole meeting:

```bash
curl -X POST https://asr.haroldpoi.dev/transcribe \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"audioUrl":"https://data.opencouncil.gr/audio/<segment>.mp3","language":"el"}'
```

By upload:

```bash
curl -X POST https://asr.haroldpoi.dev/transcribe/upload \
  -H "X-API-Key: $KEY" -F "file=@clip.wav"
```

The OpenAI-compatible route exists so tools that speak the OpenAI STT API can
use this as a provider — that is how the model registers in the OpenCouncil
benchmark, as an `openai-compatible` provider pointing at
`https://asr.haroldpoi.dev/v1`.

## What comes back

A `Transcript` object: `metadata` plus `transcription` with `full_transcript`
and `utterances`. Each word carries a real faster-whisper probability.
`speaker`, `channel`, and `drift` are `0` — diarization happens downstream
(pyannote), not here.

Send already-segmented audio. The server transcribes the whole file it receives;
it does not split. Upload cap is 500 MB (`OC_ASR_MAX_BYTES`).

## Operating it

Two systemd user services on the mini-PC, both `Restart=always` with lingering
on, so they survive crashes and reboots:

```bash
systemctl --user status oc-asr oc-asr-tunnel
systemctl --user restart oc-asr oc-asr-tunnel
journalctl --user -u oc-asr -f
```

Key rotation, model rebuild (`build_model.sh`), and the Docker Compose
alternative are in [serve/oc-asr/README.md](../../serve/oc-asr/README.md).

## Speed

CPU only, so this is for async/batch transcription, not real-time: a ~9s clip
takes about 6s. A GPU host would be much faster if throughput ever matters.

## Serving on RunPod (GPU)

The mini-PC is fine for one-off clips but too slow behind Cloudflare for long
benchmark clips (multi-minute clips hit the ~100s edge timeout and return 502/524).
Two GPU options on RunPod, both serving the same model.

### Serverless endpoint (pay-per-use, scale-to-zero)

A live RunPod Serverless endpoint runs the same model and bills only for the seconds
a GPU spends transcribing, nothing while idle. Source and build (GitHub Actions ->
`ghcr.io`) are in [github.com/angelospk/oc-asr-serverless](https://github.com/angelospk/oc-asr-serverless).

- Endpoint ID: `o1jda6sxo85dnk`, 24GB Ampere pool, workers min 0 / max 1.
- Auth is the RunPod account key (`~/.runpod/config.toml`), not the mini-PC key.
- Compute type is `float16`; `int8_float16` throws `CUBLAS_STATUS_NOT_SUPPORTED` on
  the serverless CUDA image.

```bash
RUNPOD_API_KEY=$(grep apikey ~/.runpod/config.toml | sed "s/.*'\(.*\)'.*/\1/")
curl -X POST https://api.runpod.ai/v2/o1jda6sxo85dnk/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"audioUrl":"https://data.opencouncil.gr/audio/<segment>.mp3","language":"el"}}'
```

The response `output` is the same `Transcript` object. Cold start adds ~20-40s to
wake a worker from zero; warm workers respond immediately.

**This serverless endpoint cannot be a benchmark provider as-is.** The benchmark
(`bench.opencouncil.gr`) only reaches a model through an `openai-compatible`,
`hf-endpoint`, or `huggingface` provider. RunPod Serverless speaks its own
`POST /v2/<id>/run(sync)` API with the input wrapped in `{"input": {...}}` and the
RunPod account key, and replies with a `{"status", "output", ...}` envelope instead
of the OpenAI transcription shape. So the benchmark cannot call it directly. Use the
temporary pod below for benchmark runs, or put a small `openai-compatible` shim in
front of the serverless endpoint.

### Temporary GPU pod (for benchmark re-runs)

For a benchmark run, spin up a RunPod GPU pod (not serverless) running this same
`oc_asr_server.py` with `OC_ASR_DEVICE=cuda`. It exposes the `openai-compatible`
`/v1/audio/transcriptions` route the benchmark expects, so it registers as a normal
provider. A full 260-clip run is roughly 1.5h of pod time (~$0.35 on a community
RTX 3090). The pod bills continuously while it exists, so terminate it right after
the run (`runpodctl remove pod <id>`).

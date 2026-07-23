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

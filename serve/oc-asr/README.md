# OpenCouncil self-hosted ASR

This folder is a self-contained setup that serves the fine-tuned Greek
`whisper-large-v3` model as an HTTP endpoint. It runs on the mini-PC (CPU only),
uses faster-whisper for speed, and is exposed at `https://asr.haroldpoi.dev`
through a Cloudflare Tunnel. The response matches the OpenCouncil tasks-server
`Transcript` schema, so it can stand in for the ElevenLabs Scribe transcriber.

## What's here

- `oc_asr_server.py` — the FastAPI server.
- `run-oc-asr.sh` — start the server by hand.
- `build_model.sh` — rebuild the serving model from the adapter.
- `requirements.txt` — Python deps.
- `Dockerfile`, `compose.yml`, `config.docker.yml` — the container option.
- `merged/`, `ct2/` — the model (merged HF weights and the CTranslate2 int8 build).
- `asr.env`, `api_key.txt` — the API key.

This is the copy checked into the vault; it holds the code only. The model
directories and the two key files exist on the mini-PC at `~/oc-asr-serve/` and
are deliberately not synced here (7 GB of weights and a secret). Rebuild the
model with `build_model.sh`, and see "The API key" below for the key.

## Using the endpoint

Every request needs the API key in an `X-API-Key` header.

Transcribe a segment by URL (this is how OpenCouncil would call it, with a
pre-cut segment URL, not a whole meeting):

```bash
KEY=$(cat ~/oc-asr-serve/api_key.txt)
curl -X POST https://asr.haroldpoi.dev/transcribe \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"audioUrl":"https://data.opencouncil.gr/audio/<segment>.mp3","language":"el"}'
```

Or upload a local clip:

```bash
curl -X POST https://asr.haroldpoi.dev/transcribe/upload \
  -H "X-API-Key: $KEY" -F "file=@clip.wav"
```

Health check (no key needed): `curl https://asr.haroldpoi.dev/health`.

There is also an OpenAI-compatible route, `POST /v1/audio/transcriptions`
(multipart `file`, `model`, `language`, returns `{"text": ...}`), so tools that
speak the OpenAI STT API can use this as a provider. It accepts the key as either
`X-API-Key` or `Authorization: Bearer`. This is what registers the model in the
OpenCouncil benchmark as an `openai-compatible` provider pointing at
`https://asr.haroldpoi.dev/v1`.

The reply is the `Transcript` object: `metadata` plus `transcription` with
`full_transcript` and `utterances`. Each word carries a real faster-whisper
probability. `speaker`, `channel`, and `drift` are 0 because diarization happens
downstream (pyannote), not here. Send already-segmented audio: the server
transcribes the whole file it receives.

## The API key

The key is just a random secret string that guards the endpoint. It was created
with:

```bash
python -c "import secrets; print('oc-' + secrets.token_hex(5))"
```

It lives in two files: `api_key.txt` (plain, for you to read) and `asr.env`
(read by the service). Treat it like a password: share it privately, not in a
public repo or chat. To rotate it, put a new value in both files and restart:

```bash
NEW=$(python -c "import secrets; print('oc-'+secrets.token_hex(5))")
printf '%s\n' "$NEW" > ~/oc-asr-serve/api_key.txt
printf 'OC_ASR_API_KEY=%s\nOC_ASR_MODEL_DIR=/home/harold/oc-asr-serve/ct2\n' "$NEW" > ~/oc-asr-serve/asr.env
systemctl --user restart oc-asr
```

## How it stays running

Two systemd user services keep it alive:

- `oc-asr` runs the server.
- `oc-asr-tunnel` runs the Cloudflare tunnel.

Both have `Restart=always`, and user lingering is on, so they come back after a
crash and start automatically when the mini-PC boots. Useful commands:

```bash
systemctl --user status oc-asr oc-asr-tunnel
systemctl --user restart oc-asr oc-asr-tunnel
journalctl --user -u oc-asr -f
```

## Rebuilding the model

If `ct2/` is lost, rebuild it from the public adapter:

```bash
ADAPTER=opencouncil/whisper-large-v3-el-council-lora ./build_model.sh
```

## Container option

If you install Docker later, the same stack runs in containers with the same
auto-restart behavior:

```bash
docker compose up -d --build
```

That starts the server and the tunnel as containers with
`restart: unless-stopped`. If you use containers, you can turn off the systemd
services to avoid running both (`systemctl --user disable --now oc-asr oc-asr-tunnel`).

## Note on speed

The mini-PC is CPU only, so this is meant for async/batch transcription, not
real-time. A ~9s clip transcribes in about 6s. A GPU host would be much faster if
throughput ever matters.

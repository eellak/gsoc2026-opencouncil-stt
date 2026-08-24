# Serverless ASR endpoint

The v2 adapter served on a GPU that scales to zero. Nothing runs, and nothing is
billed, between calls. Cold start is about 10 s once the node has the image;
the first pull of the image took roughly 20 minutes.

- endpoint `o1jda6sxo85dnk`, template `cilsob176z`
- image `ghcr.io/angelospk/oc-asr-serverless@sha256:79a3afef...` (deployed by digest)
- source `github.com/angelospk/oc-asr-serverless`
- key `RUNPOD_API_KEY` in `.env`

## Auth

RunPod's own endpoint key. There is no second application key. A call with no key
and a call with a wrong key both return 401.

## Calling it

Provenance only, no audio, no GPU work worth speaking of. Use this first:

```bash
curl -s -X POST https://api.runpod.ai/v2/o1jda6sxo85dnk/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{"input":{"op":"provenance"}}'
```

A short clip, waiting for the answer. `wait` is in **milliseconds**, maximum
300000:

```bash
curl -s -X POST "https://api.runpod.ai/v2/o1jda6sxo85dnk/runsync?wait=280000" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{"input":{"audioBase64":"<base64 wav>","language":"el","suffix":".wav"}}'
```

A whole meeting. Submit, then poll; a finished async result is kept for 30
minutes, so collect it:

```bash
ID=$(curl -s -X POST https://api.runpod.ai/v2/o1jda6sxo85dnk/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{"input":{"audioUrl":"https://data.opencouncil.gr/audio/<file>.mp3","language":"el"}}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')

curl -s "https://api.runpod.ai/v2/o1jda6sxo85dnk/status/$ID" -H "Authorization: Bearer $RUNPOD_API_KEY"
```

The response is the OpenCouncil `Transcript` schema plus a `provenance` block
naming the base commit, the adapter commit and the CTranslate2 model hash.

## How long a meeting takes

Measured 2026-08-24 on AMPERE_24, beam 2, word timestamps on: **2.1x real time**.
A 50.8-minute meeting took 24.2 minutes of GPU and returned 543 utterances
covering 3046.8 of 3046.9 seconds.

Set `executionTimeoutMs` to at least three times the audio duration. A 2h31m
meeting was killed at a 1-hour limit and the hour was billed for nothing. The
endpoint is now set to 3 hours.

## Changing the model

`saveEndpoint` silently ignores a changed `templateId` on an existing endpoint,
while accepting other fields in the same call. Change the image by mutating the
template instead:

```bash
curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
  -H 'Content-Type: application/json' -d '{"query":"mutation($input: SaveTemplateInput!) { saveTemplate(input: $input) { id imageName } }","variables":{"input":{"id":"cilsob176z","name":"oc-asr-serverless","imageName":"ghcr.io/angelospk/oc-asr-serverless@sha256:<digest>","containerDiskInGb":25,"volumeInGb":0,"dockerArgs":"","isServerless":true,"env":[{"key":"COMPUTE","value":"float16"},{"key":"DEVICE","value":"cuda"}]}}}'
```

Then read the endpoint back and run the provenance call. Deploy by digest, never
by tag: an earlier image floated on an unpinned `main` and served an adapter
nobody can now identify.

## What this endpoint is not

It is not a measurement stack. It runs float16 on CUDA; the canonical evaluation
numbers are CPU int8 on the mini-PC, and the two give different tokens from the
same weights.

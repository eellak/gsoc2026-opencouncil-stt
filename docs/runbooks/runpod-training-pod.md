# Runbook: a RunPod GPU pod for training / A-B runs

For fine-tuning and controlled training experiments. For *serving* the model, see
[self-hosted-asr-endpoint](self-hosted-asr-endpoint.md).

**The pod bills continuously from creation to termination, not per GPU-second.**
Terminate it the moment the run ends: `runpodctl remove pod <id>`.

## Create

```bash
runpodctl create pod --name oc-ab-labelbug \
  --gpuType 'NVIDIA A40' \
  --imageName 'runpod/pytorch:1.1.0-rc.154-cu1290-torch291-ubuntu2404' \
  --containerDiskSize 150 --volumeSize 0 --mem 60 --vcpu 9 \
  --secureCloud --startSSH --ports '22/tcp'
```

- **A40 (48 GB, $0.44/hr)** fits whisper-large-v3 LoRA at batch 2 with gradient
  checkpointing, with room to spare. A 3090 (24 GB) also works and is marginally
  cheaper; an A100 is ~3x the price for maybe 2x the speed.
- **`--volumeSize 0`** — a persistent volume survives termination and keeps billing.
  Container disk dies with the pod, which is what you want for a one-off run. Size it
  for the audio cache: ~3.5 GB per 25 meetings of mp3, plus ~1.5 MB per cached
  large-v3 feature vector.
- **Image tag matters.** `runpod/pytorch:2.4` and older break PEFT with DTensor
  errors. The `cu1290-torch291` tag above works with the pins below. Tags are listed
  at `hub.docker.com/v2/repositories/runpod/pytorch/tags`; the plain `runpod/pytorch:latest`
  style names in older notes no longer resolve.

SSH endpoint (it appears a minute or two after `desiredStatus: RUNNING`; `runtime`
is `null` until then):

```bash
RUNPOD_API_KEY=$(grep apikey ~/.runpod/config.toml | sed "s/.*'\(.*\)'.*/\1/")
curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query":"query{pod(input:{podId:\"<POD_ID>\"}){runtime{ports{ip publicPort privatePort type}}}}"}'
```

Use the `isIpPublic: true` / `type: tcp` entry: `ssh -p <publicPort> root@<ip>`.

## Set up

```bash
apt-get update -qq && apt-get install -y -qq ffmpeg
pip install --break-system-packages -U "numpy<3" \
  "transformers==5.6.2" "peft==0.19.1" datasets accelerate librosa soundfile requests
```

`--break-system-packages` is required: the image's Python is an externally-managed
(PEP 668) system install and plain `pip install` fails with a wall of text that ends
in `ModuleNotFoundError`, which reads like a missing dependency rather than a refused
install. The image also ships **without numpy**.

Copy the code up rather than cloning, so an uncommitted experiment can run:

```bash
ssh -p <port> root@<ip> 'mkdir -p /workspace/oc/notebooks /workspace/oc/eval/ab_label_bug'
scp -P <port> notebooks/train_runpod.py       root@<ip>:/workspace/oc/notebooks/
scp -P <port> eval/ab_label_bug/run_ab.py     root@<ip>:/workspace/oc/eval/ab_label_bug/
```

## Run

Always `nohup` and log to a file — an SSH drop must not kill an 8-hour run.

```bash
cd /workspace/oc
# 1. cache clips + features first (network- and CPU-bound, no GPU used)
AB_PREPARE_ONLY=1 WORK_DIR=/workspace/ab-run nohup python eval/ab_label_bug/run_ab.py \
  > /workspace/prep.log 2>&1 &
# 2. then the run itself (reuses the cache)
WORK_DIR=/workspace/ab-run nohup python eval/ab_label_bug/run_ab.py \
  > /workspace/ab.log 2>&1 &
```

Splitting prep from the run is worth it: the audio download and decode take longer
than people expect (~40 meetings ≈ 30-60 min) and produce no GPU work, so the cache
can warm while the training code is still being reviewed. Prefetch the base model at
the same time — 3 GB, and otherwise it delays the first arm:

```bash
python -c "from transformers import WhisperForConditionalGeneration as W; W.from_pretrained('openai/whisper-large-v3')"
```

## Retrieve and terminate

```bash
scp -P <port> root@<ip>:/workspace/ab-run/results_ab_detail.json ./local-only/
scp -P <port> root@<ip>:/workspace/ab.log ./local-only/
runpodctl remove pod <POD_ID>      # do this immediately
runpodctl get pod                  # verify it is gone
```

Per-utterance transcripts are **PII** — they stay out of the repo (the 2026-07-21
GDPR purge removed exactly this kind of file from the history). Only aggregate
metrics belong in a tracked results file.

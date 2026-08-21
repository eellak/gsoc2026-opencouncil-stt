# Portable training bundle for RunPod

Use this before creating a billable pod. The bundle contains the dense packs, both
control parquets, the trainer, and the exact pod dependencies. It contains council
audio and transcripts, so it lives only under `~/.cache/oc-public/` and never in git.

## Build once, locally

```bash
python3 scripts/pod_bundle.py build --archive
```

The output is content-addressed and checksummed. Re-running with unchanged inputs
reuses the same directory and archive. The portable pack manifest uses paths relative
to itself; the trainer resolves those paths after extraction or a volume mount.

## Storage choice

Prefer a RunPod network volume when more than one pod/run will reuse the data. It
keeps the input bundle and Hugging Face cache adjacent to the GPU. Cloudflare R2 is
only a transport origin: every new pod would still download and unpack it, and the
currently available Cloudflare credentials are for Tunnel, not verified R2 access.

Official RunPod pricing checked 2026-08-19 is $0.07/GB/month for the first TB,
charged while idle; volumes are Secure Cloud only, datacenter-locked, and must be
attached when the pod is created. A 100 GB screen volume therefore has a $7/month
ceiling if retained for a full month. See the official
[network-volume documentation](https://docs.runpod.io/storage/network-volumes).

Mount the volume at `/runpod`. Immediately after pod creation, arm the local hard
deadline before setup or upload:

```bash
nohup scripts/pod_hard_deadline.sh POD_ID HOURS dense-screen \
  >~/.cache/oc-public/POD_ID-watchdog.log 2>&1 &
```

Install the two small transfer tools on the pod (`rsync`, `zstd`), then stage the
archive. The transfer resumes after interruption and both the archive and every file
inside it are verified before `CURRENT_BUNDLE` is written:

```bash
scripts/stage_pod_bundle.sh POD_ID HOST SSH_PORT \
  ~/.cache/oc-public/pod-bundles/dense-screen-BUNDLE_ID.tar.zst
```

The preferred pre-seed path does not require a running pod. Create a separate RunPod
S3 API key under Settings → S3 API Keys, then store it in the gitignored `.env`:

```bash
RUNPOD_S3_ACCESS_KEY_ID=user_...
RUNPOD_S3_SECRET_ACCESS_KEY=rps_...
```

Upload with multipart support through the volume's official S3 endpoint:

```bash
scripts/stage_volume_s3.sh \
  ~/.cache/oc-public/pod-bundles/dense-screen-97a2b4ba687fb406.tar.zst
```

`RUNPOD_API_KEY` cannot authenticate this endpoint. The S3 upload stores the archive
and checksum without GPU billing; the first pod verifies and extracts them on-volume.

## Pod bootstrap and paths

```bash
BUNDLE="$(cat /runpod/oc-bundles/CURRENT_BUNDLE)"
apt-get update -qq && apt-get install -y -qq ffmpeg
pip install --break-system-packages -r "$BUNDLE/code/notebooks/requirements-runpod.txt"
export HF_HOME=/runpod/hf-cache
```

Reuse preprocessing across paired seeds. Keep the two arms separate so their
`manifest.json` and feature files cannot collide:

```bash
export PREP_CACHE_DIR=/runpod/prep/control
export FEATURE_CACHE_DIR=/runpod/features/control
# dense arm instead:
export PREP_CACHE_DIR=/runpod/prep/dense
export FEATURE_CACHE_DIR=/runpod/features/dense
```

Dense arm inputs:

```bash
PACK_MANIFEST="$BUNDLE/packs/manifest.jsonl"
```

Single-utterance control inputs:

```bash
DATA_DIR="$BUNDLE/data/public"
```

The bundle makes the data/code transfer reproducible. Experiment commands, paired
seed order, validation decode, and the GPU-stage approval remain governed by
`docs/decisions/training-evidence.md`; staging a bundle is not approval to train.

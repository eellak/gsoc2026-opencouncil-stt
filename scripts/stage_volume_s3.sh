#!/usr/bin/env bash
# Pre-seed a RunPod network volume without starting a GPU pod.
# Credentials come from .env and are distinct from RUNPOD_API_KEY.
set -euo pipefail

ARCHIVE="${1:?bundle .tar.zst required}"
VOLUME_ID="${RUNPOD_VOLUME_ID:-lyaqmog1jy}"
REGION="${RUNPOD_VOLUME_REGION:-EU-RO-1}"
ENDPOINT="${RUNPOD_S3_ENDPOINT:-https://s3api-eu-ro-1.runpod.io/}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
[[ -f "$ARCHIVE" && -f "$ARCHIVE.sha256" ]] || {
  echo "FATAL: archive and $ARCHIVE.sha256 are required" >&2
  exit 2
}
: "${RUNPOD_S3_ACCESS_KEY_ID:?Create a RunPod S3 API key and set RUNPOD_S3_ACCESS_KEY_ID in .env}"
: "${RUNPOD_S3_SECRET_ACCESS_KEY:?Set RUNPOD_S3_SECRET_ACCESS_KEY in .env}"
command -v uvx >/dev/null || { echo "FATAL: uvx is required" >&2; exit 2; }

NAME="$(basename "$ARCHIVE")"
DEST="s3://$VOLUME_ID/oc-bundles/incoming"
export AWS_ACCESS_KEY_ID="$RUNPOD_S3_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$RUNPOD_S3_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="$REGION"

uvx --from awscli aws s3 cp "$ARCHIVE" "$DEST/$NAME" \
  --endpoint-url "$ENDPOINT" --region "$REGION" --no-progress
uvx --from awscli aws s3 cp "$ARCHIVE.sha256" "$DEST/$NAME.sha256" \
  --endpoint-url "$ENDPOINT" --region "$REGION" --no-progress
uvx --from awscli aws s3 ls "$DEST/" --endpoint-url "$ENDPOINT" --region "$REGION"

echo "seeded volume $VOLUME_ID; the pod must sha256sum -c before extraction"

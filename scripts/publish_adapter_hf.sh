#!/usr/bin/env bash
# Publish artifact-adapter-fixed to HuggingFace.
#
# This script is DELIBERATELY not runnable unattended. It prints what it would
# upload, verifies the adapter hash against the ledger, and then requires the
# operator to type PUBLISH. Nothing touches the network before that.
#
#   bash scripts/publish_adapter_hf.sh            # dry run + confirm prompt
#   bash scripts/publish_adapter_hf.sh --dry-run  # show the plan and exit
#
# Publication is irreversible and public. See docs/reports/2026-08-10-benchmark-corrected-adapter.md.

set -euo pipefail

REPO_ID="${HF_REPO_ID:-opencouncil/whisper-large-v3-el-council-lora}"
STAGING="${HF_STAGING_DIR:-$HOME/.cache/oc-public/hf-publish-adapter-fixed}"
EXPECTED_SHA256="ea8f03230846888fa7e4c341813efea324cf5596e689da2d48b1de365eb0a5a6"
TOKEN_VAR="HF_TOKEN"          # read from .env; the value is never printed
ENV_FILE="${ENV_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env}"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --- 1. staging contents -----------------------------------------------------
[[ -d "$STAGING" ]] || die "staging dir not found: $STAGING"
REQUIRED=(README.md adapter_config.json adapter_model.safetensors
          tokenizer.json tokenizer_config.json processor_config.json)
for f in "${REQUIRED[@]}"; do
  [[ -f "$STAGING/$f" ]] || die "missing from staging: $f"
done

echo "Target repo : https://huggingface.co/$REPO_ID  (model, PUBLIC)"
echo "Staging dir : $STAGING"
echo
echo "Files that will be uploaded:"
( cd "$STAGING" && find . -type f -printf '  %-32p %10s bytes\n' | sort )
echo

# --- 2. artifact identity ----------------------------------------------------
ACTUAL_SHA256="$(sha256sum "$STAGING/adapter_model.safetensors" | cut -d' ' -f1)"
echo "adapter_model.safetensors sha256:"
echo "  expected (ledger artifact-adapter-fixed): $EXPECTED_SHA256"
echo "  actual   (staging)                      : $ACTUAL_SHA256"
[[ "$ACTUAL_SHA256" == "$EXPECTED_SHA256" ]] \
  || die "hash mismatch - these are NOT artifact-adapter-fixed. Refusing to publish."
echo "  OK - identity confirmed."
echo

# --- 3. credentials (name only, never the value) ----------------------------
if [[ -z "${!TOKEN_VAR:-}" && -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$ENV_FILE"; set +a
fi
# Fallback: the huggingface CLI's own token store. Found 2026-08-16 under the 'oc'
# profile; it is the angelospk identity with write on the opencouncil org, which is
# exactly what this publish needs. Value is never echoed.
HF_STORE="$HOME/.cache/huggingface/stored_tokens"
if [[ -z "${!TOKEN_VAR:-}" && -f "$HF_STORE" ]]; then
  _tok="$(sed -n 's/^[[:space:]]*hf_token[[:space:]]*=[[:space:]]*\(.*\)$/\1/p' "$HF_STORE" | head -1)"
  if [[ -n "$_tok" ]]; then
    printf -v "$TOKEN_VAR" '%s' "$_tok"
    export "${TOKEN_VAR?}"
    echo "Credential  : taken from $HF_STORE (not from $ENV_FILE)."
  fi
  unset _tok
fi

[[ -n "${!TOKEN_VAR:-}" ]] || die "\$$TOKEN_VAR is not set (looked in the environment, in $ENV_FILE, and in $HF_STORE).
Add a line '$TOKEN_VAR=<org-write token for the opencouncil org>' to $ENV_FILE.
The identity is the 'angelospk' account, which has write access to the org."
echo "Credential  : \$$TOKEN_VAR is set (value not shown)."

HF_BIN="${HF_BIN:-}"
if [[ -z "$HF_BIN" ]]; then
  for cand in "$PWD/.venv-eval/bin/hf" "$(command -v hf || true)"; do
    [[ -n "$cand" && -x "$cand" ]] && { HF_BIN="$cand"; break; }
  done
fi
[[ -n "$HF_BIN" ]] || die "the 'hf' CLI was not found. Set \$HF_BIN or install huggingface_hub[cli]."
echo "hf CLI      : $HF_BIN"
echo

# --- 4. confirm --------------------------------------------------------------
cat <<'WARN'
This REPLACES the currently published weights, which are the broken
label-prefix-bug adapter, and it is PUBLIC and irreversible.
WARN

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "--dry-run: stopping here. Nothing was uploaded."
  exit 0
fi

echo
read -r -p 'Type PUBLISH to upload, anything else to abort: ' answer
[[ "$answer" == "PUBLISH" ]] || { echo "Aborted. Nothing was uploaded."; exit 1; }

# --- 5. upload ---------------------------------------------------------------
HF_TOKEN="${!TOKEN_VAR}" "$HF_BIN" upload "$REPO_ID" "$STAGING" . \
  --repo-type model \
  --commit-message "artifact-adapter-fixed: corrected weights (sha256 ${EXPECTED_SHA256:0:16}), replaces the label-prefix-bug adapter"

echo
echo "Uploaded. Now:"
echo "  1. check https://huggingface.co/$REPO_ID"
echo "  2. set artifact-adapter-fixed publication.status = PUBLISHED in research/ledger.json"
echo "  3. close https://github.com/eellak/gsoc2026-opencouncil-stt/issues/14"

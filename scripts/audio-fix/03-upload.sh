#!/usr/bin/env bash
# Uploads re-encoded MP3s to a GitHub Release and writes data/audio-fix/url-map.json.
# Prerequisites: gh CLI authenticated, a public repo with a release tag ready.
#
# Usage:
#   GH_REPO=yourusername/opencouncil-audio-fixes \
#   GH_RELEASE_TAG=audio-fixes \
#   bash scripts/audio-fix/03-upload.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ORIG_DIR="$REPO_ROOT/data/audio-fix/original"
FIXED_DIR="$REPO_ROOT/data/audio-fix/fixed"
INVENTORY="$REPO_ROOT/data/audio-fix/inventory.json"
URL_MAP="$REPO_ROOT/data/audio-fix/url-map.json"

GH_REPO="${GH_REPO:-}"
GH_RELEASE_TAG="${GH_RELEASE_TAG:-audio-fixes}"

if [[ -z "$GH_REPO" ]]; then
  echo "Set GH_REPO=owner/repo before running this script."
  echo "Example: GH_REPO=myname/opencouncil-audio-fixes bash scripts/audio-fix/03-upload.sh"
  exit 1
fi

command -v gh >/dev/null 2>&1 || { echo "gh CLI not found — install it from https://cli.github.com"; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "jq not found — install it"; exit 1; }

[[ -f "$INVENTORY" ]] || { echo "Run 01-inspect.sh first to generate inventory.json"; exit 1; }

shopt -s nullglob
files=("$FIXED_DIR"/*.mp3)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No fixed MP3s found in $FIXED_DIR. Run 02-reencode.sh first."
  exit 1
fi

# Warn if total size is large
total_kb=$(du -sk "$FIXED_DIR" | awk '{print $1}')
total_mb=$((total_kb / 1024))
if [[ $total_mb -gt 1500 ]]; then
  echo "WARNING: Total size ${total_mb} MB exceeds 1.5 GB. GitHub Releases may be slow."
  echo "Consider Cloudflare R2 or DigitalOcean Spaces instead."
  read -p "Continue anyway? [y/N] " confirm
  [[ "$confirm" =~ ^[Yy]$ ]] || exit 1
fi

echo "Uploading to github.com/$GH_REPO release '$GH_RELEASE_TAG'..."
for f in "${files[@]}"; do
  echo "  → $(basename "$f")"
  gh release upload "$GH_RELEASE_TAG" "$f" --repo "$GH_REPO" --clobber
done

echo ""
echo "Building url-map.json..."
MAP="{}"
for entry in $(jq -c '.[]' "$INVENTORY"); do
  orig=$(echo "$entry" | jq -r '.originalUrl')
  hash=$(echo "$entry" | jq -r '.hash')
  mp3="$FIXED_DIR/$hash.mp3"
  [[ -f "$mp3" ]] || continue
  # GitHub Release download URL (direct asset URL via objects.githubusercontent.com after redirect)
  fixed_url="https://github.com/$GH_REPO/releases/download/$GH_RELEASE_TAG/$hash.mp3"
  MAP=$(echo "$MAP" | jq --arg k "$orig" --arg v "$fixed_url" '. + {($k): $v}')
done

echo "$MAP" | jq '.' > "$URL_MAP"
echo "url-map.json written to $URL_MAP"
echo ""
echo "Restart the dev server ('cd ui && npm run dev') to pick up the new URL mappings."
echo "The broken-audio proxy will automatically serve the fixed files."

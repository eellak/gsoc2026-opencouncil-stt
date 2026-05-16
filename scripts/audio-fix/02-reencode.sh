#!/usr/bin/env bash
# Re-encodes each file in data/audio-fix/original/ to 128 kbps mono MP3.
# Outputs: data/audio-fix/fixed/{hash}.mp3
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ORIG_DIR="$REPO_ROOT/data/audio-fix/original"
FIXED_DIR="$REPO_ROOT/data/audio-fix/fixed"

command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg not found — install it first"; exit 1; }

mkdir -p "$FIXED_DIR"

shopt -s nullglob
files=("$ORIG_DIR"/*.bin)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No files in $ORIG_DIR. Run 01-inspect.sh first."
  exit 1
fi

for src in "${files[@]}"; do
  hash=$(basename "$src" .bin)
  dest="$FIXED_DIR/$hash.mp3"
  if [[ -f "$dest" ]]; then
    echo "Already encoded $hash.mp3 — skipping"
    continue
  fi
  echo "Encoding $hash..."
  ffmpeg -loglevel error -i "$src" \
    -codec:a libmp3lame -b:a 128k -ar 44100 -ac 1 \
    -f mp3 "$dest"
  echo "  → $dest ($(du -h "$dest" | awk '{print $1}'))"
done

echo ""
echo "Done. Run 03-upload.sh to upload to GitHub Releases."

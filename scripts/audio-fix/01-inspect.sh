#!/usr/bin/env bash
# Reads data/decode-failures.jsonl, downloads each unique audio file, and runs ffprobe.
# Outputs: data/audio-fix/inventory.json
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FAILURES="$REPO_ROOT/data/decode-failures.jsonl"
ORIG_DIR="$REPO_ROOT/data/audio-fix/original"
INVENTORY="$REPO_ROOT/data/audio-fix/inventory.json"

if [[ ! -f "$FAILURES" ]]; then
  echo "No failures file found at $FAILURES. Browse a few corrections in the review UI first."
  exit 1
fi

command -v ffprobe >/dev/null 2>&1 || { echo "ffprobe not found — install ffmpeg"; exit 1; }

mkdir -p "$ORIG_DIR"

# Collect unique originalUrls
mapfile -t URLS < <(jq -r 'select(.originalUrl != null) | .originalUrl' "$FAILURES" | sort -u)
echo "Found ${#URLS[@]} unique failed URLs"

ENTRIES="[]"
for url in "${URLS[@]}"; do
  hash=$(echo -n "$url" | shasum -a 1 | awk '{print $1}')
  dest="$ORIG_DIR/$hash.bin"

  if [[ ! -f "$dest" ]]; then
    echo "Downloading $url → $hash.bin"
    curl -sSL --max-time 120 -o "$dest" "$url" || { echo "  FAILED to download, skipping"; rm -f "$dest"; continue; }
  else
    echo "Already have $hash.bin — skipping download"
  fi

  probe=$(ffprobe -v quiet -print_format json -show_streams -show_format "$dest" 2>/dev/null) || probe='{}'
  codec=$(echo "$probe" | jq -r '.streams[0].codec_name // "unknown"')
  bitrate=$(echo "$probe" | jq -r '.format.bit_rate // "unknown"')
  size=$(du -h "$dest" | awk '{print $1}')
  format=$(echo "$probe" | jq -r '.format.format_name // "unknown"')

  echo "  codec=$codec  bitrate=$bitrate  format=$format  size=$size"

  entry=$(jq -n \
    --arg url "$url" --arg hash "$hash" \
    --arg codec "$codec" --arg bitrate "$bitrate" \
    --arg format "$format" --arg size "$size" \
    '{originalUrl: $url, hash: $hash, codec: $codec, bitrate: $bitrate, format: $format, size: $size}')
  ENTRIES=$(echo "$ENTRIES" | jq --argjson e "$entry" '. + [$e]')
done

echo "$ENTRIES" | jq '.' > "$INVENTORY"
echo ""
echo "Inventory written to $INVENTORY"
echo "Run 02-reencode.sh next."

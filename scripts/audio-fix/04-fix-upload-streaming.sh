#!/usr/bin/env bash
# Streams problematic audio repairs one file at a time:
# download -> re-encode clean CBR MP3 -> upload to GitHub Release -> update url-map.json -> cleanup.
#
# Usage:
#   GH_REPO=owner/repo GH_RELEASE_TAG=audio-fixes bash scripts/audio-fix/04-fix-upload-streaming.sh
#
# Optional:
#   AUDIO_FIX_LIMIT=10 bash scripts/audio-fix/04-fix-upload-streaming.sh
#   AUDIO_FIX_KEEP_FIXED=1 bash scripts/audio-fix/04-fix-upload-streaming.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
AUDIO_FIX_DIR="$REPO_ROOT/data/audio-fix"
PROBLEMATIC="$AUDIO_FIX_DIR/problematic-urls.txt"
WORK_DIR="$AUDIO_FIX_DIR/streaming-work"
URL_MAP="$AUDIO_FIX_DIR/url-map.json"

GH_REPO="${GH_REPO:-}"
GH_RELEASE_TAG="${GH_RELEASE_TAG:-audio-fixes}"
LIMIT="${AUDIO_FIX_LIMIT:-}"
KEEP_FIXED="${AUDIO_FIX_KEEP_FIXED:-0}"

[[ -n "$GH_REPO" ]] || { echo "Set GH_REPO=owner/repo"; exit 1; }
[[ -f "$PROBLEMATIC" ]] || { echo "Missing $PROBLEMATIC. Run 00-check-all.sh first."; exit 1; }

command -v curl >/dev/null 2>&1 || { echo "curl not found"; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg not found"; exit 1; }
command -v ffprobe >/dev/null 2>&1 || { echo "ffprobe not found"; exit 1; }
command -v gh >/dev/null 2>&1 || { echo "gh CLI not found"; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "jq not found"; exit 1; }

if [[ -n "$LIMIT" && ! "$LIMIT" =~ ^[0-9]+$ ]]; then
	echo "AUDIO_FIX_LIMIT must be a positive integer"
	exit 1
fi

mkdir -p "$WORK_DIR"
if [[ ! -f "$URL_MAP" ]]; then echo '{}' > "$URL_MAP"; fi

if ! gh release view "$GH_RELEASE_TAG" --repo "$GH_REPO" >/dev/null 2>&1; then
	echo "Release '$GH_RELEASE_TAG' does not exist - creating it."
	gh release create "$GH_RELEASE_TAG" --repo "$GH_REPO" --title "$GH_RELEASE_TAG" --notes "Re-encoded OpenCouncil audio files for browser-compatible decoding."
fi

URLS=()
while IFS= read -r url; do
	[[ -n "$url" ]] || continue
	URLS+=("$url")
done < "$PROBLEMATIC"

if [[ -n "$LIMIT" ]]; then
	URLS=("${URLS[@]:0:$LIMIT}")
fi

processed=0
skipped=0
uploaded=0
failed=0

for url in "${URLS[@]}"; do
	processed=$((processed + 1))
	hash=$(echo -n "$url" | shasum -a 1 | awk '{print $1}')
	orig="$WORK_DIR/$hash.original"
	fixed="$WORK_DIR/$hash.mp3"
	fixed_url="https://github.com/$GH_REPO/releases/download/$GH_RELEASE_TAG/$hash.mp3"

	if jq -e --arg k "$url" 'has($k)' "$URL_MAP" >/dev/null; then
		echo "[$processed/${#URLS[@]}] already mapped: $url"
		skipped=$((skipped + 1))
		continue
	fi

	echo "[$processed/${#URLS[@]}] downloading $url"
	rm -f "$orig" "$fixed"
	if ! curl -sSL --fail --max-time 600 -o "$orig" "$url"; then
		echo "  download failed"
		failed=$((failed + 1))
		rm -f "$orig" "$fixed"
		continue
	fi

	echo "  re-encoding $hash.mp3"
	if ! ffmpeg -loglevel error -i "$orig" \
		-map_metadata -1 \
		-codec:a libmp3lame -b:a 128k -ar 44100 -ac 1 \
		-write_xing 1 -id3v2_version 3 \
		-f mp3 "$fixed"; then
		echo "  re-encode failed"
		failed=$((failed + 1))
		rm -f "$orig" "$fixed"
		continue
	fi

	echo "  verifying fixed MP3"
	if ! ffmpeg -v error -i "$fixed" -f null - >/dev/null 2>&1; then
		echo "  fixed decode verification failed"
		failed=$((failed + 1))
		rm -f "$orig" "$fixed"
		continue
	fi

	if ffprobe -v error -show_format -print_format json "$fixed" | jq -e '(.format.tags.major_brand // "") != "" or (.format.tags.compatible_brands // "") != ""' >/dev/null; then
		echo "  fixed MP3 still has MP4-origin tags"
		failed=$((failed + 1))
		rm -f "$orig" "$fixed"
		continue
	fi

	echo "  uploading $hash.mp3"
	if ! gh release upload "$GH_RELEASE_TAG" "$fixed" --repo "$GH_REPO" --clobber; then
		echo "  upload failed"
		failed=$((failed + 1))
		rm -f "$orig"
		[[ "$KEEP_FIXED" == "1" ]] || rm -f "$fixed"
		continue
	fi

	# Free the large local files before writing the tiny URL map update. On low-disk machines,
	# keeping both the original and fixed MP3 around can make even the jq tmp file fail.
	rm -f "$orig"
	if [[ "$KEEP_FIXED" != "1" ]]; then rm -f "$fixed"; fi

	tmp_map="$URL_MAP.tmp"
	jq --arg k "$url" --arg v "$fixed_url" '. + {($k): $v}' "$URL_MAP" > "$tmp_map"
	mv "$tmp_map" "$URL_MAP"
	uploaded=$((uploaded + 1))
	echo "  mapped -> $fixed_url"
done

echo ""
echo "Processed: $processed"
echo "Uploaded:  $uploaded"
echo "Skipped:   $skipped"
echo "Failed:    $failed"
echo "URL map:   $URL_MAP"

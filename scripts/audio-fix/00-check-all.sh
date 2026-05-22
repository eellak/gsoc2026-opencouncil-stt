#!/usr/bin/env bash
# Proactively checks every distinct audio_url in the corrections DB.
# Outputs:
#   data/audio-fix/all-audio-urls.txt
#   data/audio-fix/problematic-urls.txt
#   data/audio-fix/decode-report.jsonl
#
# Optional:
#   AUDIO_FIX_LIMIT=10 bash scripts/audio-fix/00-check-all.sh
#   AUDIO_FIX_URLS_FILE=/path/to/urls.txt bash scripts/audio-fix/00-check-all.sh
#   AUDIO_FIX_KEEP_DOWNLOADS=1 bash scripts/audio-fix/00-check-all.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
UI_DIR="$REPO_ROOT/ui"
AUDIO_FIX_DIR="$REPO_ROOT/data/audio-fix"
CACHE_DIR="$AUDIO_FIX_DIR/check-cache"
ALL_URLS="$AUDIO_FIX_DIR/all-audio-urls.txt"
PROBLEMATIC="$AUDIO_FIX_DIR/problematic-urls.txt"
REPORT="$AUDIO_FIX_DIR/decode-report.jsonl"
LIMIT="${AUDIO_FIX_LIMIT:-}"
URLS_FILE="${AUDIO_FIX_URLS_FILE:-}"
KEEP_DOWNLOADS="${AUDIO_FIX_KEEP_DOWNLOADS:-0}"

command -v bun >/dev/null 2>&1 || { echo "bun not found"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl not found"; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg not found - install ffmpeg"; exit 1; }
command -v ffprobe >/dev/null 2>&1 || { echo "ffprobe not found - install ffmpeg"; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "jq not found - install jq"; exit 1; }

if [[ -n "$LIMIT" && ! "$LIMIT" =~ ^[0-9]+$ ]]; then
	echo "AUDIO_FIX_LIMIT must be a positive integer"
	exit 1
fi

mkdir -p "$CACHE_DIR"

echo "Querying distinct audio URLs..."
if [[ -n "$URLS_FILE" ]]; then
	[[ -f "$URLS_FILE" ]] || { echo "AUDIO_FIX_URLS_FILE does not exist: $URLS_FILE"; exit 1; }
	sed '/^[[:space:]]*$/d' "$URLS_FILE" | sort -u > "$ALL_URLS.tmp"
else
	(cd "$UI_DIR" && bun scripts/list-audio-urls.ts) > "$ALL_URLS.tmp"
fi
mv "$ALL_URLS.tmp" "$ALL_URLS"

total_urls=$(wc -l < "$ALL_URLS" | tr -d ' ')
if [[ -n "$LIMIT" ]]; then
	echo "Found $total_urls URLs; checking first $LIMIT due to AUDIO_FIX_LIMIT"
	URLS=()
	while IFS= read -r url; do
		URLS+=("$url")
	done < <(head -n "$LIMIT" "$ALL_URLS")
else
	echo "Found $total_urls URLs"
	URLS=()
	while IFS= read -r url; do
		URLS+=("$url")
	done < "$ALL_URLS"
fi

: > "$PROBLEMATIC"
: > "$REPORT"

write_report() {
	local url="$1"
	local status="$2"
	local hash="$3"
	local message="$4"
	local probe_file="$5"
	if [[ -f "$probe_file" ]]; then
		jq -c \
			--arg url "$url" \
			--arg status "$status" \
			--arg hash "$hash" \
			--arg message "$message" \
			--arg checkedAt "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
			'{url: $url, status: $status, hash: $hash, message: $message, checkedAt: $checkedAt, probe: .}' \
			"$probe_file" >> "$REPORT"
	else
		jq -nc \
			--arg url "$url" \
			--arg status "$status" \
			--arg hash "$hash" \
			--arg message "$message" \
			--arg checkedAt "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
			'{url: $url, status: $status, hash: $hash, message: $message, checkedAt: $checkedAt}' \
			>> "$REPORT"
	fi
}

has_mp4_origin_tags() {
	local probe_file="$1"
	jq -e '
		(.format.format_name // "" | contains("mp3")) and
		(
			((.format.tags.major_brand // "") | length > 0) or
			((.format.tags.compatible_brands // "") | length > 0)
		)
	' "$probe_file" >/dev/null
}

problem_count=0
checked_count=0

for url in "${URLS[@]}"; do
	[[ -n "$url" ]] || continue
	checked_count=$((checked_count + 1))
	hash=$(echo -n "$url" | shasum -a 1 | awk '{print $1}')
	dest="$CACHE_DIR/$hash.bin"
	curl_log="$CACHE_DIR/$hash.curl.log"
	ffmpeg_log="$CACHE_DIR/$hash.ffmpeg.log"
	probe_file="$CACHE_DIR/$hash.probe.json"

	printf '[%s/%s] %s\n' "$checked_count" "${#URLS[@]}" "$url"

	if [[ ! -f "$dest" ]]; then
		set +e
		curl -sSL --fail --max-time 120 -o "$dest" "$url" 2> "$curl_log"
		curl_status=$?
		set -e
		if [[ $curl_status -ne 0 ]]; then
			rm -f "$dest"
			message=$(cat "$curl_log")
			echo "  download failed"
			printf '%s\n' "$url" >> "$PROBLEMATIC"
			write_report "$url" "download_failed" "$hash" "$message" ""
			problem_count=$((problem_count + 1))
			continue
		fi
	else
		echo "  cached download"
	fi

	set +e
	ffmpeg_output=$(ffmpeg -v error -i "$dest" -f null - 2>&1)
	ffmpeg_status=$?
	set -e
	printf '%s\n' "$ffmpeg_output" > "$ffmpeg_log"

	if [[ $ffmpeg_status -ne 0 ]] || grep -qi "Error" "$ffmpeg_log"; then
		echo "  decode failed"
		printf '%s\n' "$url" >> "$PROBLEMATIC"
		write_report "$url" "decode_failed" "$hash" "$ffmpeg_output" ""
		problem_count=$((problem_count + 1))
		if [[ "$KEEP_DOWNLOADS" != "1" ]]; then rm -f "$dest"; fi
	else
		ffprobe -v error -show_format -show_streams -print_format json "$dest" > "$probe_file" || echo '{}' > "$probe_file"
		if has_mp4_origin_tags "$probe_file"; then
			echo "  browser-suspect: MP4-origin metadata tags in MP3"
			printf '%s\n' "$url" >> "$PROBLEMATIC"
			write_report "$url" "browser_suspect_mp4_tags" "$hash" "MP3 has MP4-origin metadata tags; ffmpeg decodes it, but browser WebAudio may reject it." "$probe_file"
			problem_count=$((problem_count + 1))
		else
			echo "  ok"
			write_report "$url" "ok" "$hash" "" "$probe_file"
		fi
		if [[ "$KEEP_DOWNLOADS" != "1" ]]; then rm -f "$dest"; fi
	fi
done

echo ""
echo "Checked $checked_count URLs."
echo "Problematic URLs: $problem_count"
echo "All URLs: $ALL_URLS"
echo "Problematic URLs file: $PROBLEMATIC"
echo "Report: $REPORT"

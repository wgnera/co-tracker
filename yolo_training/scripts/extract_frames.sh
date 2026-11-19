#!/usr/bin/env bash
# Extract frames 3, 6, 9, ... as sequentially numbered PNG images.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: extract_frames.sh [INPUT_VIDEO [OUTPUT_DIR]]

Extract every third frame (3, 6, 9, ...) using ffmpeg.
Default input: dataset/holdout_test/epee_video.ogv
Default output: <input directory>/<input filename without extension>_frames/
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# > 2 )); then
    usage >&2
    exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
input="${1:-$script_dir/../dataset/holdout_test/epee_video.ogv}"
output="${2:-${input%.*}_frames}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    printf 'Error: ffmpeg is required.\n' >&2
    exit 1
fi
if [[ ! -f "$input" ]]; then
    printf 'Error: input video not found: %s\n' "$input" >&2
    exit 1
fi

mkdir -p -- "$output"
# ffmpeg numbers source frames from zero; disable output frame duplication.
ffmpeg -hide_banner -loglevel error -nostdin -n \
    -i "$input" -map 0:v:0 -vf 'select=not(mod(n+1\,3))' \
    -vsync 0 -start_number 1 "$output/frame_%06d.png"

printf 'Extracted every third frame to %s\n' "$output"

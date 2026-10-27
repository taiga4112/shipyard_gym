#!/usr/bin/env bash
# Record the README demo loops: the same policy / scene in Isaac Sim and MuJoCo, cut to a
# 10 s window and converted to looping animated WebP under docs/media/. (GIF would be
# 6-10 MB per loop because the follow camera changes every pixel each frame; WebP is
# ~1 MB and renders inline on GitHub like a GIF.)
#
#   scripts/make_readme_media.sh <policy> <scene> <name> [start_s] [camera]
#   scripts/make_readme_media.sh go2_perceptive_mid360_phase5 basic/5_3 go2 4 rear_left
#
# Needs: the Isaac Lab env at $ISAACLAB_ENV (default ~/isaacsim/env_isaaclab), uv, ffmpeg.
set -euo pipefail

POLICY=$1
SCENE=$2
NAME=$3
START=${4:-4}          # seconds into the run where the 10 s loop starts
CAMERA=${5:-rear_left}
LENGTH=10
WIDTH=720
FPS=15
ISAACLAB_ENV=${ISAACLAB_ENV:-$HOME/isaacsim/env_isaaclab}

cd "$(dirname "$0")/.."
OUT=docs/media
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"
DURATION=$((START + LENGTH + 1))
COMMON=(--policy "$POLICY" --scene "$SCENE" --headless --duration "$DURATION" --command 0.6,0,0
        --camera "$CAMERA" --no-overlay --width 1280 --height 720)

( source "$ISAACLAB_ENV/bin/activate" && python scripts/run_isaac.py "${COMMON[@]}" --record "$TMP/isaac.mp4" )
uv run scripts/run_mujoco.py "${COMMON[@]}" --record "$TMP/mujoco.mp4"

for backend in isaac mujoco; do
  ffmpeg -v error -y -ss "$START" -t "$LENGTH" -i "$TMP/$backend.mp4" \
    -vf "fps=$FPS,scale=$WIDTH:-1:flags=lanczos" \
    -c:v libwebp_anim -lossless 0 -q:v 75 -compression_level 6 -loop 0 "$OUT/${NAME}_${backend}.webp"
  echo "wrote $OUT/${NAME}_${backend}.webp ($(du -h "$OUT/${NAME}_${backend}.webp" | cut -f1))"
done

#!/bin/bash
# Pre-build the valid-index filter cache for the audio screen as a CPU-only
# job, so every GPU screen item starts with an instant cache hit instead of a
# multi-hour cold filter at dataset-init time.
#
# The cache key (midigpt.utils.file_cache_key) is path+size+min_bars+min_tracks
# +ts, deliberately mtime-independent -> portable across clusters. To guarantee
# the .npy this builds is HIT (not rebuilt) by the screen, the filter params
# here must match train_audio_screen.sh exactly:
#   --encoder-config models/amber_encoder.json   (same file the screen passes;
#                     supplies num_bars_map -> min_bars and time_signatures -> ts)
#   --audio-conditioned                          (enforces min_tracks=2, as the
#                     audio screen does)
#   --parquet data/v2.0.0/train/*.parquet        (same train shards)
# Keep these three in lockstep with train_audio_screen.sh.
#
# CPU-only (no GPU requested) so it schedules on the fast CPU partition and
# never competes with the screen for the contended GPU queue. Nested
# parallelism: --workers pools across the 10 shards; within each shard the
# phase-2 filter fans out over CHUNK_WORKERS threads (dataset.py default
# min(8, cpus)). workers x CHUNK_WORKERS is sized to ~match allocated cores.
set -euo pipefail

# pyarrow lives in the Alliance `arrow` module (these venvs are
# --system-site-packages), not pip-installed. Only fir auto-loads it via
# global.yaml modules:, so load the stack here exactly like
# train_audio_screen.sh does — otherwise dataset.py's `import pyarrow` dies
# with ModuleNotFoundError on nibi/rorqual/narval.
#
# ORDER MATTERS: activate the venv AFTER the module load, not before. prism
# already activates the venv for us, but `module load python/3.11.5` prepends
# the module python to PATH and shadows the venv's python — which drops the
# -e-installed `midigpt` (ModuleNotFoundError: No module named 'midigpt').
# Re-sourcing the venv here puts the venv python back in front, exactly as the
# screen script does.
module purge
module load python/3.11.5 gcc/12.3 arrow/19.0.1 cmake/3.31.0
source /scratch/triana24/MIDI-GPT/venvs/venv-audio-conditioning/bin/activate

PROJECT="/scratch/triana24/MIDI-GPT/worktree-audio-conditioning"
ENC_CFG="$PROJECT/models/amber_encoder.json"
TRAIN_GLOB="/scratch/triana24/MIDI-GPT/data/v2.0.0/train/*.parquet"
WORKERS="${WORKERS:-2}"   # shard-pool width; 2 x default-8 chunk-threads ~= 16 cores

export MIDIGPT_CACHE="${MIDIGPT_CACHE:-/scratch/triana24/MIDI-GPT/.midigpt_cache}"

cd "$PROJECT"
echo "preprocess_cache: enc_cfg=$ENC_CFG workers=$WORKERS cache=$MIDIGPT_CACHE"
echo "preprocess_cache: train glob=$TRAIN_GLOB"

python -m midigpt.training.preprocess \
    --parquet "$TRAIN_GLOB" \
    --encoder-config "$ENC_CFG" \
    --audio-conditioned \
    --workers "$WORKERS"

echo "preprocess_cache: done; cache .npy count=$(ls "$MIDIGPT_CACHE"/valid_*.npy 2>/dev/null | wc -l)"

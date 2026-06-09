#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute valid-index cache for all GigaMIDI v2.0.0 train shards.
#
# Run once before submitting training jobs. Safe to re-run — cache hits are
# instant no-ops. Uses spawn-based multiprocessing (one worker per shard).
#
# Usage:
#   sbatch slurm/preprocess.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --account=def-pasquier
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --job-name=midigpt-preprocess

set -e

# ── environment ───────────────────────────────────────────────────────────────

module purge
module load arrow/19.0.1

PROJECT="$HOME/projects/def-pasquier/$USER/MIDI-GPT"
VENV="$HOME/scratch/MIDI-GPT/.venv"
DATA_DIR="$SCRATCH/MIDI-GPT/data/v2.0.0"
export MIDIGPT_CACHE="$SCRATCH/.midigpt"

source "$VENV/bin/activate"
cd "$PROJECT"

[ -f .env ] && set -a && source .env && set +a

mkdir -p slurm/logs "$MIDIGPT_CACHE"

echo "Data   : $DATA_DIR/train/*.parquet"
echo "Cache  : $MIDIGPT_CACHE"
echo "Workers: 10"
echo ""

python3 -m midigpt.training.preprocess \
    --parquet "$DATA_DIR/train/*.parquet" \
    --encoder-config models/yellow_encoder.json \
    --workers 10

echo ""
echo "Done. Cache written to $MIDIGPT_CACHE"

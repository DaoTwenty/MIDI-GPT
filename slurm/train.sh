#!/bin/bash
set -e
# ─────────────────────────────────────────────────────────────────────────────
# Generic MidiGPT training job — single H100 (80 GB), 3 days.
#
# Usage:
#   sbatch slurm/train.sh slurm/configs/yellow_h100.json
#   sbatch slurm/train.sh slurm/configs/yellow_h100_small.json
#   sbatch slurm/train.sh slurm/configs/yellow_h100_medium.json
#
# Preprocess once on a login/CPU node before submitting:
#   python -m midigpt.training.preprocess \
#       --parquet "$SCRATCH/MIDI-GPT/data/v2.0.0/train/*.parquet" \
#       --encoder-config models/yellow_encoder.json
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --account=def-pasquier
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

CONFIG="${1:?Usage: sbatch slurm/train.sh <config.json>}"

# ── environment ───────────────────────────────────────────────────────────────

module purge
module load arrow/19.0.1

PROJECT="$HOME/projects/def-pasquier/$USER/MIDI-GPT"
VENV="$HOME/scratch/MIDI-GPT/.venv"
DATA_DIR="$SCRATCH/MIDI-GPT/data/v2.0.0"
export HF_DATASETS_CACHE="$SCRATCH/huggingface/datasets"
export MIDIGPT_CACHE="$SCRATCH/MIDI-GPT/.midigpt"

CONFIG_NAME=$(basename "$CONFIG" .json)
RUN_ID="${CONFIG_NAME}-$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="$SCRATCH/MIDI-GPT/runs/$RUN_ID"

source "$VENV/bin/activate"
cd "$PROJECT"

# shellcheck source=../.env
[ -f .env ] && set -a && source .env && set +a

mkdir -p "$OUTPUT_DIR" slurm/logs

echo "Run    : $RUN_ID"
echo "Config : $CONFIG"
echo "Data   : $DATA_DIR/train/*.parquet"
echo "Output : $OUTPUT_DIR"
echo "GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo ""

# ── download GigaMIDI if not present ─────────────────────────────────────────

if [ ! -d "$DATA_DIR/train" ]; then
    echo "Downloading GigaMIDI v2.0.0 …"
    DATA_DIR="$DATA_DIR" python3 - << 'PYEOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Metacreation/GigaMIDI",
    repo_type="dataset",
    revision="v2.0.0",
    local_dir=os.environ["DATA_DIR"],
    ignore_patterns=["*.md", "*.txt"],
)
PYEOF
fi

# ── preprocess (no-op on cache hit) ──────────────────────────────────────────

echo "Preprocessing shards …"
python3 -m midigpt.training.preprocess \
    --parquet "$DATA_DIR/train/*.parquet" \
    --encoder-config models/yellow_encoder.json

# ── train ─────────────────────────────────────────────────────────────────────

echo ""
echo "Starting training …"
python3 -m midigpt.training.trainer \
    --config     "$CONFIG" \
    --train-data "$DATA_DIR/train/*.parquet" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Done. Output: $OUTPUT_DIR"

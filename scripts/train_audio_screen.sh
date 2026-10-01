#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 extractor screen: short adapter_only runs across all real audio
# extractors, same data/training-mode/everything else held fixed, to rank
# extractors before committing to a full-scale training-mode x data sweep on
# the 2-3 best. Not a final-quality run -- MAX_STEPS is a short screening
# budget, not convergence.
#
# Takes a single NAME (a PRISM distribute-mode matrix value -- see
# .prism/jobs/audio_screen.yaml) and resolves (venv, extractor, weights,
# model-size) from the lookup table below, rather than threading 4-5
# separate positional args through {matrix} -- PRISM's matrix substitution
# is a single flat value per item, and using short names keeps distribute
# mode's auto-generated per-item job labels (<job-name>-<value>) clean.
#
# Different extractors cost very different FLOPs/step (58M-params
# Music2Latent vs. 852M-params SAME-L) even though all are frozen -- MAX_STEPS
# is fixed and identical across all 8 (equal adapter gradient-update count,
# the fair unit of comparison here), and the wall-clock budget in
# .prism/jobs/audio_screen.yaml's resources.time is deliberately generous
# and uniform across the whole matrix (PRISM distribute mode applies one
# resources: block to every item, no native per-item override) rather than
# sized per-extractor -- cheap ones just finish early and exit.
#
# Usage: bash scripts/train_audio_screen.sh <name>
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail

NAME="$1"
if [ -z "$NAME" ]; then
    echo "Usage: $0 <name>" >&2
    echo "Valid names: yourmt3 muscriptor muscriptor-large mert sheetsage2 dac muq music2latent same" >&2
    exit 2
fi

# name -> "venv|extractor|weights|model-size" (model-size may be empty)
case "$NAME" in
    yourmt3)
        VENV_NAME=venv-audio-conditioning; AUDIO_EXTRACTOR=yourmt3
        AUDIO_WEIGHTS="/scratch/triana24/MIDI-GPT/hf_cache/yourmt3/logs/2024/notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72/checkpoints/model.ckpt"
        MODEL_SIZE=""
        ;;
    muscriptor)
        VENV_NAME=venv-audio-conditioning; AUDIO_EXTRACTOR=muscriptor
        AUDIO_WEIGHTS="/scratch/triana24/MIDI-GPT/hf_cache/muscriptor-medium/model.safetensors"
        MODEL_SIZE="medium"
        ;;
    muscriptor-large)
        VENV_NAME=venv-audio-conditioning; AUDIO_EXTRACTOR=muscriptor
        AUDIO_WEIGHTS="/scratch/triana24/MIDI-GPT/hf_cache/muscriptor-large/model.safetensors"
        MODEL_SIZE="large"
        ;;
    mert)
        VENV_NAME=venv-mert-check; AUDIO_EXTRACTOR=mert
        AUDIO_WEIGHTS="/scratch/triana24/MIDI-GPT/hf_cache/mert-v2-fullsong"
        MODEL_SIZE="fullsong"
        ;;
    sheetsage2)
        VENV_NAME=venv-sheetsage2-check; AUDIO_EXTRACTOR=sheetsage2
        # Bare repo ID, not the local hf_cache dir: local-directory
        # trust_remote_code loading only copies modeling_sheetsage2.py's
        # direct imports, not transitive ones (tokenization_sheetsage2.py's
        # own imports) -- confirmed broken locally, confirmed working via
        # the bare repo ID (transformers fetches the full file listing).
        # Trades a network fetch of the MERT2 backbone at job start (the
        # factory doesn't expose base_model_path) for correctness.
        AUDIO_WEIGHTS="m-a-p/SheetSage2"
        MODEL_SIZE=""
        ;;
    dac)
        VENV_NAME=venv-mert-check; AUDIO_EXTRACTOR=dac
        AUDIO_WEIGHTS="/scratch/triana24/MIDI-GPT/hf_cache/dac_44khz"
        MODEL_SIZE="44khz"
        ;;
    muq)
        VENV_NAME=venv-muq-check; AUDIO_EXTRACTOR=muq
        AUDIO_WEIGHTS="/scratch/triana24/MIDI-GPT/hf_cache/muq-large-msd-iter"
        MODEL_SIZE=""
        ;;
    music2latent)
        VENV_NAME=venv-audio-conditioning; AUDIO_EXTRACTOR=music2latent
        AUDIO_WEIGHTS=""  # auto-downloads on first use
        MODEL_SIZE=""
        ;;
    same)
        VENV_NAME=venv-same-check; AUDIO_EXTRACTOR=same
        AUDIO_WEIGHTS="/scratch/triana24/MIDI-GPT/hf_cache/same-l"
        MODEL_SIZE="same-l"
        ;;
    *)
        echo "Unknown name: $NAME" >&2
        echo "Valid names: yourmt3 muscriptor muscriptor-large mert sheetsage2 dac muq music2latent same" >&2
        exit 2
        ;;
esac

module purge
module load python/3.11.5 gcc/12.3 arrow/19.0.1 cmake/3.31.0

PROJECT="/scratch/triana24/MIDI-GPT/worktree-audio-conditioning"
VENV="/scratch/triana24/MIDI-GPT/venvs/${VENV_NAME}"
ENC_CFG="$PROJECT/models/amber_encoder.json"
MIDI_CKPT="/scratch/triana24/MIDI-GPT/runs/yellow_medium-20260621-115325/model_final.safetensors"
TRAIN_DATA="/scratch/triana24/MIDI-GPT/data/v2.0.0/train/*.parquet"
SLAKH_ROOT="/scratch/triana24/MIDI-GPT/data/slakh2100/slakh2100_flac_redux/train"
OUTPUT_DIR="/scratch/triana24/MIDI-GPT/runs/audio_screen_${NAME}"
MAX_STEPS="${MAX_STEPS:-1500}"

export LD_LIBRARY_PATH="/scratch/triana24/fluidsynth-install/lib64:${LD_LIBRARY_PATH:-}"
export PATH="/scratch/triana24/fluidsynth-install/bin:${PATH:-}"
export MIDIGPT_CACHE="/scratch/triana24/MIDI-GPT/.midigpt_cache"

source "$VENV/bin/activate"
cd "$PROJECT"
mkdir -p /scratch/triana24/MIDI-GPT/logs "$OUTPUT_DIR"

echo "Screen: $NAME"
echo "Venv: $VENV"
echo "Extractor: $AUDIO_EXTRACTOR (weights=$AUDIO_WEIGHTS, model_size=$MODEL_SIZE)"
echo "Max steps: $MAX_STEPS"
echo "Output dir: $OUTPUT_DIR"
nvidia-smi -L

EXTRA_ARGS=()
[ -n "$AUDIO_WEIGHTS" ] && EXTRA_ARGS+=(--audio-weights "$AUDIO_WEIGHTS")
[ -n "$MODEL_SIZE" ] && EXTRA_ARGS+=(--model-size "$MODEL_SIZE")

python scripts/train_audio_conditioned.py \
    --train-data "$TRAIN_DATA" \
    --slakh-root "$SLAKH_ROOT" \
    --midi-ckpt "$MIDI_CKPT" \
    --encoder-config "$ENC_CFG" \
    --synthesizer soundfont \
    --soundfont-dir soundfonts/ \
    --audio-extractor "$AUDIO_EXTRACTOR" \
    "${EXTRA_ARGS[@]}" \
    --batch-size 16 \
    --max-steps "$MAX_STEPS" \
    --grad-accum 1 \
    --num-workers 8 \
    --precision bf16 \
    --training-mode adapter_only \
    --eval-data "/scratch/triana24/MIDI-GPT/data/v2.0.0/validation/*.parquet" \
    --eval-steps 500 \
    --output-dir "$OUTPUT_DIR" \
    --wandb --wandb-project midigpt --wandb-run-name "audio_screen_${NAME}"

echo "Screen $NAME exited with status $?"

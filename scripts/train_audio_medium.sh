#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Amber (audio-conditioned) fine-tune of yellow_medium -- 3-way training-mode
# ablation (adapter_only / lora / full), batch_size=16 across all three for a
# clean comparison (chosen from the 36-config H100 sweep, job 20660254, +
# peak-memory recheck, job 20671433). Headroom at bs16 on the 10-step smoke
# probe: full ~6%, adapter_only ~8%, lora ~24% -- full and adapter_only are
# deliberately close to the OOM wall; drop --batch-size to 8 (~50GB headroom)
# if either OOMs mid-run.
#
# No single GPU partition here (nibi) allows more than 12h before a slower QOS
# tier, and train_audio_conditioned.py has no native multi-day budget, so this
# self-chains via job dependencies (see bottom of script) instead of
# requesting one long walltime. Each link resumes from
# <output-dir>/checkpoints/last.ckpt (--resume-from auto). The chain stops on
# its own once model_final.safetensors exists, or after CHAIN_REMAINING links
# are exhausted (investigate manually if that happens without a final
# checkpoint).
#
# Ported from MIDI-GPT-infra/slurm/audio_train_medium_{full,lora,adapter_only}.sh
# (three near-duplicate scripts, one per mode) into this one parameterized
# version, git-tracked in this worktree instead of a separate untracked-by-
# design infra repo -- see conversation history for the "drop MIDI-GPT-infra,
# use PRISM+worktree instead" decision.
#
# PRISM's own module/venv setup is a harmless no-op on top of this (fully
# self-contained, same convention as this project's other training scripts) --
# but note the self-chain below resubmits via a BARE `sbatch "$SELF"` call,
# NOT through `prism submit` -- so the #SBATCH directives here are the real,
# active resource request for every link after the first (PRISM's own
# generated wrapper resources only govern the FIRST link, submitted via
# `prism submit`). Do not `prism submit` this job again while a chain is
# still alive for the same training-mode -- it would start a second,
# independently-resubmitting process against the same output dir.
#
# Usage:
#   bash scripts/train_audio_medium.sh <full|lora|adapter_only>
#   MAX_EPOCHS=5 bash scripts/train_audio_medium.sh full   # override
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --account=def-pasquier_gpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/triana24/MIDI-GPT/logs/audio-medium-%x-%j.out
#SBATCH --error=/scratch/triana24/MIDI-GPT/logs/audio-medium-%x-%j.err
#SBATCH --signal=B:USR1@180

set -uo pipefail  # no -e: chain-management logic below must run even if training exits non-zero

MODE="$1"
case "$MODE" in
  full|lora|adapter_only) ;;
  *) echo "Usage: $0 <full|lora|adapter_only>" >&2; exit 2 ;;
esac

module purge
module load python/3.11.5 gcc/12.3 arrow/19.0.1 cmake/3.31.0

PROJECT="/scratch/triana24/MIDI-GPT/worktree-audio-conditioning"
VENV="/scratch/triana24/MIDI-GPT/venvs/venv-audio-conditioning"
ENC_CFG="$PROJECT/models/amber_encoder.json"
export MIDI_CKPT="${MIDI_CKPT:-/scratch/triana24/MIDI-GPT/runs/yellow_medium-20260621-115325/model_final.safetensors}"
export TRAIN_DATA="${TRAIN_DATA:-/scratch/triana24/MIDI-GPT/data/v2.0.0/train/*.parquet}"
export SLAKH_ROOT="${SLAKH_ROOT:-/scratch/triana24/MIDI-GPT/data/slakh2100/slakh2100_flac_redux/train}"
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/triana24/MIDI-GPT/runs/audio_medium_${MODE}}"
export MAX_EPOCHS="${MAX_EPOCHS:-3}"
export CHAIN_REMAINING="${CHAIN_REMAINING:-10}"  # 10 links x 12h = up to 120h wall-clock budget

export LD_LIBRARY_PATH="/scratch/triana24/fluidsynth-install/lib64:${LD_LIBRARY_PATH:-}"
export PATH="/scratch/triana24/fluidsynth-install/bin:${PATH:-}"
export MIDIGPT_CACHE="/scratch/triana24/MIDI-GPT/.midigpt_cache"

SELF="/scratch/triana24/MIDI-GPT/worktree-audio-conditioning/scripts/train_audio_medium.sh"

source "$VENV/bin/activate"
cd "$PROJECT"
mkdir -p /scratch/triana24/MIDI-GPT/logs "$OUTPUT_DIR"

FINAL="$OUTPUT_DIR/model_final.safetensors"
if [ -f "$FINAL" ]; then
    echo "Already complete: $FINAL exists. Not resubmitting."
    exit 0
fi

echo "Mode: $MODE"
echo "Chain link remaining: $CHAIN_REMAINING"
echo "Output dir: $OUTPUT_DIR"
nvidia-smi -L

# A TIMEOUT-killed job never reaches the STATUS/resubmit logic below --
# slurmstepd tears down the whole batch-script process (not just the
# training subprocess) once the time limit hits, so the plain "check $?
# after python returns" pattern silently drops the chain (observed: amber
# lora job 20993767 hit TIMEOUT at 12h with no resubmit ever queued,
# 2026-09-02). --signal above asks Slurm to send SIGUSR1 3 min before that
# kill; the trap below submits the next chain link in that window, before
# we can be killed. Losing up to one checkpoint interval (500 steps) of
# progress is an acceptable tradeoff for the chain actually continuing.
resubmit_on_timeout() {
    echo "Received pre-timeout warning -- resubmitting next chain link before this job is killed."
    if [ -f "$FINAL" ]; then
        echo "Already complete: $FINAL exists. Not resubmitting."
    elif [ "$CHAIN_REMAINING" -le 1 ]; then
        echo "Chain exhausted -- not resubmitting. Investigate before continuing."
    else
        sbatch --dependency=afterany:"$SLURM_JOB_ID" \
            --export=ALL,CHAIN_REMAINING=$((CHAIN_REMAINING - 1)) \
            "$SELF" "$MODE"
    fi
    exit 0
}
trap resubmit_on_timeout USR1

python scripts/train_audio_conditioned.py \
    --train-data "$TRAIN_DATA" \
    --slakh-root "$SLAKH_ROOT" \
    --midi-ckpt "$MIDI_CKPT" \
    --encoder-config "$ENC_CFG" \
    --synthesizer soundfont \
    --soundfont-dir soundfonts/ \
    --batch-size 16 \
    --max-epochs "$MAX_EPOCHS" \
    --grad-accum 1 \
    --num-workers 12 \
    --precision bf16 \
    --training-mode "$MODE" \
    --output-dir "$OUTPUT_DIR" \
    --resume-from auto \
    --wandb --wandb-project midigpt --wandb-run-name "amber_medium_${MODE}" &
PYPID=$!
wait "$PYPID"
STATUS=$?
echo "Training process exited with status $STATUS"

if [ -f "$FINAL" ]; then
    echo "Training complete: $FINAL"
    exit 0
fi

if [ "$CHAIN_REMAINING" -le 1 ]; then
    echo "Chain exhausted without reaching model_final.safetensors -- not resubmitting. Investigate before continuing."
    exit 1
fi

# Only chain on a clean exit (STATUS==0): that's the walltime-continuation
# case this chain exists for -- the process made progress and stopped on
# its own, --resume-from auto picks it back up next link. A nonzero STATUS
# (OOM, crash, signal) is not a transient condition: rerunning the same
# code against the same data reproduces it, so resubmitting on failure just
# repeats the failure on a fresh GPU allocation instead of catching it.
if [ "$STATUS" -ne 0 ]; then
    echo "Training exited with a non-zero status ($STATUS) -- not resubmitting. Investigate before continuing."
    exit 1
fi

echo "Resubmitting chain link (remaining after this: $((CHAIN_REMAINING - 1)))"
sbatch --dependency=afterany:"$SLURM_JOB_ID" \
    --export=ALL,CHAIN_REMAINING=$((CHAIN_REMAINING - 1)) \
    "$SELF" "$MODE"

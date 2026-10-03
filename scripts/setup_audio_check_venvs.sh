#!/bin/bash
# Build the 4 per-extractor "-check" venvs the audio screen uses
# (train_audio_screen.sh selects one per extractor; their transformers pins
# conflict, so they must be separate venvs):
#   venv-mert-check      .[mert]        (also used by the dac extractor)
#   venv-sheetsage2-check .[sheetsage2]
#   venv-muq-check       .[muq]
#   venv-same-check      .[same]
#
# Wired as a PRISM setup step (one line per cluster) so `prism install`
# reproduces them. Idempotent: skips a venv that already imports its extra.
#
# Hard-won recipe details:
#   --system-site-packages : use Alliance's +computecanada transformers/torch
#                            wheels instead of rebuilding them.
#   symusic via --only-binary=:all: : grab the prebuilt Alliance wheel; its
#                            sdist build does a CMake FetchContent clone of
#                            3rdparty/Catch2 that fails (and, on an
#                            inode-tight /scratch, dies with "Disk quota
#                            exceeded" mid-clone).
#   scikit-build-core + pybind11 pre-installed + --no-build-isolation : midigpt's
#                            native _core (CMake) can't see the module pyarrow/
#                            cmake inside pip's isolated build env otherwise.
#   rustup default stable : some transitive deps build a Rust sdist (tokenizers).
#   pyarrow constraint : pin to the Arrow module version so pip never tries to
#                        build Alliance's dummy pyarrow-noinstall wheel.
#
# NOTE each venv is ~tens of thousands of files; needs /scratch inode headroom
# (Alliance caps /scratch at 1e6 files/user). A cluster already near that limit
# can't host all 5 audio venvs.
set -u
module load python/3.11.5 gcc/12.3 cmake/3.31.0 arrow/19.0.1
rustup default stable >/dev/null 2>&1 || true

WT=/scratch/triana24/MIDI-GPT/worktree-audio-conditioning
VD=/scratch/triana24/MIDI-GPT/venvs

# venv-audio-conditioning hosts three extractors directly (yourmt3, muscriptor,
# music2latent). The base `pip install -e .[train]` does NOT pull their runtime
# packages, so add them here (idempotent). muscriptor + einops + music2latent
# install cleanly on top of the Alliance transformers wheel (no conflict).
if [ -d "$VD/venv-audio-conditioning/bin" ]; then
  # shellcheck disable=SC1091
  source "$VD/venv-audio-conditioning/bin/activate"
  python -c "import muscriptor, einops, music2latent, sympy" 2>/dev/null \
    || pip install -q muscriptor einops music2latent sympy
  deactivate 2>/dev/null || true
fi
TMP="${TMPDIR:-/scratch/triana24/.tmp}"; mkdir -p "$TMP"
printf "pyarrow==%s\n" "$(python3 -c 'import pyarrow; print(pyarrow.__version__)')" > "$TMP/pyarrow_constraint.txt"

build() {  # $1 = venv name, $2 = pip extra
  local v="$1" x="$2"
  if [ -d "$VD/$v/bin" ] && "$VD/$v/bin/python" -c "import midigpt" 2>/dev/null; then
    # midigpt present, but still ensure the runtime deps that the extractor
    # extra (.[$x]) does NOT pull but training needs in EVERY extractor venv:
    #   pyfluidsynth : synth.py renders MIDI->audio (soundfont) in every venv.
    #   lightning/wandb/einops : train_audio_conditioned.py runs the full
    #     Lightning loop IN the selected venv, so the -check venvs need the
    #     training-loop deps too, not just their extractor. Installed
    #     explicitly (NOT via .[train], which would repin transformers and
    #     break the extractor the -check venv exists to isolate).
    "$VD/$v/bin/python" -c "import fluidsynth" 2>/dev/null || "$VD/$v/bin/pip" install -q pyfluidsynth
    "$VD/$v/bin/python" -c "import lightning, wandb, einops, datasets, dotenv" 2>/dev/null || "$VD/$v/bin/pip" install -q lightning wandb einops datasets python-dotenv
    echo "[$v] present (runtime deps ensured)"; return 0
  fi
  echo "[$v] building (.[$x]) ..."
  python3 -m venv --system-site-packages "$VD/$v"
  # shellcheck disable=SC1091
  source "$VD/$v/bin/activate"
  pip install -q --upgrade pip "scikit-build-core>=0.9" pybind11
  pip install -q --only-binary=:all: symusic
  ( cd "$WT" && pip install --no-build-isolation -e ".[$x]" -c "$TMP/pyarrow_constraint.txt" )
  pip install -q pyfluidsynth          # MIDI->audio synth (synth.py) — every extractor venv
  # training-loop deps (not pulled by .[$x]): lightning/wandb from [train],
  # datasets+python-dotenv also from [train] (dataset.py:493 imports datasets),
  # einops for yourmt3/sheetsage2. transformers left to the extractor pin.
  pip install -q lightning wandb einops datasets python-dotenv
  deactivate 2>/dev/null || true
  echo "[$v] done"
}

build venv-mert-check       mert
build venv-sheetsage2-check  sheetsage2
build venv-muq-check        muq
build venv-same-check       same
echo "setup_audio_check_venvs: all 4 extractor venvs ready"

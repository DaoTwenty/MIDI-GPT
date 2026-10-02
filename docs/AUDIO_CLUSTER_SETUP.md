# Audio-conditioning cluster setup — the full dependency closure

The audio-extractor screen (`scripts/train_audio_screen.sh`, job
`.prism/jobs/audio-screen.yaml`) needs more than code + a venv. This file is the
**golden manifest** so a new cluster is a one-shot `prism install` + a few asset
syncs instead of a day of discover-by-failure. Captured 2026-10-02 after
bringing all 5 clusters (nibi, fir, rorqual, narval, trillium) online.

## 1. Virtual environments (5)

The 8 extractors have conflicting `transformers` pins, so the screen selects a
venv per extractor (see the lookup table in `train_audio_screen.sh`). All live
under `$SCRATCH/MIDI-GPT/venvs/`:

| venv | hosts extractors | built by |
|------|------------------|----------|
| `venv-audio-conditioning` | yourmt3, muscriptor, music2latent | `prism install` step `pip install -e .[train]` **+** `scripts/setup_audio_check_venvs.sh` (adds `muscriptor einops music2latent`) |
| `venv-mert-check` | mert, dac | `setup_audio_check_venvs.sh` → `.[mert]` |
| `venv-sheetsage2-check` | sheetsage2 | `setup_audio_check_venvs.sh` → `.[sheetsage2]` |
| `venv-muq-check` | muq | `setup_audio_check_venvs.sh` → `.[muq]` |
| `venv-same-check` | same | `setup_audio_check_venvs.sh` → `.[same]` |

Exact package sets (reference locks, narval a100 2026-10-02):
`requirements/audio-venv-locks/*.lock.txt`. Build gotchas are documented in the
header of `scripts/setup_audio_check_venvs.sh` (symusic Alliance wheel,
`--no-build-isolation`, rustup, pyarrow constraint).

**Import smoke test** (run WITH `module load python/3.11.5 gcc/12.3 cmake/3.31.0
arrow/19.0.1` — `--system-site-packages` deps are invisible without it):
```bash
venv-audio-conditioning: python -c "import lightning, muscriptor, einops, music2latent"
venv-mert-check:         python -c "import lightning, transformers, torchaudio"
# etc. per extractor
```

## 2. Data + weight assets (NOT in git — sync per cluster)

| asset | path (under `$SCRATCH/MIDI-GPT/`) | size | source / how to fetch |
|-------|----------------------------------|------|-----------------------|
| MIDI-GPT backbone | `runs/yellow_medium-20260621-115325/model_final.safetensors` | 335M | nibi only → `globus`/`prism data sync` from nibi |
| extractor weights | `hf_cache/{dac_44khz,mert-v2-*,muq-large-msd-iter,same-{l,s},sheetsage2,yourmt3,muscriptor-*}` | 21G | public HF repos (`descript/dac_44khz`, `m-a-p/MERT-v2-FullSong`, `m-a-p/SheetSage2`, `OpenMuQ/MuQ-large-msd-iter`, `stabilityai/SAME-L`, `mimbres/yourmt3-small`); `muscriptor-*` via the `muscriptor` pkg. Fetch per-cluster or Globus from nibi |
| music2latent weights | `hf_cache/hub/models--SonyCSLParis--music2latent/` | ~small | `HF_HOME=$SCRATCH/MIDI-GPT/.hf_cache python -c "from huggingface_hub import snapshot_download; snapshot_download('SonyCSLParis/music2latent')"` on the login node (air-gapped compute can't download at runtime) |
| Slakh2100 render | `data/slakh2100/slakh2100_flac_redux/` (1289/270/151 train/val/test tracks) | 101G | public Zenodo record `4599666` (`slakh2100_flac_redux.tar.gz`) → per-cluster wget+untar, OR Globus the extracted tree from nibi |
| GigaMIDI | `data/v2.0.0/{train,validation}/*.parquet` | 8.3G | from nibi via Globus |
| encoder config | `models/amber_encoder.json` | tiny | in the repo (committed) |

## 3. Resource budget per cluster

- **Space:** ~130 GB (101G slakh + 21G weights + 8.3G GigaMIDI + venvs).
- **Inodes:** the extracted Slakh is ~100k+ small files and each venv is ~tens
  of thousands — **Alliance caps /scratch at 1e6 files/user**. The full audio
  stack + 3 project worktrees + other venvs can exhaust it (fir hit 998K/1M). A
  "Disk quota exceeded" mid-`pip`/mid-build is usually the **inode** limit —
  check `diskusage_report` files column, not just space.

## 4. Pre-flight before submitting the screen (would've saved a day)

On each target cluster, confirm: the 5 venvs import their extractor packages
(with module loaded); the backbone `.safetensors` exists and is nonzero; the
`hf_cache` extractor dirs + the music2latent hub cache exist; the Slakh train
dir has 1289 tracks with real (~45M) content, not just empty track folders; and
`diskusage_report` has inode headroom. See the companion `requires_files`/
`requires_imports`/`requires_inodes` pre-flight proposal in
`~/Workspace/PRISM_FEATURE_REQUESTS.md`.

#!/usr/bin/env python3
"""Pre-synthesize every track of a GigaMIDI-format dataset across every
soundfont in a pool, populating the on-disk AudioRenderCache that
AudioTransitionDataset consults at training time.

This is the same cache in both places: this script only *writes* it ahead of
time so the dataloader's hot path can *read* pre-rendered audio instead of
calling FluidSynth live during training.

Usage:
    python scripts/presynthesize_audio_cache.py \
        --train-data /scratch/triana24/MIDI-GPT/data/v2.0.0/train/*.parquet \
        --soundfont-dir soundfonts/ \
        --cache-dir /scratch/triana24/MIDI-GPT/data/audio_cache \
        --max-samples 5000 \
        --num-workers 8
"""

from __future__ import annotations

import argparse
import glob
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Set by _worker_init in each subprocess — a per-process SoundFontSynthesizer
# so FluidSynth handles and loaded .sf2 fonts are reused across scores
# instead of being torn down and rebuilt for every single task.
_worker_synth = None


def _worker_init(soundfont_dir: str, sample_rate: int) -> None:
    global _worker_synth
    from midigpt.audio.synth import SoundFontSynthesizer

    _worker_synth = SoundFontSynthesizer(soundfont_dir, sample_rate=sample_rate)


def _render_score_task(
    music_bytes: bytes, score_id: str, cache_dir: str, sample_rate: int, overwrite: bool,
) -> dict:
    """Render every track x every soundfont for one score. Runs in a worker process."""
    from midigpt._types import Score as PyScore
    from midigpt.audio.render_cache import AudioRenderCache
    from midigpt.audio.synth import get_bar_intervals

    stats = {"rendered": 0, "skipped": 0, "errors": 0}
    cache = AudioRenderCache(cache_dir)
    synth = _worker_synth

    try:
        score = PyScore.from_bytes(music_bytes)
    except Exception:
        stats["errors"] += 1
        return stats

    if not cache.has_bar_intervals(score_id) or overwrite:
        try:
            cache.save_bar_intervals(score_id, get_bar_intervals(score))
        except Exception:
            stats["errors"] += 1
            return stats

    for track_idx in range(len(score.tracks)):
        for sf2_idx, sf2_name in enumerate(synth.soundfont_names):
            if not overwrite and cache.has(score_id, track_idx, sf2_name):
                stats["skipped"] += 1
                continue
            try:
                wav = synth.render_track(score, track_idx, sf2_index=sf2_idx)
                cache.save(score_id, track_idx, sf2_name, wav)
                stats["rendered"] += 1
            except Exception:
                stats["errors"] += 1

    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-synthesize every track x every soundfont into the on-disk audio render cache."
    )
    parser.add_argument("--train-data", required=True, help="Parquet shard(s): path, list, or glob")
    parser.add_argument("--soundfont-dir", default="soundfonts", help="Directory of .sf2 files")
    parser.add_argument("--cache-dir", required=True, help="Output cache directory")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit to N scores (0=all)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Re-render even if already cached")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s", datefmt="%H:%M:%S")
    args = parse_args()

    import datasets as hf

    paths = []
    for p in args.train_data.split(","):
        p = p.strip()
        paths.extend(sorted(glob.glob(p)) if "*" in p else [p])
    log.info("Loading parquets: %s", paths)
    data = hf.load_dataset("parquet", data_files=paths, split="train")
    n_total = len(data) if args.max_samples <= 0 else min(args.max_samples, len(data))
    log.info("Pre-synthesizing %d / %d scores", n_total, len(data))

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    totals = {"rendered": 0, "skipped": 0, "errors": 0}
    start = time.time()
    done = 0

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=_worker_init,
        initargs=(args.soundfont_dir, args.sample_rate),
    ) as pool:
        futures = {
            pool.submit(
                _render_score_task,
                data[i]["music"],
                data[i]["md5"],
                args.cache_dir,
                args.sample_rate,
                args.overwrite,
            ): i
            for i in range(n_total)
        }
        for future in as_completed(futures):
            stats = future.result()
            for k in totals:
                totals[k] += stats[k]
            done += 1
            if done % args.progress_every == 0 or done == n_total:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0.0
                log.info(
                    "  %d/%d scores (%.1f/s) — rendered=%d skipped=%d errors=%d",
                    done, n_total, rate, totals["rendered"], totals["skipped"], totals["errors"],
                )

    elapsed = time.time() - start
    log.info(
        "Done in %.1fs. rendered=%d skipped=%d errors=%d",
        elapsed, totals["rendered"], totals["skipped"], totals["errors"],
    )


if __name__ == "__main__":
    main()

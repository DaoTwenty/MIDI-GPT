"""Generate models/amber_encoder.json — yellow's vocab plus an AudioContext token.

amber = yellow + audio tracks. It is a genuinely different encoder config, not a
patched yellow_encoder.json: yellow_small/medium/large keep pointing at the
original, unmodified yellow_encoder.json.

Why this can't be a full regenerate via EncoderConfig.derive_token_domains():
yellow_encoder.json uses the explicit "token_domains" (legacy fixture) path, and
its list does not match what derive_token_domains() produces today (e.g. NumBars
domain_size is 2 here vs. len(num_bars_map)=16 if derived). Re-deriving from
scratch would silently reassign nearly every token's ID.

Why AudioContext must be appended, not inserted where derive_token_domains()
would put it (right after BarEnd): GPT2LMHeadModel.resize_token_embeddings()
(src/python/midigpt/inference/model/gpt2.py:298) does a naive index-based copy —
new_wte.weight[:n_copy] = old_wte.weight[:n_copy] — so it is only safe when new
tokens are appended strictly at the tail (positions >= yellow's vocab_size). This
is also why Track's domain_size stays 2 here instead of growing to 3 for the
audio track value: Track sits near the front of the list, so widening it in
place would shift every following token's ID and corrupt yellow_small/medium/
large's pretrained embeddings if warm-started via _load_midi_ckpt. Audio tracks
are still fully identifiable via the per-bar AudioContext token; the Track-level
value distinguishing audio from drum is a follow-up (would need a real vocab
redesign + checkpoint remap, deliberately out of scope here).

num_bars_map and time_signatures are left byte-identical to yellow's on purpose:
_load_or_build_valid_indices' cache key (src/python/midigpt/training/dataset.py)
is derived only from those two fields (plus min_tracks/parquet path), not from
token_domains, so the existing filter cache is reused rather than rebuilt.
"""
import json
from pathlib import Path

import midigpt._core as _core

REPO_ROOT = Path(__file__).resolve().parent.parent
YELLOW_PATH = REPO_ROOT / "models" / "yellow_encoder.json"
AMBER_PATH = REPO_ROOT / "models" / "amber_encoder.json"


def main() -> None:
    yellow_dict = json.loads(YELLOW_PATH.read_text())

    amber_dict = json.loads(json.dumps(yellow_dict))  # deep copy
    amber_dict["token_domains"].append({"domain_size": 1, "type": "AudioContext"})

    yellow_cfg = _core.EncoderConfig.from_json(json.dumps(yellow_dict))
    amber_cfg = _core.EncoderConfig.from_json(json.dumps(amber_dict))
    yellow_vocab = _core.Vocabulary(yellow_cfg)
    amber_vocab = _core.Vocabulary(amber_cfg)

    # Every token type yellow already had must keep the exact same offset in amber.
    for entry in yellow_dict["token_domains"]:
        token_type = getattr(_core.TokenType, entry["type"])
        yellow_offset, _ = yellow_vocab.range(token_type)
        amber_offset, _ = amber_vocab.range(token_type)
        assert amber_offset == yellow_offset, (
            f"{entry['type']} offset moved: yellow={yellow_offset} amber={amber_offset}"
        )
        assert amber_vocab.domain_size(token_type) == entry["domain_size"], (
            f"{entry['type']} domain_size changed unexpectedly"
        )

    assert amber_vocab.has(_core.TokenType.AudioContext)
    audio_ctx_offset, _ = amber_vocab.range(_core.TokenType.AudioContext)
    assert audio_ctx_offset == yellow_vocab.size(), (
        "AudioContext must land strictly at yellow's old vocab tail"
    )
    assert amber_vocab.size() == yellow_vocab.size() + 1

    # Fields the filter cache key is derived from must stay byte-identical.
    assert amber_dict["num_bars_map"] == yellow_dict["num_bars_map"]
    assert amber_dict["time_signatures"] == yellow_dict["time_signatures"]

    AMBER_PATH.write_text(json.dumps(amber_dict, indent=4, sort_keys=True) + "\n")
    print(f"Wrote {AMBER_PATH}")
    print(f"  yellow vocab_size = {yellow_vocab.size()}")
    print(f"  amber  vocab_size = {amber_vocab.size()}")
    print(f"  AudioContext token id = {audio_ctx_offset}")


if __name__ == "__main__":
    main()

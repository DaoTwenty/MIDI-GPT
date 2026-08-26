"""
Compare two MIDI-GPT checkpoints for weight equivalence under possible
vocab reindexing.  Embedding rows (wte, lm_head) may be permuted if token
IDs were renumbered; all other tensors must match exactly.

Usage:
    python scripts/compare_checkpoints.py <path_a> <path_b>
"""
import sys
import torch


EMBED_KEYS = {"transformer.wte.weight", "lm_head.weight"}


TORCHSCRIPT_PREFIX_MAP = {
    # TorchScript key prefix → packed-bundle key prefix
    "wte.":  "transformer.wte.",
    "wpe.":  "transformer.wpe.",
    "h.":    "transformer.h.",
    "ln_f.": "transformer.ln_f.",
}


def _normalize_key(k: str) -> str:
    """Map TorchScript key names to packed-bundle key names."""
    for src, dst in TORCHSCRIPT_PREFIX_MAP.items():
        if k.startswith(src):
            return dst + k[len(src):]
    return k  # lm_head.weight already matches


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = torch.load(path, map_location="cpu", weights_only=False)

    # Packed bundle
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
        return {
            k: v for k, v in sd.items()
            if not k.endswith(("attn.bias", "attn.masked_bias"))
        }

    # TorchScript archive — torch.load dispatches to jit.load
    try:
        m = torch.jit.load(path, map_location="cpu")
        sd = m.state_dict()
        return {
            _normalize_key(k): v for k, v in sd.items()
            if not k.endswith(("attn.bias", "attn.masked_bias"))
        }
    except Exception:
        pass

    raise ValueError(f"Unrecognised checkpoint format in {path}")


def compare_non_embed(a: dict, b: dict) -> list[str]:
    issues = []
    keys_a = {k for k in a if k not in EMBED_KEYS}
    keys_b = {k for k in b if k not in EMBED_KEYS}
    if keys_a != keys_b:
        issues.append(f"Key mismatch — only in A: {keys_a - keys_b}; only in B: {keys_b - keys_a}")
        return issues
    for k in sorted(keys_a):
        ta, tb = a[k], b[k]
        if ta.shape != tb.shape:
            issues.append(f"{k}: shape {ta.shape} vs {tb.shape}")
        elif not torch.allclose(ta.float(), tb.float(), atol=1e-6):
            diff = (ta.float() - tb.float()).abs()
            issues.append(
                f"{k}: max_diff={diff.max():.2e}  mean_diff={diff.mean():.2e}"
            )
    return issues


def row_fingerprints(t: torch.Tensor) -> torch.Tensor:
    """Stable per-row fingerprint: concat [sum, sumsq, first-4-values]."""
    f = t.float()
    s = f.sum(dim=1, keepdim=True)
    ss = (f ** 2).sum(dim=1, keepdim=True)
    head = f[:, :4] if f.shape[1] >= 4 else f
    return torch.cat([s, ss, head], dim=1)


def compare_embed(key: str, a: dict, b: dict) -> list[str]:
    issues = []
    if key not in a and key not in b:
        return issues
    if key not in a or key not in b:
        issues.append(f"{key}: present in one checkpoint only")
        return issues

    ta, tb = a[key].float(), b[key].float()
    if ta.shape != tb.shape:
        issues.append(f"{key}: shape {ta.shape} vs {tb.shape}")
        return issues

    # Fast path: identical
    if torch.allclose(ta, tb, atol=1e-6):
        print(f"  {key}: identical (same token ordering)")
        return issues

    # Sort rows by fingerprint and compare
    fp_a = row_fingerprints(ta)
    fp_b = row_fingerprints(tb)

    # Lexicographic sort on fingerprints
    def lex_sort(fp):
        cols = [fp[:, i] for i in reversed(range(fp.shape[1]))]
        idx = torch.zeros(fp.shape[0], dtype=torch.long)
        # numpy argsort is simpler for lex sort
        import numpy as np
        arr = fp.numpy()
        idx = np.lexsort(arr[:, ::-1].T)
        return torch.from_numpy(idx)

    idx_a = lex_sort(fp_a)
    idx_b = lex_sort(fp_b)

    sorted_a = ta[idx_a]
    sorted_b = tb[idx_b]

    if torch.allclose(sorted_a, sorted_b, atol=1e-6):
        print(f"  {key}: same rows, different ordering (vocab reindex confirmed)")
    else:
        diff = (sorted_a - sorted_b).abs()
        issues.append(
            f"{key}: rows differ after sorting — "
            f"max_diff={diff.max():.2e}  mean_diff={diff.mean():.2e}"
        )
    return issues


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_checkpoints.py <path_a> <path_b>")
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]
    print(f"A: {path_a}")
    print(f"B: {path_b}")

    sd_a = load_state_dict(path_a)
    sd_b = load_state_dict(path_b)

    print(f"\nKeys in A: {len(sd_a)}  Keys in B: {len(sd_b)}")

    issues: list[str] = []

    print("\n-- Non-embedding layers --")
    issues += compare_non_embed(sd_a, sd_b)

    print("\n-- Embedding / lm_head layers --")
    for key in sorted(EMBED_KEYS):
        issues += compare_embed(key, sd_a, sd_b)

    if issues:
        print("\n[FAIL] Differences found:")
        for msg in issues:
            print(f"  • {msg}")
        sys.exit(1)
    else:
        print("\n[PASS] Checkpoints are weight-equivalent.")


if __name__ == "__main__":
    main()

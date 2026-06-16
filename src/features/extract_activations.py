"""Stage 3 (GPU): extract hidden-state activations and random-project them (report 3.3.1).

For each input: forward pass with output_hidden_states, take layers 8-23 (16 layers),
project 4096 -> 256 with a fixed Gaussian random matrix (seed 42), stack to
(N, 16, 64, 256) and save. Lazy transformers import; not run in CI.

Usage (GPU):
    python -m src.features.extract_activations --in data/labeled.jsonl --out data/activations.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LAYERS = tuple(range(8, 24))   # 16 middle layers
MAX_TOKENS = 64
RAW_DIM = 4096
PROJ_DIM = 256
SEED = 42
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"


def random_projection_matrix() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return (rng.standard_normal((RAW_DIM, PROJ_DIM)) / np.sqrt(RAW_DIM)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("data/labeled.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/activations.npz"))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, load_in_8bit=True, device_map="auto")
    model.eval()

    w_proj = torch.from_numpy(random_projection_matrix()).to(model.device)
    rows = [json.loads(line) for line in args.inp.read_text(encoding="utf-8").splitlines()]

    tensors, labels = [], []
    for r in rows:
        enc = tok(
            r["question"], max_length=MAX_TOKENS, truncation=True,
            padding="max_length", return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        # hidden_states: tuple of (1, T, 4096); pick layers 8-23
        hs = torch.stack([out.hidden_states[ell][0] for ell in LAYERS])  # (16, T, 4096)
        projected = (hs @ w_proj).cpu().numpy().astype(np.float32)        # (16, T, 256)
        tensors.append(projected)
        labels.append(r["label"])

    X = np.stack(tensors)              # (N, 16, 64, 256)
    y = np.asarray(labels, dtype=np.int64)
    assert not np.isnan(X).any() and not np.isinf(X).any(), "NaN/Inf in activations"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y)


if __name__ == "__main__":
    main()

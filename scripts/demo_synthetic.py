"""Try the whole pipeline WITHOUT a GPU or the Llama model.

Generates synthetic activations with a planted, separable signal in layers 12-18
(mimicking the thesis finding), then runs the full 4-probe + rank-ensemble pipeline
and prints AUC/F1 for each probe and the ensemble.

This is a *pipeline* demo on fake data — NOT a measure of real hallucination-detection
performance (that needs real Llama-3.1-8B activations from extract_activations.py on a GPU).
It proves the code path works end-to-end and that the ensemble beats individual probes.

    python -m scripts.demo_synthetic
    python -m scripts.demo_synthetic --n 600 --signal 0.20
"""

from __future__ import annotations

import argparse

import numpy as np

from src.train import run


def make_synthetic(n: int = 400, signal: float = 0.25, seed: int = 0):
    """(N, 16, 64, 256) activations; hallucinated class gets a bump in layers 12-18."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 16, 64, 256)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    X[y == 1, 12:18, :, :] += signal  # planted signal where the thesis localizes it
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--signal", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 70)
    print("SYNTHETIC pipeline demo — fake activations, real code path (no GPU/LLM).")
    print(f"  n={args.n}, planted signal={args.signal} in layers 12-18")
    print("  (Real numbers require Llama-3.1-8B activations; see README.)")
    print("=" * 70)
    X, y = make_synthetic(args.n, args.signal, args.seed)
    run(X, y, seed=args.seed)


if __name__ == "__main__":
    main()

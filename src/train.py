"""Train the four probes and the rank ensemble on extracted activations (report Ch. 3-4).

Runnable end-to-end on CPU once data/activations.npz exists (produced on GPU by
extract_activations.py). The neural probes are small enough to train on CPU/4GB GPU.

Usage:
    python -m src.train --data data/activations.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.evaluate import binary_metrics, print_report
from src.features.build_features import (
    compact_features,
    pca_source,
    rich_features,
)
from src.models.classifiers import (
    HallucinationClassifier,
    TinyConvProbe,
    build_compact_lr,
    build_rbf_svm,
    count_params,
)
from src.models.ensemble import RankEnsemble


def _train_torch_probe(model, Xtr, ytr, epochs=100, lr=3e-4, weight_decay=5e-3, patience=20):
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    xb = torch.from_numpy(Xtr).float()
    yb = torch.from_numpy(ytr).float()
    best, best_state, waited = float("inf"), None, 0
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        if loss.item() < best - 1e-4:
            best, best_state, waited = loss.item(), model.state_dict(), 0
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _probe_proba(model, X):
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(X).float())).numpy()


def run(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[RankEnsemble, dict]:
    """Full pipeline: 4 probes + rank ensemble. Returns (ensemble, ensemble_metrics).

    Works on any (N, 16, 64, 256) activation tensor — real (from extract_activations)
    or synthetic (scripts/demo_synthetic.py). Pure CPU.
    """
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split

    X = X.astype(np.float32)
    y = y.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    # --- classical probes on engineered features ---
    pca = PCA(n_components=128, random_state=seed).fit(pca_source(Xtr))
    lr = build_compact_lr().fit(compact_features(Xtr, pca), ytr)
    svm = build_rbf_svm().fit(rich_features(Xtr), ytr)

    # --- neural probes on raw tensor ---
    mlp = _train_torch_probe(HallucinationClassifier(), Xtr, ytr)
    conv = _train_torch_probe(TinyConvProbe(), Xtr, ytr)
    print(f"AttentionMLP params={count_params(mlp)}  TinyConvProbe params={count_params(conv)}")

    probs = {
        "lr": lr.predict_proba(compact_features(Xte, pca))[:, 1],
        "svm": svm.predict_proba(rich_features(Xte))[:, 1],
        "mlp": _probe_proba(mlp, Xte),
        "conv": _probe_proba(conv, Xte),
    }
    for name, p in probs.items():
        print_report(name, binary_metrics(yte, p))

    ens = RankEnsemble()
    fused = ens.fuse(probs)
    ens.select_threshold(fused, yte)
    metrics = binary_metrics(yte, fused, threshold=ens.threshold)
    print_report("ensemble", metrics)
    return ens, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/activations.npz"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    blob = np.load(args.data)
    run(blob["X"], blob["y"], args.seed)


if __name__ == "__main__":
    main()

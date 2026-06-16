"""Rank-based ensemble of the four probes (report Section 3.4.2).

Rank-normalising each model's probabilities removes calibration differences between
models trained with different losses/feature scales, then fuses with fixed weights.
"""

from __future__ import annotations

import numpy as np

# Fixed fusion weights (report): LR 0.15, SVM 0.30, MLP 0.35, Conv 0.20
DEFAULT_WEIGHTS: dict[str, float] = {"lr": 0.15, "svm": 0.30, "mlp": 0.35, "conv": 0.20}


def rank_normalize(probs: np.ndarray) -> np.ndarray:
    """Map scores to rank/N in [1/N, 1]. Ties broken by position (stable)."""
    probs = np.asarray(probs, dtype=float)
    order = probs.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(probs) + 1)
    return ranks / len(probs)


class RankEnsemble:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.threshold: float = 0.5

    def fuse(self, prob_dict: dict[str, np.ndarray]) -> np.ndarray:
        """prob_dict maps model name -> 1D array of P(hallucinated). All same length."""
        total: np.ndarray | None = None
        for name, w in self.weights.items():
            contribution = rank_normalize(prob_dict[name]) * w
            total = contribution if total is None else total + contribution
        assert total is not None, "no models provided to fuse"
        return total

    def select_threshold(
        self, fused: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None
    ) -> tuple[float, float]:
        """Pick the F1-maximising threshold on a validation set (report: tau* = 0.57)."""
        from sklearn.metrics import f1_score

        grid = grid if grid is not None else np.arange(0.05, 0.95, 0.01)
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = f1_score(y, (fused >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = float(f1), float(t)
        self.threshold = best_t
        return best_t, best_f1

    def predict(self, fused: np.ndarray) -> np.ndarray:
        return (fused >= self.threshold).astype(int)

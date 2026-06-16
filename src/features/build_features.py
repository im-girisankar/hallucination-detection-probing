"""Feature engineering from the activation tensor (report Section 3.3.2).

Two representations:
  - rich_features:    8,976 dims (per-layer stats + trajectory + volatility + norms)
  - compact (191):    PCA(128) of the mean+std source  ++  63 hand-designed features

The PCA block is fit on the training split (see train.py); the hand features and the
PCA *source* are computed deterministically here.

All inputs are X of shape (N, L, T, D) = (samples, 16 layers, 64 tokens, 256 proj dim).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def _per_layer_mean_std(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    M = X.mean(axis=2)  # (N, L, D)
    S = X.std(axis=2)   # (N, L, D)
    return M, S


def _cosine(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    num = (a * b).sum(axis=axis)
    den = np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis) + EPS
    return num / den


def _token_entropy(X: np.ndarray) -> np.ndarray:
    """Per (sample, layer) entropy of the token activation-magnitude distribution. -> (N, L)."""
    mag = np.linalg.norm(X, axis=3)                       # (N, L, T)
    p = mag / (mag.sum(axis=2, keepdims=True) + EPS)
    return -(p * np.log(p + EPS)).sum(axis=2)             # (N, L)


def rich_features(X: np.ndarray) -> np.ndarray:
    """8,976-dim rich feature vector."""
    X = np.asarray(X, dtype=np.float32)
    N, L, _, D = X.shape
    M, S = _per_layer_mean_std(X)
    mean_flat = M.reshape(N, L * D)              # 4096
    std_flat = S.reshape(N, L * D)               # 4096
    cross_layer_stab = M.std(axis=1)             # (N, D) = 256
    trajectory = M[:, -4:, :].mean(axis=1) - M[:, :4, :].mean(axis=1)  # 256
    volatility = np.abs(np.diff(M, axis=1)).mean(axis=1)               # 256
    l2 = np.linalg.norm(M, axis=2)               # (N, L) = 16
    return np.concatenate(
        [mean_flat, std_flat, cross_layer_stab, trajectory, volatility, l2], axis=1
    )


def compact_hand_features(X: np.ndarray) -> np.ndarray:
    """The 63 non-PCA components of the compact vector (15 + 16 + 16 + 16)."""
    X = np.asarray(X, dtype=np.float32)
    N, L, _, _ = X.shape
    M, _ = _per_layer_mean_std(X)
    adj_cos = np.stack([_cosine(M[:, ell, :], M[:, ell + 1, :]) for ell in range(L - 1)], axis=1)  # 15
    l2 = np.linalg.norm(M, axis=2)                       # 16
    ent = _token_entropy(X)                              # 16
    last_tok = X[:, :, -1, :]                            # (N, L, D)
    lt_cos = _cosine(last_tok, M, axis=-1)               # 16
    return np.concatenate([adj_cos, l2, ent, lt_cos], axis=1)  # 63


def pca_source(X: np.ndarray) -> np.ndarray:
    """The mean+std source that PCA(128) is fit on -> (N, L*D*2) = (N, 8192)."""
    X = np.asarray(X, dtype=np.float32)
    N = X.shape[0]
    M, S = _per_layer_mean_std(X)
    return np.concatenate([M.reshape(N, -1), S.reshape(N, -1)], axis=1)


def compact_features(X: np.ndarray, pca) -> np.ndarray:
    """Full 191-dim compact vector given an already-fitted sklearn PCA(128)."""
    reduced = pca.transform(pca_source(X))               # (N, 128)
    return np.concatenate([reduced, compact_hand_features(X)], axis=1)  # 191

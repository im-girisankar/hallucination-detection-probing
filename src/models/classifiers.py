"""The four complementary probes (report Section 3.4).

Two neural probes operate on the raw activation tensor X of shape (B, L, T, D)
  L = 16 extracted layers (8-23), T = 64 tokens, D = 256 (projected).
Two classical probes (LR, SVM) operate on engineered features (see ../features).

Param counts are asserted/printed by the tests rather than trusted from the report —
the report's Table B.2 lists TinyConvProbe components summing to ~51K while its stated
total says "~13K"; the listed component breakdown is the faithful one and is what this
implementation reproduces.
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_LAYERS = 16
PROJ_DIM = 256
POOL_DIM = 2 * PROJ_DIM  # per-layer [mean ; std] over tokens -> 512


def token_pool(x: torch.Tensor) -> torch.Tensor:
    """(B, L, T, D) -> (B, L, 2D): per-layer mean & std over the token axis."""
    mean = x.mean(dim=2)
    std = x.std(dim=2)
    return torch.cat([mean, std], dim=-1)


class HallucinationClassifier(nn.Module):
    """Attention MLP — learns per-layer attention weights over the 16 layers,
    then classifies the attended representation. ~34.5K params (report Table B.1)."""

    def __init__(
        self,
        pool_dim: int = POOL_DIM,
        hidden: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.layer_attn = nn.Linear(pool_dim, 1)
        self.layer_norm = nn.LayerNorm(pool_dim)
        self.linear1 = nn.Linear(pool_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        p = token_pool(x)                        # (B, L, 2D)
        scores = self.layer_attn(p)              # (B, L, 1)
        alpha = torch.softmax(scores, dim=1)     # (B, L, 1) — attention over layers
        z = (alpha * p).sum(dim=1)               # (B, 2D)
        z = self.layer_norm(z)
        h = self.dropout(self.act(self.bn1(self.linear1(z))))
        logit = self.linear2(h).squeeze(-1)      # (B,)
        if return_attn:
            return logit, alpha.squeeze(-1)      # alpha: (B, L)
        return logit


class TinyConvProbe(nn.Module):
    """1D-conv probe — treats the 16 layers as a temporal sequence to capture
    layer-transition patterns that attention pooling averages away."""

    def __init__(self, pool_dim: int = POOL_DIM, embed: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.layer_embed = nn.Linear(pool_dim, embed, bias=False)
        self.conv1 = nn.Conv1d(embed, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(64, 1)  # 32 mean-pool + 32 max-pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = token_pool(x)                # (B, L, 2D)
        e = self.layer_embed(p)          # (B, L, embed)
        e = e.transpose(1, 2)            # (B, embed, L)
        h = self.dropout(self.act(self.conv1(e)))   # (B, 64, L)
        h = self.act(self.conv2(h))                 # (B, 32, L)
        mean_pool = h.mean(dim=2)                    # (B, 32)
        max_pool = h.max(dim=2).values               # (B, 32)
        z = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 64)
        return self.head(z).squeeze(-1)               # (B,)


def build_compact_lr():
    """Compact logistic regression on the 191-d feature vector (saga, C=0.5, balanced)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="saga", C=0.5, class_weight="balanced", max_iter=2000)),
        ]
    )


def build_rbf_svm(n_components: int = 128):
    """RBF-SVM on the rich feature vector after PCA(128) (C=5, gamma=scale, balanced)."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
            ("clf", SVC(kernel="rbf", C=5, gamma="scale", class_weight="balanced", probability=True)),
        ]
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

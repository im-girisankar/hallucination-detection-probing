"""Synthetic-activation smoke tests — verify the CPU-runnable core actually works.

No GPU, no Llama, no API. These cover the feature engineering, the four-probe forward/
train paths, the rank ensemble, and the suppression hook. CI runs exactly this.
"""

from __future__ import annotations

import numpy as np
import torch

from src.evaluate import binary_metrics
from src.features.build_features import (
    compact_hand_features,
    pca_source,
    rich_features,
)
from src.intervention.suppression_hook import SuppressionHook
from src.models.classifiers import (
    HallucinationClassifier,
    TinyConvProbe,
    count_params,
)
from src.models.ensemble import RankEnsemble, rank_normalize

N, L, T, D = 24, 16, 64, 256


def _synth(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, L, T, D)).astype(np.float32)
    y = rng.integers(0, 2, size=N)
    X[y == 1, 12:18, :, :] += 0.3  # weak learnable signal in layers 12-18
    return X, y


def test_rich_features_dim():
    X, _ = _synth()
    assert rich_features(X).shape == (N, 8976)


def test_compact_feature_dims():
    X, _ = _synth()
    assert compact_hand_features(X).shape == (N, 63)        # 15+16+16+16
    assert pca_source(X).shape == (N, L * D * 2)            # 8192 -> PCA(128) = 191 total


def test_attention_mlp_forward_and_param_count():
    X, _ = _synth()
    m = HallucinationClassifier()
    p = count_params(m)
    assert 30_000 < p < 40_000, f"expected ~34.5K params, got {p}"  # report Table B.1
    logit, attn = m(torch.from_numpy(X), return_attn=True)
    assert logit.shape == (N,)
    assert attn.shape == (N, L)
    assert torch.allclose(attn.sum(dim=1), torch.ones(N), atol=1e-4)  # softmax over layers


def test_tinyconv_forward_and_param_count():
    X, _ = _synth()
    m = TinyConvProbe()
    p = count_params(m)
    # Report Table B.2 lists components summing to 51,361 (its "~13K" total is a typo);
    # this faithful build reproduces the component breakdown.
    assert 45_000 < p < 60_000, f"got {p}"
    assert m(torch.from_numpy(X)).shape == (N,)


def test_attention_mlp_trains_one_epoch():
    X, y = _synth()
    m = HallucinationClassifier()
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    xb, yb = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    m.train()
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = loss_fn(m(xb), yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.isfinite(losses[-1])
    assert losses[-1] <= losses[0]  # loss did not diverge


def test_rank_normalize_range():
    r = rank_normalize(np.array([0.9, 0.1, 0.5, 0.3]))
    assert r.min() > 0 and r.max() <= 1.0
    assert np.argmax(r) == 0 and np.argmin(r) == 1


def test_ensemble_fuse_threshold_predict():
    rng = np.random.default_rng(1)
    n = 60
    probs = {k: rng.random(n) for k in ("lr", "svm", "mlp", "conv")}
    y = rng.integers(0, 2, n)
    ens = RankEnsemble()
    fused = ens.fuse(probs)
    assert fused.shape == (n,)
    t, f1 = ens.select_threshold(fused, y)
    assert 0.0 <= t <= 1.0 and 0.0 <= f1 <= 1.0
    assert set(np.unique(ens.predict(fused))).issubset({0, 1})


def test_binary_metrics_keys():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 40)
    p = rng.random(40)
    m = binary_metrics(y, p, threshold=0.5)
    assert {"auc", "f1", "precision", "recall", "tp", "fp", "tn", "fn"} <= set(m)


def test_suppression_hook_scales_when_active():
    hook = SuppressionHook(scale=0.5)
    out = torch.ones(2, 4, 8)

    # inactive -> passthrough
    assert torch.allclose(hook(None, None, out), out)

    # active, plain tensor
    hook.active = True
    assert torch.allclose(hook(None, None, out), out * 0.5)

    # active, tuple output (Llama attention returns a tuple)
    tup = (torch.ones(2, 4, 8), "extra")
    scaled = hook(None, None, tup)
    assert torch.allclose(scaled[0], tup[0] * 0.5) and scaled[1] == "extra"

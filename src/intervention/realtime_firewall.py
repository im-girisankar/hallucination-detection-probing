"""
Real-time hallucination firewall  —  ROADMAP phases 1–3 (WORK IN PROGRESS).

This is a *design skeleton*, not a finished system. It encodes the architecture from
ROADMAP.md so the moving parts and their contracts are explicit. Methods raise
NotImplementedError where the implementation is still being built.

Idea
----
Stream the Hallucination Risk Index (HRI) token-by-token during generation. As the model
starts to drift:
  - softly suppress the implicated layers (12-18, learned in the thesis), and/or
  - escalate to retrieval (RAG) — a high HRI is the model signalling "I don't know this".

Phase 0 (thesis) established the pieces this depends on:
  - the probes that produce HRI from layers 8-23,
  - that layers 12-18 carry the signal,
  - that scaling those layers reduces hallucination causally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

# --- Layers identified in the thesis as carrying the hallucination signal ---
SIGNAL_LAYERS: tuple[int, ...] = (12, 13, 14, 15, 16, 17, 18)
DEFAULT_HRI_SUPPRESS = 0.45   # rising-risk band → start suppressing
DEFAULT_HRI_RETRIEVE = 0.65   # high-risk band → escalate to RAG


class HRIEstimator(Protocol):
    """Anything that maps a window of hidden states to a scalar risk in [0, 1].

    In practice this wraps the trained TinyConvProbe (0.05 ms CPU) so the monitor
    stays real-time inside the decode loop.
    """

    def score(self, hidden_states: tuple[torch.Tensor, ...]) -> float:
        ...


class Retriever(Protocol):
    """RAG backend. Returns context to re-ground generation when HRI is high."""

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        ...


@dataclass
class FirewallConfig:
    suppress_threshold: float = DEFAULT_HRI_SUPPRESS
    retrieve_threshold: float = DEFAULT_HRI_RETRIEVE
    signal_layers: tuple[int, ...] = SIGNAL_LAYERS
    score_every_k_tokens: int = 4           # phase 1: HRI cadence
    max_suppression: float = 0.5            # guardrail: don't go below this scale (incoherence at <=0.5 in thesis)
    window_tokens: int = 32                 # rolling activation buffer size


@dataclass
class FirewallTrace:
    """Per-generation telemetry — feeds Langfuse / plots of the live HRI curve."""

    hri_curve: list[float] = field(default_factory=list)
    suppression_events: list[tuple[int, float]] = field(default_factory=list)   # (token_idx, scale)
    retrieval_events: list[tuple[int, str]] = field(default_factory=list)       # (token_idx, query)


class RealTimeFirewall:
    """Online HRI monitor + adaptive suppression + HRI-gated RAG routing.

    Usage (target API):
        fw = RealTimeFirewall(model, tokenizer, hri=tiny_conv_probe, retriever=my_rag)
        text, trace = fw.generate("Who painted the Mona Lisa?", max_new_tokens=128)
    """

    def __init__(
        self,
        model,
        tokenizer,
        hri: HRIEstimator,
        retriever: Retriever | None = None,
        config: FirewallConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.hri = hri
        self.retriever = retriever
        self.cfg = config or FirewallConfig()
        self._handles: list = []

    # ---- Phase 2: adaptive suppression ------------------------------------
    def _suppression_scale(self, hri: float) -> float:
        """Map HRI -> activation scale. Stronger risk => stronger suppression,
        clamped at cfg.max_suppression to preserve fluency."""
        if hri < self.cfg.suppress_threshold:
            return 1.0
        # linear ramp from 1.0 down to max_suppression across the [suppress, retrieve] band
        span = max(self.cfg.retrieve_threshold - self.cfg.suppress_threshold, 1e-6)
        frac = min((hri - self.cfg.suppress_threshold) / span, 1.0)
        return 1.0 - frac * (1.0 - self.cfg.max_suppression)

    def _register_suppression_hooks(self, scale: float) -> None:
        """Attach forward hooks scaling self_attn output on signal layers.
        Mirrors thesis SuppressionHook but with a per-step scale. TODO: implement."""
        raise NotImplementedError("Phase 2: dynamic per-step suppression hook")

    def _clear_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ---- Phase 1: streaming HRI -------------------------------------------
    def _score_window(self, hidden_states) -> float:
        """Run the streaming probe over the rolling activation window. TODO."""
        raise NotImplementedError("Phase 1: streaming token-level HRI")

    # ---- Phase 3: HRI-gated RAG -------------------------------------------
    def _escalate_to_rag(self, query: str, trace: FirewallTrace, token_idx: int) -> list[str]:
        """High HRI -> retrieve and re-ground. TODO: inject context + continue span."""
        if self.retriever is None:
            return []
        ctx = self.retriever.retrieve(query)
        trace.retrieval_events.append((token_idx, query))
        return ctx

    # ---- Main loop --------------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int = 128) -> tuple[str, FirewallTrace]:
        """Custom decode loop that scores HRI every k tokens and intervenes.

        Target behaviour:
          for each step:
            - decode next token, capturing hidden states (layers 8-23)
            - every k tokens: hri = self._score_window(...)
            - if hri >= retrieve_threshold and retriever: escalate_to_rag(...)
            - elif hri >= suppress_threshold: re-register hooks at _suppression_scale(hri)
            - else: clear hooks
        """
        raise NotImplementedError(
            "Phase 1-3 decode loop. See ROADMAP.md. The thesis two-pass version lives in "
            "src/intervention/suppression_hook.py and is the reference implementation."
        )

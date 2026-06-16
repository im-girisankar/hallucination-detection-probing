"""Post-hoc activation suppression (report Section 4.2.5 / Appendix E).

When the ensemble HRI exceeds tau*, scale the output activations of the top attention-weighted
layers (12-18) by a factor s (default 0.8). Targeted suppression reduced hallucination 29.9%
vs 6.5% for uniform suppression in the thesis.

This is the reference two-pass implementation. The online/streaming version is in
realtime_firewall.py (roadmap phases 1-3).
"""

from __future__ import annotations

import torch

# Top-6 attention-weighted layers identified in the thesis as carrying the strongest signal.
DEFAULT_TARGET_LAYERS = (12, 13, 14, 15, 16, 17, 18)


class SuppressionHook:
    """Scales a transformer sub-layer's output activations by `scale` when active."""

    def __init__(self, scale: float = 0.8) -> None:
        self.scale = scale
        self.active = False  # toggled by the ensemble at runtime

    def __call__(self, module, inputs, output):
        if not self.active:
            return output
        if isinstance(output, tuple):
            # Llama-3.1 attention returns (hidden_state, ...)
            return (output[0] * self.scale,) + output[1:]
        return output * self.scale


def register_suppression_hooks(model, target_layers=DEFAULT_TARGET_LAYERS, scale: float = 0.8):
    """Attach a SuppressionHook to each target block's self-attention. Returns (hooks, handles)."""
    hooks, handles = [], []
    for idx in target_layers:
        hook = SuppressionHook(scale=scale)
        handle = model.model.layers[idx].self_attn.register_forward_hook(hook)
        hooks.append(hook)
        handles.append(handle)
    return hooks, handles


def run_with_suppression(model, tokenizer, prompt, ensemble, hooks, threshold: float = 0.57):
    """Two-pass inference: probe pass for HRI, then a generative pass with hooks active if risky."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Step 1: probe pass (no generation, hooks inactive)
    with torch.no_grad():
        probe_out = model(**inputs, output_hidden_states=True)
    hri = ensemble.predict_proba(probe_out.hidden_states)

    # Step 2: enable hooks only if high-risk
    for hook in hooks:
        hook.active = float(hri) >= threshold

    # Step 3: generative pass
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=128)

    for hook in hooks:
        hook.active = False

    return tokenizer.decode(gen_ids[0], skip_special_tokens=True), hri

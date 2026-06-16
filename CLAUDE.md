# Repo conventions for Claude Code

This repo recreates an M.Tech dissertation pipeline (activation-probing hallucination
detection on Llama-3.1-8B). Keep it faithful to the report in `report/` and verifiable.

## Accuracy & verification (non-negotiable)
- Never state metrics, param counts, pricing, SLAs, or library behavior from memory.
  Re-derive numbers from source/code or run it; if you can't, say "unverified".
- Self-check every figure (param counts, AUC, feature dims): re-compute and sanity-check
  the magnitude before writing it down. (The report's Table B.2 param sum is internally
  inconsistent — trust the runnable code, print the real count, note the discrepancy.)
- Do not claim "done"/"works" without running it. Paste real output.

## Code quality
- `ruff` clean (lint + format). Type-hint public functions.
- Every non-GPU module is covered by a test that runs on CPU with synthetic tensors.
- GPU/LLM stages (data gen, activation extraction) must at least import-clean and be
  unit-tested with mocked model I/O. Real runs happen on Kaggle/GPU — document the command.

## Decisions
- Lock architecture/shape choices (tensor layout, feature dims) once; only change with a
  written reason in the commit message. No silent flip-flops.

## Shapes (single source of truth)
- Activation tensor `X`: `(N, 16, 64, 256)` = (samples, layers 8–23, tokens, projected dim).
- Rich features: 8976 dims. Compact features: 191 dims. Signal layers: 12–18.

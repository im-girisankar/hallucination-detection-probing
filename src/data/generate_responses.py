"""Stage 1 (GPU): generate factual QA responses with Llama-3.1-8B (report 3.2.1).

Runs on Kaggle's free T4. Not exercised by CI (needs the gated model + GPU);
imports of transformers/datasets are lazy so the module stays import-clean on CPU.

Usage (on GPU):
    python -m src.data.generate_responses --n_trivia 1000 --n_nq 500 --out data/responses.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROMPT = "Answer the following question concisely.\nQuestion: {question}\nAnswer:"
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"


def load_questions(n_trivia: int, n_nq: int):
    from datasets import load_dataset

    trivia = load_dataset("trivia_qa", "rc.nocontext", split=f"validation[:{n_trivia}]")
    nq = load_dataset("nq_open", split=f"validation[:{n_nq}]")
    items = []
    for r in trivia:
        items.append({"question": r["question"], "gold": r["answer"]["value"]})
    for r in nq:
        items.append({"question": r["question"], "gold": r["answer"][0] if r["answer"] else ""})
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trivia", type=int, default=1000)
    ap.add_argument("--n_nq", type=int, default=500)
    ap.add_argument("--out", type=Path, default=Path("data/responses.jsonl"))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, load_in_8bit=True, device_map="auto")
    model.eval()

    items = load_questions(args.n_trivia, args.n_nq)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            inputs = tok(PROMPT.format(question=it["question"]), return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=64, do_sample=False)  # greedy
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            f.write(json.dumps({**it, "response": text.strip()}) + "\n")
            if (i + 1) % 200 == 0:
                f.flush()  # incremental save against session timeout


if __name__ == "__main__":
    main()

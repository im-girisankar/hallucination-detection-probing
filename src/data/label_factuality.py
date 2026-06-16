"""Stage 2: auto-label responses CORRECT / HALLUCINATED with a GPT-4o-mini judge (report 3.2.2).

Lazy openai import; needs OPENAI_API_KEY. Not run in CI.

Usage:
    python -m src.data.label_factuality --in data/responses.jsonl --out data/labeled.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

JUDGE_PROMPT = (
    "You are an expert fact-checker. Given a question and a model's answer, "
    "determine if the answer is factually correct or hallucinated.\n\n"
    "Question: {question}\nGold Answer: {gold}\nModel Answer: {response}\n\n"
    "Respond with ONLY one word: CORRECT or HALLUCINATED"
)


def normalize(raw: str) -> int:
    """Map a (possibly malformed) judge reply to 1=hallucinated, 0=correct."""
    token = raw.strip().upper().split()[0] if raw.strip() else "CORRECT"
    return 1 if token.startswith("HALL") else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("data/responses.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/labeled.jsonl"))
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI()
    rows = [json.loads(line) for line in args.inp.read_text(encoding="utf-8").splitlines()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            msg = JUDGE_PROMPT.format(question=r["question"], gold=r["gold"], response=r["response"])
            resp = client.chat.completions.create(
                model=args.model, messages=[{"role": "user", "content": msg}], temperature=0
            )
            r["label"] = normalize(resp.choices[0].message.content or "")
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

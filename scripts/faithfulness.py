"""Generate answers from retrieved context (Haiku), judge faithfulness to that context (Sonnet, different model to limit self-preference)."""
import json
import sys
from pathlib import Path

import anthropic
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rageval.data import chunk, load_docs
from rageval.evaluate import load_qa
from rageval.retrieve import Retriever

GEN, JUDGE = "claude-haiku-5-5", "claude-sonnet-5-5"
c = anthropic.Anthropic()
ask = lambda m, p: c.messages.create(model=m, max_tokens=400, messages=[{"role": "user", "content": p}]).content[0].text

r = Retriever(chunk(load_docs(), 600, 0))
out = []
for q in load_qa():
    ctx = "\n\n".join(f"[{i+1}] {r.chunks[j][1]}" for i, j in enumerate(r.search(q["question"], 5, "hybrid", True)))
    ans = ask(GEN, f"Answer using ONLY the context. If it is insufficient, say so.\n\nContext:\n{ctx}\n\nQuestion: {q['question']}")
    v = ask(JUDGE, f"Context:\n{ctx}\n\nAnswer:\n{ans}\n\nIs every factual claim in the Answer supported by the Context? Reply with one short reason, then a final line exactly 'VERDICT: SUPPORTED' or 'VERDICT: UNSUPPORTED'.")
    out.append({"id": q["id"], "answer": ans, "judge": v, "faithful": v.strip().endswith("VERDICT: SUPPORTED")})
    print(q["id"], out[-1]["faithful"], flush=True)
Path("results/faithfulness.jsonl").write_text("\n".join(json.dumps(o) for o in out))
f = np.array([o["faithful"] for o in out], float)
idx = np.random.default_rng(0).integers(0, len(f), (2000, len(f)))
print("faithfulness", f.mean().round(3), np.percentile(f[idx].mean(1), [2.5, 97.5]).round(3))

import json
from pathlib import Path

import numpy as np

from .data import norm

QA = Path(__file__).resolve().parents[2] / "data_eval/qa.jsonl"


def load_qa(path: Path = QA) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def ranks(retriever, qa: list[dict], k: int, **kw) -> list[int | None]:
    """1-based rank of first chunk containing the gold evidence string (and from the gold file), else None."""
    out = []
    for item in qa:
        ev = norm(item["evidence"])
        ids = retriever.search(item["question"], k=k, **kw)
        out.append(next((r + 1 for r, i in enumerate(ids) if ev in retriever.chunks[i][1]), None))
    return out


def metrics(rk: list[int | None], k: int, n_boot: int = 2000, seed: int = 0) -> dict:
    hit = np.array([1.0 if r and r <= k else 0.0 for r in rk])
    rr = np.array([1.0 / r if r and r <= k else 0.0 for r in rk])
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(rk), (n_boot, len(rk)))  # bootstrap over questions
    ci = lambda a: tuple(np.percentile(a[idx].mean(1), [2.5, 97.5]).round(3))
    return {"hit": hit.mean().round(3), "hit_ci": ci(hit), "mrr": rr.mean().round(3), "mrr_ci": ci(rr)}

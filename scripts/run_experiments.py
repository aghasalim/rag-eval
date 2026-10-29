"""Grid over chunk size x overlap x retrieval mode x rerank. Writes results/retrieval.csv."""
import csv
import itertools
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rageval.data import chunk, load_docs
from rageval.evaluate import load_qa, metrics, ranks
from rageval.retrieve import Retriever

random.seed(0); np.random.seed(0)
docs, qa = load_docs(), load_qa()
rows = []
for size, ov_frac in itertools.product([300, 600, 1200, 2400], [0.0, 0.2]):
    ov = int(size * ov_frac)
    r = Retriever(chunk(docs, size, ov))
    for mode, rerank in itertools.product(["dense", "bm25", "hybrid"], [False, True]):
        rk = ranks(r, qa, k=10, mode=mode, rerank=rerank)
        row = {"size": size, "overlap": ov, "n_chunks": len(r.chunks), "mode": mode, "rerank": rerank}
        for k in (1, 5, 10):
            m = metrics(rk, k)
            row[f"hit@{k}"] = m["hit"]
            if k == 5:
                row["hit@5_ci"] = m["hit_ci"]; row["mrr@5"] = m["mrr"]; row["mrr@5_ci"] = m["mrr_ci"]
        rows.append(row); print(row, flush=True)
Path("results").mkdir(exist_ok=True)
with open("results/retrieval.csv", "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)

"""README figures: retrieval pipeline diagram + plots from results/retrieval.csv. Run: python scripts/make_figures.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results"
MODES = {"bm25": "#4c72b0", "dense": "#dd8452", "hybrid": "#55a868"}

# 1. Pipeline diagram (mirrors src/rageval/retrieve.py and evaluate.py).
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.set_xlim(0, 11.4); ax.set_ylim(0, 4); ax.axis("off")


def box(x, y, text, w=2.0, h=0.9, fc="#eef2f8"):
    ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, fc=fc, ec="#333", lw=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5)


def arrow(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color="#333", lw=1))


box(1.0, 2.5, "46 .rst files\nsklearn user guide", w=1.8)
box(3.3, 2.5, "fixed-size char chunks\nsize 300 to 2400\noverlap 0 or 20%", w=2.2, h=1.0)
box(5.9, 3.3, "dense: MiniLM-L6\n+ FAISS inner product", w=2.3)
box(5.9, 1.7, "BM25", w=2.3)
box(8.3, 2.5, "hybrid: RRF (c=60)\nover top 50 of each", w=2.0)
box(10.4, 2.5, "optional rerank:\ncross-encoder\non top 20", w=1.7, h=1.0)
box(9.6, 0.5, "hit@k, MRR@k on 35 questions:\ndoes a chunk contain the gold evidence?", w=3.4, h=0.8, fc="#f6f6f6")
arrow(1.9, 2.5, 2.2, 2.5)
arrow(4.4, 2.7, 4.75, 3.2); arrow(4.4, 2.3, 4.75, 1.8)
arrow(7.05, 3.2, 7.3, 2.8); arrow(7.05, 1.8, 7.3, 2.2)
arrow(9.3, 2.5, 9.55, 2.5)
arrow(10.4, 2.0, 10.4, 0.9)
ax.text(4.6, 0.5, "dense or bm25 mode: top-k from that retriever alone\nhybrid mode: fuse both rankings", ha="center", va="center", fontsize=8, color="#555")
fig.tight_layout()
fig.savefig(OUT / "pipeline.png", dpi=150)

# 2. MRR@5 vs chunk size, per mode, rerank off/on, overlap 0, with 95% bootstrap CIs.
d = pd.read_csv(OUT / "retrieval.csv")
ci = d["mrr@5_ci"].str.findall(r"float64\(([\d.]+)\)").apply(lambda v: [float(x) for x in v])
d["lo"], d["hi"] = ci.str[0], ci.str[1]
z = d[d.overlap == 0]
fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharey=True)
for ax, rr in zip(axes, [False, True]):
    for j, (mode, c) in enumerate(MODES.items()):
        g = z[(z["mode"] == mode) & (z.rerank == rr)].sort_values("size")
        x = np.arange(len(g)) + (j - 1) * 0.08
        ax.errorbar(x, g["mrr@5"], yerr=[g["mrr@5"] - g.lo, g.hi - g["mrr@5"]], fmt="o-", color=c, label=mode, capsize=2, lw=1.2, ms=4)
    ax.set_xticks(range(4), sorted(z["size"].unique()))
    ax.set_xlabel("chunk size (chars)")
    ax.set_title("with rerank" if rr else "no rerank")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("MRR@5 (95% CI)")
axes[0].legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "mrr_by_size.png", dpi=150)

# 3. Rerank effect: MRR@5 with rerank minus without, per size and mode, mean over the two overlaps.
p = d.pivot_table(index=["size", "mode"], columns="rerank", values="mrr@5").reset_index()
p["delta"] = p[True] - p[False]
fig, ax = plt.subplots(figsize=(7, 3.6))
sizes = sorted(p["size"].unique())
for j, (mode, c) in enumerate(MODES.items()):
    g = p[p["mode"] == mode].sort_values("size")
    ax.bar(np.arange(len(sizes)) + (j - 1) * 0.27, g.delta, 0.27, color=c, label=mode)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(sizes)), sizes)
ax.set_xlabel("chunk size (chars)")
ax.set_ylabel("MRR@5 change from rerank")
ax.legend(frameon=False, ncol=3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "rerank_delta.png", dpi=150)
print("wrote results/pipeline.png results/mrr_by_size.png results/rerank_delta.png")

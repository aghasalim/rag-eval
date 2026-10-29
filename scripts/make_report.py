"""results/retrieval.csv -> results/summary.md (factor means + top configs)."""
import pandas as pd

d = pd.read_csv("results/retrieval.csv")
for c in ("hit@5_ci", "mrr@5_ci"):
    d[c] = d[c].str.replace("np.float64", "").str.replace(r"[()]", "", regex=True).apply(lambda x: "[" + x + "]")
d["overlap_frac"] = (d.overlap / d["size"]).round(1)
out = ["### Mean over all other settings (hit@5 / MRR@5)\n"]
for col in ["size", "overlap_frac", "mode", "rerank"]:
    g = d.groupby(col)[["hit@5", "mrr@5"]].mean().round(3)
    out += [f"**{col}**\n", g.to_markdown(), ""]
out += ["### Top 8 configs by MRR@5\n", d.sort_values("mrr@5", ascending=False).head(8)[["size", "overlap", "mode", "rerank", "hit@1", "hit@5", "hit@5_ci", "mrr@5", "mrr@5_ci", "hit@10"]].to_markdown(index=False)]
open("results/summary.md", "w").write("\n".join(out))
print("\n".join(out))

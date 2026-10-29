"""Log each row of results/retrieval.csv as an MLflow run (local ./mlruns store)."""
import mlflow
import pandas as pd

mlflow.set_experiment("rag-retrieval-grid")
for _, r in pd.read_csv("results/retrieval.csv").iterrows():
    with mlflow.start_run(run_name=f"{r['size']}-{r['overlap']}-{r['mode']}-rr{r['rerank']}"):
        mlflow.log_params({k: r[k] for k in ("size", "overlap", "mode", "rerank")})
        mlflow.log_metrics({k.replace("@", "_at_"): float(r[k]) for k in ("hit@1", "hit@5", "hit@10", "mrr@5")})
print("logged; view with: mlflow ui")

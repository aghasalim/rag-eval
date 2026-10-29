---
title: rag-eval
sdk: docker
app_port: 7860
---

# rag-eval: retrieval experiments for a RAG system over the scikit-learn user guide

**Problem.** Which retrieval settings (chunk size, overlap, dense vs BM25 vs hybrid, cross-encoder rerank) actually help a RAG system, measured on a hand-built Q&A set rather than by eyeballing answers.

**Corpus.** 46 `.rst` files of the scikit-learn user guide (`doc/modules`, BSD-3), pinned to commit `d685424`.

**Approach.** Fixed-size character chunks -> `all-MiniLM-L6-v2` embeddings in FAISS (exact inner product) and/or BM25 -> optional reciprocal-rank fusion (c=60) -> optional `ms-marco-MiniLM-L-6-v2` cross-encoder rerank of the top 20.
Relevance is defined by an **evidence string**: a retrieved chunk is correct if it contains the gold sentence fragment. This keeps the metric independent of how the corpus is chunked, so chunk-size experiments are comparable.

## Evaluation set
35 hand-written questions (`data_eval/qa.jsonl`), each with a gold evidence fragment verified to exist in its source file (`tests/`). Metrics: hit@k, MRR@k, 95% bootstrap CIs over questions (2000 resamples).

## Results (retrieval; 48 configs, all run, none dropped)
Full grid: `results/retrieval.csv`. Regenerate with `make experiments`.

### Mean over all other settings (hit@5 / MRR@5)

**size**

|   size |   hit@5 |   mrr@5 |
|-------:|--------:|--------:|
|    300 |   0.693 |   0.581 |
|    600 |   0.867 |   0.766 |
|   1200 |   0.867 |   0.746 |
|   2400 |   0.771 |   0.652 |

**overlap_frac**

|   overlap_frac |   hit@5 |   mrr@5 |
|---------------:|--------:|--------:|
|            0   |   0.798 |   0.681 |
|            0.2 |   0.801 |   0.691 |

**mode**

| mode   |   hit@5 |   mrr@5 |
|:-------|--------:|--------:|
| bm25   |   0.83  |   0.731 |
| dense  |   0.759 |   0.624 |
| hybrid |   0.809 |   0.704 |

**rerank**

| rerank   |   hit@5 |   mrr@5 |
|:---------|--------:|--------:|
| False    |   0.799 |   0.659 |
| True     |   0.8   |   0.713 |

### Top 8 configs by MRR@5

|   size |   overlap | mode   | rerank   |   hit@1 |   hit@5 | hit@5_ci       |   mrr@5 | mrr@5_ci       |   hit@10 |
|-------:|----------:|:-------|:---------|--------:|--------:|:---------------|--------:|:---------------|---------:|
|   2400 |       480 | bm25   | False    |   0.771 |   0.943 | [0.857, 1.0]   |   0.837 | [0.73, 0.938]  |    0.971 |
|   1200 |       240 | bm25   | False    |   0.8   |   0.914 | [0.828, 1.0]   |   0.835 | [0.723, 0.94]  |    0.971 |
|   1200 |         0 | hybrid | False    |   0.771 |   0.914 | [0.829, 1.0]   |   0.831 | [0.721, 0.931] |    0.971 |
|   1200 |         0 | bm25   | False    |   0.771 |   0.914 | [0.828, 1.0]   |   0.83  | [0.721, 0.934] |    1     |
|   2400 |         0 | bm25   | False    |   0.743 |   0.943 | [0.857, 1.0]   |   0.822 | [0.709, 0.924] |    0.971 |
|    600 |       120 | bm25   | True     |   0.743 |   0.886 | [0.771, 0.971] |   0.814 | [0.7, 0.914]   |    0.886 |
|    600 |         0 | dense  | True     |   0.771 |   0.886 | [0.771, 0.971] |   0.81  | [0.683, 0.919] |    0.971 |
|    600 |       120 | hybrid | True     |   0.743 |   0.886 | [0.771, 0.971] |   0.81  | [0.69, 0.914]  |    0.886 |

### What I can and cannot conclude
- **n=35, so CIs are wide (roughly +/-0.1).** The top configs are statistically indistinguishable; the ranking among them is noise.
- Very small chunks (300 chars) are clearly worse (hit@5 ~0.69 vs ~0.87 at 600-1200). That gap is outside the CIs.
- **Caveat on chunk size:** larger chunks make the evidence-in-chunk test easier and put more text behind each hit (top-5 at 2400 chars = 12k chars of context). "Bigger is better" here is partly an artifact; a downstream answer-quality metric is needed to settle it.
- BM25 beat dense retrieval here. My questions were written while reading the source sentences, so they share vocabulary with it; this likely favours BM25. A set written by someone else (or paraphrased by an LLM) would be a fairer test.
- Reranking helps most for small chunks and for dense retrieval (MRR@5 +0.05 on average) and can hurt for large chunks and BM25 (e.g. 2400/0 BM25: 0.822 -> 0.650). The MS-MARCO cross-encoder truncates long inputs.
- Overlap of 20% made no measurable difference (0.681 vs 0.691 MRR@5 on average).

## Faithfulness: NOT YET RUN
`scripts/faithfulness.py` (Haiku generates from the top-5 chunks; Sonnet judges whether each claim is supported by the context) is written but has not produced results: the API key available in my environment was rejected (HTTP 401). No faithfulness number is reported until it runs. Config fixed in advance: size 600, overlap 0, hybrid + rerank.

## Run it
```bash
make setup data test     # env, pinned docs, tests
make experiments         # ~3 min on a laptop CPU
ANTHROPIC_API_KEY=... make faithfulness
```
Docker: `docker build -t rag-eval . && docker run -p 7860:7860 rag-eval` then `curl "localhost:7860/search?q=how+does+RFE+work"` (verified locally: image 3.3 GB, /health and /search work).

## Project 2: serving + tracking
- `src/rageval/api.py`: FastAPI `GET /health`, `GET /search?q=...&k=5` (retrieval only; config 600/0/hybrid/rerank). Tested with `TestClient` (`tests/test_api.py`).
- `scripts/log_mlflow.py`: logs all 48 grid runs to a local MLflow store (`mlflow.db`; view with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).
- `Dockerfile`: serves on `$PORT` (default 7860, the Hugging Face Docker Space port).
- **Live demo (Google Cloud Run, europe-west1):** https://rag-eval-327755438.europe-west1.run.app/docs . Try `/search?q=when+does+5-fold+CV+overestimate+error`. Scale-to-zero, so the first request after idle is slow (models reload). Hugging Face Docker Spaces now require PRO, hence Cloud Run.

## Limitations
Small, author-written eval set; single corpus; lexical-overlap bias in questions; retrieval-only metrics so far; one embedding model and one reranker; seeds do not matter (all components deterministic) so CIs reflect question sampling only; no MLflow tracking yet (planned for Project 2).

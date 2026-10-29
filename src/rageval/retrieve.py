import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

EMB = "sentence-transformers/all-MiniLM-L6-v2"
RERANK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_models: dict = {}


def _get(cls, name):
    if name not in _models:
        _models[name] = cls(name)
    return _models[name]


def tok(s: str) -> list[str]:
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split()]


class Retriever:
    def __init__(self, chunks: list[tuple[str, str]]):
        self.chunks = chunks
        texts = [t for _, t in chunks]
        emb = _get(SentenceTransformer, EMB).encode(texts, batch_size=64, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(np.asarray(emb, dtype="float32"))
        self.bm25 = BM25Okapi([tok(t) for t in texts])

    def _dense(self, q: str, n: int) -> list[int]:
        v = _get(SentenceTransformer, EMB).encode([q], normalize_embeddings=True)
        return self.index.search(np.asarray(v, dtype="float32"), n)[1][0].tolist()

    def _bm25(self, q: str, n: int) -> list[int]:
        return np.argsort(-self.bm25.get_scores(tok(q)))[:n].tolist()

    def search(self, q: str, k: int = 5, mode: str = "dense", rerank: bool = False, pool: int = 20) -> list[int]:
        n = max(k, pool) if rerank else k
        if mode == "dense":
            ids = self._dense(q, n)
        elif mode == "bm25":
            ids = self._bm25(q, n)
        elif mode == "hybrid":  # reciprocal rank fusion, c=60 (Cormack et al. 2009)
            score: dict[int, float] = {}
            for lst in (self._dense(q, 50), self._bm25(q, 50)):
                for r, i in enumerate(lst):
                    score[i] = score.get(i, 0) + 1 / (60 + r)
            ids = sorted(score, key=score.get, reverse=True)[:n]
        else:
            raise ValueError(mode)
        if rerank:
            s = _get(CrossEncoder, RERANK).predict([(q, self.chunks[i][1]) for i in ids])
            ids = [ids[j] for j in np.argsort(-s)]
        return ids[:k]

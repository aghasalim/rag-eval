from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from .data import chunk, load_docs
from .retrieve import Retriever

# Config fixed before looking at generation results; see README.
SIZE, OVERLAP, MODE, RERANK = 600, 0, "hybrid", True
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["r"] = Retriever(chunk(load_docs(), SIZE, OVERLAP))
    yield


app = FastAPI(title="rag-eval retrieval API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["r"].chunks)}


@app.get("/search")
def search(q: str = Query(min_length=3, max_length=500), k: int = Query(5, ge=1, le=20)):
    r = state["r"]
    ids = r.search(q, k=k, mode=MODE, rerank=RERANK)
    if not ids:
        raise HTTPException(404, "no results")
    return {"query": q, "results": [{"source": r.chunks[i][0], "text": r.chunks[i][1]} for i in ids]}

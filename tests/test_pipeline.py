import pytest

from rageval.data import chunk, load_docs, norm
from rageval.evaluate import load_qa, metrics, ranks
from rageval.retrieve import Retriever


@pytest.fixture(scope="module")
def docs():
    return load_docs()


def test_qa_shape_and_size():
    qa = load_qa()
    assert len(qa) >= 30
    assert all({"question", "evidence", "file"} <= q.keys() for q in qa)


def test_every_evidence_string_is_in_its_file(docs):
    for q in load_qa():
        assert norm(q["evidence"]) in docs[q["file"]], q["id"]


def test_chunking_overlap_and_coverage():
    d = {"a.rst": "x" * 1000}
    c = chunk(d, 300, 50)
    assert all(len(t) <= 300 for _, t in c)
    assert sum(len(t) for _, t in c) >= 1000


def test_metrics_known_values():
    m = metrics([1, 2, None, 6], k=5)
    assert m["hit"] == 0.5 and m["mrr"] == pytest.approx(0.375)


def test_smoke_retrieval(docs):
    sub = {k: docs[k] for k in ("calibration.rst", "tree.rst", "svm.rst")}
    r = Retriever(chunk(sub, 500, 100))
    qa = [q for q in load_qa() if q["file"] in sub]
    rk = ranks(r, qa, k=5, mode="hybrid")
    assert metrics(rk, 5)["hit"] > 0.5

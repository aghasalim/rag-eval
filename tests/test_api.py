from fastapi.testclient import TestClient

from rageval.api import app


def test_api_health_and_search():
    with TestClient(app) as c:
        assert c.get("/health").json()["chunks"] > 1000
        r = c.get("/search", params={"q": "Which calibration method is more prone to overfitting?", "k": 3})
        assert r.status_code == 200 and len(r.json()["results"]) == 3
        assert "calibration.rst" in {x["source"] for x in r.json()["results"]}


def test_api_validation():
    with TestClient(app) as c:
        assert c.get("/search", params={"q": "ab"}).status_code == 422
        assert c.get("/search", params={"q": "valid question", "k": 99}).status_code == 422

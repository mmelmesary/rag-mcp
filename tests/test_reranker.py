"""Reranker: endpoint building, response parsing, and the best-effort contract
(a bad key / down endpoint / malformed body must raise RerankError so the caller
falls back to dense order — reranking never breaks a search)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reranker  # noqa: E402


def test_disabled_by_default():
    # Default env has RERANK_PROVIDER unset -> "none".
    assert reranker.enabled() is False


def test_rerank_raises_when_disabled(monkeypatch):
    monkeypatch.setattr(reranker, "PROVIDER", "none")
    monkeypatch.setattr(reranker.httpx, "post", _boom)  # must not even call out
    with pytest.raises(reranker.RerankError):
        reranker.rerank("q", ["a", "b"])


@pytest.mark.parametrize("base,expected", [
    ("https://api.cohere.com", "https://api.cohere.com/v2/rerank"),
    ("https://api.jina.ai/v1", "https://api.jina.ai/v1/rerank"),
    ("https://x/v2", "https://x/v2/rerank"),
    ("https://x/rerank", "https://x/rerank"),
    ("https://api.cohere.com/", "https://api.cohere.com/v2/rerank"),
])
def test_endpoint_building(monkeypatch, base, expected):
    monkeypatch.setattr(reranker, "BASE_URL", base)
    assert reranker._endpoint() == expected


def test_rerank_orders_by_relevance_and_maps_indices(monkeypatch):
    monkeypatch.setattr(reranker, "PROVIDER", "cohere")
    # Out-of-order results: doc index 2 is most relevant, then 0, then 1.
    payload = {"results": [
        {"index": 0, "relevance_score": 0.4},
        {"index": 2, "relevance_score": 0.9},
        {"index": 1, "relevance_score": 0.1},
    ]}
    monkeypatch.setattr(reranker.httpx, "post", _fake_post(payload))
    order = reranker.rerank("q", ["a", "b", "c"])
    assert [i for i, _ in order] == [2, 0, 1]        # best-first
    assert order[0][1] == 0.9


def test_rerank_ignores_out_of_range_indices(monkeypatch):
    monkeypatch.setattr(reranker, "PROVIDER", "cohere")
    payload = {"results": [{"index": 5, "relevance_score": 0.9},
                           {"index": 0, "relevance_score": 0.3}]}
    monkeypatch.setattr(reranker.httpx, "post", _fake_post(payload))
    order = reranker.rerank("q", ["a", "b"])
    assert order == [(0, 0.3)]                        # index 5 dropped


def test_rerank_empty_docs_returns_empty(monkeypatch):
    monkeypatch.setattr(reranker, "PROVIDER", "cohere")
    monkeypatch.setattr(reranker.httpx, "post", _boom)
    assert reranker.rerank("q", []) == []             # no call, no raise


def test_rerank_network_error_raises(monkeypatch):
    monkeypatch.setattr(reranker, "PROVIDER", "cohere")
    monkeypatch.setattr(reranker.httpx, "post", _boom_request)
    with pytest.raises(reranker.RerankError):
        reranker.rerank("q", ["a"])


def test_rerank_malformed_body_raises(monkeypatch):
    monkeypatch.setattr(reranker, "PROVIDER", "cohere")
    monkeypatch.setattr(reranker.httpx, "post", _fake_post({"nope": True}))
    with pytest.raises(reranker.RerankError):
        reranker.rerank("q", ["a"])


# ---- helpers ----------------------------------------------------------------

def _boom(*_a, **_k):
    raise AssertionError("httpx.post should not have been called")


def _boom_request(*_a, **_k):
    raise reranker.httpx.RequestError("connection refused")


def _fake_post(payload):
    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    def _post(*_a, **_k):
        return _Resp()

    return _post

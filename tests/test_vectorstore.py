"""Vectorstore: named-vector construction and the hybrid/dense query routing.

FastEmbed is not required for these — the sparse layer is mocked. Verifies that
hybrid uses prefetch + RRF fusion, that dense-only is used when sparse is absent
or hybrid=False, and that the sparse helpers degrade to None without a model."""

import sys
from pathlib import Path

from qdrant_client.models import FusionQuery, SparseVector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vectorstore  # noqa: E402


class _FakeResult:
    def __init__(self, points):
        self.points = points


class _FakeClient:
    def __init__(self):
        self.calls: list[dict] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResult(["point"])


def test_named_vectors_dense_only():
    assert vectorstore.named_vectors([0.1, 0.2], None) == {vectorstore.DENSE: [0.1, 0.2]}


def test_named_vectors_includes_sparse():
    sv = SparseVector(indices=[1, 5], values=[0.5, 0.9])
    nv = vectorstore.named_vectors([0.1], sv)
    assert nv[vectorstore.DENSE] == [0.1]
    assert nv[vectorstore.SPARSE] is sv


def test_query_dense_fallback_when_no_sparse(monkeypatch):
    monkeypatch.setattr(vectorstore, "embed_query_sparse", lambda _t: None)
    client = _FakeClient()
    points = vectorstore.query(client, "kb", [0.1, 0.2], "q", query_filter=None, limit=5)
    assert points == ["point"]
    call = client.calls[0]
    assert call["using"] == vectorstore.DENSE
    assert call["query"] == [0.1, 0.2]
    assert "prefetch" not in call


def test_query_hybrid_uses_prefetch_and_fusion(monkeypatch):
    monkeypatch.setattr(
        vectorstore, "embed_query_sparse",
        lambda _t: SparseVector(indices=[1], values=[0.9]),
    )
    client = _FakeClient()
    vectorstore.query(client, "kb", [0.1], "CrashLoopBackOff", query_filter=None, limit=7)
    call = client.calls[0]
    assert len(call["prefetch"]) == 2                      # dense + sparse
    assert isinstance(call["query"], FusionQuery)          # RRF fusion
    assert call["limit"] == 7


def test_query_hybrid_false_forces_dense(monkeypatch):
    # hybrid=False must not even compute a sparse vector.
    def _boom(_t):
        raise AssertionError("sparse must not be embedded when hybrid=False")

    monkeypatch.setattr(vectorstore, "embed_query_sparse", _boom)
    client = _FakeClient()
    vectorstore.query(client, "kb", [0.1], "q", query_filter=None, limit=3, hybrid=False)
    assert client.calls[0]["using"] == vectorstore.DENSE


def test_sparse_helpers_degrade_without_model(monkeypatch):
    monkeypatch.setattr(vectorstore, "_load_bm25", lambda: None)
    assert vectorstore.sparse_available() is False
    assert vectorstore.embed_documents_sparse(["a", "b"]) == [None, None]
    assert vectorstore.embed_query_sparse("q") is None

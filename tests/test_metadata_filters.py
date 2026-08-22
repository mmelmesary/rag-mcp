"""Metadata filters on the search tools: `cluster` (SOFT narrow) + `component` (HARD).

Covers `server._search`'s Qdrant filter construction, the soft-narrow retry that
keeps a fleet-wide precedent visible when a same-cluster search is empty, and the
same behavior on the recurring-detection path (`capture.find_similar`). Qdrant
and the embedding model are mocked — nothing here touches a real store.

Soft-narrow contract (design: the "has this happened before ON THIS CLUSTER?"
feature): a cluster filter must NEVER hide a fleet-wide precedent. If the
cluster-scoped search comes back empty, retry without the cluster condition
(keeping any hard filters) and report `cluster_narrowed: false` so callers can
tell "no precedent on this cluster" from "no precedent anywhere".
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import capture  # noqa: E402
import server  # noqa: E402


def _point(payload=None, score=0.9):
    return types.SimpleNamespace(payload=payload or {}, score=score, id="pt")


class _QueryRecorder:
    """Stands in for `vectorstore.query`: records each call's filter and returns
    one configured point-list per call (empty list once exhausted)."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, client, collection, vector, query_text, *, query_filter, limit, **kw):
        self.calls.append((query_filter, limit))
        return self.results.pop(0) if self.results else []


def _conditions(query_filter):
    if query_filter is None:
        return {}
    return {c.key: c.match.value for c in query_filter.must}


def _patch_search(monkeypatch, results):
    monkeypatch.setattr(server.embeddings, "embed", lambda q, role: [0.1, 0.2])
    rec = _QueryRecorder(results)
    monkeypatch.setattr(server.vectorstore, "query", rec)
    return rec


def test_search_no_filters_no_query_filter(monkeypatch):
    rec = _patch_search(monkeypatch, [[_point()]])
    out = server._search("crashloop", None, None, None, 5)
    assert rec.calls[0][0] is None
    assert out["status"] == "ok"
    assert out["count"] == 1
    assert out["cluster"] is None
    assert out["cluster_narrowed"] is None
    assert out["component"] is None


def test_doc_type_and_component_build_hard_filters(monkeypatch):
    rec = _patch_search(monkeypatch, [[_point()]])
    server._search("volume stuck", "incident", None, "longhorn", 5)
    conds = _conditions(rec.calls[0][0])
    assert conds == {"doc_type": "incident", "component": "longhorn"}


def test_cluster_scoped_results_stay_narrowed(monkeypatch):
    rec = _patch_search(monkeypatch, [[_point()]])
    out = server._search("volume stuck", "incident", "prod-01", None, 5)
    assert len(rec.calls) == 1
    conds = _conditions(rec.calls[0][0])
    assert conds["cluster"] == "prod-01"
    assert conds["doc_type"] == "incident"
    assert out["cluster_narrowed"] is True
    assert "note" not in out


def test_cluster_empty_falls_back_fleet_wide(monkeypatch):
    # Scoped call returns nothing -> retry without the cluster condition and say so.
    rec = _patch_search(monkeypatch, [[], [_point(payload={"title": "prior"})]])
    out = server._search("volume stuck", "incident", "prod-02", None, 5)
    assert len(rec.calls) == 2
    assert _conditions(rec.calls[0][0])["cluster"] == "prod-02"
    assert "cluster" not in _conditions(rec.calls[1][0])
    assert out["count"] == 1
    assert out["cluster_narrowed"] is False
    assert "note" in out


def test_fallback_keeps_hard_filters(monkeypatch):
    # The retry drops ONLY the cluster narrow; doc_type/component stay.
    rec = _patch_search(monkeypatch, [[], [_point()]])
    server._search("volume stuck", "runbook", "prod-01", "longhorn", 5)
    assert len(rec.calls) == 2
    assert _conditions(rec.calls[1][0]) == {"doc_type": "runbook", "component": "longhorn"}


def test_component_empty_does_not_retry(monkeypatch):
    # Component is a HARD filter: an empty result stays empty — no fallback.
    rec = _patch_search(monkeypatch, [[]])
    out = server._search("volume stuck", None, None, "longhorn", 5)
    assert len(rec.calls) == 1
    assert out["count"] == 0
    assert out["cluster_narrowed"] is None


def test_search_incidents_shortcut_scopes_doc_type_and_cluster(monkeypatch):
    rec = _patch_search(monkeypatch, [[_point()]])
    out = server.search_incidents("stuck attaching", "prod-01")
    conds = _conditions(rec.calls[0][0])
    assert conds["doc_type"] == "incident"
    assert conds["cluster"] == "prod-01"
    assert out["status"] == "ok"


def test_search_runbooks_shortcut_scopes_component(monkeypatch):
    rec = _patch_search(monkeypatch, [[_point()]])
    server.search_runbooks("rebuild procedure", component="longhorn")
    conds = _conditions(rec.calls[0][0])
    assert conds["doc_type"] == "runbook"
    assert conds["component"] == "longhorn"


def test_search_incidents_fleet_fallback(monkeypatch):
    rec = _patch_search(monkeypatch, [[], [_point()]])
    out = server.search_incidents("stuck attaching", "prod-03")
    assert len(rec.calls) == 2
    assert out["cluster_narrowed"] is False


# --- recurring-detection path (capture.find_similar) -------------------------


def _patch_similar(monkeypatch, results):
    monkeypatch.setattr(capture.embeddings, "embed", lambda q, role: [0.1, 0.2])
    rec = _QueryRecorder(results)
    monkeypatch.setattr(capture.vectorstore, "query", rec)
    return rec


def test_find_similar_no_cluster_no_narrow(monkeypatch):
    rec = _patch_similar(monkeypatch, [[_point()]])
    out = capture.find_similar("volume stuck", cluster=None)
    assert len(rec.calls) == 1
    assert "cluster" not in _conditions(rec.calls[0][0])
    assert out["cluster_narrowed"] is None
    assert out["count"] == 1


def test_find_similar_cluster_narrowed_when_hits(monkeypatch):
    rec = _patch_similar(monkeypatch, [[_point()]])
    out = capture.find_similar("volume stuck", cluster="prod-01")
    assert "cluster" in _conditions(rec.calls[0][0])
    assert out["cluster_narrowed"] is True
    assert out["count"] == 1


def test_find_similar_cluster_empty_falls_back_fleet_wide(monkeypatch):
    rec = _patch_similar(monkeypatch, [[], [_point(payload={"title": "prior"})]])
    out = capture.find_similar("volume stuck", cluster="prod-02")
    assert len(rec.calls) == 2
    assert "cluster" in _conditions(rec.calls[0][0])
    assert "cluster" not in _conditions(rec.calls[1][0])
    assert out["cluster_narrowed"] is False
    assert out["count"] == 1


def test_find_similar_min_score_still_filters(monkeypatch):
    # A cluster match that clears no match because scores are below the threshold
    # is treated like an empty same-cluster result -> fleet-wide fallback.
    rec = _patch_similar(monkeypatch, [[_point(score=0.4)], [_point(score=0.9)]])
    out = capture.find_similar("volume stuck", cluster="prod-02", min_score=0.75)
    assert len(rec.calls) == 2
    assert out["cluster_narrowed"] is False
    assert out["count"] == 1

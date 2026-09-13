"""Knowledge-capture write path for the RAG memory (the "flywheel").

This is the DELIBERATE, trusted write path that closes the loop: after the agent
finishes an investigation, the resulting RCA is written back here so the next
similar alert can be recognised as "seen before". It complements the batch
`ingest.py` (runbooks/docs) with per-incident capture.

IMPORTANT — read-only MCP surface is preserved
----------------------------------------------
None of this is exposed as an ``@mcp.tool()``. The LLM only ever sees the search
tools in ``server.py``; capture/feedback are plain HTTP routes (see
``server.py`` ``@mcp.custom_route``) called by the trusted agent process, never
by the model. This keeps the LLM-facing server strictly read-only — no write
tools are ever offered to the model.

Embeddings + Qdrant live here (the knowledge service owns them), so capture and
query embed identically — the hard rule from ``embeddings.py`` (ingest and query
MUST share provider + model) is satisfied by construction.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
)

import embeddings
import reranker
import vectorstore
from ingest import _chunk  # reuse the exact chunking used by batch ingest (DRY)

log = logging.getLogger("rag-capture")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
COLLECTION = os.environ.get("QDRANT_COLLECTION")
HTTP_TIMEOUT = float(os.environ.get("RAG_TIMEOUT_SECONDS", "60"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

# Same stable namespace as ingest.py so IDs are drawn from one space.
_ID_NAMESPACE = uuid.UUID("6f3a9c1e-9b2d-5a44-8c11-a1b2c3d4e5f6")

# Payload keys we index so recurring-incident filters (cluster/alert/component)
# are cheap. Creating an index that already exists is a no-op we swallow.
_INDEXED_FIELDS = (
    "doc_type", "fingerprint", "status", "cluster",
    "namespace", "alertname", "component",
)

_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=HTTP_TIMEOUT)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(doc: dict[str, Any]) -> str:
    """Stable identity for a recurring incident. Prefer the alert fingerprint;
    fall back to a deterministic hash of the identifying labels so manual
    captures still de-duplicate."""
    fp = (doc.get("fingerprint") or "").strip()
    if fp:
        return fp
    basis = "|".join(str(doc.get(k, "")) for k in ("alertname", "cluster", "namespace", "component", "title"))
    return f"auto:{uuid.uuid5(_ID_NAMESPACE, basis)}"


def _ensure_collection(dim: int) -> None:
    # Named dense (+ BM25 sparse when hybrid is live) schema + keyword indexes,
    # shared with ingest.py so write paths never drift.
    vectorstore.ensure_collection(_client, COLLECTION, dim, payload_indexes=_INDEXED_FIELDS)


def _searchable_text(doc: dict[str, Any]) -> str:
    """The text we embed for retrieval — the symptom + cause + fix, so a future
    alert with the same symptom matches. The full proposal is stored separately
    in the payload for display."""
    parts = [
        doc.get("title", ""),
        doc.get("symptom", ""),
        f"Root cause: {doc.get('root_cause', '')}",
        f"Fix: {doc.get('proposed_fix', '')}",
        " ".join(doc.get("tags", []) or []),
    ]
    return "\n".join(p for p in parts if p and p.strip())


def _existing_history(fingerprint: str) -> dict[str, Any]:
    """Return {occurrence_count, first_seen} for a prior capture of this
    fingerprint, so a recurrence increments rather than resets."""
    try:
        found, _ = _client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="fingerprint", match=MatchValue(value=fingerprint))]),
            limit=1,
            with_payload=True,
        )
    except Exception:  # noqa: BLE001 - collection may not exist yet
        return {}
    if not found:
        return {}
    p = found[0].payload or {}
    return {"occurrence_count": p.get("occurrence_count", 1), "first_seen": p.get("first_seen")}


def capture_incident(doc: dict[str, Any]) -> dict[str, Any]:
    """Upsert one incident/RCA into the knowledge base, keyed by fingerprint.

    Idempotent + recurrence-aware: re-capturing the same fingerprint replaces its
    chunks and bumps ``occurrence_count`` / ``last_seen`` (first_seen preserved).
    """
    fingerprint = _fingerprint(doc)
    body = (doc.get("body") or doc.get("proposal") or _searchable_text(doc)).strip()
    if not body:
        return {"status": "error", "error": "nothing to capture (empty body)"}

    chunks = _chunk(body, CHUNK_SIZE, CHUNK_OVERLAP)
    # First chunk carries the composed searchable text so symptom-based retrieval
    # hits even when the body is a long free-form proposal.
    embed_texts = [_searchable_text(doc) or chunks[0]] + chunks[1:]
    try:
        vectors = [embeddings.embed(t, "document") for t in embed_texts]
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"embedding failed: {exc}"}

    _ensure_collection(len(vectors[0]))

    prior = _existing_history(fingerprint)
    occurrence = int(prior.get("occurrence_count") or 0) + 1
    first_seen = prior.get("first_seen") or _now()
    now = _now()
    status = doc.get("status") or ("recurring" if occurrence > 1 else "open")

    payload_base = {
        "doc_type": "incident",
        "fingerprint": fingerprint,
        "title": doc.get("title") or doc.get("alertname") or "incident",
        "source": doc.get("source") or "agent-investigation",
        "status": status,
        "cluster": doc.get("cluster"),
        "namespace": doc.get("namespace"),
        "alertname": doc.get("alertname"),
        "component": doc.get("component"),
        "workload": doc.get("workload"),
        "root_cause": doc.get("root_cause"),
        "proposed_fix": doc.get("proposed_fix"),
        "confidence": doc.get("confidence"),
        "risk": doc.get("risk"),
        "tags": doc.get("tags") or [],
        "proposal_id": doc.get("proposal_id"),
        "occurrence_count": occurrence,
        "first_seen": first_seen,
        "last_seen": now,
    }

    # Replace any prior chunks for this fingerprint so a shorter re-capture
    # doesn't leave stale chunks behind.
    try:
        _client.delete(
            collection_name=COLLECTION,
            points_selector=Filter(must=[FieldCondition(key="fingerprint", match=MatchValue(value=fingerprint))]),
        )
    except Exception:  # noqa: BLE001
        pass

    # Sparse vectors from the SAME texts the dense vectors describe (embed_texts),
    # so both signals point at the composed searchable content. [None...] if hybrid off.
    sparse = vectorstore.embed_documents_sparse(embed_texts)
    points = [
        PointStruct(
            id=str(uuid.uuid5(_ID_NAMESPACE, f"incident:{fingerprint}#{i}")),
            vector=vectorstore.named_vectors(vec, sp),
            payload={**payload_base, "text": chunk, "chunk": i},
        )
        for i, (chunk, vec, sp) in enumerate(zip(chunks, vectors, sparse))
    ]
    _client.upsert(collection_name=COLLECTION, points=points)
    log.info("captured incident fp=%s (occurrence=%d, %d chunk(s))", fingerprint, occurrence, len(points))
    return {
        "status": "ok",
        "fingerprint": fingerprint,
        "occurrence_count": occurrence,
        "first_seen": first_seen,
        "last_seen": now,
        "incident_status": status,
        "chunks": len(points),
    }


def find_similar(query: str, doc_type: str = "incident", limit: int = 3, min_score: float = 0.0,
                 cluster: str | None = None) -> dict[str, Any]:
    """Semantic lookup for the recurring-incident pre-check. Returns matches at or
    above ``min_score`` with the structured payload fields the notifier surfaces.

    ``cluster`` is a SOFT narrow, mirroring the search tools: when a cluster-scoped
    lookup comes back empty it is retried fleet-wide and the response reports
    ``cluster_narrowed: false`` — a precedent seen on another cluster must still be
    surfaced the first time a symptom appears on a new cluster.
    """
    query = (query or "").strip()
    if not query:
        return {"status": "error", "error": "query must be non-empty"}
    cluster = (cluster or "").strip() or None
    try:
        vector = embeddings.embed(query, "query")
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"embedding failed: {exc}"}

    def _run(narrow_cluster: str | None) -> list[Any]:
        conditions: list[Any] = []
        if doc_type:
            conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
        if narrow_cluster:
            conditions.append(FieldCondition(key="cluster", match=MatchValue(value=narrow_cluster)))
        query_filter = Filter(must=conditions) if conditions else None
        # Dense-only: recurring detection thresholds on cosine (min_score), which
        # RRF fusion would not preserve. The LLM-facing search uses hybrid.
        return list(vectorstore.query(
            _client, COLLECTION, vector, query,
            query_filter=query_filter, limit=max(1, limit), hybrid=False,
        ))

    def _hits(points: list[Any]) -> list[dict[str, Any]]:
        # De-duplicate by fingerprint (an incident has several chunks) keeping best score.
        best: dict[str, dict[str, Any]] = {}
        for point in points:
            if point.score < min_score:
                continue
            p = point.payload or {}
            fp = p.get("fingerprint") or p.get("source") or str(point.id)
            if fp in best and best[fp]["score"] >= round(point.score, 4):
                continue
            best[fp] = {
                "score": round(point.score, 4),
                "fingerprint": p.get("fingerprint"),
                "title": p.get("title"),
                "cluster": p.get("cluster"),
                "root_cause": p.get("root_cause"),
                "proposed_fix": p.get("proposed_fix"),
                "status": p.get("status"),
                "occurrence_count": p.get("occurrence_count"),
                "last_seen": p.get("last_seen"),
                "proposal_id": p.get("proposal_id"),
            }
        return sorted(best.values(), key=lambda h: h["score"], reverse=True)

    try:
        points = _run(cluster)
        # None = no cluster requested; True = scoped to the cluster; False = the
        # soft-narrow fallback to fleet-wide fired.
        cluster_narrowed: bool | None = None
        hits = _hits(points)
        if cluster is not None:
            cluster_narrowed = True
            if not hits:
                points = _run(None)
                cluster_narrowed = False
                hits = _hits(points)
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            return {"status": "ok", "count": 0, "results": []}  # empty KB is not an error
        return {"status": "error", "error": f"qdrant error: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"qdrant search failed: {exc}"}

    return {"status": "ok", "query": query, "count": len(hits),
            "cluster": cluster, "cluster_narrowed": cluster_narrowed, "results": hits}


def record_feedback(fingerprint: str, status: str | None = None, confidence: str | None = None,
                    note: str | None = None) -> dict[str, Any]:
    """Attach a human decision back onto a captured incident (the feedback loop).
    Updates every chunk sharing the fingerprint."""
    fingerprint = (fingerprint or "").strip()
    if not fingerprint:
        return {"status": "error", "error": "fingerprint required"}
    patch: dict[str, Any] = {"feedback_at": _now()}
    if status:
        patch["status"] = status
    if confidence:
        patch["confidence"] = confidence
    if note:
        patch["feedback_note"] = note
    try:
        _client.set_payload(
            collection_name=COLLECTION,
            payload=patch,
            points=Filter(must=[FieldCondition(key="fingerprint", match=MatchValue(value=fingerprint))]),
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"feedback update failed: {exc}"}
    log.info("recorded feedback fp=%s status=%s", fingerprint, status)
    return {"status": "ok", "fingerprint": fingerprint, "updated": patch}


def stats() -> dict[str, Any]:
    """Knowledge-base counts for an admin dashboard or UI."""
    try:
        total = _client.count(COLLECTION, exact=True).count
    except Exception:  # noqa: BLE001
        return {"status": "ok", "collection": COLLECTION, "exists": False}

    def _count(doc_type: str) -> int | None:
        try:
            return _client.count(
                COLLECTION, exact=True,
                count_filter=Filter(
                    must=[FieldCondition(key="doc_type", match=MatchValue(value=doc_type))]
                ),
            ).count
        except Exception:  # noqa: BLE001 - a per-type count is best-effort
            return None

    return {
        "status": "ok", "collection": COLLECTION, "exists": True,
        "points": total,
        "incident_points": _count("incident"),
        "runbook_points": _count("runbook"),
        "embeddings": embeddings.describe(),
        "reranker": reranker.describe(),
        "retrieval": vectorstore.describe(),
    }

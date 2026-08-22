"""RAG-memory MCP server — read-only semantic search over a Qdrant
knowledge base of runbooks, past incidents, and RCAs.


Read-only posture
-----------------
This server exposes SEARCH tools only. The knowledge base is populated by the
out-of-band `ingest.py` job (see README.md), so the LLM-facing surface stays
read-only.

Vendor-neutral by design
------------------------
Nothing here is bound to a specific LLM, UI, or embedding vendor. The chat LLM
is chosen by whatever MCP client connects (LibreChat, Open WebUI via mcpo, a
custom UI/CLI, ...). Embeddings go through the pluggable provider in
`embeddings.py` (Ollama / any OpenAI-compatible endpoint), so the same server
works offline with Ollama or against a hosted provider without code changes.
"""

from __future__ import annotations

import hmac
import logging
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import FieldCondition, Filter, MatchValue
from starlette.requests import Request
from starlette.responses import JSONResponse

import capture
import embeddings
import reranker
import vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("rag-mcp")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
COLLECTION = os.environ.get("QDRANT_COLLECTION")

HTTP_TIMEOUT = float(os.environ.get("RAG_TIMEOUT_SECONDS", "30"))
DEFAULT_LIMIT = int(os.environ.get("RAG_DEFAULT_LIMIT", "5"))
MAX_LIMIT = int(os.environ.get("RAG_MAX_LIMIT", "20"))
SNIPPET_CHARS = int(os.environ.get("RAG_SNIPPET_CHARS", "1200"))

MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.environ.get("MCP_PORT", "8084"))

mcp = FastMCP("rag", host=MCP_HOST, port=MCP_PORT)

# One long-lived client; Qdrant connections are cheap to keep open.
_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=HTTP_TIMEOUT)


def _clamp_limit(limit: int) -> int:
    if limit < 1:
        return 1
    return min(limit, MAX_LIMIT)


def _build_conditions(doc_type: str | None, component: str | None,
                      cluster: str | None) -> list[Any]:
    """Qdrant payload conditions for the hard filters plus the (optional) cluster
    narrow. `doc_type` and `component` are HARD: an empty result stays empty.
    `cluster` is passed by the caller as a SOFT narrow and dropped on the fallback
    retry (see `_search`)."""
    conditions = []
    if doc_type:
        conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
    if component:
        conditions.append(FieldCondition(key="component", match=MatchValue(value=component)))
    if cluster:
        conditions.append(FieldCondition(key="cluster", match=MatchValue(value=cluster)))
    return conditions


def _search(query: str, doc_type: str | None, cluster: str | None, component: str | None,
            limit: int) -> dict[str, Any]:
    """Shared retrieval path used by every search tool.

    `doc_type` and `component` are hard filters. `cluster` is a SOFT narrow: if a
    cluster-scoped search comes back empty it is retried fleet-wide (keeping the
    hard filters), so a real precedent on another cluster is never hidden the
    first time a symptom appears on a new cluster — the same empty-result
    self-correction the agent prompt applies to Kubernetes/Prometheus. The
    response reports `cluster_narrowed` so callers can tell "no precedent on this
    cluster" from "no precedent anywhere".
    """
    query = (query or "").strip()
    if not query:
        return {"status": "error", "error": "query must be a non-empty string"}
    doc_type = (doc_type or "").strip() or None
    cluster = (cluster or "").strip() or None
    component = (component or "").strip() or None

    limit = _clamp_limit(limit)
    # With a reranker on, fetch a wider candidate set (dense recall) and let the
    # cross-encoder pick the final top-`limit` (precision). Off => fetch exactly limit.
    fetch = max(limit, reranker.CANDIDATES) if reranker.enabled() else limit

    try:
        vector = embeddings.embed(query, "query")
    except embeddings.EmbeddingError as exc:
        return {"status": "error", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - surface any embedding failure verbatim
        return {"status": "error", "error": f"embedding failed: {exc}"}

    def _run(narrow_cluster: str | None) -> list[Any]:
        conditions = _build_conditions(doc_type, component, narrow_cluster)
        query_filter = Filter(must=conditions) if conditions else None
        # Hybrid (dense + BM25 sparse, RRF-fused) when hybrid is live, else dense.
        return list(vectorstore.query(
            _qdrant, COLLECTION, vector, query,
            query_filter=query_filter, limit=fetch,
        ))

    try:
        points = _run(cluster)
        # None = no cluster requested; True = scoped to the cluster; False = the
        # soft-narrow fallback to fleet-wide fired.
        cluster_narrowed: bool | None = None
        if cluster is not None:
            cluster_narrowed = True
            if not points:
                # Soft narrow: no same-cluster match — retry fleet-wide before
                # answering "no prior occurrence" (a fleet-wide precedent must stay
                # visible the first time a symptom appears on a new cluster).
                points = _run(None)
                cluster_narrowed = False
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            return {
                "status": "error",
                "error": (
                    f"collection '{COLLECTION}' not found in Qdrant. Populate it "
                    f"first with the ingestion job (see rag-mcp/README.md: "
                    f"`python ingest.py --path ./knowledge`)."
                ),
            }
        return {"status": "error", "error": f"qdrant error: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"qdrant search failed: {exc}"}
    # Rerank the candidate set on FULL chunk text (not the snippet). Best-effort:
    # any failure falls back to the dense order so a search never breaks.
    rerank_scores: list[float | None] = [None] * len(points)
    reranked = False
    if reranker.enabled() and len(points) > 1:
        docs = [(p.payload or {}).get("text", "") for p in points]
        try:
            order = reranker.rerank(query, docs)
            points = [points[i] for i, _ in order]
            rerank_scores = [s for _, s in order]
            reranked = True
        except reranker.RerankError as exc:
            log.warning("rerank failed (%s); falling back to dense order", exc)
            rerank_scores = [None] * len(points)

    points = points[:limit]
    rerank_scores = rerank_scores[:limit]

    hits = []
    for point, rr in zip(points, rerank_scores):
        payload = point.payload or {}
        text = payload.get("text", "")
        hit = {
            "score": round(point.score, 4),
            "doc_type": payload.get("doc_type"),
            "title": payload.get("title"),
            "source": payload.get("source"),
            "tags": payload.get("tags"),
            "text": text[:SNIPPET_CHARS],
            "truncated": len(text) > SNIPPET_CHARS,
        }
        if rr is not None:
            hit["rerank_score"] = round(rr, 4)
        hits.append(hit)

    response: dict[str, Any] = {
        "status": "ok",
        "collection": COLLECTION,
        "query": query,
        "doc_type": doc_type,
        "cluster": cluster,
        "cluster_narrowed": cluster_narrowed,
        "component": component,
        "reranked": reranked,
        "count": len(hits),
        "results": hits,
    }
    if cluster is not None and cluster_narrowed is False:
        response["note"] = (
            f"no matching knowledge tagged cluster={cluster!r}; expanded the search "
            f"across all clusters"
        )
    return response


@mcp.tool()
def rag_search(
    query: str,
    doc_type: str | None = None,
    cluster: str | None = None,
    component: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Semantic search across the knowledge base (runbooks, incidents, RCAs).

    Call this while diagnosing an issue to pull in historical context — prior
    incidents with the same symptom, the runbook for a component, past root
    causes. Retrieval is by meaning, not keywords, so describe the symptom.

    Args:
        query: What you're looking for, in natural language. Example:
            'Longhorn volume stuck in attaching state after node reboot'.
        doc_type: Optional filter — 'incident', 'runbook', 'rca', or a custom
            type used at ingestion time. Omit to search everything.
        cluster: Optional SOFT filter — restrict to knowledge tagged with this
            cluster (e.g. 'prod-01'). If a cluster-scoped search comes back empty
            the server retries across ALL clusters (keeping the other filters)
            and reports `cluster_narrowed: false`, so a fleet-wide precedent is
            never hidden behind an empty same-cluster result. Omit for fleet-wide.
        component: Optional HARD filter — restrict to knowledge tagged with this
            component (e.g. 'longhorn'). Unlike `cluster` there is no empty-result
            fallback: empty means nothing is tagged with it.
        limit: Max results to return (1-20). Defaults to 5.
    """
    return _search(query, doc_type, cluster, component, limit)


@mcp.tool()
def search_incidents(
    query: str,
    cluster: str | None = None,
    component: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Search only past incidents / RCAs for ones matching the current symptom.

    Shortcut for rag_search(..., doc_type='incident'). Use this when you want
    'has this happened before?' rather than 'what's the documented procedure?'.
    Pass the target `cluster` to ask 'has this happened before ON THIS CLUSTER?'
    (a soft narrow — empty same-cluster results fall back to all clusters).

    Args:
        query: The symptom or error, in natural language.
        cluster: Optional SOFT filter — cluster-tagged incidents only, with a
            fleet-wide fallback when the same-cluster search is empty.
        component: Optional HARD filter — incidents tagged with this component.
        limit: Max results (1-20). Defaults to 5.
    """
    return _search(query, "incident", cluster, component, limit)


@mcp.tool()
def search_runbooks(
    query: str,
    cluster: str | None = None,
    component: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Search only runbooks / documented procedures.

    Shortcut for rag_search(..., doc_type='runbook'). Use this when you want the
    established procedure for a component or task. Pass `component` (e.g.
    'longhorn') to scope to one component's runbooks.

    Args:
        query: The component or task, in natural language.
        cluster: Optional SOFT filter — cluster-tagged runbooks only, with a
            fleet-wide fallback when the same-cluster search is empty.
        component: Optional HARD filter — runbooks tagged with this component.
        limit: Max results (1-20). Defaults to 5.
    """
    return _search(query, "runbook", cluster, component, limit)


@mcp.tool()
def rag_collections() -> dict[str, Any]:
    """List Qdrant collections and the point count of the active knowledge base.

    Use this first to confirm the knowledge base exists and has been populated
    before running searches.
    """
    try:
        names = [c.name for c in _qdrant.get_collections().collections]
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": f"could not reach Qdrant at {QDRANT_URL}: {exc}",
        }

    active_points: int | None = None
    if COLLECTION in names:
        try:
            active_points = _qdrant.count(COLLECTION, exact=True).count
        except Exception:  # noqa: BLE001 - count is best-effort
            active_points = None

    return {
        "status": "ok",
        "qdrant_url": QDRANT_URL,
        "active_collection": COLLECTION,
        "active_collection_exists": COLLECTION in names,
        "active_collection_points": active_points,
        "collections": names,
    }


@mcp.tool()
def rag_health() -> dict[str, Any]:
    """Reachability check for both dependencies: Qdrant and the embedding model.

    Returns per-dependency status. Call this first if searches are failing to
    tell whether the problem is the vector DB or the local embedding model.
    """
    health: dict[str, Any] = {"status": "ok"}

    try:
        _qdrant.get_collections()
        health["qdrant"] = {"reachable": True, "url": QDRANT_URL}
    except Exception as exc:  # noqa: BLE001
        health["status"] = "degraded"
        health["qdrant"] = {"reachable": False, "url": QDRANT_URL, "error": str(exc)}

    emb = embeddings.describe()
    try:
        embeddings.embed("healthcheck", "query")
        health["embeddings"] = {"reachable": True, **emb}
    except Exception as exc:  # noqa: BLE001
        health["status"] = "degraded"
        health["embeddings"] = {"reachable": False, **emb, "error": str(exc)}

    # Reranking is optional and best-effort; report config only (no live probe).
    health["reranker"] = reranker.describe()
    health["retrieval"] = vectorstore.describe()  # hybrid on/off + sparse model

    return health


# ---------------------------------------------------------------------------
# Internal write API (NOT MCP tools — invisible to the LLM)
# ---------------------------------------------------------------------------
# The knowledge "flywheel": the trusted agent process captures RCAs and records
# human feedback here after an investigation, and does the recurring-incident
# pre-check via /similar. These are plain HTTP routes, so the read-only MCP tool
# surface above is unchanged — the model can search but can never write.
# Optionally gated by RAG_INTERNAL_TOKEN (set it in prod; blank = open for dev).
INTERNAL_TOKEN = os.environ.get("RAG_INTERNAL_TOKEN", "")


def _authorized(request: Request) -> bool:
    if not INTERNAL_TOKEN:
        return True  # dev: open
    presented = request.headers.get("x-internal-token") or ""
    auth = request.headers.get("authorization", "")
    if not presented and auth.lower().startswith("bearer "):
        presented = auth[7:]
    return bool(presented) and hmac.compare_digest(presented, INTERNAL_TOKEN)


async def _guarded(request: Request, fn) -> JSONResponse:
    if not _authorized(request):
        return JSONResponse({"status": "error", "error": "unauthorized"}, status_code=401)
    try:
        body = await request.json() if request.method == "POST" else {}
    except Exception:  # noqa: BLE001
        body = {}
    try:
        return JSONResponse(fn(body))
    except Exception as exc:  # noqa: BLE001 - never 500 the caller opaquely
        log.exception("internal knowledge route failed")
        return JSONResponse({"status": "error", "error": str(exc)}, status_code=500)


@mcp.custom_route("/internal/knowledge/capture", methods=["POST"])
async def _capture_route(request: Request) -> JSONResponse:
    return await _guarded(request, lambda b: capture.capture_incident(b))


@mcp.custom_route("/internal/knowledge/similar", methods=["POST"])
async def _similar_route(request: Request) -> JSONResponse:
    return await _guarded(request, lambda b: capture.find_similar(
        b.get("query", ""), b.get("doc_type", "incident"),
        int(b.get("limit", 3)), float(b.get("min_score", 0.0)), b.get("cluster"),
    ))


@mcp.custom_route("/internal/knowledge/feedback", methods=["POST"])
async def _feedback_route(request: Request) -> JSONResponse:
    return await _guarded(request, lambda b: capture.record_feedback(
        b.get("fingerprint", ""), b.get("status"), b.get("confidence"), b.get("note"),
    ))


@mcp.custom_route("/internal/knowledge/stats", methods=["GET"])
async def _stats_route(request: Request) -> JSONResponse:
    return await _guarded(request, lambda _b: capture.stats())


def main() -> None:
    emb = embeddings.describe()
    rr = reranker.describe()
    log.info(
        "starting rag-mcp on %s:%s (qdrant=%s, collection=%s, embed=%s:%s@%s, "
        "hybrid=%s, rerank=%s)",
        MCP_HOST, MCP_PORT, QDRANT_URL, COLLECTION,
        emb["provider"], emb["model"], emb["base_url"],
        vectorstore.sparse_available(),  # loads BM25 at boot (fail fast)
        f"{rr['provider']}:{rr['model']}" if rr["enabled"] else "off",
    )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

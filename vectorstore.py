"""Shared Qdrant vector-store layer: named dense + BM25 sparse (hybrid search).

Retrieval step 1. Dense vectors capture meaning but miss exact tokens that matter
in ops text — error strings (`CrashLoopBackOff`), resource ids (`c-xxxxx`),
component names (`Longhorn`). A sparse BM25 vector recovers those. We store BOTH
per chunk (Qdrant named vectors) and fuse them at query time with Reciprocal Rank
Fusion(RRF), so recall benefits from both signals.

Sparse vectors are produced LOCALLY with FastEmbed's `Qdrant/bm25` model — no API
key, offline-friendly, matching the rest of the stack. IDF is applied server-side
via the collection's `Modifier.IDF`, so the query side only needs term presence.

Single source of truth: `ingest.py`, `capture.py`, and `server.py` all go through
here so the collection schema and the vector names never drift between the write
paths and the read path.

Graceful degradation: if FastEmbed can't be imported/loaded, or hybrid is turned
off (`RAG_HYBRID=false`), the collection is dense-only and queries fall back to a
plain dense search. Nothing breaks — you just lose the keyword signal.

SCHEMA NOTE: this uses NAMED vectors (`dense`), which is not compatible with the
old unnamed-vector collections. Moving to hybrid requires a one-time re-ingest
(`python ingest.py --recreate`).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    Modifier,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

log = logging.getLogger("rag-vectorstore")

# Named-vector keys. Kept as constants so every read/write path agrees.
DENSE = "dense"
SPARSE = "bm25"

# Hybrid can be forced off without code change (dense-only, but still the named
# schema). Default on; the effective state also depends on FastEmbed loading.
_HYBRID_REQUESTED = os.environ.get("RAG_HYBRID", "true").strip().lower() not in (
    "0", "false", "no", "off", ""
)
BM25_MODEL = os.environ.get("RAG_SPARSE_MODEL", "Qdrant/bm25")

_bm25 = None            # lazily-loaded FastEmbed model
_bm25_loaded = False    # have we attempted to load it yet?


def _load_bm25() -> Any | None:
    """Lazily import + construct the FastEmbed BM25 model. Cached. Returns None
    (and logs once) if hybrid is off or FastEmbed is unavailable."""
    global _bm25, _bm25_loaded
    if _bm25_loaded:
        return _bm25
    _bm25_loaded = True
    if not _HYBRID_REQUESTED:
        log.info("hybrid search disabled (RAG_HYBRID=false); using dense-only")
        return None
    try:
        from fastembed import SparseTextEmbedding

        _bm25 = SparseTextEmbedding(model_name=BM25_MODEL)
        log.info("hybrid search enabled (sparse model=%s)", BM25_MODEL)
    except Exception as exc:  # noqa: BLE001 - any failure => dense-only, never fatal
        log.warning("FastEmbed unavailable (%s); falling back to dense-only", exc)
        _bm25 = None
    return _bm25


def sparse_available() -> bool:
    """True when BM25 sparse vectors can be produced (hybrid is live)."""
    return _load_bm25() is not None


def describe() -> dict[str, Any]:
    return {"hybrid": sparse_available(), "sparse_model": BM25_MODEL if _HYBRID_REQUESTED else None}


def _to_sparse(embedding: Any) -> SparseVector:
    return SparseVector(
        indices=embedding.indices.tolist(), values=embedding.values.tolist()
    )


def embed_documents_sparse(texts: list[str]) -> list[SparseVector | None]:
    """Sparse vectors for stored chunks. Returns [None, ...] when hybrid is off."""
    model = _load_bm25()
    if model is None:
        return [None] * len(texts)
    return [_to_sparse(e) for e in model.embed(texts)]


def embed_query_sparse(text: str) -> SparseVector | None:
    """Sparse vector for a query (IDF is applied server-side via Modifier.IDF)."""
    model = _load_bm25()
    if model is None:
        return None
    return _to_sparse(next(iter(model.query_embed(text))))


def named_vectors(dense: list[float], sparse: SparseVector | None) -> dict[str, Any]:
    """Build the PointStruct.vector mapping for one chunk."""
    vectors: dict[str, Any] = {DENSE: dense}
    if sparse is not None:
        vectors[SPARSE] = sparse
    return vectors


def ensure_collection(
    client: QdrantClient, collection: str, dim: int, payload_indexes: tuple[str, ...] = ()
) -> None:
    """Create the collection with a named dense vector (+ BM25 sparse when hybrid
    is live) and any keyword payload indexes. No-op if it already exists."""
    if not client.collection_exists(collection):
        sparse_config = (
            {SPARSE: SparseVectorParams(modifier=Modifier.IDF)}
            if sparse_available()
            else None
        )
        client.create_collection(
            collection_name=collection,
            vectors_config={DENSE: VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config=sparse_config,
        )
        log.info(
            "created collection %s (dim=%d, cosine, hybrid=%s)",
            collection, dim, sparse_available(),
        )
    for field in payload_indexes:
        try:
            client.create_payload_index(
                collection, field_name=field, field_schema="keyword"
            )
        except Exception:  # noqa: BLE001 - already-exists / older server: best-effort
            pass


def query(
    client: QdrantClient,
    collection: str,
    dense_vector: list[float],
    query_text: str,
    *,
    query_filter: Any | None,
    limit: int,
    hybrid: bool = True,
):
    """Retrieve `limit` points.

    hybrid=True (default): dense + BM25 sparse fused with RRF when sparse is
    available — best recall; `point.score` is the (small) RRF fusion score.
    hybrid=False: plain dense search — `point.score` is cosine similarity. Use
    this when a caller thresholds on a cosine value (e.g. recurring-incident
    detection), since fusion scores are on a different scale.
    Falls back to dense-only whenever sparse is unavailable."""
    sparse = embed_query_sparse(query_text) if (hybrid and query_text) else None

    if sparse is not None:
        result = client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(query=dense_vector, using=DENSE, limit=limit, filter=query_filter),
                Prefetch(query=sparse, using=SPARSE, limit=limit, filter=query_filter),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
    else:
        result = client.query_points(
            collection_name=collection,
            query=dense_vector,
            using=DENSE,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
    return result.points

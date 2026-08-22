"""Provider-agnostic cross-encoder reranking for the RAG memory (retrieval step 2).

Dense vector search is good at recall but weak at ordering: the best chunk is
often in the top-30 but not the top-5. A reranker scores each (query, chunk) pair
with a cross-encoder and reorders, which is the single biggest precision win for
the least effort — it is query-time only and needs NO re-ingest.

Posture
-------
- OFF by default (`RERANK_PROVIDER=none`) so the stack behaves exactly as before
  until an operator opts in — set it in .env, no code change.
- BEST-EFFORT: any failure (bad key, endpoint down, malformed response) raises
  RerankError and the caller falls back to the original dense order. Reranking
  must never break a search.

Providers
---------
  none    — disabled (default).
  cohere  — Cohere/Jina-compatible rerank API: POST {base}/rerank with
            {model, query, documents} -> {results:[{index, relevance_score}]}.
            Covers Cohere (default base) and Jina (set RERANK_BASE_URL +
            RERANK_MODEL). One standard shape, like the openai embeddings path.

Env
---
  RERANK_PROVIDER    none | cohere            (default none)
  RERANK_MODEL       rerank-english-v3.0      (Cohere) / jina-reranker-v2-... (Jina)
  RERANK_BASE_URL    https://api.cohere.com   (or https://api.jina.ai/v1)
  RERANK_API_KEY     provider API key
  RERANK_CANDIDATES  how many dense hits to fetch before reranking (default 30)
  RERANK_TIMEOUT     HTTP timeout seconds (default 30)
"""

from __future__ import annotations

import os

import httpx

PROVIDER = os.environ.get("RERANK_PROVIDER", "none").strip().lower()
MODEL = os.environ.get("RERANK_MODEL", "rerank-english-v3.0")
BASE_URL = os.environ.get("RERANK_BASE_URL") or "https://api.cohere.com"
API_KEY = os.environ.get("RERANK_API_KEY", "")
CANDIDATES = int(os.environ.get("RERANK_CANDIDATES", "30"))
HTTP_TIMEOUT = float(os.environ.get("RERANK_TIMEOUT", "30"))


class RerankError(RuntimeError):
    """Raised when reranking cannot be produced; caller falls back to dense order."""


def enabled() -> bool:
    return PROVIDER not in ("", "none", "off", "false")


def describe() -> dict[str, object]:
    """Non-secret summary of the active rerank config (for health/logs)."""
    return {"provider": PROVIDER, "model": MODEL, "candidates": CANDIDATES,
            "enabled": enabled()}


def _endpoint() -> str:
    base = BASE_URL.rstrip("/")
    if base.endswith("/rerank"):
        return base
    if base.endswith(("/v1", "/v2")):
        return f"{base}/rerank"
    return f"{base}/v2/rerank"  # Cohere default


def rerank(query: str, documents: list[str]) -> list[tuple[int, float]]:
    """Score (query, doc) pairs and return (original_index, score) sorted best-first.

    Raises RerankError on any failure so the caller can fall back to the input
    order. Returns at most len(documents) items."""
    if not enabled():
        raise RerankError("reranker disabled")
    if not documents:
        return []

    if PROVIDER != "cohere":
        raise RerankError(f"unknown RERANK_PROVIDER={PROVIDER!r}; use 'none' or 'cohere'")

    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    try:
        resp = httpx.post(
            _endpoint(),
            headers=headers,
            json={"model": MODEL, "query": query, "documents": documents},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results")
    except httpx.HTTPStatusError as exc:
        raise RerankError(
            f"rerank endpoint returned HTTP {exc.response.status_code}: "
            f"{exc.response.text[:200]}"
        ) from exc
    except httpx.RequestError as exc:
        raise RerankError(f"could not reach the rerank endpoint ({exc})") from exc
    except Exception as exc:  # noqa: BLE001 - malformed JSON etc.
        raise RerankError(f"rerank response could not be parsed ({exc})") from exc

    if not isinstance(results, list):
        raise RerankError("rerank response missing a 'results' list")

    ranked: list[tuple[int, float]] = []
    for item in results:
        idx = item.get("index")
        score = item.get("relevance_score", item.get("score"))
        if isinstance(idx, int) and 0 <= idx < len(documents) and score is not None:
            ranked.append((idx, float(score)))
    if not ranked:
        raise RerankError("rerank response contained no usable results")

    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked

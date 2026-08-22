"""
Provider-agnostic embeddings for the RAG memory.

Supports:
- Ollama            (/api/embeddings, single-prompt)
- OpenAI-compatible (/v1/embeddings, native batch input)

Features:
- Symmetric vs. asymmetric models handled in code: asymmetric models (e.g. nomic)
  get distinct query/document task prefixes applied automatically; symmetric models
  (e.g. OpenAI text-embedding-*) get none. An explicit EMBED_QUERY_PREFIX /
  EMBED_DOC_PREFIX env always overrides the auto default (escape hatch for families
  the auto-detect doesn't know, e.g. e5/bge which use "query:"/"passage:").
- Single public entry point `embed(text, kind)` used across rag-mcp, plus
  `embed_query` / `embed_document` / `embed_documents` convenience wrappers.
- HTTP errors surface the status code + response body so endpoint/model problems
  (wrong model, 404, auth) are debuggable.

IMPORTANT: ingest and query MUST use the SAME provider + model. The Qdrant vectors
are model-specific, so changing the model requires a re-ingest. See rag-mcp/README.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


# =========================================================
# Config
# =========================================================

@dataclass
class EmbeddingConfig:
    provider: str
    model: str
    base_url: str
    api_key: str | None
    query_prefix: str = ""
    doc_prefix: str = ""
    is_asymmetric: bool = False


def _build_config() -> EmbeddingConfig:
    provider = os.environ.get("EMBEDDINGS_PROVIDER", "ollama").strip().lower()
    model = os.environ.get("EMBEDDINGS_MODEL", "nomic-embed-text")

    ollama_url = os.environ.get("OLLAMA_BASE_URL") or "http://host.docker.internal:11434"
    openai_url = os.environ.get("EMBEDDINGS_BASE_URL") or "https://api.openai.com"
    api_key = os.environ.get("EMBEDDINGS_API_KEY")

    # Auto-detect asymmetric models (query/doc need different task prefixes). nomic
    # is the common offline default; other families (e5, bge, gte) also need
    # prefixes but with different strings — set EMBED_*_PREFIX for those.
    is_asymmetric = "nomic" in model.lower()

    query_prefix = os.environ.get(
        "EMBED_QUERY_PREFIX",
        "search_query: " if is_asymmetric else "",
    )
    doc_prefix = os.environ.get(
        "EMBED_DOC_PREFIX",
        "search_document: " if is_asymmetric else "",
    )

    base_url = ollama_url if provider == "ollama" else openai_url

    return EmbeddingConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        query_prefix=query_prefix,
        doc_prefix=doc_prefix,
        is_asymmetric=is_asymmetric,
    )


CONFIG = _build_config()
HTTP_TIMEOUT = float(os.environ.get("RAG_TIMEOUT_SECONDS", "60"))


# =========================================================
# Errors
# =========================================================

class EmbeddingError(RuntimeError):
    pass


def _http_error(provider: str, exc: httpx.HTTPStatusError) -> EmbeddingError:
    """Preserve the status code + response body so a bad model/endpoint is
    diagnosable (e.g. Ollama 404 = model not pulled)."""
    body = (exc.response.text or "").strip()
    detail = f" — {body}" if body else ""
    return EmbeddingError(f"{provider} embeddings HTTP {exc.response.status_code}{detail}")


# =========================================================
# Helpers
# =========================================================

def describe() -> dict[str, str]:
    return {
        "provider": CONFIG.provider,
        "model": CONFIG.model,
        "base_url": CONFIG.base_url,
        "asymmetric": str(CONFIG.is_asymmetric),
    }


def _apply_prefix(text: str, kind: str) -> str:
    # Apply whatever prefix is configured — "" for symmetric models is a no-op.
    # `is_asymmetric` only picks the DEFAULT prefixes in _build_config; gating here
    # would silently ignore an explicit EMBED_*_PREFIX set for a family the
    # auto-detect doesn't know (e5/bge/gte), so we don't gate on it.
    if kind not in ("query", "document"):
        raise ValueError("kind must be 'query' or 'document'")
    prefix = CONFIG.query_prefix if kind == "query" else CONFIG.doc_prefix
    return f"{prefix}{text}"


def _openai_endpoint() -> str:
    base = CONFIG.base_url.rstrip("/")
    return f"{base}/embeddings" if base.endswith("/v1") else f"{base}/v1/embeddings"


# =========================================================
# Providers
# =========================================================

def _embed_ollama_one(text: str) -> list[float]:
    """Ollama's legacy /api/embeddings is single-prompt: {"prompt": str} ->
    {"embedding": [...]}. It does NOT accept a list, so batch is done by looping."""
    try:
        resp = httpx.post(
            f"{CONFIG.base_url.rstrip('/')}/api/embeddings",
            json={"model": CONFIG.model, "prompt": text},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise _http_error("ollama", exc) from exc
    except httpx.RequestError as exc:
        raise EmbeddingError(f"ollama connection failed: {exc}") from exc

    vector = resp.json().get("embedding")
    if not vector:
        raise EmbeddingError(f"ollama returned no embedding for model {CONFIG.model!r}")
    return vector


def _embed_openai(texts: list[str]) -> list[list[float]]:
    """OpenAI-compatible /v1/embeddings accepts a batch `input` array natively."""
    headers = {"Authorization": f"Bearer {CONFIG.api_key}"} if CONFIG.api_key else {}
    try:
        resp = httpx.post(
            _openai_endpoint(),
            headers=headers,
            json={"model": CONFIG.model, "input": texts},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise _http_error("openai", exc) from exc
    except httpx.RequestError as exc:
        raise EmbeddingError(f"openai connection failed: {exc}") from exc

    # Realign by `index` — the API may return items out of request order.
    data = sorted(resp.json().get("data") or [], key=lambda d: d.get("index", 0))
    vectors = [item["embedding"] for item in data]
    if len(vectors) != len(texts):
        raise EmbeddingError(
            f"openai returned {len(vectors)} embeddings for {len(texts)} input(s)"
        )
    return vectors


def _embed_batch(texts: list[str]) -> list[list[float]]:
    if CONFIG.provider == "ollama":
        return [_embed_ollama_one(t) for t in texts]
    if CONFIG.provider == "openai":
        return _embed_openai(texts)
    raise EmbeddingError(f"unknown embeddings provider: {CONFIG.provider!r}")


# =========================================================
# Public API — what the rest of rag-mcp calls
# =========================================================

def embed(text: str, kind: str) -> list[float]:
    """Embed one text. `kind` is 'query' or 'document' and selects the asymmetric
    task prefix. This is the entry point used by server.py / ingest.py / capture.py."""
    return _embed_batch([_apply_prefix(text, kind)])[0]


def embed_query(text: str) -> list[float]:
    return embed(text, "query")


def embed_document(text: str) -> list[float]:
    return embed(text, "document")


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Batch document embedding (one HTTP call for OpenAI; looped for Ollama)."""
    return _embed_batch([_apply_prefix(t, "document") for t in texts])

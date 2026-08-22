# rag-mcp — design (Qdrant, read-only)

Semantic search over a **Qdrant** knowledge base of runbooks, past incidents,
and RCAs, exposed as an MCP server so the agent can pull historical context
while it debugs a live cluster (agentic retrieval).

## Design

- **Read-only surface.** The MCP tools only *search*. The knowledge base is
  written out-of-band by [`ingest.py`](../ingest.py), keeping the LLM-facing
  surface read-only.
- **Vendor-neutral.** Nothing is bound to a specific LLM, UI, or embedding
  vendor. The chat LLM is chosen by whatever MCP client connects. Embeddings go
  through a pluggable provider ([`embeddings.py`](./embeddings.py)) — `ollama`
  (offline default) or `openai` (any OpenAI-compatible endpoint). Same code,
  one env var. Prefixing is automatic: asymmetric models (nomic) get
  `search_query:`/`search_document:` prefixes, symmetric models (OpenAI
  `text-embedding-*`) get none — no config needed. Override with
  `EMBED_QUERY_PREFIX`/`EMBED_DOC_PREFIX` for other asymmetric families the
  auto-detect doesn't know (e5/bge use `query:`/`passage:`).
- **Storage.** One Qdrant collection (`rag_kb`) with **named vectors** — a
  `dense` cosine vector and (when hybrid is on) a `bm25` sparse vector — plus a
  `doc_type` payload field used to filter incidents vs runbooks.

### Retrieval pipeline (`vectorstore.py` + `reranker.py`)

1. **Hybrid search** (recall) — dense embeddings + local **BM25 sparse** vectors
   (FastEmbed `Qdrant/bm25`, offline, no key), fused with Reciprocal Rank Fusion
   via Qdrant's Query API. BM25 recovers exact tokens dense vectors miss
   (`CrashLoopBackOff`, `c-xxxxx`, `Longhorn`). ON by default (`RAG_HYBRID`);
   degrades to dense-only if FastEmbed is unavailable.
2. **Reranking** (precision, optional) — over-fetch `RERANK_CANDIDATES`, reorder
   with a Cohere/Jina-compatible cross-encoder, keep the top few. OFF by default
   (`RERANK_PROVIDER`); best-effort — any failure falls back to the fused order.
3. **Section-aware chunking** — `ingest.py` splits docs on markdown headings and
   prepends the heading to each chunk; optional `component`/`severity`/`cluster`
   front-matter is stored and keyword-indexed for pre-filtering.

The recurring-incident pre-check (`find_similar`) stays **dense-only** so its
cosine `min_score` threshold keeps its meaning (RRF scores are on another scale).

#### How a query flows through the pipeline (plain language)

Calling `rag_search("Longhorn volume stuck attaching")` turns the query text
into **two different representations in parallel**, because the two search
halves understand text differently:

- **Dense vector (`nomic-embed-text`)** — the *meaning* of the query becomes a
  list of ~768 numbers. Qdrant compares it against every stored chunk using
  **cosine similarity** over an **HNSW** graph index. HNSW is an *approximate*
  nearest-neighbor search: it is fast and scales to millions of points, but the
  returned top-k is not guaranteed to be the mathematically exact top-k — it is
  extremely close in practice (exact scan only runs if you explicitly ask for
  it).
- **BM25 sparse vector (FastEmbed `Qdrant/bm25`)** — the *exact words* of the
  query become a bag of `{term_id: weight}` pairs. Qdrant looks the terms up in
  an inverted index, catching exact tokens dense vectors miss
  (`CrashLoopBackOff`, `c-xxxxx`, `Longhorn`).

Both searches run inside Qdrant at the same time (one `Prefetch` each) and each
produces its own ranked list. **RRF (Reciprocal Rank Fusion)** then merges the
two lists by *rank position*, not by score: a chunk that ranked near the top of
either list gets a strong fused score (`score ≈ Σ 1/(60 + rank)`). That is why
the final `point.score` is a small fusion number rather than a cosine
similarity, and why `find_similar` stays dense-only (it thresholds on real
cosine values).

Finally, **if a reranker is enabled** (`RERANK_PROVIDER=cohere`), the fused
top-`RERANK_CANDIDATES` (30) hits are sent to a cross-encoder (Cohere/Jina)
that scores each `(query, chunk)` pair and re-orders them. This is the biggest
precision win for the least effort — dense search has good *recall* but weak
*ordering*, and the reranker fixes ordering. Any failure falls back to the
fused order, so reranking never breaks a search.

##### Mental model

| Term | Plain meaning | In this stack |
|------|---------------|---------------|
| dense vector | meaning / semantics | `nomic-embed-text` → ~768 floats |
| sparse vector | exact words / keywords | FastEmbed BM25 → `{term_id: weight}` |
| HNSW | approximate nearest-neighbor graph index | fast cosine search, near-exact top-k |
| RRF | rank-fusion that merges two lists by position | combines dense + sparse ordering |
| reranker | cross-encoder re-ordering `(query, chunk)` pairs | Cohere/Jina (optional) |
| asymmetric prefixes | `search_query:` / `search_document:` task tags | nomic needs them; symmetric models (OpenAI) don't |

> ℹ️ **The embedding model is pre-trained and frozen** — it is *not* trained on
> your data. It converts text to numbers using generic semantic knowledge and
> never learns organizational or industry-specific context. That context comes
> from the **documents you ingest into Qdrant** and (optionally) from the
> reranker — not from the embedding weights. Because Qdrant vectors are
> model-specific, the model must never change between ingest and query (see the
> warnings below).

> ⚠️ **Ingest and query must use the same provider + model.** Vectors from
> different models have different dimensions/semantics and are not comparable —
> mixing them silently breaks retrieval. Change the model ⇒ re-ingest.
>
> ⚠️ **Enabling/disabling hybrid changes the collection schema** (named vs.
> unnamed vectors). Switching it requires a one-time rebuild:
> `docker compose run --rm rag-ingest --recreate`.

## Tools

| Tool | Purpose |
|------|---------|
| `rag_search(query, doc_type?, cluster?, component?, limit?)` | Semantic search across the whole KB |
| `search_incidents(query, cluster?, component?, limit?)` | "Has this happened before?" — filters `doc_type=incident` |
| `search_runbooks(query, cluster?, component?, limit?)` | "What's the procedure?" — filters `doc_type=runbook` |
| `rag_collections()` | List collections + active point count (confirm the KB is populated) |
| `rag_health()` | Reachability of both Qdrant and the embedding model |

All three search tools accept `cluster` and `component` filters matched on the
indexed payload fields (`cluster`, `component`). **`cluster` is a *soft* narrow**:
a same-cluster query that comes back empty is retried fleet-wide and the response
reports `cluster_narrowed: false`, so "has this happened before on this cluster?"
still surfaces a precedent seen on another cluster. `doc_type` and `component`
are **hard** filters — an empty result stays empty. Responses include
`cluster_narrowed` (`None` when no cluster was requested) so callers can tell a
same-cluster miss from a fleet-wide one.

## Internal write API — the knowledge flywheel (NOT MCP tools)

Beyond the read-only search tools, `rag-mcp` serves a small **internal HTTP API**
so institutional memory grows from every investigation. These are plain HTTP
routes (`@mcp.custom_route`), **not `@mcp.tool()`** — so the LLM can never see
or call them and the model-facing surface stays strictly read-only. Only a
trusted agent process should call them.

| Route (POST unless noted) | Purpose |
|------|---------|
| `/internal/knowledge/capture` | Upsert one incident/RCA (idempotent by `fingerprint`; bumps `occurrence_count`) |
| `/internal/knowledge/similar` | Recurring pre-check — "have we seen this symptom?" (accepts `cluster` for soft scoping) |
| `/internal/knowledge/feedback` | Attach a human approve/reject decision to a captured incident |
| `/internal/knowledge/stats` (GET) | KB counts (total / incidents) for admin/UI |

Gate them in production with `RAG_INTERNAL_TOKEN` (the agent presents the same
value as `X-Internal-Token`); blank = open for local dev. Embeddings + Qdrant are
owned here, so capture and query embed **identically** — the hard rule below is
satisfied by construction. Captured incidents carry structured payload fields
(`cluster`, `namespace`, `alertname`, `component`, `status`, `root_cause`, …) with
Qdrant payload indexes for cheap filtering.

## Prerequisites

- Qdrant running (the `qdrant` service in [`docker-compose.yml`](../docker-compose.yml)).
- An embedding provider:
  - **Ollama** (default, offline) reachable from the container, with the model pulled:
    ```bash
    ollama pull nomic-embed-text
    ```
  - **or** any OpenAI-compatible endpoint — set `EMBEDDINGS_PROVIDER=openai`
    (see [Configuration](#configuration-env)).

## Populate the knowledge base

Documents are **markdown** with optional YAML front matter, or **PDFs** (text is
extracted page-by-page; each page becomes a `# [Page N]` section). A single
`knowledge/` tree can mix both formats:

```markdown
---
title: Longhorn volume stuck attaching
type: incident        # incident | runbook | rca | ...  (default: folder name)
tags: [longhorn, storage]
---
# Body...
```

PDFs carry no front matter: `type` is inferred from the containing folder
(`knowledge/runbooks/*.pdf` → `runbook`), and the title from the file name.

`type` defaults to the containing folder name (`knowledge/incidents/*` →
`incident`), so you can also just drop files into typed folders. Then ingest:

```bash
# Recommended: use the one-shot Compose job. It has the required Python
# dependencies and uses the exact same embedding configuration as rag-mcp.
docker compose run --rm rag-ingest

# Full rebuild after changing embedding provider or model:
docker compose run --rm rag-ingest --recreate
```

In the Compose stack, `rag-ingest` bind-mounts `./knowledge` read-only
over `/knowledge`, so adding a document and running the job is enough — **no
rebuild required**:

```bash
cp my-runbook.md knowledge/runbooks/
docker compose run --rm rag-ingest
```

Always check the job's summary line (`done: N file(s), M chunk(s)`) against what
you expect. If your new document isn't in the count, it was skipped — the log
line above it says why (unsupported extension, empty file, or a PDF with no
extractable text).

The image also bakes `knowledge/` at build time (`Dockerfile` `COPY knowledge
/knowledge`) for runs outside this Compose stack. Without the bind mount above,
a newly added document stays invisible until the **`rag-ingest`** image itself is
rebuilt — rebuilding only `rag-mcp` does not help, and the job then reports
success having silently skipped the file. If you run the image directly, rebuild
both:

```bash
docker compose build rag-mcp rag-ingest
docker compose run --rm rag-ingest
```

For a host-only workflow, install the dependencies first and point the script at
the exposed Qdrant endpoint:

```bash
python3 -m pip install -r requirements.txt
QDRANT_URL=http://localhost:6333 python3 ingest.py --path knowledge
```

Ingestion is idempotent — chunk IDs are derived from `(source, chunk index)`,
so re-running updates existing points instead of duplicating them. Run it from
CI or a cron job whenever the `knowledge/` docs change.

If a document **shrinks** between runs (a runbook edited down, a PDF re-exported
with fewer pages), the tail chunks from the longer version would otherwise
survive as stale search hits with no file behind them — same `(source, index)`
scheme, but nothing overwrites index 20..29 when the doc now ends at 19. Each
file's chunks beyond its current count are therefore deleted right after its
upsert, so no `--recreate` is needed just to clear stale text.

**Deleting** a document is not handled: with no file left to ingest, nothing
knows its chunks are orphaned, so they stay searchable. Remove them explicitly
(delete by the `source` payload field) or rebuild with `--recreate`. The same
applies when you **move or rename** a file — `source` changes, so the old path's
chunks remain alongside the new ones.

Per-document writes are batched to keep any one HTTP request bounded — embeddings
at `EMBED_BATCH_SIZE` chunks per request, upserts at `QDRANT_UPSERT_BATCH` points.
This matters most for PDFs: a 100-page document is hundreds of chunks, each
carrying a dense vector, a BM25 sparse vector and its text, which is several
megabytes if sent as one call against `RAG_TIMEOUT_SECONDS`. Batching also means
a failure part-way through a large file leaves the earlier batches committed
instead of losing the whole document.

> **Note on `EMBED_BATCH_SIZE` and Ollama:** it only helps providers with native
> batch input (OpenAI-compatible `/v1/embeddings`). Ollama's `/api/embeddings` is
> single-prompt, so on the default provider the chunks are embedded one HTTP call
> at a time no matter what this is set to — raising it changes nothing there.

## Run

```bash
cp .env.example .env    # edit as needed
docker compose up -d --build
```

The MCP endpoint is then available at `http://localhost:${RAG_MCP_PORT:-8084}/mcp`
(streamable-http). See the top-level [README](../README.md) for wiring it into
MCP clients.

## Configuration (env)

| Var | Default | Notes |
|-----|---------|-------|
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant REST endpoint |
| `QDRANT_COLLECTION` | `rag_kb` | Collection name |
| `QDRANT_API_KEY` | _(unset)_ | If Qdrant auth is enabled |
| `EMBEDDINGS_PROVIDER` | `ollama` | `ollama` or `openai` (OpenAI-compatible) |
| `EMBEDDINGS_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | `ollama` provider host |
| `EMBEDDINGS_BASE_URL` | `https://api.openai.com` | `openai` provider base (e.g. a LiteLLM proxy) |
| `EMBEDDINGS_API_KEY` | _(unset)_ | `openai` provider key |
| `EMBED_QUERY_PREFIX` / `EMBED_DOC_PREFIX` | auto (set by model: nomic → `search_query: `/`search_document: `, symmetric → blank) | Override only for asymmetric families the auto-detect misses (e5/bge → `query: `/`passage: `) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1500` / `100` | Ingestion chunking |
| `EMBED_BATCH_SIZE` | `32` | Chunks per embeddings request. No effect on the `ollama` provider (single-prompt API) |
| `QDRANT_UPSERT_BATCH` | `64` | Points per Qdrant upsert request during ingestion |
| `RAG_DEFAULT_LIMIT` / `RAG_MAX_LIMIT` | `5` / `20` | Search result caps |
| `MCP_PORT` | `8084` | Server port |

### Using a hosted / OpenAI-compatible embedding provider

```bash
EMBEDDINGS_PROVIDER=openai \
EMBEDDINGS_BASE_URL=https://api.openai.com \   # or your LiteLLM proxy, etc.
EMBEDDINGS_API_KEY=sk-... \
EMBEDDINGS_MODEL=text-embedding-3-small \       # symmetric → prefixes auto-blank, no need to set them
  python ingest.py --path knowledge
```

Set the identical vars on the `rag-mcp` service so queries embed the same way.

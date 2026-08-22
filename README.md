# rag-mcp

A self-hosted **RAG memory server for AI assistants**, exposed over the
[Model Context Protocol](https://modelcontextprotocol.io) (MCP). Give any MCP-capable
LLM client (Claude Desktop / Claude Code, LibreChat Agents, Open WebUI via a proxy,
your own tooling) searchable long-term memory over your own documents — runbooks,
incident reports, RCAs, wiki exports, anything in markdown or PDF.

- **Read-only LLM surface** — the model can only *search*; writes happen out-of-band
  through a batch ingestion job or a token-gated internal API.
- **Hybrid retrieval** — dense semantic vectors **+ BM25 keyword sparse vectors**,
  fused with Reciprocal Rank Fusion. Exact tokens (`CrashLoopBackOff`, error codes,
  resource names) are found even when embeddings miss them.
- **Optional cross-encoder reranking** (Cohere/Jina-compatible) for precision.
- **Pluggable embeddings** — local [Ollama](https://ollama.com) (offline default) or
  any OpenAI-compatible `/v1/embeddings` endpoint. One env var to switch.
- **Markdown + PDF ingestion** with YAML front matter, heading-aware chunking,
  idempotent re-runs.
- **Vendor-neutral MCP** — works with any client that speaks streamable-http MCP.

## Architecture

```
 markdown/PDF docs            ┌──────────────────────────────────────┐
 knowledge/**  ──────────────►│ rag-ingest (one-shot job)            │
                              │ front-matter → section-aware chunks  │
                              │ → dense (Ollama/OpenAI-compat)       │
                              │   + BM25 (FastEmbed, offline)        │
                              │ → idempotent upsert into Qdrant      │
                              └──────────────────┬───────────────────┘
                                                 ▼
                              ┌──────────────────────────────────────┐
   MCP clients                │            Qdrant                    │
   ─────────────────          │  collection: named vectors           │
   Claude Code ──────────────►│    dense (cosine HNSW)               │
   LibreChat Agents ────────► │    bm25  (sparse, IDF)               │
   your agent ──────────────► │                                      │
                              └──────────────────▲───────────────────┘
                                                 │ hybrid query (Prefetch ×2 + RRF)
                              ┌──────────────────┴───────────────────┐
                              │ rag-mcp :8084 (FastMCP)              │
                              │ READ-ONLY tools:                     │
                              │  rag_search / search_incidents /     │
                              │  search_runbooks / collections /     │
                              │  health   (+ optional rerank step)   │
                              └──────────────────────────────────────┘
```

## Quickstart

Prerequisites: Docker (+ Compose v2), and for the default offline embedding path
[Ollama](https://ollama.com) on the host:

```bash
ollama pull nomic-embed-text     # once

git clone https://github.com/<you>/rag-mcp.git && cd rag-mcp
cp .env.example .env             # defaults work out of the box

docker compose up -d --build     # starts Qdrant + rag-mcp

docker compose run --rm rag-ingest   # index the sample docs in ./knowledge
```

Verify:

```bash
curl -s http://localhost:8084/mcp            # MCP endpoint (400 without handshake = alive)
curl -s http://localhost:6333/collections    # rag_kb exists with points
```

## Wire it into an MCP client

The server speaks **streamable-http** at `http://localhost:8084/mcp`.

**Claude Code** (`.mcp.json` in your project):

```json
{
  "mcpServers": {
    "rag": {
      "type": "http",
      "url": "http://localhost:8084/mcp"
    }
  }
}
```

**LibreChat** (`librechat.yaml`):

```yaml
mcpServers:
  rag:
    type: streamable-http
    url: http://rag-mcp-server:8084/mcp   # container name when compose-networked
```

Any other MCP client: register an HTTP/streamable-http server at the URL above.

### Tools exposed to the model

| Tool | Purpose |
|------|---------|
| `rag_search(query, doc_type?, cluster?, component?, limit?)` | Semantic search across the whole KB |
| `search_incidents(query, cluster?, component?, limit?)` | "Has this happened before?" — `doc_type=incident` |
| `search_runbooks(query, cluster?, component?, limit?)` | "What's the procedure?" — `doc_type=runbook` |
| `rag_collections()` | List collections + point counts |
| `rag_health()` | Reachability of Qdrant and the embedding provider |

Filters: `cluster` is a *soft* narrow (empty same-cluster result retries fleet-wide);
`component`/`doc_type` are hard filters.

## Add your own documents

Drop markdown or PDFs under `knowledge/` (subfolders become the default `doc_type`,
e.g. `knowledge/incidents/*` → `incident`). Markdown supports optional YAML front
matter:

```markdown
---
title: Longhorn volume stuck attaching
type: incident
tags: [longhorn, storage]
component: longhorn
cluster: prod-eu
---
# Body...
```

Then re-index — no rebuild needed (the job bind-mounts `./knowledge`):

```bash
docker compose run --rm rag-ingest
```

Ingestion is idempotent (chunk IDs derive from `(source, chunk index)`); shrunken
docs get their stale tail chunks deleted. Run it from CI/cron whenever docs change.
Known limitations: deleting or renaming a file does not remove its old chunks — use
`docker compose run --rm rag-ingest --recreate` after removals/renames.

## Switching to a hosted embedding provider

```bash
# .env
EMBEDDINGS_PROVIDER=openai
EMBEDDINGS_BASE_URL=https://api.openai.com    # or LiteLLM proxy / Azure gateway / TEI
EMBEDDINGS_API_KEY=sk-...
EMBEDDINGS_MODEL=text-embedding-3-small
```

Then rebuild the collection (ingest and query must always share provider+model):

```bash
docker compose run --rm rag-ingest --recreate
```

## Configuration

All configuration is environment-driven — see [.env.example](.env.example) for the
full annotated list and [docs/DESIGN.md](docs/DESIGN.md) for the design rationale,
retrieval-pipeline deep-dive, internal write API, and complete env table.

## Internal write API (optional)

Besides the read-only MCP tools, the server exposes plain-HTTP routes
(`/internal/knowledge/capture|similar|feedback|stats`) so a *trusted* automation
process can write incident records ("knowledge flywheel"). These are **not**
visible to the LLM. Gate them with `RAG_INTERNAL_TOKEN`. See
[docs/DESIGN.md](docs/DESIGN.md).

## Development

```bash
python3 -m pip install -r requirements.txt pytest
python3 -m pytest tests/
python3 server.py                       # run against a local Qdrant on :6333
QDRANT_URL=http://localhost:6333 python3 ingest.py --path knowledge
```

## License

TBD — add a license before sharing publicly if you want others to reuse this code.

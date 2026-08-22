"""DevXOps knowledge-base ingestion job.

Reads markdown and PDF documents (runbooks, past incidents, RCAs, reference
material), chunks them, embeds each chunk with Ollama, and upserts into Qdrant.
This is the deliberate, out-of-band write path for the RAG memory — the MCP
server itself is read-only.

Run it whenever the knowledge base changes (locally, from CI, or as a cron job):

    python ingest.py --path ./knowledge

Idempotent: chunk IDs are derived from (source, chunk index), so re-running
updates existing points instead of creating duplicates. If a document shrinks
between runs, its leftover tail chunks are deleted rather than left behind as
stale search hits (see `_delete_orphan_chunks`).

Markdown document format (front-matter is optional but recommended):

    ---
    title: Longhorn volume stuck attaching
    type: incident          # incident | runbook | rca | ...  (default: folder name, else "note")
    tags: [longhorn, storage, node-reboot]
    ---
    # Body markdown...

PDF documents are extracted page-by-page (each page becomes a `# [Page N]`
section so page context survives chunking). PDFs have no front matter: `type`
is always inferred from the containing folder name, and the title from the
file name.

If `type` is omitted, it is inferred from the containing folder name
(e.g. knowledge/incidents/* -> "incident", knowledge/runbooks/* -> "runbook").
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    Range,
)

import embeddings
import vectorstore

# Doc metadata worth indexing for cheap pre-filtering (step 3). Keyword indexes;
# creating one that exists is a no-op handled in vectorstore.ensure_collection.
# `source` is indexed for the stale-chunk cleanup filter, not for query filters.
_INDEXED_FIELDS = ("doc_type", "component", "cluster", "source")

# File types the ingester understands: markdown (front-matter aware, section
# chunking) and PDF (page-by-page text extraction). Everything else is skipped.
_SUPPORTED_SUFFIXES = (".md", ".pdf")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("rag-ingest")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
COLLECTION = os.environ.get("QDRANT_COLLECTION", "rag_kb")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))
HTTP_TIMEOUT = float(os.environ.get("RAG_TIMEOUT_SECONDS", "60"))

# Batch sizes for the two per-document round trips. Both cap work per HTTP
# request so one large document (a long runbook, a 100-page PDF) can't turn into
# a single oversized, all-or-nothing call.
#
# EMBED_BATCH_SIZE: chunks per embeddings request. Only matters for providers
#   with native batch input (OpenAI-compatible `/v1/embeddings`, capped at 2048
#   inputs and a per-request token ceiling well under a big doc's chunk count).
#   Ollama's `/api/embeddings` is single-prompt and loops internally regardless,
#   so on the default provider this value changes nothing.
# UPSERT_BATCH_SIZE: points per Qdrant upsert. Each point carries a dense vector,
#   a BM25 sparse vector and the chunk text, so a few hundred chunks in one body
#   is multiple megabytes against RAG_TIMEOUT_SECONDS.
EMBED_BATCH_SIZE = max(1, int(os.environ.get("EMBED_BATCH_SIZE", "32")))
UPSERT_BATCH_SIZE = max(1, int(os.environ.get("QDRANT_UPSERT_BATCH", "64")))

# Stable namespace so re-ingesting the same file overwrites its points.
_ID_NAMESPACE = uuid.UUID("6f3a9c1e-9b2d-5a44-8c11-a1b2c3d4e5f6")


def _parse_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    """Split optional YAML front matter from the body. Returns (meta, body)."""
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) == 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                meta = {}
            if isinstance(meta, dict):
                return meta, parts[2].strip()
    return {}, raw.strip()


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    """Paragraph-aware char chunking with overlap. Keeps whole paragraphs
    together when they fit; falls back to hard splits for oversized ones."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > size:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), size - overlap):
                chunks.append(para[i : i + size])
            continue
        if len(current) + len(para) + 2 > size:
            chunks.append(current)
            # carry the tail of the previous chunk for context continuity
            current = (current[-overlap:] + "\n\n" + para) if overlap else para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current:
        chunks.append(current)
    return chunks or [text]


def _chunk_document(body: str) -> list[str]:
    """Section-aware chunking (step 3): split on markdown headings so a runbook
    step or incident section stays intact, prepend the heading to each of its
    chunks for standalone context, then fall back to the paragraph chunker within
    an oversized section. Bodies with no headings behave exactly as before."""
    sections: list[tuple[str, list[str]]] = []
    heading = ""
    buf: list[str] = []
    for line in body.splitlines():
        if line.lstrip().startswith("#"):
            if heading or buf:
                sections.append((heading, buf))
            heading, buf = line.strip(), []
        else:
            buf.append(line)
    if heading or buf:
        sections.append((heading, buf))

    chunks: list[str] = []
    for head, body_lines in sections:
        text = "\n".join(body_lines).strip()
        if not text and not head:
            continue
        for c in _chunk(text, CHUNK_SIZE, CHUNK_OVERLAP) if text else [""]:
            chunks.append(f"{head}\n{c}".strip() if head else c)
    return [c for c in chunks if c] or _chunk(body, CHUNK_SIZE, CHUNK_OVERLAP)


def _infer_doc_type(meta: dict[str, Any], file: Path, root: Path) -> str:
    if meta.get("type"):
        return str(meta["type"])
    rel = file.relative_to(root)
    if len(rel.parts) > 1:
        folder = rel.parts[0].rstrip("s")  # incidents -> incident, runbooks -> runbook
        return folder
    return "note"


def _extract_pdf_text(path: Path) -> str:
    """Extract a PDF's text as one `# [Page N]` section per non-empty page.

    Page markers are markdown headings so `_chunk_document` keeps page context
    (layout and reading order rarely survive a page break; sections do).
    Returns "" for scanned/image-only PDFs with no extractable text.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"# [Page {i}]\n{text}")
    return "\n\n".join(pages)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Dense vectors for every chunk, EMBED_BATCH_SIZE chunks per request.

    Goes through `embeddings.embed_documents` rather than looping `embed()` so
    providers with native batch input spend one HTTP round trip per batch instead
    of one per chunk. Documents are embedded with the "document" task type; the
    provider and model are whatever embeddings.py is configured for. Ingest and
    query MUST share that config (see embeddings.py) or retrieval breaks.

    Order is preserved: `_embed_openai` realigns its response by `index`, and
    batches are concatenated in slice order, so vectors[i] belongs to texts[i].
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        vectors.extend(embeddings.embed_documents(texts[start : start + EMBED_BATCH_SIZE]))
    return vectors


def _upsert_points(client: QdrantClient, points: list[PointStruct]) -> None:
    """Upsert in UPSERT_BATCH_SIZE-point requests, waiting for each to be applied.

    `wait=True` keeps the per-file "ingested" log line honest — it means the
    chunks are actually queryable, not just queued — and means a failure part-way
    through a large document leaves the earlier batches committed instead of
    losing the whole file to one rejected request.
    """
    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        client.upsert(
            collection_name=COLLECTION,
            points=points[start : start + UPSERT_BATCH_SIZE],
            wait=True,
        )


def _delete_orphan_chunks(client: QdrantClient, source: str, kept: int) -> None:
    """Drop points left over from a previous, longer ingest of the same source.

    Point IDs are uuid5(f"{source}#{index}"), which makes re-ingest idempotent
    only while a document's chunk count never falls. Edit a runbook down from 30
    chunks to 20 — or re-export a PDF with fewer pages — and chunks 20..29 stay
    in the collection forever, still matching searches with stale text. Runs
    after the upsert so current chunks are already in place; best-effort, because
    an orphan is a stale search hit, not a reason to abort the whole ingest.
    """
    stale = Filter(
        must=[
            FieldCondition(key="source", match=MatchValue(value=source)),
            FieldCondition(key="chunk", range=Range(gte=kept)),
        ]
    )
    try:
        orphans = client.count(COLLECTION, count_filter=stale, exact=True).count
        if not orphans:
            return
        client.delete(collection_name=COLLECTION, points_selector=FilterSelector(filter=stale))
        log.info("removed %d stale chunk(s) from a previous ingest of %s", orphans, source)
    except Exception as exc:  # noqa: BLE001 - cleanup is best-effort, never fatal
        log.warning("stale-chunk cleanup failed for %s: %s", source, exc)


def _ensure_collection(client: QdrantClient, dim: int, recreate: bool) -> None:
    if client.collection_exists(COLLECTION) and recreate:
        log.info("recreating collection %s", COLLECTION)
        client.delete_collection(COLLECTION)
    # Named dense (+ BM25 sparse when hybrid is live) schema, shared with capture.
    vectorstore.ensure_collection(client, COLLECTION, dim, payload_indexes=_INDEXED_FIELDS)


def _discover_files(path: Path) -> list[Path]:
    """All supported (.md/.pdf) files under `path`, sorted for stable IDs."""
    return sorted(
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    )


def ingest(path: Path, recreate: bool) -> None:
    files = _discover_files(path)
    if not files:
        log.warning("no .md or .pdf files found under %s", path)
        return

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=HTTP_TIMEOUT)
    collection_ready = False
    total_chunks = 0

    for file in files:
        if file.suffix.lower() == ".pdf":
            meta: dict[str, Any] = {}
            body = _extract_pdf_text(file)
            if not body:
                log.warning("skipping PDF with no extractable text %s", file)
                continue
        else:
            raw = file.read_text(encoding="utf-8")
            meta, body = _parse_front_matter(raw)
            if not body:
                log.warning("skipping empty file %s", file)
                continue

        doc_type = _infer_doc_type(meta, file, path)
        source = str(file.relative_to(path)).replace(os.sep, "/")
        title = meta.get("title") or file.stem
        tags = meta.get("tags") or []

        chunks = _chunk_document(body)
        dense = _embed_batch(chunks)
        sparse = vectorstore.embed_documents_sparse(chunks)

        if not collection_ready:
            _ensure_collection(client, len(dense[0]), recreate)
            collection_ready = True

        # Pass through optional doc metadata for pre-filtering (step 3).
        extra = {k: meta[k] for k in ("component", "severity", "cluster") if meta.get(k)}
        points = [
            PointStruct(
                id=str(uuid.uuid5(_ID_NAMESPACE, f"{source}#{i}")),
                vector=vectorstore.named_vectors(d, s),
                payload={
                    "text": chunk,
                    "doc_type": doc_type,
                    "title": title,
                    "source": source,
                    "tags": tags,
                    "chunk": i,
                    **extra,
                },
            )
            for i, (chunk, d, s) in enumerate(zip(chunks, dense, sparse))
        ]
        _upsert_points(client, points)
        _delete_orphan_chunks(client, source, len(points))
        total_chunks += len(points)
        log.info("ingested %s (%s, %d chunk(s))", source, doc_type, len(points))

    log.info("done: %d file(s), %d chunk(s) into '%s'", len(files), total_chunks, COLLECTION)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest markdown/PDF docs into the Qdrant knowledge base.")
    parser.add_argument(
        "--path",
        default="./knowledge",
        help="Directory tree of .md and .pdf documents to ingest (default: ./knowledge).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before ingesting (full rebuild).",
    )
    args = parser.parse_args()

    root = Path(args.path).resolve()
    if not root.is_dir():
        parser.error(f"path not found or not a directory: {root}")

    emb = embeddings.describe()
    log.info(
        "ingesting from %s -> qdrant=%s collection=%s embed=%s:%s@%s",
        root, QDRANT_URL, COLLECTION, emb["provider"], emb["model"], emb["base_url"],
    )
    ingest(root, args.recreate)


if __name__ == "__main__":
    main()

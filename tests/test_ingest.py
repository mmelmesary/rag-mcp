"""Ingest: supported-file discovery, PDF text extraction, and write batching.

PDF extraction is exercised against a fake pypdf so no real PDF is needed.
Verifies that pages become `# [Page N]` sections (so section-aware chunking
keeps page context), that empty/image-only pages are skipped, and that
discovery picks up .md and .pdf files while ignoring everything else.

The batching tests cover the per-document write path against a fake Qdrant
client: embeddings go out in EMBED_BATCH_SIZE-sized requests with chunk order
intact, upserts are capped at UPSERT_BATCH_SIZE points, and a document that
shrinks between runs has its orphaned tail chunks deleted."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ingest  # noqa: E402


def test_discover_files_picks_md_and_pdf_only(tmp_path):
    (tmp_path / "runbooks").mkdir()
    (tmp_path / "runbooks" / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "runbooks" / "b.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "runbooks" / "c.PDF").write_bytes(b"%PDF-1.4")
    (tmp_path / "runbooks" / "d.txt").write_text("x", encoding="utf-8")
    (tmp_path / "runbooks" / "e.yaml").write_text("x", encoding="utf-8")

    found = [p.name for p in ingest._discover_files(tmp_path)]
    assert found == ["a.md", "b.pdf", "c.PDF"]
    assert "d.txt" not in found and "e.yaml" not in found


class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakePdfReader:
    def __init__(self, pages):
        self.pages = pages


def _patch_pdf(monkeypatch, pages):
    import types

    fake = types.SimpleNamespace(PdfReader=lambda _path: _FakePdfReader(pages))
    monkeypatch.setitem(sys.modules, "pypdf", fake)


def test_extract_pdf_text_builds_page_sections(monkeypatch):
    _patch_pdf(monkeypatch, [_FakePage("Page one body"), _FakePage("Page two body")])
    text = ingest._extract_pdf_text(Path("unused.pdf"))
    assert text == "# [Page 1]\nPage one body\n\n# [Page 2]\nPage two body"


def test_extract_pdf_text_skips_empty_pages(monkeypatch):
    _patch_pdf(monkeypatch, [_FakePage("   "), _FakePage("Real text"), _FakePage(None)])
    text = ingest._extract_pdf_text(Path("unused.pdf"))
    assert text == "# [Page 2]\nReal text"


def test_extract_pdf_text_empty_for_image_only_pdf(monkeypatch):
    _patch_pdf(monkeypatch, [_FakePage(""), _FakePage(" \n ")])
    assert ingest._extract_pdf_text(Path("unused.pdf")) == ""


# ---------------------------------------------------------------------------
# Embedding batches
# ---------------------------------------------------------------------------

def test_embed_batch_splits_requests_and_keeps_order(monkeypatch):
    """One request per EMBED_BATCH_SIZE chunks — not one per chunk — and the
    returned vectors stay aligned with the input chunks."""
    sizes = []

    def fake_embed_documents(texts):
        sizes.append(len(texts))
        # A vector that encodes its own text, so misordering is detectable.
        return [[float(len(t)), float(ord(t[0]))] for t in texts]

    monkeypatch.setattr(ingest.embeddings, "embed_documents", fake_embed_documents)
    monkeypatch.setattr(ingest, "EMBED_BATCH_SIZE", 3)

    texts = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff", "g"]
    vectors = ingest._embed_batch(texts)

    assert sizes == [3, 3, 1]  # batched, not 7 single-text calls
    assert vectors == [[float(len(t)), float(ord(t[0]))] for t in texts]


def test_embed_batch_empty_input_makes_no_requests(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ingest.embeddings, "embed_documents", lambda texts: calls.append(texts) or []
    )
    assert ingest._embed_batch([]) == []
    assert calls == []


# ---------------------------------------------------------------------------
# Upsert batches + stale-chunk cleanup
# ---------------------------------------------------------------------------

class _FakeCount:
    def __init__(self, count):
        self.count = count


class _FakeClient:
    """Records upsert/count/delete calls. `orphans` is what count() reports."""

    def __init__(self, orphans=0, count_raises=None):
        self.upserts = []          # list of point-id lists, one per request
        self.deletes = []          # captured points_selector values
        self.orphans = orphans
        self.count_raises = count_raises
        self.count_filters = []

    def upsert(self, collection_name, points, wait=None):
        self.upserts.append([p.id for p in points])

    def count(self, collection_name, count_filter=None, exact=True):
        self.count_filters.append(count_filter)
        if self.count_raises:
            raise self.count_raises
        return _FakeCount(self.orphans)

    def delete(self, collection_name, points_selector):
        self.deletes.append(points_selector)


def _points(n):
    from qdrant_client.models import PointStruct

    return [
        PointStruct(id=i + 1, vector={"dense": [0.0, 1.0]}, payload={"chunk": i})
        for i in range(n)
    ]


def test_upsert_points_caps_points_per_request(monkeypatch):
    monkeypatch.setattr(ingest, "UPSERT_BATCH_SIZE", 4)
    client = _FakeClient()

    ingest._upsert_points(client, _points(10))

    assert [len(batch) for batch in client.upserts] == [4, 4, 2]
    # Every point written exactly once, in order.
    assert [pid for batch in client.upserts for pid in batch] == list(range(1, 11))


def test_upsert_points_single_request_when_under_batch_size(monkeypatch):
    monkeypatch.setattr(ingest, "UPSERT_BATCH_SIZE", 64)
    client = _FakeClient()

    ingest._upsert_points(client, _points(5))

    assert len(client.upserts) == 1


def test_delete_orphan_chunks_targets_only_the_tail():
    client = _FakeClient(orphans=7)

    ingest._delete_orphan_chunks(client, "runbooks/longhorn.md", kept=20)

    assert len(client.deletes) == 1
    conditions = client.deletes[0].filter.must
    assert conditions[0].key == "source"
    assert conditions[0].match.value == "runbooks/longhorn.md"
    # Only chunks at/after the current count — the surviving 0..19 are untouched.
    assert conditions[1].key == "chunk"
    assert conditions[1].range.gte == 20


def test_delete_orphan_chunks_noop_when_nothing_stale():
    client = _FakeClient(orphans=0)

    ingest._delete_orphan_chunks(client, "runbooks/longhorn.md", kept=20)

    assert client.deletes == []  # no delete request at all


def test_delete_orphan_chunks_survives_qdrant_error(caplog):
    """Cleanup is best-effort: a stale hit is not worth aborting the ingest."""
    client = _FakeClient(count_raises=RuntimeError("qdrant down"))

    ingest._delete_orphan_chunks(client, "runbooks/longhorn.md", kept=3)

    assert client.deletes == []
    assert "stale-chunk cleanup failed" in caplog.text

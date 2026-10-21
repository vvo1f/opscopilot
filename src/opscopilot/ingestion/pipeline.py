"""parse -> chunk -> embed -> store, for one file."""
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from opscopilot import store
from opscopilot.config import settings
from opscopilot.embeddings import get_embedder
from opscopilot.ingestion.chunker import add_headers, chunk_pages, document_header
from opscopilot.ingestion.parsers import parse_file

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    document_id: int
    filename: str
    collection: str
    chunks: int
    skipped: bool  # True when the same file was already stored


def ingest_file(path: Path, collection: str, filename: str | None = None) -> IngestResult:
    filename = filename or path.name
    # The hash identifies the content, not the name: renaming a file doesn't
    # make it new, and editing it does.
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    with store.connect() as conn:  # one transaction: all rows are saved, or none
        existing = store.find_by_hash(conn, sha256)
        if existing:
            log.info("skip %s (already stored as id %s)", filename, existing["id"])
            return IngestResult(existing["id"], existing["filename"], existing["collection"], 0, True)

        pages = parse_file(path)
        chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
        if settings.chunk_header:
            chunks = add_headers(chunks, document_header(pages))
        vectors = get_embedder().embed_passages([c.text for c in chunks])

        document_id = store.insert_document(conn, filename, collection, sha256)
        store.insert_chunks(conn, document_id, chunks, vectors)

    log.info("ingested %s: %d pages, %d chunks", filename, len(pages), len(chunks))
    return IngestResult(document_id, filename, collection, len(chunks), False)

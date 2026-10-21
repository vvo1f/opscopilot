"""parse -> chunk -> embed -> classify/extract -> store, for one file."""
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from opscopilot import store
from opscopilot.config import settings
from opscopilot.docintel.analyze import analyze_text
from opscopilot.embeddings import get_embedder
from opscopilot.ingestion.chunker import add_headers, chunk_pages, document_header
from opscopilot.ingestion.parsers import parse_file
from opscopilot.llm.factory import get_llm

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    document_id: int
    filename: str
    collection: str
    chunks: int
    skipped: bool              # True when the same file was already stored
    label: str | None = None
    analysis_error: str | None = None


def _needs_analysis(existing: dict) -> bool:
    # retry documents that were never analysed or whose analysis failed
    return existing["label"] is None or bool((existing["fields"] or {}).get("error"))


def ingest_file(path: Path, collection: str, filename: str | None = None,
                analyze: bool = True, reanalyze: bool = False) -> IngestResult:
    filename = filename or path.name
    # The hash identifies the content, not the name: renaming a file doesn't
    # make it new, and editing it does.
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    with store.connect() as conn:  # one transaction: all rows are saved, or none
        existing = store.find_by_hash(conn, sha256)
        if existing:
            # Same file again: no new chunks. But if its analysis is missing or
            # failed (quota, network), redo just that, without re-embedding.
            result = IngestResult(existing["id"], existing["filename"], existing["collection"],
                                  0, True, existing["label"],
                                  (existing["fields"] or {}).get("error"))
            if analyze and (reanalyze or _needs_analysis(existing)):
                pages = parse_file(path)
                analysis = analyze_text("\n".join(p.text for p in pages), get_llm())
                store.set_analysis(conn, existing["id"], analysis.label, analysis.fields,
                                   analysis.error)
                result.label, result.analysis_error = analysis.label, analysis.error
            else:
                log.info("skip %s (already stored as id %s)", filename, existing["id"])
            return result

        pages = parse_file(path)
        chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
        if settings.chunk_header:
            chunks = add_headers(chunks, document_header(pages))
        vectors = get_embedder().embed_passages([c.text for c in chunks])

        # analyse the clean page text, not the header-prefixed chunks
        analysis = (analyze_text("\n".join(p.text for p in pages), get_llm()) if analyze else None)

        document_id = store.insert_document(conn, filename, collection, sha256)
        store.insert_chunks(conn, document_id, chunks, vectors)
        if analysis:
            store.set_analysis(conn, document_id, analysis.label, analysis.fields, analysis.error)

    log.info("ingested %s: %d pages, %d chunks", filename, len(pages), len(chunks))
    return IngestResult(document_id, filename, collection, len(chunks), False,
                        analysis.label if analysis else None,
                        analysis.error if analysis else None)

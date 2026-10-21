"""An in-memory copy of the retrieval index, for experiments.

It chunks and embeds the documents with the same code as ingestion, but keeps
the vectors in a numpy array. That lets us try other chunk sizes, with or
without headers, without touching the database. At this size Postgres does an
exact scan anyway, so the ranking is the same as the real system's.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opscopilot.config import settings
from opscopilot.ingestion.chunker import add_headers, chunk_pages, document_header
from opscopilot.ingestion.parsers import parse_file
from opscopilot.rag.retriever import Source


@dataclass
class MemoryIndex:
    chunks: list[dict]      # filename, collection, page, text
    vectors: np.ndarray     # one normalised row per chunk
    embedder: object

    def search(self, question: str, k: int | None = None, collection: str | None = None) -> list[Source]:
        k = k or settings.top_k
        # vectors are normalised, so the dot product is the cosine similarity
        scores = self.vectors @ self.embedder.embed_query(question)
        results: list[Source] = []
        for i in np.argsort(-scores):
            c = self.chunks[i]
            if collection and c["collection"] != collection:
                continue
            results.append(Source(n=len(results) + 1, document_id=0, filename=c["filename"],
                                  collection=c["collection"], page=c["page"], text=c["text"],
                                  score=float(scores[i])))
            if len(results) == k:
                break
        return results


def build_index(folder: Path, embedder, chunk_size: int | None = None, overlap: int | None = None,
                header: bool | None = None) -> MemoryIndex:
    chunk_size = chunk_size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap
    header = settings.chunk_header if header is None else header

    chunks = []
    for path in sorted(p for p in folder.glob("*/*") if p.is_file()):
        pages = parse_file(path)
        pieces = chunk_pages(pages, chunk_size, overlap)
        if header:
            pieces = add_headers(pieces, document_header(pages))
        chunks += [{"filename": path.name, "collection": path.parent.name,
                    "page": c.page, "text": c.text} for c in pieces]
    vectors = embedder.embed_passages([c["text"] for c in chunks])
    return MemoryIndex(chunks, vectors, embedder)

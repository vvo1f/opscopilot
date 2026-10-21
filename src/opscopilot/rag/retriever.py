"""Question -> the most similar chunks."""
from dataclasses import dataclass

from opscopilot import store
from opscopilot.config import settings
from opscopilot.embeddings import get_embedder


@dataclass
class Source:
    n: int            # rank, starting at 1; this is the number the LLM cites as [n]
    document_id: int
    filename: str
    collection: str
    page: int
    text: str
    score: float      # cosine similarity, higher is closer


def search(query: str, k: int | None = None, collection: str | None = None) -> list[Source]:
    k = k or settings.top_k
    # the same model embeds the question and the chunks; that is what makes
    # their vectors comparable
    query_vec = get_embedder().embed_query(query)
    with store.connect() as conn:
        rows = store.search_chunks(conn, query_vec, k, collection)
    return [Source(n=i, document_id=r["document_id"], filename=r["filename"],
                   collection=r["collection"], page=r["page"], text=r["content"],
                   score=float(r["score"]))
            for i, r in enumerate(rows, start=1)]

"""All SQL for the app database lives here.

Keeping it in one place means the rest of the code never builds queries, and
swapping pgvector for another vector store would touch only this file.
"""
import logging

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from opscopilot.config import settings

log = logging.getLogger(__name__)

# One row per uploaded file. sha256 is unique, which is what makes ingestion
# idempotent: the same file can't be stored twice.
SCHEMA = [
    """CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        filename TEXT NOT NULL,
        collection TEXT NOT NULL,
        sha256 CHAR(64) UNIQUE NOT NULL,
        label TEXT,
        fields JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )""",
    # One row per chunk. The vector size is fixed by the embedding model, so
    # the column type depends on EMBEDDING_DIM.
    """CREATE TABLE IF NOT EXISTS chunks (
        id SERIAL PRIMARY KEY,
        document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        page INT NOT NULL,
        chunk_index INT NOT NULL,
        content TEXT NOT NULL,
        embedding vector({dim}) NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id)",
    # HNSW = approximate nearest-neighbour graph index; vector_cosine_ops makes
    # it match the <=> (cosine distance) operator we query with in phase 4.
    "CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops)",
]


def connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)  # lets psycopg send numpy arrays as pgvector values
    return conn


def init_schema() -> None:
    # register_vector needs the extension to exist, so create it on a plain connection first
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        for statement in SCHEMA:
            conn.execute(statement.format(dim=settings.embedding_dim))
        # for the vector type, atttypmod holds the declared dimension
        found = conn.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'").fetchone()[0]
    if found != settings.embedding_dim:
        raise RuntimeError(
            f"chunks.embedding has {found} dimensions but EMBEDDING_DIM is "
            f"{settings.embedding_dim}. Run: python scripts/ingest_folder.py --reset")


def reset_schema() -> None:
    """Drop both tables. Needed when the embedding model (and its dimension) changes."""
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute("DROP TABLE IF EXISTS documents")
    init_schema()


def find_by_hash(conn: psycopg.Connection, sha256: str) -> dict | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, filename, collection, label, fields FROM documents WHERE sha256 = %s",
                    (sha256,))
        return cur.fetchone()


def insert_document(conn: psycopg.Connection, filename: str, collection: str, sha256: str) -> int:
    row = conn.execute(
        "INSERT INTO documents (filename, collection, sha256) VALUES (%s, %s, %s) RETURNING id",
        (filename, collection, sha256)).fetchone()
    return row[0]


def insert_chunks(conn: psycopg.Connection, document_id: int, chunks: list, vectors) -> None:
    rows = [(document_id, c.page, c.index, c.text, vec) for c, vec in zip(chunks, vectors)]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (document_id, page, chunk_index, content, embedding) "
            "VALUES (%s, %s, %s, %s, %s)", rows)


def set_analysis(conn: psycopg.Connection, document_id: int, label: str | None,
                 fields: dict | None, error: str | None) -> None:
    """Save the classification and the extracted fields.

    A failure is stored too, as {"error": "..."} in fields, so the API can show it
    and a later run knows which documents to retry.
    """
    payload = {"error": error} if error else (fields or {})
    conn.execute("UPDATE documents SET label = %s, fields = %s WHERE id = %s",
                 (label, Jsonb(payload), document_id))


def list_documents(conn: psycopg.Connection) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT d.id, d.filename, d.collection, d.label, d.created_at, count(c.id) AS chunks "
            "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
            "GROUP BY d.id ORDER BY d.id")
        return cur.fetchall()


def get_document(conn: psycopg.Connection, document_id: int) -> dict | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT d.id, d.filename, d.collection, d.label, d.fields, d.created_at, "
            "count(c.id) AS chunks FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
            "WHERE d.id = %s GROUP BY d.id", (document_id,))
        return cur.fetchone()


def search_chunks(conn: psycopg.Connection, query_vec, k: int, collection: str | None) -> list[dict]:
    """The k chunks closest to query_vec, optionally inside one collection.

    <=> is pgvector's cosine distance (0 = same direction, 2 = opposite), so
    1 - distance is the cosine similarity. Ordering by the raw distance is what
    lets Postgres use the HNSW index.
    """
    # With an HNSW index a WHERE filter is applied after the index scan, so a
    # selective filter can leave fewer than k rows. A wider search (ef_search)
    # makes that less likely. SET LOCAL only lasts for this transaction.
    conn.execute("SET LOCAL hnsw.ef_search = 100")
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT c.id AS chunk_id, c.document_id, d.filename, d.collection,
                   c.page, c.chunk_index, c.content,
                   1 - (c.embedding <=> %(q)s) AS score
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE %(collection)s::text IS NULL OR d.collection = %(collection)s
            ORDER BY c.embedding <=> %(q)s
            LIMIT %(k)s
            """,
            {"q": query_vec, "collection": collection, "k": k})
        return cur.fetchall()

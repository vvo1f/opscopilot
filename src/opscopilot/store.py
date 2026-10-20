"""All SQL for the app database lives here.

Keeping it in one place means the rest of the code never builds queries, and
swapping pgvector for another vector store would touch only this file.
"""
import logging

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

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
        cur.execute("SELECT id, filename, collection FROM documents WHERE sha256 = %s", (sha256,))
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

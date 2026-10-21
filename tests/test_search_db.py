"""Similarity search against the real database, using hand-made vectors.

With basis vectors the cosine similarities are known exactly, so the ordering,
the scores and the collection filter can be checked without any embedding model.
"""
import uuid

import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient

from opscopilot import store
from opscopilot.config import settings
from opscopilot.ingestion.chunker import Chunk
from opscopilot.main import app

pytestmark = pytest.mark.integration


def unit(i):
    v = np.zeros(settings.embedding_dim, dtype=np.float32)
    v[i] = 1.0
    return v


@pytest.fixture
def data():
    try:
        psycopg.connect(settings.database_url, connect_timeout=2).close()
    except psycopg.Error:
        pytest.skip("database not reachable")
    store.init_schema()
    tag = uuid.uuid4().hex[:8]
    mine, other = f"t{tag}a", f"t{tag}b"
    rows = [  # (collection, file, text, vector)
        (mine, "far.txt", "far away", unit(0)),
        (mine, "near.txt", "very close", unit(1)),
        (other, "also_near.txt", "close but elsewhere", unit(1)),
    ]
    with store.connect() as conn:
        for collection, name, text, vec in rows:
            doc_id = store.insert_document(conn, f"{tag}_{name}", collection, uuid.uuid4().hex * 2)
            store.insert_chunks(conn, doc_id, [Chunk(1, 0, text)], [vec])
    yield mine, other
    with store.connect() as conn:
        conn.execute("DELETE FROM documents WHERE collection IN (%s, %s)", (mine, other))


def query_vector():
    q = 0.9 * unit(1) + 0.1 * unit(0)
    return (q / np.linalg.norm(q)).astype(np.float32)


def test_nearest_first_with_cosine_scores(data):
    mine, _ = data
    with store.connect() as conn:
        rows = store.search_chunks(conn, query_vector(), 5, mine)
    assert [r["content"] for r in rows] == ["very close", "far away"]
    assert rows[0]["score"] == pytest.approx(0.9 / np.sqrt(0.82), abs=1e-4)
    assert rows[1]["score"] == pytest.approx(0.1 / np.sqrt(0.82), abs=1e-4)


def test_collection_filter_never_leaks(data):
    _, other = data
    with store.connect() as conn:
        rows = store.search_chunks(conn, query_vector(), 5, other)
    assert {r["collection"] for r in rows} == {other}


def test_search_and_ask_endpoints(data, monkeypatch):
    mine, _ = data

    class FakeEmbedder:
        def embed_query(self, text):
            return query_vector()

    class FakeLLM:
        def chat(self, messages, temperature=0.0, max_tokens=None):
            return "The close one [1]."

    monkeypatch.setattr("opscopilot.rag.retriever.get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr("opscopilot.main.get_llm", lambda: FakeLLM())

    with TestClient(app) as client:
        hits = client.post("/search", json={"query": "anything", "k": 2, "collection": mine}).json()
        assert [h["text"] for h in hits] == ["very close", "far away"]
        assert hits[0]["n"] == 1

        reply = client.post("/ask", json={"question": "anything", "collection": mine}).json()
        assert reply["found"] is True
        assert [c["text"] for c in reply["citations"]] == ["very close"]

        assert client.post("/search", json={"query": ""}).status_code == 422

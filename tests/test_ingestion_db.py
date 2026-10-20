"""Needs the Postgres container. Skipped when the database is not reachable.

The embedding model is replaced with a fake, so the tests don't download or
load anything big. They use a throwaway collection name and delete it at the end.
"""
import uuid

import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient

from opscopilot import store
from opscopilot.config import settings
from opscopilot.datagen.documents import make_invoice
from opscopilot.datagen.writers import write_doc
from opscopilot.main import app

pytestmark = pytest.mark.integration


class FakeEmbedder:
    def embed_passages(self, texts, batch_size=16):
        rng = np.random.default_rng(0)
        return rng.random((len(texts), settings.embedding_dim), dtype=np.float32)


@pytest.fixture
def client(monkeypatch):
    try:
        psycopg.connect(settings.database_url, connect_timeout=2).close()
    except psycopg.Error:
        pytest.skip("database not reachable")
    monkeypatch.setattr("opscopilot.ingestion.pipeline.get_embedder", lambda: FakeEmbedder())
    with TestClient(app) as c:  # the with-block runs startup, which creates the tables
        yield c


@pytest.fixture
def collection(client):
    name = "t" + uuid.uuid4().hex[:10]
    yield name
    with store.connect() as conn:  # chunks go too: ON DELETE CASCADE
        conn.execute("DELETE FROM documents WHERE collection = %s", (name,))


def upload(client, collection, name, content, mime="text/plain"):
    return client.post("/documents", data={"collection": collection},
                       files={"file": (name, content, mime)})


def test_upload_then_duplicate_is_skipped(client, collection):
    content = f"Unique policy text {uuid.uuid4()}. " * 40
    first = upload(client, collection, "a.txt", content.encode())
    assert first.status_code == 200 and first.json()["skipped"] is False
    assert first.json()["chunks"] >= 2

    again = upload(client, collection, "renamed.txt", content.encode())  # same bytes, new name
    assert again.json()["skipped"] is True
    assert again.json()["document_id"] == first.json()["document_id"]

    doc = client.get(f"/documents/{first.json()['document_id']}").json()
    assert doc["chunks"] == first.json()["chunks"]
    assert any(d["id"] == doc["id"] for d in client.get("/documents").json())


def test_pdf_upload(client, collection, tmp_path):
    path = write_doc(make_invoice(0), tmp_path)
    resp = upload(client, collection, path.name, path.read_bytes() + b"\n%" + uuid.uuid4().bytes,
                  "application/pdf")
    assert resp.status_code == 200, resp.text
    assert resp.json()["chunks"] >= 1


def test_rejections(client, collection):
    assert upload(client, collection, "x.exe", b"data").status_code == 415
    assert upload(client, collection, "x.txt", b"  \n ").status_code == 422
    assert upload(client, "Bad Name!", "x.txt", b"text").status_code == 422
    assert client.get("/documents/99999999").status_code == 404

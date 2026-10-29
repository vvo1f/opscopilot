import logging
import re
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

import psycopg
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from opscopilot import store
from opscopilot.config import settings
from opscopilot.ingestion.parsers import SUPPORTED, EmptyDocument, UnsupportedFileType
from opscopilot.ingestion.pipeline import ingest_file
from opscopilot.llm.client import LLMError
from opscopilot.llm.factory import get_llm
from opscopilot.rag.answer import ask
from opscopilot.rag.retriever import search
from opscopilot.triage.executor import simulate
from opscopilot.triage.service import triage_email

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create the tables on startup; if the database is down the API still
    # starts, and /health will say so
    try:
        store.init_schema()
    except (psycopg.Error, RuntimeError) as exc:
        log.warning("could not initialise the database schema: %s", exc)
    yield


app = FastAPI(title="OpsCopilot", lifespan=lifespan)


def db_ok(url: str) -> bool:
    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except psycopg.Error:
        return False


@app.get("/health")
def health():
    # geo_db is optional (compose profile "geo"), so false there is not an error yet
    return {
        "app_db": db_ok(settings.database_url),
        "geo_db": db_ok(settings.geo_database_url),
    }


# These endpoints are plain `def`, not `async def`: parsing and embedding block
# the CPU, and FastAPI runs sync endpoints in a thread pool so they don't freeze
# the event loop.
@app.post("/documents")
def upload_document(file: Annotated[UploadFile, File()], collection: Annotated[str, Form()],
                    analyze: Annotated[bool, Form()] = True):
    if not re.fullmatch(r"[a-z0-9_-]{1,32}", collection):
        raise HTTPException(422, "collection must be 1-32 chars: lowercase letters, digits, - or _")
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(415, f"unsupported file type '{suffix}'. Use: {sorted(SUPPORTED)}")

    data = file.file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"file larger than {settings.max_upload_mb} MB")

    # the parsers want a real path, so write the upload to a temp folder first
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        path.write_bytes(data)
        try:
            result = ingest_file(path, collection, filename, analyze=analyze)
        except (UnsupportedFileType, EmptyDocument) as exc:
            raise HTTPException(422, str(exc)) from exc
    return {"document_id": result.document_id, "filename": result.filename,
            "collection": result.collection, "chunks": result.chunks, "skipped": result.skipped,
            "label": result.label, "analysis_error": result.analysis_error}


@app.get("/documents")
def list_documents():
    with store.connect() as conn:
        return store.list_documents(conn)


@app.get("/documents/{document_id}")
def get_document(document_id: int):
    with store.connect() as conn:
        doc = store.get_document(conn, document_id)
    if doc is None:
        raise HTTPException(404, "document not found")
    return doc


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(5, ge=1, le=20)
    collection: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(5, ge=1, le=20)
    collection: str | None = None  # "hr" switches to the HR assistant


@app.post("/search")
def search_chunks(req: SearchRequest):
    return [asdict(s) for s in search(req.query, req.k, req.collection)]


@app.post("/ask")
def ask_question(req: AskRequest):
    try:
        return asdict(ask(req.question, get_llm(), req.collection, req.k))
    except LLMError as exc:
        raise HTTPException(502, f"language model unavailable: {exc}") from exc


class TriageRequest(BaseModel):
    sender: str = Field(min_length=1, max_length=200)
    subject: str = Field(max_length=300)
    body: str = Field(min_length=1, max_length=20000)


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]


@app.post("/emails/triage")
def triage(req: TriageRequest):
    """Propose an action for an email. Nothing is executed until a person approves it."""
    try:
        proposal, flags = triage_email(req.sender, req.subject, req.body, get_llm())
    except LLMError as exc:
        raise HTTPException(502, f"could not triage the email: {exc}") from exc
    with store.connect() as conn:
        return store.insert_proposal(conn, req.sender, req.subject, req.body,
                                     proposal.model_dump(), suspicious=bool(flags))


@app.get("/emails/proposals")
def list_email_proposals(status: Literal["pending", "approved", "rejected"] | None = None):
    with store.connect() as conn:
        return store.list_proposals(conn, status)


@app.post("/emails/proposals/{proposal_id}/decision")
def decide(proposal_id: int, req: DecisionRequest):
    with store.connect() as conn:
        proposal = store.get_proposal(conn, proposal_id)
        if proposal is None:
            raise HTTPException(404, "proposal not found")
        if proposal["status"] != "pending":
            raise HTTPException(409, f"proposal is already {proposal['status']}")
        if req.decision == "approve":
            updated = store.decide_proposal(conn, proposal_id, "approved", simulate(proposal))
        else:
            updated = store.decide_proposal(conn, proposal_id, "rejected", None)
    if updated is None:  # someone decided it between our check and our update
        raise HTTPException(409, "proposal was decided in the meantime")
    return updated

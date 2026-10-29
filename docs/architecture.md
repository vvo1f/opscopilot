# Architecture

```
Client (Swagger UI / curl / scripts)
        |
   FastAPI app (src/opscopilot/main.py)
        |
        +-- ingestion  parsers -> chunker (+header) -> embedder -> store
        +-- docintel   classify -> extract (pydantic schemas, repair loop)
        +-- rag        retriever (cosine search) -> grounded prompt -> answer + citations
        |
   store.py -----> PostgreSQL + pgvector  (documents, chunks, HNSW index)
   llm/client.py -> Ollama | Gemini | Groq   (OpenAI-compatible API)
   embeddings.py -> sentence-transformers (multilingual-e5-small, local)
```

## Data model

- `documents(id, filename, collection, sha256 UNIQUE, label, fields JSONB, created_at)`
- `chunks(id, document_id -> documents, page, chunk_index, content, embedding vector(EMBEDDING_DIM))`
- HNSW index on `chunks.embedding` with cosine distance.

`sha256` makes ingestion idempotent. `fields` holds the extracted values, or `{"error": ...}` when the
analysis failed, so a later run knows what to retry.

## Flows

**Ingestion.** Hash the file; if known, only retry a missing analysis. Otherwise parse pages, chunk each page
(paragraph, line, sentence, word boundaries, with overlap), prepend the document header to chunks that lack it,
embed in batches, classify and extract with the LLM, and store everything in one transaction.

**Question answering.** Embed the question with the same model, order chunks by cosine distance (optionally in one
collection), put the top k into a numbered prompt, and ask the model to answer only from them and cite `[n]`.
`NOT_FOUND`, an empty result or a score below `MIN_SIMILARITY` produces a fixed "not found" answer.

**Evaluation.** `eval/run_eval.py` rebuilds the index in memory (same chunking and embeddings), so chunk size and
header experiments never touch the database. Classification and extraction are scored from what ingestion stored.

## Configuration

Everything is read from `.env` (see `env.example`): LLM endpoint and model, embedding model and dimension,
database URLs, chunk size and overlap, `CHUNK_HEADER`, `TOP_K`, `MIN_SIMILARITY`.
Changing the embedding model changes the vector size: run `python scripts/ingest_folder.py --reset`.

# OpsCopilot

LLM-based assistant for internal business processes: document ingestion,
classification and extraction, semantic search with RAG, and (later) email
triage and natural-language SQL. Runs locally with Ollama, no paid APIs.

Status: phases 0-5 (setup, data, LLM client, embeddings, ingestion, RAG, classification and extraction).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev,data,ml]"
cp env.example .env                 # Windows: copy env.example .env

docker compose up -d               # Postgres + pgvector
# edit .env: put your Gemini key (or use Ollama, see env.example)

python scripts/llm_ping.py
python scripts/generate_data.py        # synthetic documents + ground truth + eval sets
python scripts/ingest_folder.py        # parse, chunk, embed, classify and store them (needs the ml extra + LLM)
uvicorn opscopilot.main:app --reload   # then open http://localhost:8000/health
pytest
```

Embeddings need the `ml` extra (PyTorch). On Windows the default wheel is CPU-only.
On Linux install the CPU build first to avoid a multi-GB CUDA download:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,ml]"
python scripts/embed_ping.py
```

## Layout

```
src/opscopilot/
  config.py        settings from .env
  llm/client.py    OpenAI-compatible client, retries, JSON repair loop
  llm/factory.py   builds the client from settings
  embeddings.py    sentence-transformers wrapper
  main.py          FastAPI app
  datagen/         synthetic documents, ground truth, eval sets
  store.py         all SQL (tables, pgvector index, inserts)
  ingestion/       parsers.py, chunker.py, pipeline.py
  rag/             retriever.py, answer.py
  docintel/        schemas.py, analyze.py (classification + extraction)
  prompts/         *.txt, one file per prompt
scripts/           generate_data.py, llm_ping.py, embed_ping.py
data/              ground_truth/ and eval_sets/ are committed; documents/ is generated
tests/             unit tests (no LLM, no database needed)
```

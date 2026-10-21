# OpsCopilot

LLM-based assistant for internal business processes: document ingestion,
classification and extraction, semantic search with RAG, and (later) email
triage and natural-language SQL. Runs locally with Ollama, no paid APIs.

Status: phases 0-6 (setup, data, LLM client, embeddings, ingestion, RAG, classification and extraction, evaluation).

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
eval/              run_eval.py (all metrics), cache_llm.py (answers cached on disk)
scripts/           generate_data.py, llm_ping.py, embed_ping.py
data/              ground_truth/ and eval_sets/ are committed; documents/ is generated
tests/             unit tests (no LLM, no database needed)
```

## Evaluation

```bash
python eval/run_eval.py retrieval        # hit@k and MRR, no LLM
python eval/run_eval.py chunks           # chunk size x header sweep, no LLM
python eval/run_eval.py answers          # RAG answers, 30 LLM calls (cached in eval/cache)
python eval/run_eval.py answers --prompt rag_plain
python eval/run_eval.py classification   # reads what ingestion stored
python eval/run_eval.py extraction
python eval/run_eval.py report           # writes eval/results/summary.md
```

The datasets are small and synthetic (28 documents, 30 questions), so scores are
optimistic and differences of a few points are noise.

# Deployment notes

Status: **documented, not deployed.** The project has been run locally only.

## Container

`Dockerfile` builds the API with CPU-only PyTorch. `docker compose --profile app up -d --build` starts the API
next to the pgvector database. The embedding model is downloaded on first use into the `hf_cache` volume.
If the LLM is Ollama on the host, set `LLM_BASE_URL=http://host.docker.internal:11434/v1` and start Ollama with
`OLLAMA_HOST=0.0.0.0` so the container can reach it.

## Cloud

The image runs on any container service (Azure Container Apps, AWS App Runner, Google Cloud Run). Managed
PostgreSQL on all three clouds supports the pgvector extension, so `DATABASE_URL` is the only change for the
database. The LLM is already a hosted API (Gemini or Groq), configured through environment variables.

Before a real deployment: add authentication, put secrets in the platform's secret store, move ingestion to a
background queue, and cap upload sizes at the gateway.

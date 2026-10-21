"""Application settings.

Every field is filled from the environment variable with the same name
(case-insensitive), then from the .env file, then from the default below.
pydantic casts the types for us, so EMBEDDING_DIM=384 arrives as an int.

Switching from the local model to a hosted API is only a .env change.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# the .env next to pyproject.toml, wherever the script is run from
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    # LLM, reached through an OpenAI-compatible endpoint (Ollama, Groq, Gemini...)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen2.5:3b"
    llm_timeout: float = 180.0  # seconds; small local models can be slow on CPU

    # Embeddings. The dimension must match the model, and it is also the size of
    # the vector column in Postgres, so changing model means re-creating the table.
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384

    # Chunking. The sample documents are only 250-550 characters long, so the chunks
    # are small on purpose: at 600 every document would be a single chunk and
    # retrieval would just be picking a whole document.
    chunk_size: int = 300       # max characters per chunk
    chunk_overlap: int = 60     # characters repeated at the start of the next chunk
    # Contextual header: the document's first lines are put in front of every chunk, so
    # a chunk like "Total: 1.234,56 EUR" still carries the invoice number it belongs to.
    chunk_header: bool = True

    # How much of a document is sent to the LLM for classification and extraction
    analysis_chars: int = 6000

    # Retrieval / answering
    top_k: int = 5
    # Chunks whose best cosine similarity is below this are not sent to the LLM.
    # 0 = off. E5 models give similar scores to almost everything (roughly 0.7-0.9),
    # so the right value has to be calibrated on the eval set in phase 6, not guessed.
    min_similarity: float = 0.0

    max_upload_mb: int = 10

    database_url: str = "postgresql://opscopilot:opscopilot@localhost:5434/opscopilot"
    geo_database_url: str = "postgresql://geo_admin:geo_admin@localhost:5433/geo"


settings = Settings()

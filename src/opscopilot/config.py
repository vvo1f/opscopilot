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
    max_upload_mb: int = 10

    database_url: str = "postgresql://opscopilot:opscopilot@localhost:5434/opscopilot"
    geo_database_url: str = "postgresql://geo_admin:geo_admin@localhost:5433/geo"


settings = Settings()

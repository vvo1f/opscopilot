"""Text -> vector, using a Hugging Face sentence-transformers model.

An embedding model maps a text to a fixed-size vector so that texts with a
similar meaning end up close to each other. Multilingual models also put an
Italian sentence near its English translation, which is what we rely on.
"""
import logging
from functools import lru_cache

import numpy as np

from opscopilot.config import settings

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self.dim = dim
        self._model = None  # loaded on first use, not at import time

        # E5 models were trained with these prefixes and give worse results
        # without them. BGE-M3 doesn't need any.
        self._e5 = "e5" in model_name.lower()

    def _load(self):
        if self._model is None:
            # Imported here because sentence-transformers pulls in PyTorch,
            # and we don't want the API (or the unit tests) to pay for it.
            from sentence_transformers import SentenceTransformer

            log.info("loading embedding model %s (first run downloads it)", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            [prefix + t for t in texts],
            batch_size=batch_size,        # small batches keep RAM usage low on a laptop
            normalize_embeddings=True,    # unit length: dot product == cosine similarity
            show_progress_bar=False,
        )
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"{self.model_name} produces {vectors.shape[1]}-dim vectors but "
                f"EMBEDDING_DIM is {self.dim}. Fix .env (and re-create the chunks table)."
            )
        return vectors

    def embed_passages(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Embed document chunks (what we store)."""
        return self._encode(texts, "passage: " if self._e5 else "", batch_size)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a user question (what we search with)."""
        return self._encode([text], "query: " if self._e5 else "", 1)[0]


@lru_cache
def get_embedder() -> Embedder:
    """One shared instance, so the model is loaded only once per process."""
    return Embedder(settings.embedding_model, settings.embedding_dim)

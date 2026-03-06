"""Generate embeddings using BAAI/bge-small-en-v1.5 via fastembed."""

import logging
from typing import Optional

import numpy as np
from fastembed import TextEmbedding

from pipeline import Chunk

logger = logging.getLogger(__name__)


class BGEEmbedder:
    """Generate embeddings using BAAI/bge-small-en-v1.5."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 256,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        logger.info(f"Loading embedding model: {model_name}")
        self.model = TextEmbedding(model_name=model_name)
        self.dimension = 384  # BGE-small dimension
        logger.info(f"Model loaded (dim={self.dimension})")

    def embed_chunks(self, chunks: list[Chunk]) -> list[np.ndarray]:
        """Generate embeddings for a list of chunks."""
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        embeddings = list(self.model.embed(texts, batch_size=self.batch_size))
        return [np.array(e, dtype=np.float32) for e in embeddings]

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        embeddings = list(self.model.embed([query]))
        return np.array(embeddings[0], dtype=np.float32)

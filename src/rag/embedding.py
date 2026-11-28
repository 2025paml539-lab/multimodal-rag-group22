"""Real sentence-embedding model for the lab (GPU if available). Tests use HashEmbedder instead."""
from __future__ import annotations
from typing import Sequence
import numpy as np


class SentenceTransformerEmbedder:
    """Satisfies the Embedder protocol: rows are L2-normalised. The model is imported lazily,
    so importing this module never loads torch."""

    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "auto",
                 batch_size: int = 64):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError as e:
            raise RuntimeError("sentence-transformers is not installed. In the lab: "
                               "pip install -r requirements-lab.txt (never reinstall torch)") from e
        self.name = name
        self.batch_size = batch_size
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model = SentenceTransformer(name, device=self.device)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True,
                                 batch_size=self.batch_size)
        return np.asarray(vecs, dtype=np.float32)

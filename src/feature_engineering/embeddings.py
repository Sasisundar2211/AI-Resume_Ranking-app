"""Deterministic hash-based embeddings for lightweight ranking."""

from __future__ import annotations

import hashlib

import numpy as np


def generate_embedding(text: str, dims: int = 64) -> np.ndarray:
    """Generate deterministic normalized embedding vector from text."""
    digest = hashlib.sha256((text or "").encode("utf-8")).digest()
    raw = np.frombuffer((digest * ((dims // len(digest)) + 1))[:dims], dtype=np.uint8)
    vector = raw.astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector

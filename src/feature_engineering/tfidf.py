"""TF-IDF feature extractor wrapper."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFExtractor:
    """Small wrapper around scikit-learn's TF-IDF vectorizer."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")

    def fit(self, documents: list[str]) -> None:
        self.vectorizer.fit(documents)

    def transform(self, document: str) -> np.ndarray:
        matrix = self.vectorizer.transform([document]).toarray()
        return np.array(matrix[0])

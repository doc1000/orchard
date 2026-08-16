"""Offline TF-IDF embedding backend (phase3 tfidf math over document text)."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

TOKEN_PATTERN = re.compile(r"(?u)\b[\w][\w-]*\b")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def tfidf_matrix(texts: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Return TF-IDF feature matrix and sorted vocabulary for raw texts."""
    documents = [tokenize(text) for text in texts]
    vocabulary = sorted({token for document in documents for token in document})
    if not vocabulary:
        raise ValueError("TF-IDF vocabulary is empty")
    index = {term: position for position, term in enumerate(vocabulary)}
    document_frequency = Counter(term for document in documents for term in set(document))
    values = np.zeros((len(documents), len(vocabulary)), dtype=np.float64)
    for row, document in enumerate(documents):
        if not document:
            raise ValueError(f"document at index {row} has no tokens")
        counts = Counter(document)
        for term, count in counts.items():
            tf = count / len(document)
            idf = math.log((1 + len(documents)) / (1 + document_frequency[term])) + 1.0
            values[row, index[term]] = tf * idf
    return values, vocabulary


@dataclass
class TfidfEmbeddingBackend:
    """Inspectable offline lexical backend.

    Semantic default is MiniLM when ``orchard[embeddings]`` is present.
    Pass this class explicitly, or set ``allow_offline_fallback=True``, for
    a TF-IDF-only semantic tree.
    """

    vocabulary_: list[str] = field(default_factory=list, init=False, repr=False)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values, vocabulary = tfidf_matrix(texts)
        self.vocabulary_ = vocabulary
        return values

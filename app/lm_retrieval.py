"""Query likelihood retrieval: rank passages by how likely their language model is to produce the question.

Each passage gets a unigram language model. A word the passage never uses would get probability
zero, so the model is smoothed with the whole collection. This is the same zero-count problem as in
n-gram language modeling, and the same remedies:

  laplace         add one to every count
  jelinek_mercer  mix the passage model with the collection model by a fixed weight
  dirichlet       like Jelinek-Mercer, but short passages lean on the collection more
"""
from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from app.text_utils import content_stems

SMOOTHING = ("dirichlet", "jelinek_mercer", "laplace")


def _analyzer(text: str) -> list[str]:
    return content_stems(text, extra_stopwords=frozenset())


class QueryLikelihoodIndex:
    def __init__(self, texts: list[str], smoothing: str = "dirichlet", mu: float = 120.0, lam: float = 0.3) -> None:
        if smoothing not in SMOOTHING:
            raise ValueError(f"smoothing must be one of {SMOOTHING}")
        self.smoothing, self.mu, self.lam = smoothing, mu, lam
        self.vectorizer = CountVectorizer(analyzer=_analyzer)
        counts = self.vectorizer.fit_transform(texts).tocsc().astype(float)
        self.counts = counts
        self.lengths = np.asarray(counts.sum(axis=1)).ravel()
        totals = np.asarray(counts.sum(axis=0)).ravel()
        self.collection_p = totals / totals.sum()
        self.vocab_size = counts.shape[1]

    def scores(self, terms: dict[str, float]) -> np.ndarray:
        """Weighted log-likelihood of the question under each passage. Higher is better."""
        result = np.zeros(self.counts.shape[0])
        lengths = np.maximum(self.lengths, 1.0)
        for term, weight in terms.items():
            column = self.vectorizer.vocabulary_.get(term)
            if column is None:
                continue
            c = self.counts[:, column].toarray().ravel()
            p = self.collection_p[column]
            if self.smoothing == "dirichlet":
                probability = (c + self.mu * p) / (lengths + self.mu)
            elif self.smoothing == "jelinek_mercer":
                probability = (1 - self.lam) * c / lengths + self.lam * p
            else:
                probability = (c + 1.0) / (lengths + self.vocab_size)
            result += weight * np.log(probability)
        return result

    def matches(self, terms: dict[str, float]) -> np.ndarray:
        """True for passages that contain at least one question word."""
        hit = np.zeros(self.counts.shape[0], dtype=bool)
        for term in terms:
            column = self.vectorizer.vocabulary_.get(term)
            if column is not None:
                hit |= self.counts[:, column].toarray().ravel() > 0
        return hit

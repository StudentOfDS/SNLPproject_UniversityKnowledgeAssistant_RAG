"""Naive Bayes text classifier written from scratch. The pipeline uses it to guess which document a question is about."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from app.text_utils import content_stems


class NaiveBayes:
    """Multinomial Naive Bayes with add-alpha smoothing, computed in log space."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, documents: list[list[str]], labels: list[str]) -> "NaiveBayes":
        self.classes = sorted(set(labels))
        self.log_prior = {c: math.log(labels.count(c) / len(labels)) for c in self.classes}
        self.word_counts: dict[str, Counter] = defaultdict(Counter)
        for tokens, label in zip(documents, labels):
            self.word_counts[label].update(tokens)
        self.vocab = {w for counter in self.word_counts.values() for w in counter}
        self.totals = {c: sum(self.word_counts[c].values()) for c in self.classes}
        return self

    def log_scores(self, tokens: list[str]) -> dict[str, float]:
        tokens = [t for t in tokens if t in self.vocab]
        v = len(self.vocab)
        scores = {}
        for c in self.classes:
            denominator = self.totals[c] + self.alpha * v
            scores[c] = self.log_prior[c] + sum(
                math.log((self.word_counts[c][t] + self.alpha) / denominator) for t in tokens
            )
        return scores

    def predict_proba(self, tokens: list[str]) -> dict[str, float]:
        scores = self.log_scores(tokens)
        top = max(scores.values())
        exp = {c: math.exp(s - top) for c, s in scores.items()}
        z = sum(exp.values())
        return {c: v / z for c, v in exp.items()}

    def predict(self, tokens: list[str]) -> str:
        scores = self.log_scores(tokens)
        return max(scores, key=scores.get)


# question routing: which document does a question belong to? -----------------


class QuestionRouter:
    def fit(self, chunks: list) -> "QuestionRouter":
        labels = [c.document_name for c in chunks]
        if len(set(labels)) < 2:
            raise ValueError("Routing needs at least two indexed documents.")
        self.model = NaiveBayes(alpha=0.3).fit([content_stems(f"{c.section} {c.text}") for c in chunks], labels)
        # equal priors: a long document should not win just because it has more passages
        self.model.log_prior = {c: -math.log(len(self.model.classes)) for c in self.model.classes}
        return self

    def route(self, question: str, top: int = 3) -> list[dict]:
        probabilities = self.model.predict_proba(content_stems(question))
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:top]
        return [{"document": name, "probability": round(p, 4)} for name, p in ranked]

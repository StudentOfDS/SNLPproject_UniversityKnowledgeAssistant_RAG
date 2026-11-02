"""Decides whether the retrieved passages really contain what the question is about.

Rank-based scores (like the fused hybrid score) can't do this: the best chunk is always
ranked first, even when nothing in the documents is relevant. So the decision uses
absolute signals instead: how many question terms exist in the documents, how many are
in the best chunk, and the raw semantic similarity when an embedding model is loaded.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from app.settings import Settings
from app.text_utils import content_stems

NOT_FOUND_MESSAGE = (
    "Sorry, I couldn't find this in the documents you've provided. "
    "Try rephrasing the question, or add the document that covers it."
)


@dataclass
class Evidence:
    sufficient: bool
    confidence: str  # HIGH | MEDIUM | LOW
    strength: float  # 0..1, shown to the user as the retrieval score
    reason: str
    known_term_ratio: float = 0.0
    term_coverage: float = 0.0
    semantic_similarity: float | None = None
    tfidf_similarity: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def assess_evidence(question: str, candidates: list, vocabulary: set[str], settings: Settings,
                    llm_available: bool = False) -> Evidence:
    """`candidates` are RetrievedChunk objects, best first."""
    terms = sorted(set(content_stems(question)))
    if not terms:
        return Evidence(False, "LOW", 0.0, "The question has no searchable words.")
    if not candidates:
        return Evidence(False, "LOW", 0.0, "Nothing was retrieved.")

    known = [t for t in terms if t in vocabulary]
    known_ratio = len(known) / len(terms)

    best_coverage = 0.0
    for candidate in candidates:
        chunk = candidate.chunk
        chunk_terms = set(content_stems(f"{chunk.document_name} {chunk.section} {chunk.text}", frozenset()))
        best_coverage = max(best_coverage, sum(t in chunk_terms for t in terms) / len(terms))

    semantic_values = [c.semantic_score for c in candidates if c.semantic_score is not None]
    semantic_top = max(semantic_values) if semantic_values else None
    tfidf_top = max(c.tfidf_score for c in candidates)

    if semantic_top is not None:
        semantic_part = _clamp((semantic_top - settings.semantic_floor) / (0.7 - settings.semantic_floor))
    else:
        semantic_part = _clamp(tfidf_top / 0.5)
    strength = round(0.5 * best_coverage + 0.5 * semantic_part, 4)

    def result(sufficient: bool, reason: str) -> Evidence:
        if not sufficient:
            label = "LOW"
        else:
            label = "HIGH" if strength >= 0.75 else "MEDIUM" if strength >= 0.5 else "LOW"
        return Evidence(sufficient, label, strength, reason, round(known_ratio, 3),
                        round(best_coverage, 3), None if semantic_top is None else round(semantic_top, 3),
                        round(tfidf_top, 3))

    if known_ratio < settings.min_known_term_ratio:
        missing = ", ".join(t for t in terms if t not in vocabulary)
        return result(False, f"Most of the question's terms never appear in the documents ({missing}).")

    strong_semantic = semantic_top is not None and semantic_top >= settings.semantic_strong
    if best_coverage < settings.min_term_coverage and not strong_semantic:
        return result(False, f"The best passage only covers {best_coverage:.0%} of the question's terms.")

    if semantic_top is not None and semantic_top < settings.semantic_floor:
        return result(False, f"The closest passage is not similar enough ({semantic_top:.2f}).")

    minimum = settings.min_strength_with_llm if llm_available else settings.min_strength_without_llm
    if strength < minimum:
        return result(False, f"The evidence is too weak ({strength:.2f} < {minimum:.2f}).")

    return result(True, "The retrieved passages cover the question.")

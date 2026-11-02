"""Extractive summaries with TextRank. Used for per-document summaries and for "tell me about" answers."""
from __future__ import annotations

import re

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.snlp.textproc import split_sentences
from app.text_utils import tfidf_analyzer


def _candidate_sentences(text: str) -> list[str]:
    seen, sentences = set(), []
    for sentence in split_sentences(re.sub(r"\s*\n\s*", " ", text)):
        words = sentence.split()
        letters = sum(c.isalpha() for c in sentence)
        key = re.sub(r"\W+", "", sentence.lower())
        if 6 <= len(words) <= 60 and letters / max(1, len(sentence)) > 0.6 and key not in seen:
            seen.add(key)
            sentences.append(sentence)
    return sentences


def summarize(text: str, sentences: int = 5) -> list[dict]:
    """Rank sentences with PageRank over a TF-IDF similarity graph, return the best in reading order."""
    candidates = _candidate_sentences(text)
    if not candidates:
        return []
    if len(candidates) <= sentences:
        return [{"position": i, "score": 1.0, "text": s} for i, s in enumerate(candidates)]

    vectors = TfidfVectorizer(analyzer=tfidf_analyzer, sublinear_tf=True).fit_transform(candidates)
    similarity = cosine_similarity(vectors)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(candidates)))
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if similarity[i, j] > 0.05:
                graph.add_edge(i, j, weight=float(similarity[i, j]))
    scores = nx.pagerank(graph, weight="weight")

    # skip sentences that say almost the same thing as one already chosen
    best: list[int] = []
    for i in sorted(scores, key=scores.get, reverse=True):
        if all(similarity[i, j] < 0.5 for j in best):
            best.append(i)
        if len(best) == sentences:
            break
    return [{"position": i, "score": round(scores[i], 5), "text": candidates[i]} for i in sorted(best)]

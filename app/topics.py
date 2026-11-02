"""Finds the topics in the indexed documents and uses them in two places.

- The Documents tab shows each document's keywords, a short summary and the topics it covers.
- When a question is not answered, the closest topics are suggested, so the person knows what the files do cover.

Topics come from k-means on LSA vectors of the passages (clustering and distributional semantics:
words that occur in the same passages end up in the same SVD dimensions).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from app.snlp import summarize
from app.text_utils import stem, tfidf_analyzer, tokenize


@dataclass
class TopicModel:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    centroids: np.ndarray  # one row per topic, in LSA space
    topics: list[dict]
    silhouette: float

    def closest(self, question: str, count: int = 3, minimum: float = 0.35) -> list[dict]:
        vector = self.vectorizer.transform([question])
        if vector.nnz == 0:
            return []
        reduced = normalize(self.svd.transform(vector))[0]
        similarity = self.centroids @ reduced
        order = np.argsort(-similarity)[:count]
        return [self.topics[i] for i in order if similarity[i] >= minimum]


def build_topics(chunks: list, k: int = 8, seed: int = 0) -> TopicModel | None:
    texts = [f"{c.section} {c.text}" for c in chunks]
    if len(texts) < 6:
        return None
    k = max(2, min(k, len(texts) // 3))

    vectorizer = TfidfVectorizer(analyzer=tfidf_analyzer, min_df=2, max_df=0.8, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(texts)
    if tfidf.shape[1] < 3:
        return None
    svd = TruncatedSVD(n_components=min(60, tfidf.shape[1] - 1), random_state=seed)
    reduced = normalize(svd.fit_transform(tfidf))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(reduced)

    terms = np.array(vectorizer.get_feature_names_out())
    unigram = np.array([" " not in t and not t.isdigit() for t in terms])
    surface: dict[str, Counter] = defaultdict(Counter)  # show real words instead of stems
    for text in texts:
        for token in tokenize(text):
            surface[stem(token)][token] += 1

    topics, centroids = [], []
    for cluster in range(k):
        members = np.where(labels == cluster)[0]
        centroid = np.asarray(tfidf[members].mean(axis=0)).ravel()
        centroid[~unigram] = 0
        words = [surface[terms[i]].most_common(1)[0][0] if surface[terms[i]] else terms[i]
                 for i in np.argsort(-centroid)[:6] if centroid[i] > 0]
        topics.append({
            "label": ", ".join(words[:4]),
            "words": words,
            "passages": int(len(members)),
            "documents": dict(Counter(chunks[i].document_name for i in members).most_common(3)),
        })
        mean = reduced[members].mean(axis=0)
        centroids.append(mean / (np.linalg.norm(mean) or 1.0))

    sample = min(500, len(texts))
    score = float(silhouette_score(reduced, labels, sample_size=sample, random_state=seed))
    order = sorted(range(k), key=lambda i: -topics[i]["passages"])
    return TopicModel(vectorizer, svd, np.array([centroids[i] for i in order]), [topics[i] for i in order], round(score, 4))


def document_profile(chunks: list, topic_model: TopicModel | None, sentences: int = 3, keywords: int = 10) -> dict:
    """Keywords (highest TF-IDF), a TextRank summary and chunk count for one document."""
    ordered = sorted(chunks, key=lambda c: (c.page, c.chunk_index))
    text = "\n".join(c.text for c in ordered)

    vectorizer = TfidfVectorizer(analyzer=tfidf_analyzer, sublinear_tf=True)
    matrix = vectorizer.fit_transform([c.text for c in ordered])
    terms = np.array(vectorizer.get_feature_names_out())
    weights = np.asarray(matrix.sum(axis=0)).ravel()
    surface: dict[str, Counter] = defaultdict(Counter)
    for token in tokenize(text):
        surface[stem(token)][token] += 1
    top = []
    for i in np.argsort(-weights):
        term = terms[i]
        if " " in term or term.isdigit() or not surface.get(term):
            continue
        top.append(surface[term].most_common(1)[0][0])
        if len(top) == keywords:
            break

    return {
        "keywords": top,
        "summary": [s["text"] for s in summarize.summarize(text, sentences)],
    }

"""Sentence splitting and basic statistics of the text collection."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from app.snlp import data
from app.text_utils import STEMMERS, STOPWORDS, tokenize

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    """Punkt if its model is available, otherwise a regex split."""
    text = text.strip()
    if not text:
        return []
    try:
        data.ensure("tokenizers/punkt_tab", "punkt_tab")
        from nltk.tokenize import sent_tokenize
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def first_look(texts: list[str], top_n: int = 15) -> dict:
    """Basic statistics for a text collection: size, vocabulary, Zipf curve, common words."""
    joined = "\n".join(texts)
    sentences = [s for text in texts for s in split_sentences(text)]
    tokens = tokenize(joined)
    if not tokens:
        return {"tokens": 0}

    counts = Counter(tokens)
    content = Counter({w: c for w, c in counts.items() if w not in STOPWORDS and not w.isdigit() and len(w) > 2})
    bigrams = Counter(
        (a, b) for a, b in zip(tokens, tokens[1:])
        if a not in STOPWORDS and b not in STOPWORDS and not a.isdigit() and not b.isdigit()
    )

    frequencies = np.array(sorted(counts.values(), reverse=True)[:1000], dtype=float)
    ranks = np.arange(1, len(frequencies) + 1)
    slope = float(np.polyfit(np.log(ranks), np.log(frequencies), 1)[0]) if len(frequencies) > 2 else 0.0

    stems = {STEMMERS["snowball"](w) for w in counts}
    return {
        "characters": len(joined),
        "sentences": len(sentences),
        "tokens": len(tokens),
        "types": len(counts),
        "type_token_ratio": round(len(counts) / len(tokens), 4),
        "hapax_legomena": sum(1 for c in counts.values() if c == 1),
        "avg_sentence_length": round(len(tokens) / max(1, len(sentences)), 1),
        "types_after_stemming": len(stems),
        "vocabulary_reduction_by_stemming": round(1 - len(stems) / len(counts), 4),
        "zipf_slope": round(slope, 3),
        "top_words": content.most_common(top_n),
        "top_bigrams": [(" ".join(b), c) for b, c in bigrams.most_common(top_n)],
        "stopword_share": round(sum(c for w, c in counts.items() if w in STOPWORDS) / len(tokens), 4),
    }

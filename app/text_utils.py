"""Tokenising, stopwords and stemming shared by retrieval, the answer check and sentence selection."""
from __future__ import annotations

import re
from functools import lru_cache

from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer

STOPWORDS = frozenset("""
a about above after again against all am an and any are aren as at be because been before being below
between both but by can cannot could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most my
myself no nor not of off on once only or other our ours ourselves out over own same she should so some
such than that the their theirs them themselves then there these they this those through to too under
until up very was we were what when where which while who whom why will with would you your yours
yourself yourselves shall may might must also etc via per
""".split())

# Words that show up in questions but say nothing about the topic.
QUESTION_FILLERS = frozenset("""
tell explain describe give list state mention please kindly show say know want need let like many much
whats hows
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STEMMERS = {
    "snowball": SnowballStemmer("english").stem,
    "porter": PorterStemmer().stem,
    "lancaster": LancasterStemmer().stem,
    "none": lambda word: word,
}
_stemmer_name = "snowball"


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def set_stemmer(name: str) -> None:
    """Choose the stemmer used for searching. Indexes are built at search time, so this takes effect at once."""
    global _stemmer_name
    if name not in STEMMERS:
        raise ValueError(f"Unknown stemmer '{name}'. Use one of: {', '.join(STEMMERS)}.")
    if name != _stemmer_name:
        _stemmer_name = name
        _stem_cached.cache_clear()


@lru_cache(maxsize=50_000)
def _stem_cached(word: str) -> str:
    return STEMMERS[_stemmer_name](word)


def stem(word: str) -> str:
    return word if word.isdigit() else _stem_cached(word)


def content_stems(text: str, extra_stopwords: frozenset[str] = QUESTION_FILLERS) -> list[str]:
    """Stemmed words with stopwords removed, in order, duplicates kept."""
    return [
        stem(token)
        for token in tokenize(text)
        if token not in STOPWORDS and token not in extra_stopwords and (len(token) > 1 or token.isdigit())
    ]


def tfidf_analyzer(text: str) -> list[str]:
    """Analyzer for TfidfVectorizer: stems plus adjacent stem pairs."""
    stems = content_stems(text, extra_stopwords=frozenset())
    return stems + [f"{a} {b}" for a, b in zip(stems, stems[1:])]


# Kept for older callers: a set of stemmed keywords.
def keywords(text: str) -> set[str]:
    return set(content_stems(text))

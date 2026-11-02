"""Picks the sentences that answer the question out of the retrieved passages.

Used in two places: as the answer itself when no LLM is configured, and as highlights shown
inside each cited passage. Sentences are scored by how many of the question's words they contain
and whether they look like the kind of thing the question asks for (a date for "when", a number
for "how many"). "Tell me about" questions get the most central sentences (TextRank) instead.
For yes/no questions a Yes or No is added when the sentence is a close match and its wording is clear.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import Citation
from app.snlp import summarize
from app.text_utils import content_stems
from app.understanding import QueryAnalysis

_DATE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b|\b\d{1,2}[-/]\w{3,9}[-/]\d{2,4}\b|\b20\d\d\b|"
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
_NUMBER = re.compile(r"\d")
# rows of a numbered date list ("21. 25-Dec-26 Friday Christmas") that were joined into one line
_LIST_ITEMS = re.compile(r"\s(?=\d{1,2}\.\s+\d{1,2}[-/][A-Za-z]{3})")
_REASON = re.compile(r"\b(because|since|due to|so that|in order to|as a result)\b", re.I)
_MIN_WORDS = 3
# A Yes or No is only given when the sentence it is based on covers most of the question.
MIN_VERDICT_COVERAGE = 0.75

# Wording that says a thing is not allowed / not the case, and wording that says it is.
_NEGATIVE = re.compile(
    r"\b(not|no|never|cannot|prohibit\w*|forbid\w*|forbidden|ban(ned)?|disallow\w*|ineligible|barred|refus\w*|"
    r"denied|rusticat\w*|penali[sz]\w*|punish\w*)\b|n't\b", re.I)
_POSITIVE = re.compile(
    r"\b(allowed|permitted|may|eligible|entitled|compulsory|mandatory|required|must|shall|is a|are a|will be)\b", re.I)


@dataclass
class Selected:
    citation_index: int  # position in the citations list
    sentence: str
    score: float
    position: int  # order inside the passage
    coverage: float = 0.0  # share of the question's words found in the sentence


def passage_sentences(text: str) -> list[str]:
    """Sentences of a passage. Table rows (cells joined by ' | ') stay whole."""
    pieces: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            from app.snlp.textproc import split_sentences
            pieces.extend(split_sentences(" ".join(paragraph)))
            paragraph.clear()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if " | " in line:
            flush()
            pieces.append(line)
        elif _LIST_ITEMS.search(line):
            flush()
            pieces.extend(p.strip() for p in _LIST_ITEMS.split(line) if p.strip())
        else:
            paragraph.append(line)
    flush()
    return [p for p in pieces if len(p.split()) >= _MIN_WORDS]


def score_sentence(sentence: str, analysis: QueryAnalysis, rank: int) -> tuple[float, float]:
    """Returns (score, coverage)."""
    terms = analysis.terms
    if not terms:
        return 0.0, 0.0
    present = set(content_stems(sentence, frozenset()))
    total = sum(terms.values())
    coverage = sum(w for t, w in terms.items() if t in present) / total

    score = coverage
    kind = analysis.question_type
    if kind == "when" and _DATE.search(sentence):
        score += 0.2
    elif kind == "quantity" and _NUMBER.search(sentence):
        score += 0.2
    elif kind == "why" and _REASON.search(sentence):
        score += 0.2
    score += 0.05 / rank  # passages ranked higher by retrieval get a small edge
    return score, coverage


def select_sentences(analysis: QueryAnalysis, citations: list[Citation], limit: int = 3,
                     neighbours: bool = True) -> list[Selected]:
    candidates: list[Selected] = []
    for index, citation in enumerate(citations):
        for position, sentence in enumerate(passage_sentences(citation.text)):
            score, coverage = score_sentence(sentence, analysis, index + 1)
            candidates.append(Selected(index, sentence, score, position, coverage))
    if not candidates:
        return []

    candidates.sort(key=lambda c: c.score, reverse=True)
    best = candidates[0].score
    chosen: list[Selected] = []
    for candidate in candidates:
        if len(chosen) == limit or candidate.score < 0.6 * best:
            break
        words = set(content_stems(candidate.sentence, frozenset()))
        if any(len(words & set(content_stems(c.sentence, frozenset()))) / max(1, len(words)) > 0.7 for c in chosen):
            continue  # nearly the same sentence is already chosen
        chosen.append(candidate)
    if neighbours:
        chosen = _add_neighbours(chosen, citations, analysis)
    return chosen


def _add_neighbours(chosen: list[Selected], citations: list[Citation], analysis: QueryAnalysis) -> list[Selected]:
    """The sentence before and after the best one often holds the number or condition the answer needs."""
    best = max(chosen, key=lambda c: c.score)
    sentences = passage_sentences(citations[best.citation_index].text)
    have = {(c.citation_index, c.position) for c in chosen}
    wanted = set(analysis.terms)
    for position in (best.position - 1, best.position + 1):
        if 0 <= position < len(sentences) and (best.citation_index, position) not in have:
            words = set(content_stems(sentences[position], frozenset()))
            if wanted & words:
                score, coverage = score_sentence(sentences[position], analysis, best.citation_index + 1)
                chosen.append(Selected(best.citation_index, sentences[position], score, position, coverage))
    return chosen


def overview(citations: list[Citation], sentences: int = 4) -> list[Selected]:
    """For "tell me about ..." questions: the most central sentences of the retrieved passages (TextRank)."""
    text = "\n".join(c.text for c in citations)
    picked = summarize.summarize(text, sentences)
    selected = []
    for item in picked:
        for index, citation in enumerate(citations):
            if item["text"][:60] in re.sub(r"\s+", " ", citation.text):
                selected.append(Selected(index, item["text"], item["score"], item["position"]))
                break
    return selected


def polarity(analysis: QueryAnalysis, chosen: list[Selected]) -> str:
    """"Yes" or "No" for a yes/no question, judged from the best sentence. Empty when the sentence gives no clear cue."""
    if analysis.question_type != "yes_no" or not chosen or chosen[0].coverage < MIN_VERDICT_COVERAGE:
        return ""
    sentence = chosen[0].sentence
    negative = bool(_NEGATIVE.search(sentence))
    positive = bool(_POSITIVE.search(sentence))
    if negative:
        return "No"
    if positive or " | " in sentence or re.search(r"\d", sentence):
        return "Yes"
    return ""


def choose_sentences(analysis: QueryAnalysis, citations: list[Citation]) -> list[Selected]:
    chosen = overview(citations) if analysis.question_type == "overview" else []
    return chosen or select_sentences(analysis, citations)


def compose_answer(analysis: QueryAnalysis, citations: list[Citation]) -> tuple[str, list[Selected]]:
    """Extractive answer: the chosen sentences, each followed by the number of the passage it came from."""
    chosen = choose_sentences(analysis, citations)
    if not chosen:
        return "", []
    # keep passages in retrieval order, sentences in reading order
    chosen.sort(key=lambda s: (s.citation_index, s.position))
    verdict = polarity(analysis, chosen)
    lines = [f"{s.sentence} [{s.citation_index + 1}]" for s in chosen]
    first = citations[chosen[0].citation_index]
    head = f"{verdict}. " if verdict else ""
    return f"{head}From {first.document_name}, page {first.page}:\n\n" + "\n\n".join(lines), chosen

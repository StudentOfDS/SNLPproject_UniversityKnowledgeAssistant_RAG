"""Reads a question before it is searched: which words matter, and what kind of answer is wanted."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from app.text_utils import content_stems

_DATE_WORDS = re.compile(r"\b(when|what date|which date|what day|which day|what time|deadline|last date)\b", re.I)
_QUANTITY = re.compile(
    r"\b(how many|how much|how long|how often|what percentage|"
    r"what is the (minimum|maximum|number|duration|fee|amount|limit))\b", re.I)
_OVERVIEW = re.compile(
    r"\b(tell me about|summari[sz]e|overview|in brief|what are the (rules|guidelines|norms|policies|policy))\b"
    r"|\bexplain\b|\bdescribe\b", re.I)
_YES_NO = re.compile(r"^(can|may|could|is|are|am|do|does|did|will|would|should|must|shall|has|have|was|were)\b", re.I)


@dataclass
class QueryAnalysis:
    question: str
    question_type: str = "what"  # when, quantity, who, where, why, how, yes_no, overview or what
    terms: dict[str, float] = field(default_factory=dict)  # stemmed content words and their weights
    pairs: list[str] = field(default_factory=list)  # adjacent stem pairs, as in the index

    def as_dict(self) -> dict:
        out = asdict(self)
        out.pop("pairs")
        out["terms"] = sorted(self.terms)
        return out


def classify_question(question: str) -> str:
    q = question.strip()
    if _OVERVIEW.search(q):
        return "overview"
    if _DATE_WORDS.search(q):
        return "when"
    if _QUANTITY.search(q):
        return "quantity"
    first = q.lower().split()[0] if q.split() else ""
    if first in {"who", "whom", "whose"}:
        return "who"
    if first == "where":
        return "where"
    if first == "why":
        return "why"
    if first == "how":
        return "how"
    if _YES_NO.match(q):
        return "yes_no"
    return "what"


def analyze(question: str) -> QueryAnalysis:
    stems = content_stems(question)
    return QueryAnalysis(
        question=question,
        question_type=classify_question(question),
        terms={t: 1.0 for t in stems},
        pairs=[f"{a} {b}" for a, b in zip(stems, stems[1:])],
    )

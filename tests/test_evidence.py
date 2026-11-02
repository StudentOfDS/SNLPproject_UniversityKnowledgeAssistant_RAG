from types import SimpleNamespace

from app.evidence import assess_evidence
from app.settings import Settings
from app.text_utils import content_stems

SETTINGS = Settings(google_api_key="", semantic_search=False)


def candidate(text: str, tfidf: float = 0.3, semantic: float | None = None):
    chunk = SimpleNamespace(document_name="rules.pdf", section="Attendance", text=text)
    return SimpleNamespace(chunk=chunk, tfidf_score=tfidf, semantic_score=semantic)


def vocab(*texts: str) -> set[str]:
    return {s for t in texts for s in content_stems(t, frozenset())}


TEXT = "Students need minimum 75 percent attendance to appear for the examination."


def test_question_covered_by_the_best_chunk_is_accepted():
    result = assess_evidence("What is the minimum attendance for the examination?", [candidate(TEXT)], vocab(TEXT), SETTINGS)
    assert result.sufficient
    assert result.confidence in {"MEDIUM", "HIGH"}


def test_question_about_something_else_is_refused():
    result = assess_evidence("Who is the prime minister of India?", [candidate(TEXT, tfidf=0.05)], vocab(TEXT), SETTINGS)
    assert not result.sufficient
    assert result.confidence == "LOW"
    assert "never appear" in result.reason


def test_known_words_in_the_wrong_context_are_refused():
    other = "The library is open on weekdays. Students may borrow five books."
    question = "What is the attendance policy for weekdays books?"
    result = assess_evidence(question, [candidate(other, tfidf=0.08)], vocab(TEXT, other), SETTINGS)
    assert not result.sufficient


def test_no_searchable_words():
    assert not assess_evidence("what is the", [candidate(TEXT)], vocab(TEXT), SETTINGS).sufficient


def test_nothing_retrieved():
    assert not assess_evidence("attendance rule", [], vocab(TEXT), SETTINGS).sufficient


def test_strong_semantic_match_rescues_a_paraphrase():
    result = assess_evidence("How often must pupils be present in class?", [candidate(TEXT, tfidf=0.05, semantic=0.7)],
                             vocab(TEXT, "pupils present class often"), SETTINGS)
    assert result.sufficient


def test_weak_semantic_match_is_refused():
    result = assess_evidence("minimum attendance", [candidate(TEXT, tfidf=0.3, semantic=0.1)], vocab(TEXT), SETTINGS)
    assert not result.sufficient


def test_without_llm_the_bar_is_higher():
    weak = candidate("attendance is recorded daily by the faculty", tfidf=0.12)
    question = "What is the minimum attendance?"
    kwargs = dict(vocabulary=vocab(TEXT, weak.chunk.text), settings=SETTINGS)
    with_llm = assess_evidence(question, [weak], llm_available=True, **kwargs)
    without_llm = assess_evidence(question, [weak], llm_available=False, **kwargs)
    assert with_llm.strength == without_llm.strength
    assert not without_llm.sufficient or with_llm.sufficient


def test_signals_are_reported():
    result = assess_evidence("minimum attendance", [candidate(TEXT)], vocab(TEXT), SETTINGS)
    assert set(result.as_dict()) >= {"sufficient", "confidence", "strength", "reason", "known_term_ratio"}

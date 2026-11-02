import pytest

from app.corpus import CorpusInsights
from app.topics import build_topics, document_profile
from tests.conftest import make_pdf

PAGES = [
    "FEES\nTuition fees are paid every semester through the portal. A late fee applies after the due date. "
    "Refunds are given only when a student withdraws before the programme starts. Scholarship fee waivers are decided by the committee.",
    "LIBRARY\nStudents can borrow four books for fourteen days. Overdue books carry a fine for each day. "
    "The reading room is open on weekdays and the library card must be shown at the desk.",
    "HOSTEL\nHostel rooms are allotted each year by the warden. Visitors are not allowed after ten at night. "
    "Rooms must be vacated within two days after the final examination of the year.",
    "EXAMINATIONS\nThe hall ticket must be carried to every examination. Students with low attendance cannot sit the exam. "
    "Re-examination is held after the results of the regular term end examination are declared.",
    "SPORTS\nThe sports complex has courts for badminton and tennis. Team trials are held at the start of the year. "
    "Players represent the university in inter college tournaments and receive attendance relief.",
    "PLACEMENTS\nThe placement cell invites companies from August. Students must register before the first drive. "
    "Offers are shared with the department and an accepted offer cannot be withdrawn later.",
]


@pytest.fixture
def indexed(kb):
    info = kb.process_and_index("Handbook.pdf", make_pdf(*PAGES))
    return kb, info["file_hash"]


def test_topic_model_labels_topics_and_finds_the_closest(indexed):
    kb, _ = indexed
    chunks = kb.all_chunks()
    model = build_topics(chunks, k=3)
    assert model and 2 <= len(model.topics) <= 3
    assert sum(t["passages"] for t in model.topics) == len(chunks)
    assert all(t["label"] for t in model.topics)
    assert isinstance(model.closest("library books fine"), list)
    assert model.closest("qqqq zzzz") == []


def test_too_little_text_gives_no_topics(kb):
    kb.process_and_index("Tiny.pdf", make_pdf("A short notice about the notice board."))
    assert build_topics(kb.all_chunks()) is None


def test_document_profile_has_keywords_and_summary(indexed):
    kb, _ = indexed
    profile = document_profile(kb.all_chunks(), build_topics(kb.all_chunks(), k=3))
    assert profile["keywords"]
    assert profile["summary"]


def test_corpus_insights_refreshes_when_documents_change(indexed):
    kb, file_hash = indexed
    insights = CorpusInsights(kb)
    first = insights.profile()
    assert first["collection"]["sentences"] > 0
    assert [d["document_name"] for d in first["documents"]] == ["Handbook.pdf"]
    assert insights.profile() is first  # cached
    kb.remove_index(file_hash)
    kb.process_and_index("Other.pdf", make_pdf(*PAGES[:3]))
    assert [d["document_name"] for d in insights.profile()["documents"]] == ["Other.pdf"]

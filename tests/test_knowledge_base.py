from app.knowledge_base import KnowledgeBase
from tests.conftest import ATTENDANCE, make_pdf


def test_file_hash_depends_on_content():
    assert KnowledgeBase.file_hash(b"a") == KnowledgeBase.file_hash(b"a")
    assert KnowledgeBase.file_hash(b"a") != KnowledgeBase.file_hash(b"b")


def test_safe_filename_drops_folders():
    assert KnowledgeBase.safe_filename("../../Rules.pdf") == "Rules.pdf"


def test_document_type_from_name():
    assert KnowledgeBase.infer_document_type("Academic Calendar - All Programs.pdf", b"") == "academic_calendar"
    assert KnowledgeBase.infer_document_type("Holiday list 2026.pdf", b"") == "academic_calendar"
    assert KnowledgeBase.infer_document_type("SRB_MPSTME_2024-25.pdf", b"") == "student_handbook"
    assert KnowledgeBase.infer_document_type("something.pdf", b"") == "unclassified"


def test_headings():
    assert KnowledgeBase.looks_like_heading("ATTENDANCE RULES")
    assert KnowledgeBase.looks_like_heading("3.2 Examination Eligibility")
    assert not KnowledgeBase.looks_like_heading("Students must keep 75 percent attendance.")
    assert not KnowledgeBase.looks_like_heading("26 January")


def test_sections_are_split_on_headings():
    text = "ATTENDANCE RULES\nStudents must attend.\n\nEXAM RULES\nBring the ID card."
    names = [name for name, _ in KnowledgeBase.split_into_sections(text)]
    assert "ATTENDANCE RULES" in names and "EXAM RULES" in names


def test_chunks_respect_the_size_and_overlap(kb):
    kb.settings.chunk_size, kb.settings.chunk_overlap = 20, 5
    text = "\n".join(f"line {i} " + "word " * 4 for i in range(30))
    chunks = kb.chunk_text(text)
    assert len(chunks) > 3
    assert all(len(c.split()) <= 30 for c in chunks)
    assert chunks[0].split()[-1] in chunks[1]  # overlap


def test_index_and_search(kb, policy_pdf):
    info = kb.process_and_index("Policy.pdf", policy_pdf)
    assert info["chunk_count"] >= 2
    assert info["page_count"] == 2
    hashes = [info["file_hash"]]
    found = kb.search("What attendance do students need?", hashes, mode="tfidf", top_k=3)
    assert found
    assert "attendance" in found[0].chunk.text.lower()
    assert found[0].chunk.page == 1


def test_same_file_is_not_indexed_twice(kb, policy_pdf):
    first = kb.process_and_index("Policy.pdf", policy_pdf)
    second = kb.process_and_index("Policy.pdf", policy_pdf)
    assert not first.get("reused")
    assert second["reused"]
    assert len(kb.list_documents()) == 1


def test_changed_chunk_size_triggers_reindex(kb, policy_pdf):
    kb.process_and_index("Policy.pdf", policy_pdf)
    kb.settings.chunk_size = 40
    assert not kb.process_and_index("Policy.pdf", policy_pdf).get("reused")


def test_search_modes_without_embeddings_still_work(kb, policy_pdf):
    h = [kb.process_and_index("Policy.pdf", policy_pdf)["file_hash"]]
    for mode in ("tfidf", "semantic", "hybrid"):
        assert kb.search("late fee", h, mode=mode, top_k=2)


def test_in_document_question_is_accepted(kb, policy_pdf):
    h = [kb.process_and_index("Policy.pdf", policy_pdf)["file_hash"]]
    found = kb.search("What attendance is required to appear for the examination?", h)
    assert kb.assess("What attendance is required to appear for the examination?", found, h).sufficient


def test_out_of_document_questions_are_refused(kb, policy_pdf):
    h = [kb.process_and_index("Policy.pdf", policy_pdf)["file_hash"]]
    for question in ["Who won the football world cup?", "What is the capital of France?",
                     "How do I bake bread?", "Explain quantum entanglement"]:
        found = kb.search(question, h)
        verdict = kb.assess(question, found, h)
        assert not verdict.sufficient, question
        assert verdict.confidence == "LOW"


def test_remove_index(kb, policy_pdf):
    h = kb.process_and_index("Policy.pdf", policy_pdf)["file_hash"]
    kb.remove_index(h)
    assert kb.list_documents() == []


def test_citation_carries_the_source(kb):
    h = kb.process_and_index("Rules.pdf", make_pdf(ATTENDANCE))["file_hash"]
    cite = kb.to_citation(kb.search("attendance", [h], top_k=1)[0])
    assert cite.document_name == "Rules.pdf"
    assert cite.page == 1
    assert 0 <= cite.score <= 1

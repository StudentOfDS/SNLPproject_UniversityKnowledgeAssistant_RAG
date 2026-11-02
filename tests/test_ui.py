from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

UI = str(Path(__file__).resolve().parent.parent / "app" / "ui.py")


def test_ui_loads_and_says_so_when_the_backend_is_down(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9")
    app = AppTest.from_file(UI, default_timeout=30).run()
    assert not app.exception
    assert any("backend" in e.value.lower() for e in app.error)
    assert [t.label for t in app.tabs][:2] == ["Ask", "Documents"]


def test_empty_question_gets_a_warning(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9")
    app = AppTest.from_file(UI, default_timeout=30).run()
    next(b for b in app.button if b.label == "Ask").click().run()
    assert not app.exception
    assert any("Type a question" in w.value for w in app.warning)


class FakeResponse:
    def __init__(self, body, status=200):
        self._body, self.status_code, self.text = body, status, str(body)

    def json(self):
        return self._body


def fake_backend(method, url, **kwargs):
    path = url.split("/api/v1")[1]
    doc = {"document_id": "1", "document_name": "Rules.pdf", "document_type": "student_handbook", "file_hash": "h",
           "size_bytes": 10, "indexed_at": "now", "chunk_count": 4, "page_count": 2, "ocr_pages": [2]}
    if path == "/health":
        return FakeResponse({"status": "ok", "indexed_documents": 1, "llm_configured": False,
                             "semantic_search": False, "ocr_engine": "rapidocr"})
    if path == "/documents":
        return FakeResponse([doc])
    if path == "/corpus":
        return FakeResponse({
            "collection": {"sentences": 10, "tokens": 200, "types": 80, "vocabulary_reduction_by_stemming": 0.2},
            "documents": [{"document_name": "Rules.pdf", "keywords": ["attendance"], "summary": ["Attendance is needed."]}],
            "topics": [{"label": "attendance, exam", "passages": 4, "documents": {"Rules.pdf": 4}}],
        })
    if path == "/ask":
        found = kwargs["data"]["question"].startswith("What")
        cite = {"chunk_id": "c", "document_id": "1", "document_name": "Rules.pdf", "document_type": "x", "section": "Attendance",
                "page": 1, "score": 0.7, "text": "Students need 75 percent attendance. The library opens at nine.",
                "highlights": ["Students need 75 percent attendance."]}
        return FakeResponse({
            "query": "q", "answer": "From Rules.pdf, page 1: ..." if found else "Sorry, I couldn't find this in the documents you've provided.",
            "retrieval_score": 0.8 if found else 0.1, "confidence": "HIGH" if found else "LOW",
            "evidence_sufficient": found, "reason": "ok" if found else "Most terms never appear.", "signals": {},
            "analysis": {"question_type": "what", "terms": ["attend"]},
            "likely_documents": [{"document": "Rules.pdf", "probability": 0.9}],
            "suggestions": [] if found else ["attendance, exam"],
            "citations": [cite] if found else [], "processed_documents": ["Rules.pdf"],
        })
    raise AssertionError(path)


def test_answer_and_refusal_render(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "request", fake_backend)
    app = AppTest.from_file(UI, default_timeout=30).run()
    assert not app.exception
    app.text_area[0].set_value("What attendance is needed?")
    next(b for b in app.button if b.label == "Ask").click().run()
    assert not app.exception
    assert any("Rules.pdf" in m.value for m in app.markdown)
    app.text_area[0].set_value("Who won the world cup")
    next(b for b in app.button if b.label == "Ask").click().run()
    assert not app.exception
    assert any("couldn't find" in w.value for w in app.warning)
    assert any("attendance, exam" in m.value for m in app.markdown)

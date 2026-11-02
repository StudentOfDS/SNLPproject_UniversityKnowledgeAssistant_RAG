import json

import pytest
from fastapi.testclient import TestClient

from app.api import app, knowledge_base
from app.evidence import NOT_FOUND_MESSAGE
from tests.conftest import ATTENDANCE, FEES, image_only_pdf, make_image, make_pdf, needs_ocr, png_bytes


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clean_index():
    for doc in knowledge_base.list_documents():
        knowledge_base.remove_index(doc["file_hash"])
    yield


def upload(client, name, data, content_type="application/pdf"):
    return client.post("/api/v1/index", files=[("files", (name, data, content_type))])


def ask(client, question, history=None):
    return client.post("/api/v1/ask", data={"question": question, "history": json.dumps(history or [])})


def test_health(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["semantic_search"] is False
    assert body["llm_configured"] is False


def test_ask_with_nothing_indexed(client):
    assert ask(client, "anything").status_code == 400


def test_upload_then_ask_gives_a_cited_answer(client):
    assert upload(client, "Policy.pdf", make_pdf(ATTENDANCE, FEES)).status_code == 200
    body = ask(client, "What attendance do I need to appear for the semester end examination?").json()
    assert body["evidence_sufficient"] is True
    assert body["confidence"] in {"MEDIUM", "HIGH"}
    assert body["citations"]
    assert body["citations"][0]["document_name"] == "Policy.pdf"
    assert "75 percent" in body["answer"]


def test_question_not_in_the_documents_is_refused_without_sources(client):
    upload(client, "Policy.pdf", make_pdf(ATTENDANCE, FEES))
    for question in ["Who is the president of the United States?", "How do I bake a cake?", "What is the speed of light?"]:
        body = ask(client, question).json()
        assert body["answer"] == NOT_FOUND_MESSAGE, question
        assert body["evidence_sufficient"] is False
        assert body["confidence"] == "LOW"
        assert body["citations"] == []


def test_short_follow_up_uses_the_previous_question(client):
    upload(client, "Policy.pdf", make_pdf(ATTENDANCE, FEES))
    history = [{"question": "What is the late fee rule?", "answer": "A late fee is charged."}]
    body = ask(client, "and refunds?", history).json()
    assert body["evidence_sufficient"] is True
    assert "refund" in body["answer"].lower()


def test_off_topic_long_question_does_not_borrow_from_history(client):
    upload(client, "Policy.pdf", make_pdf(ATTENDANCE, FEES))
    history = [{"question": "What is the late fee rule?", "answer": "A late fee is charged."}]
    body = ask(client, "Who is the prime minister of Canada right now?", history).json()
    assert body["evidence_sufficient"] is False


def test_validation(client):
    upload(client, "Policy.pdf", make_pdf(ATTENDANCE))
    assert ask(client, "   ").status_code == 400
    assert client.post("/api/v1/ask", data={"question": "x", "mode": "bogus"}).status_code == 400
    assert upload(client, "notes.txt", b"hello", "text/plain").status_code == 400
    assert upload(client, "fake.pdf", b"not a pdf").status_code == 400


def test_search_endpoint(client):
    upload(client, "Policy.pdf", make_pdf(ATTENDANCE, FEES))
    body = client.post("/api/v1/search", data={"question": "late fee", "mode": "tfidf", "top_k": 2}).json()
    assert body["results"]
    assert "fee" in body["results"][0]["text"].lower()


def test_documents_list_shows_page_counts(client):
    upload(client, "Policy.pdf", make_pdf(ATTENDANCE, FEES))
    docs = client.get("/api/v1/documents").json()
    assert docs[0]["document_name"] == "Policy.pdf"
    assert docs[0]["page_count"] == 2


@needs_ocr
def test_png_upload_is_read_and_answerable(client):
    image = make_image("The library fine is five rupees per day\nBooks can be kept for fourteen days")
    assert upload(client, "notice.png", png_bytes(image), "image/png").status_code == 200
    docs = client.get("/api/v1/documents").json()
    assert docs[0]["ocr_pages"] == [1]
    body = ask(client, "What is the library fine per day?").json()
    assert body["evidence_sufficient"] is True
    assert "five rupees" in body["answer"].lower()


@needs_ocr
def test_scanned_pdf_is_read_and_answerable(client):
    pdf = image_only_pdf(make_image("Hostel rooms must be vacated within two days after the final exam"))
    assert upload(client, "scan.pdf", pdf).status_code == 200
    body = ask(client, "When must hostel rooms be vacated?").json()
    assert body["evidence_sufficient"] is True
    assert "two days" in body["answer"].lower()


def test_reindex_picks_up_files_in_documents_folder(client):
    from app.api import settings
    settings.documents_path.mkdir(parents=True, exist_ok=True)
    (settings.documents_path / "Extra.pdf").write_bytes(make_pdf(FEES))
    assert client.post("/api/v1/reindex").json()["newly_indexed"] == 1


def test_answer_carries_analysis_routing_and_highlights(client):
    upload(client, "Policy.pdf", make_pdf(ATTENDANCE))
    upload(client, "Fees.pdf", make_pdf(FEES))
    body = ask(client, "What attendance do I need to appear for the semester end examination?").json()
    assert body["analysis"]["question_type"] in {"what", "quantity"}
    assert "attend" in body["analysis"]["terms"]
    assert body["likely_documents"][0]["document"] == "Policy.pdf"
    assert any(c["highlights"] for c in body["citations"])
    assert "[1]" in body["answer"]


def test_search_mode_lm_works(client):
    upload(client, "Policy.pdf", make_pdf(ATTENDANCE, FEES))
    body = client.post("/api/v1/search", data={"question": "late fee", "mode": "lm", "top_k": 2}).json()
    assert body["results"] and "fee" in body["results"][0]["text"].lower()


def test_corpus_endpoint(client):
    assert client.get("/api/v1/corpus").status_code == 400
    upload(client, "Policy.pdf", make_pdf(ATTENDANCE, FEES))
    body = client.get("/api/v1/corpus").json()
    assert body["collection"]["sentences"] > 0
    assert body["documents"][0]["document_name"] == "Policy.pdf"
    assert body["documents"][0]["keywords"]


def test_refusal_still_carries_the_routing_guess(client):
    upload(client, "Policy.pdf", make_pdf(ATTENDANCE))
    upload(client, "Fees.pdf", make_pdf(FEES))
    body = ask(client, "Who won the football world cup?").json()
    assert body["evidence_sufficient"] is False and body["citations"] == []
    assert isinstance(body["suggestions"], list)

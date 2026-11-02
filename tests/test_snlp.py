import pytest

from app.snlp import summarize, textproc
from app.snlp.classify import NaiveBayes, QuestionRouter
from app.text_utils import content_stems, set_stemmer, stem


def test_naive_bayes_learns_a_toy_split():
    docs = [["fee", "payment", "refund"], ["fee", "late", "payment"], ["exam", "hall", "ticket"], ["exam", "marks"]]
    labels = ["fees", "fees", "exams", "exams"]
    model = NaiveBayes().fit(docs, labels)
    assert model.predict(["fee", "refund"]) == "fees"
    assert model.predict(["exam", "ticket"]) == "exams"
    assert sum(model.predict_proba(["fee"]).values()) == pytest.approx(1.0)


def test_router_points_to_the_right_document():
    class Chunk:
        def __init__(self, name, text):
            self.document_name, self.section, self.text = name, "", text

    chunks = [Chunk("Fees.pdf", "tuition fee payment due date late fine refund"),
              Chunk("Fees.pdf", "fee instalments and scholarship refund rules"),
              Chunk("Library.pdf", "books borrowing library card return fine reading room"),
              Chunk("Library.pdf", "library timings and borrowing limit for books")]
    router = QuestionRouter().fit(chunks)
    assert router.route("How many books can I borrow?")[0]["document"] == "Library.pdf"
    assert router.route("When is the fee due?")[0]["document"] == "Fees.pdf"


def test_router_needs_two_documents():
    class Chunk:
        document_name, section, text = "Only.pdf", "", "text"

    with pytest.raises(ValueError):
        QuestionRouter().fit([Chunk(), Chunk()])


def test_sentence_split_handles_abbreviation_free_text():
    assert textproc.split_sentences("First one. Second one! Third?") == ["First one.", "Second one!", "Third?"]


def test_first_look_counts_words_and_stem_reduction():
    info = textproc.first_look(["Students attend classes. The student attends class."])
    assert info["sentences"] == 2
    assert info["types_after_stemming"] < info["types"]
    assert info["top_words"]


def test_textrank_returns_sentences_in_reading_order():
    text = " ".join(f"Sentence number {i} talks about the examination schedule and the hall ticket." for i in range(8))
    picked = summarize.summarize(text, 3)
    assert len(picked) == 3
    assert [p["position"] for p in picked] == sorted(p["position"] for p in picked)


def test_switching_stemmer_changes_stems_and_restores():
    try:
        set_stemmer("none")
        assert stem("examinations") == "examinations"
        assert content_stems("examinations") == ["examinations"]
        with pytest.raises(ValueError):
            set_stemmer("bogus")
    finally:
        set_stemmer("snowball")
    assert stem("examinations") == "examin"

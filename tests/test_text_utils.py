from app.text_utils import content_stems, keywords, tfidf_analyzer, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("The Exam is on 12th May!") == ["the", "exam", "is", "on", "12th", "may"]


def test_content_stems_drops_stopwords_and_question_words():
    assert content_stems("What is the minimum attendance required?") == ["minimum", "attend", "requir"]


def test_stemming_matches_word_forms():
    assert content_stems("examinations") == content_stems("examination")


def test_analyzer_adds_adjacent_pairs():
    terms = tfidf_analyzer("late fee payment")
    assert "late" in terms and "fee" in terms
    assert any("_" in t or " " in t for t in terms)


def test_keywords_is_a_set_of_stems():
    assert keywords("Attendance attendance rules") == {"attend", "rule"}

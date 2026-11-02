import pytest

from app import answering
from app.models import Citation
from app.understanding import analyze, classify_question


@pytest.mark.parametrize("question,expected", [
    ("When is the term end exam?", "when"),
    ("How many days of leave are allowed?", "quantity"),
    ("Who is the dean?", "who"),
    ("Why is a late fee charged?", "why"),
    ("Is ragging allowed on campus?", "yes_no"),
    ("Tell me about the grading system", "overview"),
    ("What is the grading policy?", "what"),
])
def test_question_types(question, expected):
    assert classify_question(question) == expected


def test_analysis_has_stems_and_pairs():
    a = analyze("What is the minimum attendance required?")
    assert set(a.terms) == {"minimum", "attend", "requir"}
    assert "minimum attend" in a.pairs
    assert "pairs" not in a.as_dict()


def citation(text, name="Rules.pdf", page=1):
    return Citation(chunk_id="c", document_id="d", document_name=name, document_type="x",
                    section="s", page=page, score=0.5, text=text)


PASSAGE = (
    "Students must keep 75 percent attendance in every course. "
    "Those below 75 percent cannot sit the examination. "
    "The library is open from 9 am to 5 pm."
)


def test_the_relevant_sentence_is_picked():
    a = analyze("What attendance do students need?")
    chosen = answering.select_sentences(a, [citation(PASSAGE)])
    assert chosen[0].sentence.startswith("Students must keep 75 percent")
    assert all("library" not in c.sentence for c in chosen)


def test_neighbour_sentence_is_added_when_it_shares_a_word():
    a = analyze("Can students below the minimum attendance sit the examination?")
    sentences = [c.sentence for c in answering.select_sentences(a, [citation(PASSAGE)])]
    assert any("cannot sit" in s for s in sentences)


def test_date_question_prefers_sentences_with_a_date():
    text = "The exam committee meets often. The term end exam starts on 12 December 2026."
    chosen = answering.select_sentences(analyze("When does the term end exam start?"), [citation(text)], neighbours=False)
    assert "12 December" in chosen[0].sentence


def test_table_rows_stay_whole():
    sentences = answering.passage_sentences("Course | Credits | Hours\nMathematics | 4 | 60 hours total")
    assert "Mathematics | 4 | 60 hours total" in sentences


def test_numbered_date_list_is_split_into_rows():
    row = "20. 02-Nov-26 Monday Diwali 21. 25-Dec-26 Friday Christmas"
    assert len(answering.passage_sentences(row)) == 2


def test_compose_answer_names_the_source_and_numbers_passages():
    text, chosen = answering.compose_answer(analyze("What attendance do students need?"), [citation(PASSAGE, page=4)])
    assert text.startswith("From Rules.pdf, page 4:")
    assert "[1]" in text and chosen


def test_compose_answer_is_empty_when_nothing_can_be_selected():
    assert answering.compose_answer(analyze("anything"), [citation("ok")])[0] == ""


def test_yes_no_verdict_follows_the_wording():
    no = analyze("Is ragging allowed on campus?")
    chosen = answering.select_sentences(no, [citation("Ragging is not allowed on campus or inside hostels.")])
    assert answering.polarity(no, chosen) == "No"
    yes = analyze("Is a medical exemption allowed?")
    chosen = answering.select_sentences(yes, [citation("A medical exemption is allowed with a certificate.")])
    assert answering.polarity(yes, chosen) == "Yes"


def test_a_sentence_that_misses_a_question_word_gives_no_verdict():
    a = analyze("Is ragging allowed on campus?")
    chosen = answering.select_sentences(a, [citation("Ragging is strictly prohibited on campus and inside hostels.")])
    assert answering.polarity(a, chosen) == ""


def test_no_verdict_when_the_sentence_only_partly_matches():
    a = analyze("Can a student with 75 percent attendance appear for the supplementary examination in december?")
    chosen = answering.select_sentences(a, [citation("Attendance of 75 percent is needed.")])
    assert answering.polarity(a, chosen) == ""


def test_overview_questions_use_textrank():
    text = " ".join([
        "The grading system uses letter grades from O to F for every course.",
        "Each letter grade carries grade points that feed the semester grade point average.",
        "The cumulative grade point average is calculated from all completed semesters.",
        "A student needs a minimum grade point average to receive the degree.",
        "The library is open on weekdays.",
    ])
    chosen = answering.choose_sentences(analyze("Tell me about the grading system"), [citation(text)])
    assert 1 <= len(chosen) <= 4

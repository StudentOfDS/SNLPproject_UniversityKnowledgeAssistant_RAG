import numpy as np
import pytest

from app.lm_retrieval import QueryLikelihoodIndex

TEXTS = [
    "students need seventy five percent attendance to sit the examination",
    "tuition fees are paid every semester and late payment adds a fine",
    "the library is open on weekdays and closed on public holidays",
]
QUESTION = {"attend": 1.0, "examin": 1.0}


@pytest.mark.parametrize("smoothing", ["dirichlet", "jelinek_mercer", "laplace"])
def test_matching_passage_ranks_first(smoothing):
    index = QueryLikelihoodIndex(TEXTS, smoothing=smoothing, mu=5)
    assert int(np.argmax(index.scores(QUESTION))) == 0


def test_unknown_words_are_ignored_not_fatal():
    index = QueryLikelihoodIndex(TEXTS)
    assert np.allclose(index.scores({"zzzz": 1.0}), 0.0)


def test_smoothing_keeps_missing_words_from_zeroing_a_passage():
    index = QueryLikelihoodIndex(TEXTS, smoothing="dirichlet", mu=5)
    scores = index.scores({"attend": 1.0, "fee": 1.0})
    assert np.all(np.isfinite(scores))


def test_matches_flags_passages_with_a_question_word():
    index = QueryLikelihoodIndex(TEXTS)
    assert index.matches({"librari": 1.0}).tolist() == [False, False, True]


def test_unknown_smoothing_is_rejected():
    with pytest.raises(ValueError):
        QueryLikelihoodIndex(TEXTS, smoothing="bm25")

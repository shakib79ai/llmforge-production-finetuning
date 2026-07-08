import math

from llmforge.evaluation.metrics import exact_match_score, perplexity, rouge_scores


def test_exact_match_score():
    assert exact_match_score(["a", "b"], ["a", "c"]) == 0.5


def test_rouge_scores_identical_text_is_near_one():
    scores = rouge_scores(["the cat sat on the mat"], ["the cat sat on the mat"])
    assert scores["rouge1"] > 0.99


def test_perplexity_matches_exp_of_loss():
    assert math.isclose(perplexity(0.0), 1.0)

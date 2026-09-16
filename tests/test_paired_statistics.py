"""Pin known paired-test probabilities and cross-check an established library."""
import pytest

from multivon_eval.experiments import mcnemar_test


@pytest.mark.parametrize("b,c,p", [(0, 0, 1.0), (2, 2, 1.0), (100, 100, 1.0),
                                    (0, 5, 0.0625), (0, 6, 0.03125), (1, 1, 1.0)])
def test_known_binomial_probabilities_and_equal_directions(b, c, p):
    assert mcnemar_test([True] * b + [False] * c, [False] * b + [True] * c) == pytest.approx(p)


def test_unknown_outcomes_are_not_failures():
    with pytest.raises(ValueError, match="binary outcomes"):
        mcnemar_test([True, None], [False, True])


def test_exact_small_sample_results_match_scipy():
    stats = pytest.importorskip("scipy.stats")
    for b in range(15):
        for c in range(15):
            expected = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
            assert mcnemar_test([True] * b + [False] * c,
                                [False] * b + [True] * c) == pytest.approx(expected)

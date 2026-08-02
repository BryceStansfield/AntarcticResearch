"""Tests for deterministic tie-breaking in the WP -> instrument matchers.

Three matchers pick a "best" working paper with `argmax`, which returns the *first* maximal index.
First in what order? The order `load_working_papers` built, which is `get_embeddings_by_type`'s
`ORDER BY document_uuid` -- and a uuid is sha256 of the document's text. So the tie-break was
stable for a fixed store but arbitrary in meaning, and re-OCRing one paper (or changing the
segmentation) moved its uuid and silently re-resolved ties between two *other* papers that had not
changed at all.

`label_order` fixes the order to something meaningful: reorder the columns by label and `argmax`'s
"first" becomes "lexicographically smallest label".
"""
import numpy as np
import pytest

from latency_analyses.measure_wp_latency import argmax_tiebroken, label_order
from latency_analyses.latency_threshold_exploration import null_sample


# --------------------------------------------------------------------------- label_order

def test_argmax_picks_the_strict_maximum_regardless_of_labels():
    """Tie-breaking must never override a genuinely higher similarity."""
    labels = ["zzz", "aaa"]
    sims = np.array([0.9, 0.5])
    assert int(argmax_tiebroken(sims, label_order(labels))) == 0


def test_a_tie_goes_to_the_lexicographically_first_label():
    labels = ["b_paper", "a_paper"]
    sims = np.array([0.8, 0.8])
    assert int(argmax_tiebroken(sims, label_order(labels))) == 1, "a_paper wins the tie"


def test_the_choice_does_not_depend_on_array_position():
    """The regression: the same two candidates, presented in either order, must give the same
    paper. Plain argmax returns index 0 both times, i.e. a different paper each time."""
    sims = np.array([0.8, 0.8])
    forward = ["b_paper", "a_paper"]
    reverse = ["a_paper", "b_paper"]

    chosen_forward = forward[int(argmax_tiebroken(sims, label_order(forward)))]
    chosen_reverse = reverse[int(argmax_tiebroken(sims, label_order(reverse)))]
    assert chosen_forward == chosen_reverse == "a_paper"

    # Contrast, so the test is not vacuous.
    assert forward[int(np.argmax(sims))] != reverse[int(np.argmax(sims))]


def test_reordering_the_whole_candidate_set_does_not_change_the_winner():
    """A re-embed can permute the candidate array wholesale; the match must not move with it."""
    rng = np.random.default_rng(0)
    labels = np.array([f"paper_{i:03d}" for i in range(50)])
    sims = np.round(rng.random(50), 2)  # rounding manufactures ties

    winner = labels[int(argmax_tiebroken(sims, label_order(labels)))]
    for _ in range(5):
        perm = rng.permutation(50)
        p_labels, p_sims = labels[perm], sims[perm]
        assert p_labels[int(argmax_tiebroken(p_sims, label_order(p_labels)))] == winner


def test_works_row_wise_on_a_matrix():
    """The threshold sweep and the bipartite graphs call it on a whole (instrument, paper) matrix."""
    labels = ["b", "a", "c"]
    sims = np.array([[0.5, 0.5, 0.1],   # tie between b and a -> a
                     [0.1, 0.2, 0.9]])  # clear winner -> c
    assert argmax_tiebroken(sims, label_order(labels)).tolist() == [1, 2]


def test_masked_out_candidates_are_never_chosen():
    """Callers mask ineligible papers with -inf; those must stay unselected however they sort."""
    labels = ["a_masked", "z_eligible"]
    sims = np.array([-np.inf, 0.3])
    assert int(argmax_tiebroken(sims, label_order(labels))) == 1


# --------------------------------------------------------------------------- null_sample

def test_null_sample_is_deterministic():
    sims = np.arange(1000, dtype=float).reshape(20, 50)
    assert np.array_equal(null_sample(sims, size=100), null_sample(sims, size=100))


def test_null_sample_returns_everything_when_small_enough():
    sims = np.arange(60, dtype=float).reshape(6, 10)
    assert np.array_equal(null_sample(sims, size=100), sims.ravel())


def test_null_sample_draws_from_the_whole_matrix_not_a_stride():
    """Striding `sims.ravel()[::k]` walks row-major, so when k shares a factor with the row length
    it revisits the same columns forever -- sampling a few papers against every instrument rather
    than the pair space. Here k = size // 100 = 20 = the row length exactly, the degenerate case."""
    n_rows, n_cols = 100, 20
    sims = np.tile(np.arange(n_cols, dtype=float), (n_rows, 1))  # column index as the value

    strided = sims.ravel()[:: max(1, sims.size // 100)]
    assert len(set(strided.tolist())) == 1, "the stride sees exactly one column"

    sampled = null_sample(sims, size=100)
    assert len(set(sampled.tolist())) > 1, "a random draw sees the spread of columns"


def test_null_sample_size_is_respected():
    sims = np.arange(10_000, dtype=float).reshape(100, 100)
    assert null_sample(sims, size=250).shape == (250,)

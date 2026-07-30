"""Tests for the pure weighting/credit-assignment math in `measure_wp_introduction.py`.

`MeasureWPIntroducers.__init__` does real embedding-space nearest-neighbour lookups
via `embeddings.document_embeddings.DocumentTextGetter`/`EmbeddingLookerUpper` -- that
part is network/embedding-cache dependent and out of scope here. What it hands off to
`_select_introducing_neighbours` and `_weighted_country_credits`, once a WP's
`(distance, parties, year)` have already been resolved, is pure and is what these
tests exercise. Both functions are plain module-level functions (no I/O, no
randomness), so importing the module and calling them directly is enough -- unlike
`RatificationSpeed`, nothing here needs an AST-lift to reach in isolation.
"""
import pytest

from antarctic_ladder_metrics.measure_wp_introduction import (
    _select_introducing_neighbours,
    _weighted_country_credits,
)


# -------------------------------------------------- _select_introducing_neighbours

def test_non_party_author_alone_is_dropped_not_credited_to_nobody():
    """A candidate whose only author is SCAR must be excluded outright, not kept
    with an empty party list. `_weighted_country_credits` has no way to skip a kept
    tuple with no parties, so if this filtering slipped, SCAR's distance would still
    consume a `weight_normaliser` slot with nothing to credit it to."""
    candidates = [(1.0, ["scar"], 2000)]
    assert _select_introducing_neighbours(candidates, measure_year=2005, neighbours_to_weigh=3) == []


def test_non_party_authors_are_stripped_but_real_parties_survive():
    """A WP co-authored by SCAR and Chile keeps Chile and drops SCAR, rather than
    being dropped wholesale for having a non-party co-author."""
    candidates = [(1.0, ["scar", "chile"], 2000)]
    kept = _select_introducing_neighbours(candidates, measure_year=2005, neighbours_to_weigh=3)
    assert kept == [(1.0, ["chile"])]


def test_candidate_postdating_the_measure_is_skipped():
    """A WP can only introduce a measure it predates; year > measure_year disqualifies it."""
    candidates = [(1.0, ["chile"], 2010)]
    assert _select_introducing_neighbours(candidates, measure_year=2005, neighbours_to_weigh=3) == []


def test_candidate_with_no_year_is_skipped():
    candidates = [(1.0, ["chile"], None)]
    assert _select_introducing_neighbours(candidates, measure_year=2005, neighbours_to_weigh=3) == []


def test_candidate_in_the_same_year_as_the_measure_is_kept():
    """The predate check is `year > measure_year`, not `>=`, so a same-year WP counts."""
    candidates = [(1.0, ["chile"], 2005)]
    assert _select_introducing_neighbours(candidates, measure_year=2005, neighbours_to_weigh=3) == \
        [(1.0, ["chile"])]


def test_truncates_to_neighbours_to_weigh_survivors():
    """Four valid candidates with neighbours_to_weigh=3 keep only the nearest three."""
    candidates = [
        (1.0, ["chile"], 2000),
        (2.0, ["argentina"], 2000),
        (3.0, ["norway"], 2000),
        (4.0, ["australia"], 2000),
    ]
    kept = _select_introducing_neighbours(candidates, measure_year=2005, neighbours_to_weigh=3)
    assert kept == [(1.0, ["chile"]), (2.0, ["argentina"]), (3.0, ["norway"])]


def test_a_failing_candidate_does_not_consume_a_survivor_slot():
    """An excluded candidate sitting between two valid ones does not stop later valid
    candidates from filling the quota -- only surviving candidates count towards
    neighbours_to_weigh, not position in the input list."""
    candidates = [
        (1.0, ["chile"], 2000),
        (1.5, ["scar"], 2000),        # excluded: no party author left
        (2.0, ["argentina"], 2000),
        (3.0, ["norway"], 2000),      # third survivor -> stop here
        (4.0, ["australia"], 2000),   # never reached
    ]
    kept = _select_introducing_neighbours(candidates, measure_year=2005, neighbours_to_weigh=3)
    assert kept == [(1.0, ["chile"]), (2.0, ["argentina"]), (3.0, ["norway"])]


def test_candidates_after_the_quota_is_reached_are_never_inspected():
    """Characterises the early-stop behaviour precisely: iteration halts the moment
    neighbours_to_weigh candidates have survived, so a later candidate is never even
    evaluated against the filters -- not evaluated-and-passed, not evaluated-and-failed.

    A plain `year > measure_year` comparison can't distinguish "never evaluated" from
    "evaluated and happened to pass", so this rigs the year field with an object whose
    `__gt__` raises. If the loop ever reached this candidate, the comparison would blow
    up; the test passing is proof it did not.
    """
    class ExplodesIfCompared:
        def __gt__(self, other):
            raise AssertionError("year comparison ran on a candidate past the quota")

    candidates = [
        (1.0, ["chile"], 2000),
        (2.0, ["argentina"], 2000),
        (3.0, ["norway"], 2000),
        (4.0, ["australia"], ExplodesIfCompared()),
    ]
    kept = _select_introducing_neighbours(candidates, measure_year=2005, neighbours_to_weigh=3)
    assert kept == [(1.0, ["chile"]), (2.0, ["argentina"]), (3.0, ["norway"])]


def test_fewer_survivors_than_the_quota_yields_all_of_them():
    candidates = [(1.0, ["chile"], 2000), (2.0, ["scar"], 2000)]
    kept = _select_introducing_neighbours(candidates, measure_year=2005, neighbours_to_weigh=3)
    assert kept == [(1.0, ["chile"])]


def test_no_candidates_yields_empty_list():
    assert _select_introducing_neighbours([], measure_year=2005, neighbours_to_weigh=3) == []


# -------------------------------------------------------- _weighted_country_credits

def test_weights_are_inverse_distance_and_sum_to_one():
    """Two single-party candidates: their credited weights must add to 1, and the
    nearer one (smaller distance) must get the larger share."""
    kept = [(1.0, ["chile"]), (2.0, ["argentina"])]
    credits = _weighted_country_credits(kept, measure_year=2005)

    assert credits[(2005, "chile")] == pytest.approx(2 / 3)
    assert credits[(2005, "argentina")] == pytest.approx(1 / 3)
    assert sum(credits.values()) == pytest.approx(1.0)
    assert credits[(2005, "chile")] > credits[(2005, "argentina")]


def test_credit_key_is_measure_year_and_party():
    kept = [(1.0, ["chile"])]
    credits = _weighted_country_credits(kept, measure_year=1999)
    assert credits == {(1999, "chile"): pytest.approx(1.0)}


def test_multiple_parties_on_one_candidate_each_get_the_full_weight():
    """A jointly-authored WP is not split between its co-authors: Chile and Argentina
    on the same candidate each receive that candidate's whole doc_weight, so the two
    credited weights are equal, not halved, and their sum exceeds the single-candidate
    normalisation of 1."""
    kept = [(1.0, ["chile", "argentina"])]
    credits = _weighted_country_credits(kept, measure_year=2005)

    assert credits[(2005, "chile")] == pytest.approx(1.0)
    assert credits[(2005, "argentina")] == pytest.approx(1.0)


def test_a_party_credited_by_two_candidates_gets_both_weights_added():
    """The same party appearing on two different surviving WPs accumulates both
    candidates' doc_weights into one entry rather than overwriting it."""
    kept = [(1.0, ["chile"]), (2.0, ["chile"])]
    credits = _weighted_country_credits(kept, measure_year=2005)

    assert credits == {(2005, "chile"): pytest.approx(1.0)}


def test_single_candidate_gets_all_the_weight():
    kept = [(4.0, ["norway"])]
    credits = _weighted_country_credits(kept, measure_year=2005)
    assert credits == {(2005, "norway"): pytest.approx(1.0)}


def test_empty_kept_list_yields_empty_credits():
    """No survivors (e.g. every candidate failed the filters) means no credit at all
    for that measure -- weight_normaliser is a sum over zero terms and the loop over
    `kept` never runs, so this must not raise a ZeroDivisionError."""
    assert _weighted_country_credits([], measure_year=2005) == {}


def test_credits_are_rounded_to_ten_decimal_places():
    """`distance` ultimately comes from a nearest-neighbour embedding lookup, whose
    underlying numpy computation is not guaranteed bit-identical run-to-run.
    Rounding here (rather than only at CSV-write time) keeps every downstream sum
    built from already-stable numbers, so re-running the pipeline over unchanged
    data reproduces the exact same figures."""
    kept = [(3.0, ["chile"]), (7.0, ["argentina"])]
    credits = _weighted_country_credits(kept, measure_year=2005)
    for value in credits.values():
        assert value == round(value, 10)

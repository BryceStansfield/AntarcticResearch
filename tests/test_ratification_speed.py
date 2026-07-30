"""Tests for the approval-parsing helpers behind `RatificationSpeed`.

`extract_end_year`, `parse_approval_pairs` and `compute_row_delays` used to be
nested inside `RatificationSpeed.__init__`, which calls `scrape_and_enrich_measures`
and reads the whole corpus -- so exercising them meant running the entire pipeline.
They are now module-level functions with no dependency on the class, the corpus, or
the network, so they are imported and driven directly here.

`RatificationSpeed.__init__` itself, and the `add_approval` closure inside it, are
NOT covered here: instantiating the class still means running the real pipeline.
The population filter that `add_approval` applies (only countries whose name ends
in " *" -- Consultative Parties at that resolution -- are kept, and the suffix is
stripped before accumulation) is deliberately left inside `add_approval`, so
`compute_row_delays` is expected to hand back " *"-suffixed and unstarred country
names untouched; see `test_the_consultative_party_suffix_passes_through_untouched`.
"""
import pytest

from antarctic_ladder_metrics.constants import END_YEAR
from antarctic_ladder_metrics.ratification_speed import (
    compute_row_delays,
    extract_end_year,
    parse_approval_pairs,
)


# --------------------------------------------------------------- extract_end_year

@pytest.mark.parametrize("status,expected", [
    # Right-censored at the window edge.
    ("Not yet effective", END_YEAR),
    # Plain effective date.
    ("Effective 11/05/2016", 2016),
    # "(Fast Approval)" is a label, not a year -- four digits are required
    # inside the parens, which is what keeps this on the effective-date branch.
    ("Effective 19/12/2002 (Fast Approval)", 2002),
    # Both a trailing "(yyyy)" and an "Effective dd/mm/yyyy" prefix: the
    # parenthesised year wins because it closes the ratification window later.
    ("Effective 30/04/1962. No longer current:D 1 (2014)", 2014),
    # Withdrawn with no "Effective" fragment at all.
    ("Did not enter into effect. Withdrawn:M 3 (2012)", 2012),
])
def test_each_documented_status_shape_parses(status, expected):
    assert extract_end_year(status) == expected


def test_trailing_parenthesised_year_takes_precedence_over_effective_date():
    """Restates the fourth case above on its own: this is the one status shape
    where both regexes would match, so it is the only place the precedence
    between them is actually observable."""
    assert extract_end_year("Effective 30/04/1962. No longer current:D 1 (2014)") == 2014


def test_surrounding_whitespace_is_stripped():
    assert extract_end_year("  Effective 11/05/2016  ") == 2016


def test_unrecognised_status_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="Cannot extract year"):
        extract_end_year("Some entirely new status wording")


# ------------------------------------------------------------- parse_approval_pairs

def test_a_country_followed_by_a_year_line_is_a_ratification():
    pairs = parse_approval_pairs("Argentina *\n12/06/1996")
    assert pairs == [("Argentina *", "1996")]


def test_a_country_with_no_following_year_line_is_never_ratified():
    """No digit-leading line follows the country before the text ends, so it
    pairs with None -- meaning "never ratified", to be censored by the caller
    rather than treated as a zero-delay ratification."""
    pairs = parse_approval_pairs("Belgium *")
    assert pairs == [("Belgium *", None)]


def test_a_country_followed_by_another_country_is_also_never_ratified():
    """The digit-detection rule (`e[0] not in "0123456789"`) is what tells
    "the next line is a new country" apart from "the next line is this
    country's ratification date" -- a country line immediately followed by
    another country line means the first one never ratified."""
    pairs = parse_approval_pairs("Australia *\nBelgium *")
    assert pairs == [("Australia *", None), ("Belgium *", None)]


def test_multiple_countries_ratified_and_unratified_in_one_blob():
    text = "Argentina *\n12/06/1996\nAustralia *\n\nBelgium"
    pairs = parse_approval_pairs(text)
    assert pairs == [
        ("Argentina *", "1996"),
        ("Australia *", None),
        ("Belgium", None),
    ]


def test_blank_lines_are_filtered_regardless_of_position():
    text = "\n\nArgentina *\n12/06/1996\n\n"
    assert parse_approval_pairs(text) == [("Argentina *", "1996")]


def test_the_consultative_party_suffix_passes_through_untouched():
    """The " *" suffix is left exactly as it appears in the Approvals cell.

    Stripping it (and filtering out countries that never had it) is
    `add_approval`'s job, not this function's -- `compute_row_delays` and
    `parse_approval_pairs` only split and pair, they never touch the
    population. An unstarred country name is returned just as faithfully.
    """
    pairs = parse_approval_pairs("Argentina *\n12/06/1996\nSomeNonConsultativeState\n2001")
    assert pairs == [("Argentina *", "1996"), ("SomeNonConsultativeState", "2001")]

def test_empty_text_yields_no_pairs():
    """RatificationSpeed's row filter excludes blank/NaN Approvals before this
    function is ever called, so this input is never live -- but it should
    degrade gracefully rather than raising."""
    assert parse_approval_pairs("") == []


# -------------------------------------------------------------- compute_row_delays

def test_a_ratified_country_gets_the_calendar_gap_as_its_delay():
    delays = compute_row_delays("Argentina *\n12/06/1996", atcm_year=1995,
                                 status="Effective 11/05/2016")
    assert delays == [("Argentina *", 1996 - 1995)]


def test_a_never_ratified_country_is_censored_via_extract_end_year():
    """No year line follows, so the delay is `extract_end_year(status) -
    atcm_year` -- the same censoring the withdrawn-measure comment in
    ratification_speed.py describes."""
    delays = compute_row_delays("Belgium *", atcm_year=1995,
                                 status="Not yet effective")
    assert delays == [("Belgium *", END_YEAR - 1995)]


def test_mixed_ratified_and_censored_countries_in_one_row():
    text = "Argentina *\n12/06/1996\nAustralia *\n\nBelgium"
    delays = compute_row_delays(text, atcm_year=1995,
                                 status="Did not enter into effect. Withdrawn:M 3 (2012)")
    assert delays == [
        ("Argentina *", 1996 - 1995),
        ("Australia *", 2012 - 1995),
        ("Belgium", 2012 - 1995),
    ]


def test_the_consultative_party_suffix_is_still_untouched_after_delay_computation():
    """Same population-filter boundary as
    `test_the_consultative_party_suffix_passes_through_untouched`, but through
    the full `compute_row_delays` path: it must not strip " *" or drop
    unstarred countries, because `add_approval` -- called by
    `RatificationSpeed.__init__` for every pair this returns -- still does
    both of those with its own `country[-1] == "*"` check."""
    delays = compute_row_delays("Argentina *\n1995\nUnstarredState", atcm_year=1995,
                                 status="Not yet effective")
    assert delays == [("Argentina *", 0), ("UnstarredState", END_YEAR - 1995)]


def test_delay_can_be_negative_when_a_measure_is_censored_before_its_own_year():
    """Not a case that occurs for real data (a measure cannot close its
    ratification window before it was tabled), but the function does no
    bounds-checking on the arithmetic, so this pins down that it is pure
    subtraction rather than a clamped delay."""
    delays = compute_row_delays("Belgium *", atcm_year=2020,
                                 status="Did not enter into effect. Withdrawn:M 3 (2012)")
    assert delays == [("Belgium *", 2012 - 2020)]

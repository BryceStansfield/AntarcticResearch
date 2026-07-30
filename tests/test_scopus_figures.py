"""Tests for the Scopus affiliation-matching pipeline (`ScopusFigures`).

`_build_keys_by_length` and `_resolve_affiliation_country` are pure and are
exercised against small synthetic `str_to_country` fixtures rather than the real
country_meta_info data -- the real country/alias list is out of scope here, only
the matching algorithm itself is under test.

`ScopusFigures.__init__` still owns the document-level aggregation (year
filtering, one-country-per-document dedup, per-year and totals rollup). It reads
`data/scopus_export.csv` and builds `str_to_country` from country_meta_info by
default, but both are now optional constructor parameters purely so tests can
inject fixtures; omitting them reproduces the pre-refactor behaviour exactly, so
the real pipeline (`ScopusFigures()`) is unaffected.
"""
import pandas as pd
import pytest

from antarctic_ladder_metrics.constants import END_YEAR
from antarctic_ladder_metrics.scopus_figures import (
    SCOPUS_START_YEAR,
    ScopusFigures,
    _build_keys_by_length,
    _resolve_affiliation_country,
)


# --------------------------------------------------------------- _build_keys_by_length

def test_groups_keys_by_length_descending():
    str_to_country = {"a": "x", "bb": "y", "cc": "y", "ddd": "z"}
    keys_by_length = _build_keys_by_length(str_to_country)

    lengths = [len(group[0]) for group in keys_by_length]
    assert lengths == sorted(lengths, reverse=True) == [3, 2, 1]
    assert set(keys_by_length[0]) == {"ddd"}
    assert set(keys_by_length[1]) == {"bb", "cc"}
    assert set(keys_by_length[2]) == {"a"}


def test_single_length_dict_yields_one_group():
    keys_by_length = _build_keys_by_length({"norway": "norway"})
    assert keys_by_length == [["norway"]]


def test_empty_dict_yields_no_groups():
    assert _build_keys_by_length({}) == []


# ----------------------------------------------------------- _resolve_affiliation_country

def test_longer_key_wins_over_shorter_substring():
    """Reproduces the "indian river state college" case from the source comment:
    "india" is a substring of "indian river", so without length precedence the
    affiliation would wrongly resolve to India instead of the specific, correct
    "indian river" key."""
    str_to_country = {"india": "india", "indian river": "united states"}
    keys_by_length = _build_keys_by_length(str_to_country)

    country, ambiguous = _resolve_affiliation_country(
        "indian river state college, florida", keys_by_length, str_to_country)

    assert (country, ambiguous) == ("united states", False)


def test_ambiguous_when_a_length_group_matches_more_than_one_country():
    str_to_country = {"chile": "chile", "china": "china"}
    keys_by_length = _build_keys_by_length(str_to_country)

    country, ambiguous = _resolve_affiliation_country(
        "joint chile china exchange program", keys_by_length, str_to_country)

    assert country is None
    assert ambiguous is True


def test_unresolved_when_nothing_matches():
    str_to_country = {"norway": "norway"}
    keys_by_length = _build_keys_by_length(str_to_country)

    country, ambiguous = _resolve_affiliation_country(
        "unaffiliated independent researcher", keys_by_length, str_to_country)

    assert country is None
    assert ambiguous is False


def test_stops_at_first_matching_group_even_when_that_group_is_ambiguous():
    """Matching must not fall through to shorter keys once a longer group has
    matched something, even if that longer match is ambiguous and a shorter group
    would otherwise have resolved cleanly. "ch" here would cleanly match "united
    kingdom" alone, but it must never be reached because the longer "chile"/"china"
    group matches first and is ambiguous."""
    str_to_country = {"chile": "chile", "china": "china", "ch": "united kingdom"}
    keys_by_length = _build_keys_by_length(str_to_country)
    assert [len(g[0]) for g in keys_by_length] == [5, 2]  # sanity: chile/china tried first

    country, ambiguous = _resolve_affiliation_country(
        "school of ch business, chile china program", keys_by_length, str_to_country)

    assert country is None
    assert ambiguous is True


def test_single_match_within_an_otherwise_larger_group_is_unambiguous():
    """Only the keys actually found in the affiliation count toward ambiguity --
    other keys of the same length that simply don't appear are irrelevant."""
    str_to_country = {"chile": "chile", "china": "china", "spain": "spain"}
    keys_by_length = _build_keys_by_length(str_to_country)

    country, ambiguous = _resolve_affiliation_country(
        "universidad de chile", keys_by_length, str_to_country)

    assert (country, ambiguous) == ("chile", False)


# --------------------------------------------------------------------- ScopusFigures

def _figures(years, affiliations, str_to_country):
    scopus_table = pd.DataFrame({"Year": years, "Affiliations": affiliations})
    return ScopusFigures(scopus_table=scopus_table, str_to_country=str_to_country)


def test_two_affiliations_from_the_same_country_count_once():
    """`document_countries` is a set keyed per document, so a paper listing two
    Norwegian institutions must not count as two Norwegian documents."""
    sf = _figures(
        [2020],
        ["University of Oslo, Norway; Norwegian Polar Institute, Norway"],
        {"norway": "norway"})

    assert sf.country_counts_by_year[(2020, "norway")] == 1
    assert sf.country_counts["norway"] == 1


def test_affiliations_from_two_different_countries_both_count():
    sf = _figures(
        [2020],
        ["University of Oslo, Norway; Universidad de Chile, Chile"],
        {"norway": "norway", "chile": "chile"})

    assert sf.country_counts_by_year[(2020, "norway")] == 1
    assert sf.country_counts_by_year[(2020, "chile")] == 1


def test_missing_affiliations_field_contributes_no_country():
    """`Affiliations` is filled with '' before splitting, so a NaN cell (a document
    with no recorded affiliations) must not raise and must not credit any country."""
    sf = _figures([2020], [float("nan")], {"norway": "norway"})

    assert sf.country_counts_by_year == {}
    assert sf.country_counts == {}


def test_ambiguous_affiliation_is_not_counted_for_either_country():
    sf = _figures([2020], ["Joint Chile China Institute"],
                  {"chile": "chile", "china": "china"})

    assert sf.country_counts_by_year == {}
    assert sf.country_counts == {}


@pytest.mark.parametrize("year,should_be_kept", [
    (SCOPUS_START_YEAR - 1, False),  # 2012: truncated Scopus export year, excluded
    (SCOPUS_START_YEAR, True),
    (END_YEAR, True),
    (END_YEAR + 1, False),
])
def test_year_filter_boundaries(year, should_be_kept):
    sf = _figures([year], ["University, Norway"], {"norway": "norway"})

    assert ((year, "norway") in sf.country_counts_by_year) == should_be_kept


def test_country_counts_sums_across_years():
    sf = _figures(
        [SCOPUS_START_YEAR, SCOPUS_START_YEAR + 1],
        ["University, Norway", "University, Norway"],
        {"norway": "norway"})

    assert sf.country_counts_by_year[(SCOPUS_START_YEAR, "norway")] == 1
    assert sf.country_counts_by_year[(SCOPUS_START_YEAR + 1, "norway")] == 1
    assert sf.country_counts["norway"] == 2


def test_country_dict_exposes_country_counts():
    sf = _figures([SCOPUS_START_YEAR], ["University, Norway"], {"norway": "norway"})
    assert sf.country_dict() == sf.country_counts == {"norway": 1}

"""Tests for the shared country lookup helpers.

These back every figure in the Antarctic ladder: `get_country_value_from_dict` is
what turns a metric's raw country dict into a cell in ladder_results.csv, so its
missing-value and alias behaviour decides what a blank or a zero means.
"""
import math

import pytest

import country_meta_info as cmi
from country_meta_info import (CaseInsensitiveDict, check_dict_coverage,
                               get_country_value_from_dict, normalize_country_name)


# ------------------------------------------------------------- CaseInsensitiveDict

def test_keys_round_trip_regardless_of_case():
    d = CaseInsensitiveDict()
    d["Australia"] = 3
    assert d["australia"] == 3
    assert d["AUSTRALIA"] == 3
    assert "AuStRaLiA" in d


def test_get_honours_default_and_case():
    d = CaseInsensitiveDict()
    d["Chile"] = 7
    assert d.get("CHILE") == 7
    assert d.get("Peru") is None
    assert d.get("Peru", 0) == 0


def test_non_string_keys_pass_through_untouched():
    d = CaseInsensitiveDict()
    d[2024] = "x"
    assert d[2024] == "x"
    assert 2024 in d


def test_from_dict_rejects_keys_that_collide_once_lowercased():
    """Two spellings collapsing to one key would silently drop a country's value."""
    with pytest.raises(ValueError, match="Duplicate case-insensitive key"):
        CaseInsensitiveDict.from_dict({"Norway": 1, "norway": 2})


# ------------------------------------------------------------ normalize_country_name

@pytest.mark.parametrize("alias,expected", [
    ("Czech Republic", "czechia"),
    ("Russian Federation", "russia"),
    ("South Korea", "republic of korea"),
    ("Korea (ROK)", "republic of korea"),
    ("UK", "united kingdom"),
    ("NZ", "new zealand"),
    ("USA", "united states"),
    ("türkiye", "turkey"),
    ("argentino", "argentina"),
])
def test_aliases_normalise_to_their_canonical_country(alias, expected):
    assert normalize_country_name(alias) == expected


def test_unknown_names_are_merely_lowercased():
    assert normalize_country_name("Atlantis") == "atlantis"


def test_canonical_names_normalise_to_themselves():
    assert normalize_country_name("Australia") == "australia"


# ------------------------------------------------------- get_country_value_from_dict

def test_matches_the_canonical_name():
    assert get_country_value_from_dict({"norway": 5}, "Norway") == 5


def test_matches_through_an_alias():
    assert get_country_value_from_dict({"russian federation": 4}, "Russia") == 4


def test_sums_across_distinct_aliases():
    """A metric may key the same country under more than one spelling; those add."""
    assert get_country_value_from_dict({"usa": 89, "united states": 1},
                                       "United States") == 90


def test_canonical_name_is_never_counted_twice():
    """Regression: 'United States' used to appear inside its own alias list, so the
    as-is match and the alias walk both hit the same entry and doubled every US
    figure in the ladder."""
    assert get_country_value_from_dict({"united states": 1}, "United States") == 1


def test_repeated_alias_entries_cannot_double_count():
    """Guards the class of bug above, not just the one instance of it."""
    cmi.country_alternative_names["Testland"] = ["Testland", "TL", "testland", "TL"]
    try:
        assert get_country_value_from_dict({"testland": 7}, "Testland") == 7
        assert get_country_value_from_dict({"testland": 7, "tl": 3}, "Testland") == 10
    finally:
        del cmi.country_alternative_names["testland"]


def test_absent_country_defaults_to_zero():
    """Right for the count metrics: a country with no entries genuinely scored zero."""
    assert get_country_value_from_dict({"australia": 5}, "Norway") == 0


def test_absent_country_can_report_a_custom_sentinel():
    """Averaged metrics pass NaN, because 0 there is the best possible score rather
    than an absence -- that is what keeps Czechia out of the Ratification Delay
    ranking instead of topping it."""
    assert math.isnan(get_country_value_from_dict({"australia": 5.0}, "Czechia",
                                                  float("nan")))


def test_a_real_zero_is_not_mistaken_for_a_missing_country():
    """Present-with-value-0 must survive even when the sentinel is NaN."""
    assert get_country_value_from_dict({"norway": 0}, "Norway", float("nan")) == 0


def test_lookup_is_case_insensitive_on_both_sides():
    assert get_country_value_from_dict({"NORWAY": 2}, "norway") == 2


# ------------------------------------------------------------- check_dict_coverage

def test_coverage_reports_unused_keys_and_missing_countries():
    unused, not_found = check_dict_coverage({"australia": 1, "atlantis": 9},
                                            ["Australia", "Norway"])
    assert unused == ["atlantis"]
    assert not_found == ["Norway"]


def test_coverage_counts_an_alias_hit_as_found():
    unused, not_found = check_dict_coverage({"russian federation": 1}, ["Russia"])
    assert not_found == []
    assert unused == []


def test_coverage_is_clean_when_everything_lines_up():
    unused, not_found = check_dict_coverage({"chile": 1, "peru": 2},
                                            ["Chile", "Peru"])
    assert (unused, not_found) == ([], [])


# ------------------------------- CaseInsensitiveDict: the C-level entry points fold keys too

def test_constructor_folds_keys():
    """`dict.__init__` is implemented in C and stores keys directly rather than routing through
    `__setitem__`, so the type's own constructor used to build a dict that was not in fact
    case-insensitive -- while `from_dict` produced one that was."""
    d = CaseInsensitiveDict({"USA": 1})
    assert d["usa"] == 1
    assert "USA" in d


def test_constructor_accepts_pairs_and_kwargs():
    assert CaseInsensitiveDict([("Chile", 1)])["chile"] == 1
    assert CaseInsensitiveDict(Norway=2)["norway"] == 2


def test_update_folds_keys():
    """Same defect as the constructor: `dict.update` bypasses `__setitem__`."""
    d = CaseInsensitiveDict()
    d.update({"Norway": 5})
    assert d["NORWAY"] == 5

    d.update([("Chile", 6)])
    assert d["chile"] == 6


def test_setdefault_and_pop_and_delete_fold_keys():
    d = CaseInsensitiveDict({"Peru": 1})
    assert d.setdefault("PERU", 99) == 1, "must find the existing entry, not insert a second"
    assert d.setdefault("Brazil", 7) == 7
    assert d["brazil"] == 7

    assert d.pop("BRAZIL") == 7
    assert "brazil" not in d

    del d["PERU"]
    assert "peru" not in d


def test_a_dict_built_by_constructor_matches_one_built_by_from_dict():
    assert dict(CaseInsensitiveDict({"USA": 1})) == dict(CaseInsensitiveDict.from_dict({"USA": 1}))


# --------------------------------------------- aliases resolve in both directions (all_names_for)

def test_all_names_for_expands_from_the_canonical_name():
    names = [n.lower() for n in cmi.all_names_for("United States")]
    assert "united states" in names and "usa" in names and "us" in names


def test_all_names_for_expands_from_an_alias():
    """`country_alternative_names` is keyed by canonical name only, so looking up "USA" in it
    directly returns nothing. Normalising to the canonical name first is what makes every spelling
    of a country resolve to the same set."""
    assert sorted(n.lower() for n in cmi.all_names_for("USA")) == \
           sorted(n.lower() for n in cmi.all_names_for("United States"))


def test_all_names_for_passes_through_an_unknown_country():
    assert cmi.all_names_for("Atlantis") == ["Atlantis"]


def test_value_lookup_by_alias_finds_a_canonically_keyed_dict():
    """The regression. `get_list_of_country_names()` returns canonical names *and* aliases, so a
    caller iterating it asks about "USA" as readily as "United States"; asking by alias used to
    miss a dict keyed canonically and report a spurious zero."""
    assert get_country_value_from_dict({"United States": 4}, "USA") == 4
    assert get_country_value_from_dict({"united states": 4}, "US") == 4


def test_value_lookup_by_canonical_name_still_finds_an_alias_keyed_dict():
    assert get_country_value_from_dict({"USA": 4}, "United States") == 4


def test_value_lookup_by_alias_does_not_double_count():
    """A dict holding the same country under two spellings must still sum each entry once."""
    assert get_country_value_from_dict({"United States": 4, "USA": 5}, "USA") == 9


def test_missing_sentinel_still_applies_to_an_unmatched_alias():
    assert get_country_value_from_dict({"Chile": 1}, "USA") == 0
    assert math.isnan(get_country_value_from_dict({"Chile": 1}, "USA", missing=float("nan")))


def test_coverage_by_alias_finds_a_canonically_keyed_dict():
    """Symmetric to the value lookup: asking about "USA" over a dict keyed "United States" used to
    report the country as not-found *and* its key as unused -- one absence counted twice."""
    unused, not_found = check_dict_coverage({"united states": 1}, ["USA"])
    assert (unused, not_found) == ([], [])

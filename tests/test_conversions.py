"""Tests for meeting-name -> year conversion.

`actm_meeting_to_year` reads a meeting number out of a free-text title and maps it through
MEETING_YEAR_DICT. Two things make that harder than it looks, and both are pinned here:

* **SATCM contains ATCM.** The Special Consultative Meetings carry their own numbering (SATCM
  I-XII) that has nothing to do with the ordinary ATCM series, so matching "ATCM" as a substring
  reads a Special meeting's number as an ordinary one and returns a year from the wrong series.
  A blocked SATCM must fall through to the parenthesised year in its own title.
* **The number's format varies.** The corpus writes roman ("ATCM XXIV"), arabic with a space
  ("ATCM 24") and arabic without one ("ATCM24"). Accepting the no-space form means the numeral
  can no longer be anchored by whitespace, which is what lets an ordinary word starting in
  I/V/X/L be misread as a roman numeral.
"""
import pytest

from conversions import MEETING_YEAR_DICT, actm_meeting_to_year, roman_to_int


# --------------------------------------------------------------------------- roman numerals

@pytest.mark.parametrize("roman, expected", [
    ("I", 1), ("IV", 4), ("V", 5), ("IX", 9), ("X", 10),
    ("XXIV", 24), ("XL", 40), ("L", 50),
])
def test_roman_to_int(roman, expected):
    assert roman_to_int(roman) == expected


def test_roman_to_int_rejects_non_strings():
    assert roman_to_int(None) is None
    assert roman_to_int("") is None


# --------------------------------------------------------------------------- the ATCM series

@pytest.mark.parametrize("name, expected", [
    ("ATCM I", 1961),
    ("ATCM IV", 1966),
    ("ATCM XXIV", 2001),
    ("ATCM XLIV", 2022),
])
def test_roman_meeting_numbers(name, expected):
    assert actm_meeting_to_year(name) == expected


@pytest.mark.parametrize("name", ["ATCM 24", "ATCM24", "ATCM  24"])
def test_arabic_meeting_numbers_with_or_without_a_space(name):
    """The no-space form used to return None, because the pattern required whitespace."""
    assert actm_meeting_to_year(name) == MEETING_YEAR_DICT[24]


def test_meetings_past_the_table_are_extrapolated():
    """ATCM 44 is 2022 and the series is annual thereafter."""
    assert actm_meeting_to_year("ATCM 45") == 2023
    assert actm_meeting_to_year("ATCM 46") == 2024


# --------------------------------------------------------------------------- SATCM must not match

def test_satcm_does_not_read_as_atcm():
    """The regression. 'SATCM XI' contains 'ATCM XI', which mapped to the ordinary series' 1981.
    Blocked, it falls through to the year in its own title."""
    assert actm_meeting_to_year("SATCM XI (Utrecht, 1994)") == 1994


def test_satcm_without_a_year_is_unknown_rather_than_wrong():
    """With nothing else to go on, None is the honest answer -- a Special meeting's number cannot
    be mapped through the ordinary series' table at all."""
    assert actm_meeting_to_year("SATCM XI") is None


def test_a_bare_atcm_after_other_letters_does_not_match():
    assert actm_meeting_to_year("XATCM IV") is None


# ------------------------------------------------------- numerals must not be found inside words

@pytest.mark.parametrize("name", [
    "ATCM Information Paper",   # "I" of Information
    "ATCM Ixxx",                # leading I, then letters
    "ATCM Views on the matter",  # "V" of Views
    "ATCM Long paper",          # "L" of Long
])
def test_words_starting_with_a_roman_letter_are_not_meeting_numbers(name):
    """Pre-existing, not a side effect of accepting 'ATCM24': the old `ATCM\\s+([IVXL]+)` matched
    the "I" of "Information" just as readily, so 'ATCM Information Paper' returned meeting 1, i.e.
    1961. Making the whitespace optional widens the exposure, and the trailing boundary is what
    closes it in both forms."""
    assert actm_meeting_to_year(name) is None


def test_a_real_numeral_followed_by_text_still_matches():
    assert actm_meeting_to_year("ATCM XXIV (Bogota, 2001)") == 2001


# --------------------------------------------------------------------------- year fallbacks

def test_parenthesised_year_is_used_when_there_is_no_meeting_number():
    assert actm_meeting_to_year("Antarctic Treaty Meeting (1998)") == 1998


def test_city_and_year_form_is_read():
    assert actm_meeting_to_year("Some Meeting (Madrid, 1991)") == 1991


def test_unparseable_names_return_none():
    assert actm_meeting_to_year("no meeting information here") is None

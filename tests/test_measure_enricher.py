"""Tests for the measure enrichment helpers.

`roman_to_int` is pure. `enrich_measure_data` reads and writes CSVs but takes both
paths as arguments, so it can be driven end to end over a temp fixture without
touching the real corpus or the network.
"""
import pandas as pd
import pytest

from ACTM_Measure_Scraper.src.MeasureEnricher import enrich_measure_data, roman_to_int


# --------------------------------------------------------------------- roman_to_int

@pytest.mark.parametrize("roman,expected", [
    # Single symbols, the whole supported alphabet.
    ("I", 1), ("V", 5), ("X", 10), ("L", 50),
    # Purely additive.
    ("II", 2), ("III", 3), ("VI", 6), ("VII", 7), ("XV", 15), ("XXX", 30),
    # Subtractive pairs.
    ("IV", 4), ("IX", 9), ("XL", 40),
    # Subtractive pair trailing an additive run.
    ("XIV", 14), ("XXIV", 24), ("XXIX", 29), ("XLIV", 44), ("XLIX", 49),
    # Real ATCM meeting numbers seen in titles.
    ("XIX", 19), ("XXI", 21), ("XXVIII", 28), ("XLVII", 47),
])
def test_roman_to_int_parses_supported_numerals(roman, expected):
    assert roman_to_int(roman) == expected


@pytest.mark.parametrize("value", [None, "", 0, float("nan"), 12, ["X"]])
def test_roman_to_int_returns_none_for_non_numeral_input(value):
    """Falsy values and non-strings yield None rather than raising.

    `str.extract` hands this function NaN whenever the ATCM regex misses, so the
    NaN case is a live code path, not a hypothetical.
    """
    assert roman_to_int(value) is None


# ---------------------------------------------------------------- enrich_measure_data

def _write_corpus(path, rows):
    """Build a minimal MeasureCorpus.csv. Only Title and Status are read by the
    enricher, but the real column set is kept so the fixture stays representative."""
    frame = pd.DataFrame(rows, columns=["Document_Number", "Subject", "Status",
                                        "Category", "Topics", "Title", "Content",
                                        "Approvals"])
    frame.to_csv(path, index=False)


def _enrich(tmp_path, rows):
    src, dest = tmp_path / "corpus.csv", tmp_path / "enriched.csv"
    _write_corpus(src, rows)
    enrich_measure_data(src, dest)
    return pd.read_csv(dest)


def _row(title, status="Effective 11/05/2016", number=1):
    return {"Document_Number": number, "Subject": "s", "Status": status,
            "Category": "c", "Topics": "[]", "Title": title, "Content": "x",
            "Approvals": ""}


def test_modern_title_yields_year_number_type_and_meeting(tmp_path):
    out = _enrich(tmp_path, [_row("Measure 1 (1995) - ATCM XIX, Seoul")])
    assert out.loc[0, "ATCM_Year"] == 1995
    assert out.loc[0, "ATCM_Number"] == 19       # XIX
    assert out.loc[0, "Type"] == "Measure"
    assert out.loc[0, "Meeting_Type"] == "ATCM"


def test_adoption_year_comes_from_the_status_date(tmp_path):
    out = _enrich(tmp_path, [_row("Measure 1 (1995) - ATCM XIX, Seoul",
                                  status="Effective 11/05/2016")])
    assert out.loc[0, "Adoption_Year"] == 2016


def test_adoption_year_is_blank_when_status_has_no_date(tmp_path):
    out = _enrich(tmp_path, [_row("Measure 1 (1995) - ATCM XIX, Seoul",
                                  status="Not yet effective")])
    assert pd.isna(out.loc[0, "Adoption_Year"])


def test_legacy_title_with_parenthesised_city_and_year(tmp_path):
    """The year sits inside a '(ATCM I - Canberra, 1961)' group, which only the
    enricher's third regex alternative reaches."""
    out = _enrich(tmp_path, [_row("Recommendation I-I (ATCM I - Canberra, 1961)")])
    assert out.loc[0, "ATCM_Year"] == 1961
    assert out.loc[0, "ATCM_Number"] == 1
    assert out.loc[0, "Type"] == "Recommendation"


@pytest.mark.parametrize("title,expected", [
    ("Measure 1 (1995) - ATCM XIX, Seoul", "ATCM"),
    ("Adopted by SATCM I-1 (London, 1977)", "SATCM"),
    ("Adopted by Conf. CCAS-1 (London, 1972)", "CCAS"),
    ("Adopted by Conf. CCAMLR-1 (Canberra, 1980)", "CCAMLR"),
    ("Adopted by ATIP 2019/2021, Intersessional period", "ATCM"),
    ("Something entirely unrelated", "Unknown"),
])
def test_meeting_type_classification(tmp_path, title, expected):
    out = _enrich(tmp_path, [_row(title)])
    assert out.loc[0, "Meeting_Type"] == expected


def test_satcm_wins_over_the_atcm_substring(tmp_path):
    """'SATCM' contains 'ATCM', so ordering of the checks is what keeps these apart."""
    out = _enrich(tmp_path, [_row("Adopted by SATCM I-1 (London, 1977)")])
    assert out.loc[0, "Meeting_Type"] == "SATCM"


def test_ccamlr_is_not_labelled_satcm(tmp_path):
    """Regression: CCAMLR titles used to be tagged SATCM.

    CCAMLR is a separate convention, not a Special ATCM. It never reached the ladder
    either way (RatificationSpeed keeps Meeting_Type == 'ATCM'), so this is a
    correctness fix in the enriched corpus rather than a change to any figure.
    """
    out = _enrich(tmp_path, [_row("Adopted by Conf. CCAMLR-1 (Canberra, 1980)")])
    assert out.loc[0, "Meeting_Type"] == "CCAMLR"


@pytest.mark.parametrize("title,expected", [
    ("Measure 3 (1995) - ATCM XIX, Seoul", "Measure"),
    ("Decision 1 (1997) - ATCM XXI, Christchurch", "Decision"),
    ("Resolution 2 (1995) - ATCM XIX, Seoul", "Resolution"),
    ("Recommendation I-II (ATCM I - Canberra, 1961)", "Recommendation"),
])
def test_type_is_read_from_the_title_prefix(tmp_path, title, expected):
    out = _enrich(tmp_path, [_row(title)])
    assert out.loc[0, "Type"] == expected


def test_type_is_blank_when_title_does_not_start_with_a_known_instrument(tmp_path):
    """The Type regex is anchored, so a mid-string mention does not count."""
    out = _enrich(tmp_path, [_row("Adopted by SATCM I-1 (London, 1977)")])
    assert pd.isna(out.loc[0, "Type"])


def test_post_44_meeting_without_a_year_in_title_gets_no_atcm_year(tmp_path):
    """Characterises a latent bug: the extrapolation fallback is unreachable.

    `meeting_to_year` carries an `else 2022 + (meeting_number - 44)` branch meant to
    cover meetings absent from meeting_year_dictionary.csv (which stops at 44/2022).
    But the caller only invokes it when `ATCM_Number in meeting_dict_map`, so that
    branch is dead and such a row keeps a null year instead.

    No live row hits this -- every post-44 title carries its own '(yyyy)' -- so this
    test documents the gap rather than a present-day error.
    """
    out = _enrich(tmp_path, [_row("Measure 1 - ATCM XLV, Helsinki")])
    assert out.loc[0, "ATCM_Number"] == 45
    assert pd.isna(out.loc[0, "ATCM_Year"])


def test_known_meeting_number_maps_through_the_dictionary(tmp_path):
    """A title with no '(yyyy)' but a dictionary-known ATCM number is filled in."""
    out = _enrich(tmp_path, [_row("Measure 1 - ATCM XIX, Seoul")])
    assert out.loc[0, "ATCM_Number"] == 19
    assert out.loc[0, "ATCM_Year"] == 1995

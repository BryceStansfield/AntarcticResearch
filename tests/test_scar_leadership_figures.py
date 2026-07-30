"""Tests for ScarLeadershipFigures.

The class used to hardcode `open("data/SCAR_Leadership.csv", ...)` inside
`__init__`, so it could only ever be driven over the real corpus file. It now
takes an optional `csv_path` keyword argument that defaults to that same
literal path, mirroring the `measures_df_path` pattern in
`ACTM_Measure_Scraper.src.MeasureEnricher.enrich_measure_data`: the real
caller (`antarctic_ladder_figure_aggregator.py`) constructs
`ScarLeadershipFigures()` with no arguments, so the default keeps it reading
the exact same file it always did -- nothing about the real pipeline's output
changes. Tests here pass `csv_path` explicitly to point at a tmp_path
fixture instead.
"""
import csv

import pandas as pd
import pytest

from antarctic_ladder_metrics.scar_leadership_figures import ScarLeadershipFigures


# --------------------------------------------------------------------- fixtures

def _write_scar_csv(path, data_rows, header_rows=(("Year", "Chair", "Vice-Chair"),
                                                    ("", "", ""))):
    """Write a SCAR_Leadership-shaped CSV: two header rows (content is never
    parsed, only counted and skipped) followed by `data_rows`."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in header_rows:
            writer.writerow(row)
        for row in data_rows:
            writer.writerow(row)


def _scar(tmp_path, data_rows, header_rows=(("Year", "Chair", "Vice-Chair"),
                                             ("", "", ""))):
    path = tmp_path / "SCAR_Leadership.csv"
    _write_scar_csv(path, data_rows, header_rows)
    return ScarLeadershipFigures(csv_path=str(path))


# ------------------------------------------------------------- header row skip

def test_first_two_rows_are_skipped_without_being_parsed(tmp_path):
    """The header rows hold labels, not years. If `__init__` ever tried to
    `int()` row[0] on them it would raise before reaching any data row, so a
    header whose first cell is not year-shaped is exactly what proves the
    skip-first-two-rows logic runs unconditionally rather than being masked
    by header content that happens to parse."""
    scar = _scar(tmp_path,
                 header_rows=(("Year", "Chair"), ("not-a-year", "also not one")),
                 data_rows=[["2010", "Australia"]])
    assert scar.country_dict() == {"Australia": 1}


# --------------------------------------------------------------- year filtering

@pytest.mark.parametrize("year,expect_kept", [
    (1999, False),  # just below START_YEAR
    (2000, True),   # START_YEAR itself, inclusive
    (2025, True),   # END_YEAR itself, inclusive
    (2026, False),  # just above END_YEAR
])
def test_year_range_boundaries(tmp_path, year, expect_kept):
    scar = _scar(tmp_path, data_rows=[[str(year), "Australia"]])
    if expect_kept:
        assert scar.country_dict() == {"Australia": 1}
    else:
        assert scar.country_dict() == {}


# --------------------------------------------------------- AustraliaFrance typo

def test_australia_france_typo_is_split_into_two_countries(tmp_path):
    """The source data contains the literal typo 'AustraliaFrance' (no space,
    no ampersand) for what is meant to be a joint Australia/France position.
    It is special-cased into 'Australia & France' before the normal '&'-split
    runs, so it should credit both countries once each."""
    scar = _scar(tmp_path, data_rows=[["2010", "AustraliaFrance"]])
    assert scar.country_dict() == {"Australia": 1, "France": 1}


def test_typo_substitution_only_fires_on_the_exact_literal(tmp_path):
    """A near-miss -- here with leading whitespace -- is not equal to the
    literal 'AustraliaFrance' string, so the special case must not fire. The
    cell is instead treated as a single (accidentally-concatenated) country
    name after stripping, which pins down that the check is an exact `==`
    and not a substring/fuzzy match."""
    scar = _scar(tmp_path, data_rows=[["2010", " AustraliaFrance"]])
    assert scar.country_dict() == {"AustraliaFrance": 1}


# ------------------------------------------------------------ ampersand splits

def test_ampersand_joined_cell_credits_both_countries(tmp_path):
    scar = _scar(tmp_path, data_rows=[["2010", "Australia & France"]])
    assert scar.country_dict() == {"Australia": 1, "France": 1}


def test_cell_with_no_ampersand_credits_a_single_country(tmp_path):
    scar = _scar(tmp_path, data_rows=[["2010", "Australia"]])
    assert scar.country_dict() == {"Australia": 1}


@pytest.mark.parametrize("cell", ["", "   "])
def test_empty_or_whitespace_only_cells_contribute_nothing(tmp_path, cell):
    scar = _scar(tmp_path, data_rows=[["2010", cell]])
    assert scar.country_dict() == {}


def test_multiple_cells_in_one_row_each_contribute_independently(tmp_path):
    """One year-row can carry several leadership positions across its columns
    (e.g. Chair, Vice-Chair, ...). Each cell is parsed independently, so a
    blank position column alongside populated ones should neither drop nor
    merge the populated ones."""
    scar = _scar(tmp_path, data_rows=[["2010", "Australia", "France & UK", ""]])
    assert scar.country_dict() == {"Australia": 1, "France": 1, "UK": 1}


# ------------------------------------------------------- yearly/total consistency

def test_yearly_and_total_counts_stay_consistent(tmp_path):
    """`country_counts_by_years` (keyed by year+country, dumped by
    `save_full_figures`) and `_counts` (keyed by country only, returned by
    `country_dict`) are built from the same loop but summed separately --
    this pins down that the per-year breakdown always sums back to the
    totals rather than the two bookkeeping structures drifting apart."""
    scar = _scar(tmp_path, data_rows=[
        ["2010", "Australia", "France & UK"],
        ["2011", "Australia"],
        ["2012", "France"],
    ])

    out_path = tmp_path / "out.csv"
    scar.save_full_figures(str(out_path))
    yearly = pd.read_csv(out_path)

    summed_from_yearly = {country: int(total)
                          for country, total in yearly.groupby("country")["value"].sum().items()}
    assert summed_from_yearly == scar.country_dict()
    assert scar.country_dict() == {"Australia": 2, "France": 2, "UK": 1}


# ------------------------------------------------------------------ misc surface

def test_figure_title_is_constant(tmp_path):
    scar = _scar(tmp_path, data_rows=[])
    assert scar.figure_title() == "Scar Leadership Positions"

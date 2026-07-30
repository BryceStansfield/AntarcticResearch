"""Tests for `FacilityFigures` and `VesselCrewFigures`.

Both classes used to build their CSV path inline in `__init__` from
`pathlib.Path(__file__).parent.parent / "data" / ...`, which meant exercising them
meant reading the real `data/` CSVs. `infrastructure_figures.py` now accepts an
optional `facilities_path` / `vessels_path` argument that defaults to that same
expression -- real callers (`FacilityFigures()`, `VesselCrewFigures()`, as used by
`if __name__ == "__main__"` and anywhere else in the pipeline) are unaffected, but a
temp-file CSV can be driven through instead here.

`take_first_figure` was lifted out of `VesselCrewFigures.__init__` (where it was a
nested closure) to module level with its body unchanged, so it can be tested
directly rather than only indirectly through pandas' dtype inference.

Every fixture's Passenger/Crew and Peak Population columns carry at least one blank
or dash-string cell somewhere in the file, so those columns round-trip through
`read_csv` as the same dtype the real (blank- and dash-containing) corpus produces,
rather than an all-numeric dtype the real data never actually has.
"""
import pandas as pd
import pytest

from antarctic_ladder_metrics.infrastructure_figures import (
    FacilityFigures,
    VesselCrewFigures,
    take_first_figure,
)


# --------------------------------------------------------------- take_first_figure

@pytest.mark.parametrize("value", [40, 40.0, 0])
def test_take_first_figure_passes_plain_numbers_through_unchanged(value):
    assert take_first_figure(value) == value


def test_take_first_figure_takes_the_substring_before_the_first_dash():
    """Mirrors the Ukrainian vessel 'Noosfera', which cites two figures for maximum
    capacity separated by a dash; the first is taken."""
    assert take_first_figure("40-60") == "40"


def test_take_first_figure_strips_whitespace_around_a_plain_string_value():
    assert take_first_figure(" 40 ") == "40"


# --------------------------------------------------------------------- FacilityFigures

def _write_facilities(path, rows):
    columns = ["Operator (primary)", "Operator (additional)", "Peak Population", "Seasonality"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _facilities(tmp_path, rows):
    path = tmp_path / "facilities.csv"
    _write_facilities(path, rows)
    return FacilityFigures(facilities_path=path)


def _facility_row(primary, additional, population, seasonality="Year-Round"):
    return {"Operator (primary)": primary, "Operator (additional)": additional,
            "Peak Population": population, "Seasonality": seasonality}


def test_operator_additional_present_splits_75_25(tmp_path):
    figures = _facilities(tmp_path, [_facility_row("USA", "UK", "1,200")])
    assert figures.country_dict() == {"USA": 900.0, "UK": 300.0}


def test_operator_additional_blank_gives_full_credit_to_primary(tmp_path):
    """A genuinely blank CSV cell is read by pandas as float NaN, and
    `str(nan) == 'nan'` -- that is what the `!= "nan"` check in `__init__` actually
    keys off, not some explicit missing-value marker in the source data."""
    figures = _facilities(tmp_path, [_facility_row("Argentina", "", "1,500")])
    assert figures.country_dict() == {"Argentina": 1500}


def test_seasonal_facility_gets_quarter_weight_before_the_75_25_split(tmp_path):
    """The 0.25 seasonal multiplier is applied to Peak Population first, and the
    75/25 operator split is taken on that already-adjusted figure, not on the raw
    Peak Population -- so a seasonal facility's operators split 750/250 here, not
    3000/1000."""
    figures = _facilities(tmp_path, [_facility_row("Chile", "Peru", "4,000", "Seasonal")])
    assert figures.country_dict() == {"Chile": 750.0, "Peru": 250.0}


def test_comma_separated_peak_population_is_parsed_as_an_int(tmp_path):
    figures = _facilities(tmp_path, [_facility_row("Norway", "", "1,234")])
    assert figures.country_dict() == {"Norway": 1234}


def test_blank_peak_population_defaults_to_zero(tmp_path):
    """The second row keeps "Peak Population" at a string dtype across the whole
    fixture (see the module docstring) so the blank in the first row round-trips the
    same way it does in the real, comma-bearing corpus; it plays no other role."""
    figures = _facilities(tmp_path, [
        _facility_row("France", "", ""),
        _facility_row("Iceland", "", "1,000"),
    ])
    assert figures.country_dict() == {"France": 0, "Iceland": 1000}


def test_literal_lowercase_nan_string_is_also_treated_as_absent(tmp_path):
    """Characterises current behaviour: writing the literal text "nan" (rather than
    leaving the cell blank) reaches the same "absent" branch, but not because of the
    explicit `str(row[...]) != "nan"` check -- pandas' own default `na_values` list
    already includes the lowercase string "nan", so `read_csv` turns it into float
    NaN before that check ever runs. This confirms the two mechanisms agree for this
    exact spelling; the next test shows a spelling where they do not.
    """
    figures = _facilities(tmp_path, [_facility_row("Chile", "nan", "1,000")])
    assert figures.country_dict() == {"Chile": 1000}


def test_literal_uppercase_NAN_string_is_wrongly_treated_as_a_real_operator(tmp_path):
    """Characterises current behaviour -- a real (if obscure) data-entry wart.

    Pandas' default `na_values` list includes the lowercase "nan" and the title-case
    "NaN", but not the all-uppercase "NAN". So an all-caps "NAN" typed into
    "Operator (additional)" survives `read_csv` as the literal string "NAN", and the
    `str(row[...]) != "nan"` check in `__init__` is a case-sensitive comparison
    against the lowercase spelling only -- so "NAN" is treated as a genuine (if
    oddly named) additional operator and silently receives 25% of the facility's
    population instead of being ignored.
    """
    figures = _facilities(tmp_path, [_facility_row("Chile", "NAN", "1,200")])
    assert figures.country_dict() == {"Chile": 900.0, "NAN": 300.0}


# ------------------------------------------------------------------- VesselCrewFigures

def _write_vessels(path, rows):
    columns = ["Country", "Status", "Maximum Passenger", "Maximum Crew"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _vessels(tmp_path, rows):
    path = tmp_path / "vessels.csv"
    _write_vessels(path, rows)
    return VesselCrewFigures(vessels_path=path)


def _vessel_row(country, status, passenger, crew):
    return {"Country": country, "Status": status,
            "Maximum Passenger": passenger, "Maximum Crew": crew}


def test_status_filter_excludes_non_in_service_vessels(tmp_path):
    """The "Laid Up" row's large figures are there so the assertion would visibly
    fail (2020 instead of 20) if the Status filter stopped working; the third row
    (Peru, blank Passenger/Crew) exists only to keep both columns at float64 dtype
    across the fixture (see the module docstring) and is not itself the point of
    this test.
    """
    figures = _vessels(tmp_path, [
        _vessel_row("Chile", "In Service", 10, 10),
        _vessel_row("Chile", "Laid Up", 1000, 1000),
        _vessel_row("Peru", "In Service", "", ""),
    ])
    assert figures.country_dict() == {"Chile": 20, "Peru": 0}


def test_dash_style_capacity_is_reduced_to_its_first_figure(tmp_path):
    """End-to-end regression for the real Noosfera row: "40-60" contributes 40, not
    60, and does not raise. The second row's blank Crew cell keeps that column at
    float64 dtype (see the module docstring) and doubles as this test's coverage of
    blank-crew-defaults-to-zero.
    """
    figures = _vessels(tmp_path, [
        _vessel_row("Ukraine", "In Service", "40-60", 20),
        _vessel_row("Ukraine", "In Service", 5, ""),
    ])
    assert figures.country_dict() == {"Ukraine": 65}


def test_blank_passenger_and_crew_each_default_to_zero(tmp_path):
    figures = _vessels(tmp_path, [
        _vessel_row("Norway", "In Service", 5, ""),
        _vessel_row("Norway", "In Service", "", 8),
    ])
    assert figures.country_dict() == {"Norway": 13}

"""Tests for `InformationPaperAuthorship` / `compute_ip_country_authorships`.

`compute_ip_country_authorships` is the aggregation core: filter to ATCM information
papers, restrict to a year window, drop duplicate paper_ids, then credit each party
on each surviving paper's `parties` list to (year, country) and to country totals.
It was lifted out of `InformationPaperAuthorship.__init__` (which used to read the
real parquet and run this inline) into a function taking the raw dataframe plus
start/end year, mirroring `enrich_measure_data` in
ACTM_Measure_Scraper/src/MeasureEnricher.py -- paths/data go in as arguments so the
logic can be driven over a small synthetic frame instead of the real corpus.

`InformationPaperAuthorship.__init__` itself is now just "read this parquet path,
then delegate"; it is covered separately, once, via a tmp_path parquet fixture, to
confirm the wiring rather than re-testing the aggregation logic.
"""
import pandas as pd
import pytest

from antarctic_ladder_metrics.information_paper_metrics import (
    InformationPaperAuthorship, compute_ip_country_authorships)


def _row(parties, year, paper_id, meeting_type="ATCM", party_type="ip"):
    return {"meeting_type": meeting_type, "party_type": party_type,
            "parties": parties, "meeting_year": year, "paper_id": paper_id}


def _table(rows):
    return pd.DataFrame(rows)


# ------------------------------------------------------- compute_ip_country_authorships

def test_multiple_parties_on_one_paper_are_each_credited_independently():
    table = _table([_row(["Australia", "Chile"], 2001, "ip1")])
    yearly, total = compute_ip_country_authorships(table, 2000, 2002)
    assert yearly[(2001, "australia")] == 1
    assert yearly[(2001, "chile")] == 1
    assert total == {"australia": 1, "chile": 1}


def test_country_name_normalization_is_applied():
    """'UK' is an alias handled by country_meta_info.normalize_country_name; if the
    aggregation forgot to call it, this would land under the literal string 'uk'
    instead of merging with the rest of the United Kingdom's papers."""
    table = _table([_row(["UK"], 2001, "ip1"), _row(["United Kingdom"], 2001, "ip2")])
    yearly, total = compute_ip_country_authorships(table, 2000, 2002)
    assert yearly[(2001, "united kingdom")] == 2
    assert total == {"united kingdom": 2}


@pytest.mark.parametrize("meeting_type,party_type,should_count", [
    ("ATCM", "ip", True),
    ("CEP", "ip", False),   # wrong meeting type
    ("ATCM", "wp", False),  # wrong party type -- working papers, not information papers
])
def test_filters_to_atcm_information_papers_only(meeting_type, party_type, should_count):
    table = _table([_row(["Chile"], 2001, "ip1", meeting_type=meeting_type, party_type=party_type)])
    _, total = compute_ip_country_authorships(table, 2000, 2002)
    assert total == ({"chile": 1} if should_count else {})


@pytest.mark.parametrize("year,should_count", [
    (2000, True),   # start boundary, inclusive
    (2025, True),   # end boundary, inclusive
    (1999, False),  # just before the window
    (2026, False),  # just after the window
])
def test_year_window_is_a_closed_interval(year, should_count):
    table = _table([_row(["Chile"], year, "ip1")])
    _, total = compute_ip_country_authorships(table, 2000, 2025)
    assert total == ({"chile": 1} if should_count else {})


def test_duplicate_paper_id_keeps_only_the_first_row():
    """A paper_id repeated across rows (e.g. re-listed under a later meeting) must
    only be credited once, and to whichever parties its first occurrence carries --
    matching pandas' drop_duplicates(keep='first')."""
    table = _table([
        _row(["Australia"], 2001, "dup"),
        _row(["Chile"], 2001, "dup"),
    ])
    yearly, total = compute_ip_country_authorships(table, 2000, 2002)
    assert total == {"australia": 1}
    assert (2001, "chile") not in yearly


def test_a_year_with_zero_rows_is_absent_rather_than_zero():
    """The window is iterated year by year regardless of data, so an empty year must
    not raise and must not leave a spurious zero-valued entry -- it should simply
    contribute nothing to either dict."""
    table = _table([_row(["Chile"], 2001, "ip1")])
    yearly, total = compute_ip_country_authorships(table, 2000, 2003)
    assert not any(year == 2000 for year, _ in yearly)
    assert not any(year == 2003 for year, _ in yearly)
    assert total == {"chile": 1}


def test_empty_table_over_the_whole_window_yields_empty_dicts():
    table = _table([])
    # An empty frame still needs the expected columns for the boolean filters to run.
    table = table.reindex(columns=["meeting_type", "party_type", "parties",
                                    "meeting_year", "paper_id"])
    yearly, total = compute_ip_country_authorships(table, 2000, 2001)
    assert yearly == {}
    assert total == {}


def test_yearly_breakdown_sums_to_the_country_totals():
    """The invariant `save_full_figures` relies on: summing the yearly dict's values
    for a country must equal that country's entry in the total dict, i.e. the total
    is nothing more than the yearly breakdown collapsed across years."""
    table = _table([
        _row(["Chile"], 2000, "ip1"),
        _row(["Chile", "Australia"], 2001, "ip2"),
        _row(["Chile"], 2001, "ip3"),
    ])
    yearly, total = compute_ip_country_authorships(table, 2000, 2001)
    for country in total:
        assert sum(v for (y, c), v in yearly.items() if c == country) == total[country]
    assert total == {"chile": 3, "australia": 1}


def test_same_country_across_different_years_accumulates_separately_then_together():
    table = _table([
        _row(["Chile"], 2000, "ip1"),
        _row(["Chile"], 2001, "ip2"),
    ])
    yearly, total = compute_ip_country_authorships(table, 2000, 2001)
    assert yearly[(2000, "chile")] == 1
    assert yearly[(2001, "chile")] == 1
    assert total == {"chile": 2}


# --------------------------------------------------------------- InformationPaperAuthorship

def test_init_reads_the_given_parquet_path_and_delegates(tmp_path):
    """Confirms __init__ is wired correctly (reads parquet_path, passes start/end year
    through) rather than re-testing the aggregation logic covered above."""
    src = tmp_path / "document-summary.parquet"
    table = _table([
        _row(["Chile"], 2001, "ip1"),
        _row(["Australia"], 1999, "ip-out-of-range"),
    ])
    table.to_parquet(src)

    ip = InformationPaperAuthorship(parquet_path=str(src), start_year=2000, end_year=2002)

    assert ip.country_dict() == {"chile": 1}
    assert ip.figure_title() == "Information Paper Authorship"


def test_save_full_figures_breakdown_sums_to_country_dict(tmp_path):
    src = tmp_path / "document-summary.parquet"
    out = tmp_path / "figures.csv"
    table = _table([
        _row(["Chile"], 2000, "ip1"),
        _row(["Chile", "Australia"], 2001, "ip2"),
    ])
    table.to_parquet(src)

    ip = InformationPaperAuthorship(parquet_path=str(src), start_year=2000, end_year=2001)
    ip.save_full_figures(out)

    figures = pd.read_csv(out)
    summed = figures.groupby("country")["value"].sum().to_dict()
    assert summed == ip.country_dict()

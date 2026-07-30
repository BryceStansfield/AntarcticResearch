"""Tests for `WorkingPaperAuthorship`, `WPCollaborationGraphCentrality` and
`WPCollaborationDiversity` in antarctic_ladder_metrics/working_paper_metrics.py.

Two pure aggregation cores were lifted out of `__init__` bodies that used to read the
real parquet and run the logic inline, mirroring the same treatment already applied to
`InformationPaperAuthorship` (see tests/test_information_paper_metrics.py) and to
`enrich_measure_data` in ACTM_Measure_Scraper/src/MeasureEnricher.py:

- `compute_wp_country_authorships(table, start_year, end_year)` -- the filter/dedup/
  credit logic that used to live in `WorkingPaperAuthorship.__init__`. `__init__` is now
  "read this parquet path, then delegate", covered separately via a tmp_path fixture.
- `compute_wp_collaboration_diversity(author_sets)` -- the "count each country's
  distinct collaborators" loop that used to live in `WPCollaborationDiversity.__init__`.
  `__init__` still reads the real parquet path directly (it was not given a path
  parameter, since only the pure counting logic needed extracting) and so is not
  covered end to end here -- see "What is not covered" at the bottom of this file.

`WPCollaborationGraphCentrality` was NOT refactored at all. `_graph_within` and
`_centrality_within` were already separate methods that only depend on
`self._author_sets_by_year` (a plain list of `(meeting_year, parties_list)` tuples,
built by `__init__` from the parquet). Since that attribute can be set directly, tests
below construct instances via `object.__new__(WPCollaborationGraphCentrality)` and
assign `_author_sets_by_year` by hand, then call the two methods directly -- this
touches zero production code and needed no code changes.
"""
import pandas as pd
import pytest

from antarctic_ladder_metrics.working_paper_metrics import (
    WorkingPaperAuthorship,
    WPCollaborationGraphCentrality,
    compute_wp_country_authorships,
    compute_wp_collaboration_diversity,
)


def _row(parties, year, paper_id, meeting_type="ATCM", party_type="wp"):
    return {"meeting_type": meeting_type, "party_type": party_type,
            "parties": parties, "meeting_year": year, "paper_id": paper_id}


def _table(rows):
    return pd.DataFrame(rows)


# ------------------------------------------------------- compute_wp_country_authorships

def test_multiple_parties_on_one_paper_are_each_credited_independently():
    table = _table([_row(["Australia", "Chile"], 2001, "wp1")])
    yearly, total = compute_wp_country_authorships(table, 2000, 2002)
    assert yearly[(2001, "australia")] == 1
    assert yearly[(2001, "chile")] == 1
    assert total == {"australia": 1, "chile": 1}


def test_country_name_normalization_is_applied():
    """'UK' is an alias handled by country_meta_info.normalize_country_name; if the
    aggregation forgot to call it, this would land under the literal string 'uk'
    instead of merging with the rest of the United Kingdom's papers."""
    table = _table([_row(["UK"], 2001, "wp1"), _row(["United Kingdom"], 2001, "wp2")])
    yearly, total = compute_wp_country_authorships(table, 2000, 2002)
    assert yearly[(2001, "united kingdom")] == 2
    assert total == {"united kingdom": 2}


@pytest.mark.parametrize("meeting_type,party_type,should_count", [
    ("ATCM", "wp", True),
    ("CEP", "wp", False),   # wrong meeting type
    ("ATCM", "ip", False),  # wrong party type -- information papers, not working papers
])
def test_filters_to_atcm_working_papers_only(meeting_type, party_type, should_count):
    table = _table([_row(["Chile"], 2001, "wp1", meeting_type=meeting_type, party_type=party_type)])
    _, total = compute_wp_country_authorships(table, 2000, 2002)
    assert total == ({"chile": 1} if should_count else {})


@pytest.mark.parametrize("year,should_count", [
    (2000, True),   # start boundary, inclusive
    (2025, True),   # end boundary, inclusive
    (1999, False),  # just before the window
    (2026, False),  # just after the window
])
def test_year_window_is_a_closed_interval(year, should_count):
    table = _table([_row(["Chile"], year, "wp1")])
    _, total = compute_wp_country_authorships(table, 2000, 2025)
    assert total == ({"chile": 1} if should_count else {})


def test_duplicate_paper_id_keeps_only_the_first_row():
    """A paper_id repeated across rows must only be credited once, and to whichever
    parties its first occurrence carries -- matching pandas'
    drop_duplicates(keep='first')."""
    table = _table([
        _row(["Australia"], 2001, "dup"),
        _row(["Chile"], 2001, "dup"),
    ])
    yearly, total = compute_wp_country_authorships(table, 2000, 2002)
    assert total == {"australia": 1}
    assert (2001, "chile") not in yearly


def test_a_year_with_zero_rows_is_absent_rather_than_zero():
    """The window is iterated year by year regardless of data, so an empty year must
    not raise and must not leave a spurious zero-valued entry."""
    table = _table([_row(["Chile"], 2001, "wp1")])
    yearly, total = compute_wp_country_authorships(table, 2000, 2003)
    assert not any(year == 2000 for year, _ in yearly)
    assert not any(year == 2003 for year, _ in yearly)
    assert total == {"chile": 1}


def test_empty_table_over_the_whole_window_yields_empty_dicts():
    table = _table([])
    # An empty frame still needs the expected columns for the boolean filters to run.
    table = table.reindex(columns=["meeting_type", "party_type", "parties",
                                    "meeting_year", "paper_id"])
    yearly, total = compute_wp_country_authorships(table, 2000, 2001)
    assert yearly == {}
    assert total == {}


def test_yearly_breakdown_sums_to_the_country_totals():
    """The invariant `save_full_figures` relies on: summing the yearly dict's values
    for a country must equal that country's entry in the total dict."""
    table = _table([
        _row(["Chile"], 2000, "wp1"),
        _row(["Chile", "Australia"], 2001, "wp2"),
        _row(["Chile"], 2001, "wp3"),
    ])
    yearly, total = compute_wp_country_authorships(table, 2000, 2001)
    for country in total:
        assert sum(v for (y, c), v in yearly.items() if c == country) == total[country]
    assert total == {"chile": 3, "australia": 1}


def test_same_country_across_different_years_accumulates_separately_then_together():
    table = _table([
        _row(["Chile"], 2000, "wp1"),
        _row(["Chile"], 2001, "wp2"),
    ])
    yearly, total = compute_wp_country_authorships(table, 2000, 2001)
    assert yearly[(2000, "chile")] == 1
    assert yearly[(2001, "chile")] == 1
    assert total == {"chile": 2}


# --------------------------------------------------------------- WorkingPaperAuthorship

def test_init_reads_the_given_parquet_path_and_delegates(tmp_path):
    """Confirms __init__ is wired correctly (reads parquet_path, passes start/end year
    through) rather than re-testing the aggregation logic covered above."""
    src = tmp_path / "document-summary.parquet"
    table = _table([
        _row(["Chile"], 2001, "wp1"),
        _row(["Australia"], 1999, "wp-out-of-range"),
    ])
    table.to_parquet(src)

    wp = WorkingPaperAuthorship(parquet_path=str(src), start_year=2000, end_year=2002)

    assert wp.country_dict() == {"chile": 1}
    assert wp.figure_title() == "Working Paper Authorship"


def test_save_full_figures_breakdown_sums_to_country_dict(tmp_path):
    src = tmp_path / "document-summary.parquet"
    out = tmp_path / "figures.csv"
    table = _table([
        _row(["Chile"], 2000, "wp1"),
        _row(["Chile", "Australia"], 2001, "wp2"),
    ])
    table.to_parquet(src)

    wp = WorkingPaperAuthorship(parquet_path=str(src), start_year=2000, end_year=2001)
    wp.save_full_figures(out)

    figures = pd.read_csv(out)
    summed = figures.groupby("country")["value"].sum().to_dict()
    assert summed == wp.country_dict()


# ------------------------------------------------------------- WPCollaborationGraphCentrality
#
# Not refactored: instances are built with object.__new__ and _author_sets_by_year is
# set by hand, so these tests exercise _graph_within/_centrality_within directly
# without touching the parquet file or any other production code path.

def _centrality_instance(author_sets_by_year):
    instance = object.__new__(WPCollaborationGraphCentrality)
    instance._author_sets_by_year = author_sets_by_year
    return instance


def test_graph_within_merges_edges_regardless_of_party_order_across_papers():
    """(A, B) on one paper and (B, A) on another must land on the same edge, since
    _graph_within canonicalizes each pair via `sorted` (done manually with a `>`
    comparison) before keying edge_weights."""
    instance = _centrality_instance([
        (2001, ["brazil", "australia"]),
        (2002, ["australia", "brazil"]),
    ])
    party_set, edge_weights = instance._graph_within()
    assert party_set == {"australia", "brazil"}
    assert edge_weights == {("australia", "brazil"): 2}


def test_graph_within_respects_the_year_window():
    instance = _centrality_instance([
        (2000, ["australia", "brazil"]),
        (2010, ["australia", "chile"]),
    ])
    party_set, edge_weights = instance._graph_within(2005, 2015)
    assert party_set == {"australia", "chile"}
    assert edge_weights == {("australia", "chile"): 1}


def test_centrality_within_empty_window_returns_empty_dict():
    """See the "An empty window has no graph to run centrality over" comment in
    _centrality_within: an out-of-range window must short-circuit to {} rather than
    handing networkx an empty node set."""
    instance = _centrality_instance([(2001, ["australia", "brazil"])])
    assert instance._centrality_within(2050, 2060) == {}

    party_set, edge_weights = instance._graph_within(2050, 2060)
    assert party_set == set()
    assert edge_weights == {}


def test_centrality_is_deterministic_across_repeated_calls():
    """_centrality_within rebuilds a fresh networkx.Graph and reruns
    katz_centrality_numpy every call. Node insertion order (country name, descending)
    and edge insertion order (pair, descending) are fixed specifically so that this is
    reproducible; two calls over the same data must agree exactly, not just
    approximately."""
    instance = _centrality_instance([
        (2001, ["australia", "brazil", "chile"]),
        (2002, ["brazil", "chile"]),
        (2003, ["australia", "chile"]),
    ])
    first = instance._centrality_within()
    second = instance._centrality_within()
    assert first == second


def test_edge_weight_normalization_only_depends_on_relative_ratios():
    """Two graphs whose raw co-authorship counts differ in absolute scale but share
    the same ratio between edges must normalize (via division by the max edge weight)
    to the same weights, and therefore produce the same Katz centrality. This isolates
    the "Normalizing our graph decreases our eigenvalues" step from the raw counts
    computed by _graph_within."""
    small = _centrality_instance([
        (2001, ["australia", "brazil"]),   # australia-brazil: weight 1
        (2002, ["australia", "chile"]),
        (2003, ["australia", "chile"]),    # australia-chile: weight 2
    ])
    large = _centrality_instance([
        (2001, ["australia", "brazil"]),
        (2002, ["australia", "brazil"]),   # australia-brazil: weight 2
        (2003, ["australia", "chile"]),
        (2004, ["australia", "chile"]),
        (2005, ["australia", "chile"]),
        (2006, ["australia", "chile"]),    # australia-chile: weight 4
    ])

    small_centrality = small._centrality_within()
    large_centrality = large._centrality_within()

    assert small_centrality.keys() == large_centrality.keys()
    for country in small_centrality:
        assert small_centrality[country] == pytest.approx(large_centrality[country])


def test_full_window_graph_matches_the_unrestricted_centrality_input():
    """`__init__` sets self.centrality = self._centrality_within() (no bounds), and
    save_collaboration_graphs's "Full" window calls _graph_within(None, None). Both
    must be built from the exact same (unfiltered) graph, since the point of the
    Full-window edge-list export is to let readers inspect the graph that produced
    the unrestricted centrality figures."""
    instance = _centrality_instance([
        (2001, ["australia", "chile"]),
        (2010, ["chile", "argentina"]),
    ])
    instance.centrality = instance._centrality_within()  # mirrors __init__

    assert instance._graph_within() == instance._graph_within(None, None)
    assert instance.centrality == instance._centrality_within(None, None)


def test_centrality_values_are_rounded_to_ten_decimal_places():
    """`katz_centrality_numpy`'s underlying linear solve is not guaranteed
    bit-identical run-to-run (BLAS summation order), so without rounding, two
    otherwise-identical runs over the same corpus can differ in the last couple of
    significant digits. Rounding here keeps repeated pipeline runs comparable."""
    instance = _centrality_instance([
        (2001, ["australia", "brazil", "chile"]),
        (2002, ["brazil", "chile"]),
    ])
    for value in instance._centrality_within().values():
        assert value == round(value, 10)


# ----------------------------------------------------------- compute_wp_collaboration_diversity

def test_a_country_never_collaborates_with_itself():
    """The `i != j` guard: a lone author on a paper contributes no self-edge, so it
    must not appear in the resulting diversity dict at all (there is nothing to count
    it as having collaborated with)."""
    diversity = compute_wp_collaboration_diversity([["chile"]])
    assert "chile" not in diversity


def test_repeated_co_authorship_between_the_same_two_parties_counts_once():
    """Two papers by the same pair of countries must not double the diversity count:
    collaborators are tracked in a set, not a running total."""
    diversity = compute_wp_collaboration_diversity([
        ["chile", "argentina"],
        ["chile", "argentina"],
    ])
    assert diversity == {"chile": 1, "argentina": 1}


def test_diversity_accumulates_distinct_collaborators_across_papers():
    diversity = compute_wp_collaboration_diversity([
        ["chile", "argentina"],
        ["chile", "brazil"],
    ])
    assert diversity == {"chile": 2, "argentina": 1, "brazil": 1}


def test_three_way_paper_credits_every_pair_symmetrically():
    diversity = compute_wp_collaboration_diversity([["australia", "brazil", "chile"]])
    assert diversity == {"australia": 2, "brazil": 2, "chile": 2}


def test_collaboration_with_non_country_entities_is_counted():
    """Documents current, intentional behaviour (see the "Note this includes
    collaboration with agencies" comment carried over from the original inline code):
    nothing filters non-country party strings out, so an agency counts as a
    collaborator like any country would."""
    diversity = compute_wp_collaboration_diversity([["chile", "some agency"]])
    assert diversity == {"chile": 1, "some agency": 1}


def test_empty_author_sets_yields_an_empty_diversity_dict():
    assert compute_wp_collaboration_diversity([]) == {}


# What is not covered, and why
# -----------------------------
# WPCollaborationDiversity.__init__ still reads
# "data/antarctic-db/processed/document-summary.parquet" directly and was not given a
# path parameter (only the pure counting logic needed extracting to become testable),
# so instantiating the class still means running against the real corpus. It is not
# covered here, matching the existing convention for RatificationSpeed.__init__
# described in tests/README.md.

"""Tests for the measure-status year rule used to censor unratified measures.

`extract_end_year` is nested inside `RatificationSpeed.__init__`, which calls
`scrape_and_enrich_measures` and reads the whole corpus, so it cannot be imported.
Rather than restate the rule here -- a copy would drift silently and leave these
tests passing against a broken implementation -- the function's AST is lifted
straight out of the source file and executed in isolation. So these tests exercise
the real code, with no network and no corpus load.

When the parser is lifted to module level, replace `_real_extract_end_year` with a
plain import; the test bodies stay as they are.
"""
import ast
import pathlib
import re

import pandas as pd
import pytest

SOURCE = pathlib.Path("antarctic_ladder_metrics/ratification_speed.py")
END_YEAR_SENTINEL = 2025


def _real_extract_end_year(end_year=END_YEAR_SENTINEL):
    """Compile the actual `extract_end_year` out of ratification_speed.py.

    Its only free names are `re` and `END_YEAR`, both supplied here, so it runs
    standalone without importing the module (which would scrape and load the corpus).
    """
    tree = ast.parse(SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "extract_end_year":
            namespace = {"re": re, "END_YEAR": end_year}
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         str(SOURCE), "exec"), namespace)
            return namespace["extract_end_year"]
    raise AssertionError(f"extract_end_year not found in {SOURCE}")


extract_end_year = _real_extract_end_year()


def test_the_function_was_actually_located():
    """If the parser gets renamed or moved, fail loudly here rather than silently
    testing nothing."""
    assert callable(extract_end_year)
    assert extract_end_year("Not yet effective") == END_YEAR_SENTINEL


@pytest.mark.parametrize("status,expected", [
    # Right-censored at the window edge.
    ("Not yet effective", END_YEAR_SENTINEL),
    # Plain effective date.
    ("Effective 11/05/2016", 2016),
    ("Effective 06/10/2009", 2009),
    # Withdrawn or superseded: the parenthesised year closes the window.
    ("Did not enter into effect. No longer current:M 3 (2003)", 2003),
    ("Did not enter into effect. Withdrawn:M 3 (2012)", 2012),
    ("Did not enter into effect. Withdrawn:M 10 (2008)", 2008),
])
def test_each_status_shape_in_the_qualifying_population_parses(status, expected):
    """These six shapes are the ones that actually reach the metric."""
    assert extract_end_year(status) == expected


def test_surrounding_whitespace_is_tolerated():
    assert extract_end_year("  Effective 11/05/2016  ") == 2016


@pytest.mark.parametrize("status,expected", [
    ("Effective 30/04/1962. No longer current:D 1 (2014)", 2014),
    ("Effective 30/04/1962. Spent:D 3 (2002)", 2002),
    ("Effective 28/04/2005. No longer current:M 4 (2010)", 2010),
])
def test_trailing_year_takes_precedence_over_the_effective_date(status, expected):
    """A status can carry both dates. Precedence goes to the parenthesised year
    because it closes the ratification window later than the effective date does.

    This is a judgement call, not a derived fact -- but it is unobservable in the
    ladder: no status in the qualifying population carries both, which
    `test_precedence_choice_moves_no_current_figure` below pins down. These 91 rows
    are Recommendations, excluded by the Type filter.
    """
    assert extract_end_year(status) == expected


@pytest.mark.parametrize("status,expected", [
    ("Effective 19/12/2002 (Fast Approval)", 2002),
    ("Effective 01/12/2025 (Fast Approval)", 2025),
    ("Effective 28/10/2024 (Fast Approval)", 2024),
])
def test_a_parenthesised_label_is_not_mistaken_for_a_year(status, expected):
    """26 distinct live Measure statuses end in "(Fast Approval)".

    Requiring four digits inside the parentheses is what keeps these from being read
    as a year; the effective date is used instead. These measures never reach the
    metric -- fast approval means the instrument entered into effect wholesale, with
    no per-country ratification to time -- so this is robustness, not a live path.
    """
    assert extract_end_year(status) == expected


@pytest.mark.parametrize("status", [
    "Effective 01/11/1982. Terminated:ATCM VIII-2",
    "Effective 01/11/1982. Terminated:ATCM XV-7",
])
def test_a_non_numeric_termination_falls_back_to_the_effective_date(status):
    """Terminated at a named meeting with no year, so the effective date is the only
    date available."""
    assert extract_end_year(status) == 1982


@pytest.mark.parametrize("status", [
    "Some entirely new status wording",
    "Withdrawn at some point",
    "",
])
def test_an_unrecognised_status_raises_rather_than_guessing(status):
    with pytest.raises(ValueError, match="Cannot extract year"):
        extract_end_year(status)


# --------------------------------------------------------------- corpus-wide guards

def _measure_statuses():
    try:
        corpus = pd.read_csv("data/MeasureCorpusEnriched.csv")
    except FileNotFoundError:
        pytest.skip("data/MeasureCorpusEnriched.csv not built")
    statuses = corpus[corpus["Type"] == "Measure"]["Status"].dropna().unique()
    assert len(statuses) > 0, "no Measure statuses found -- fixture assumption broke"
    return corpus, statuses


def test_every_measure_status_in_the_live_corpus_parses():
    """Guards against a new status shape landing in the corpus unnoticed -- it fails
    here rather than part-way through a ladder run."""
    _, statuses = _measure_statuses()
    unparseable = []
    for status in statuses:
        try:
            extract_end_year(status)
        except ValueError:
            unparseable.append(status)
    assert unparseable == [], f"statuses the rule cannot parse: {unparseable}"


def test_precedence_choice_moves_no_current_figure():
    """The rule change must leave every reachable status parsing identically.

    Compares against the previous implementation over exactly the population
    RatificationSpeed keeps. Agreement here is what makes the change robustness-only.
    """
    corpus, _ = _measure_statuses()
    qualifying = corpus[(corpus["Meeting_Type"] == "ATCM")
                        & (corpus["ATCM_Year"] >= 1995)
                        & (corpus["Type"] == "Measure")
                        & ~corpus["Approvals"].str.contains("Fast Approval", na=False)
                        & (corpus["Approvals"] != "")
                        & ~corpus["Approvals"].isna()]

    def previous_rule(status):
        if status == "Not yet effective":
            return END_YEAR_SENTINEL
        if "Effective" in status:
            return int(status[-4:])
        if status.endswith(')'):
            return int(status[status.rfind('(') + 1:-1])
        raise ValueError(status)

    statuses = qualifying["Status"].dropna().unique()
    assert len(statuses) > 0, "qualifying population is empty -- filter assumption broke"
    for status in statuses:
        assert extract_end_year(status) == previous_rule(status), status


def test_fast_approval_measures_carry_no_per_country_dates():
    """Documents why fast-approval measures are excluded, and why that exclusion is
    definitional rather than incidental.

    Their Approvals field is a single prose line ("Fast Approval: Entered into effect
    on ...") with no country/date table at all, so there is no ratification interval
    to measure for any country. If this ever stops holding, the exclusion in
    RatificationSpeed needs revisiting.
    """
    corpus, _ = _measure_statuses()
    fast = corpus[(corpus["Type"] == "Measure")
                  & corpus["Approvals"].astype(str).str.contains("Fast Approval",
                                                                 na=False)]
    assert len(fast) > 0, "no fast-approval measures found -- fixture assumption broke"
    dated = fast["Approvals"].astype(str).str.contains(r"\d{2}/\d{2}/\d{4}", regex=True)
    assert not dated.any(), (
        f"{int(dated.sum())} fast-approval measures now list per-country dates")

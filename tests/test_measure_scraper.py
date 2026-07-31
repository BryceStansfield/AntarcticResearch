"""Tests for `MeasureScraper.parse_topics`.

The topic cell used to be split on every hyphen, which shattered hyphenated topic names: the live
corpus ended up with 'De' (15x), 'Multi' (12x), 'Non' (5x) and 'based activities' (11x) as
first-class topic labels. These pin whole names surviving.

The exact markup of the ATS topic span could not be checked (that would mean hitting the live site),
so the parser is written to work whether the topics are separate child nodes or one joined string —
both shapes are covered below.
"""
from bs4 import BeautifulSoup

from ACTM_Measure_Scraper.src.MeasureScraper import parse_topics


def _span(html):
    return BeautifulSoup(html, "html.parser").span


def test_splits_topics_that_are_separate_child_nodes():
    got = parse_topics(_span(
        "<span>Tourism Management<span> - </span>Rules of Procedure</span>"))
    assert got == ["Tourism Management", "Rules of Procedure"]


def test_splits_topics_joined_into_one_string():
    got = parse_topics(_span("<span>Tourism Management - Rules of Procedure</span>"))
    assert got == ["Tourism Management", "Rules of Procedure"]


def test_keeps_a_hyphenated_topic_whole():
    """The regression that produced 'Multi' and 'Year Strategic Work Plan' as separate topics."""
    got = parse_topics(_span("<span>Multi-Year Strategic Work Plan</span>"))
    assert got == ["Multi-Year Strategic Work Plan"]


def test_keeps_hyphenated_names_whole_while_still_splitting_between_them():
    got = parse_topics(_span(
        "<span>Vessel-based activities - Land-based activities</span>"))
    assert got == ["Vessel-based activities", "Land-based activities"]


def test_keeps_a_leading_prefix_hyphen_whole():
    """'De-listed (HSM 12)' used to contribute a bare 'De' topic."""
    got = parse_topics(_span("<span>De-listed (HSM 12) - Non-Consultative Parties</span>"))
    assert got == ["De-listed (HSM 12)", "Non-Consultative Parties"]


def test_a_single_topic_is_returned_alone():
    got = parse_topics(_span("<span>ASPAs (Antarctic Specially Protected Areas)</span>"))
    assert got == ["ASPAs (Antarctic Specially Protected Areas)"]


def test_an_empty_span_yields_no_topics():
    assert parse_topics(_span("<span></span>")) == []


def test_separator_only_pieces_are_dropped():
    assert parse_topics(_span("<span><span> - </span></span>")) == []

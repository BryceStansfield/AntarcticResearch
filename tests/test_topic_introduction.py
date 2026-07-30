"""Tests for the topic-introduction/diversity aggregation logic.

`TopicIntroduction`/`TopicDiversity.__init__` both call `get_wp_bertopic()`, which
fits a real BERTopic model over OpenRouter embeddings across the whole working
paper corpus -- heavy, network-dependent, and out of scope here, as is
`WPBertTopic` itself. What's actually interesting -- and pure -- is the
aggregation logic extracted into module-level functions:

- `_earliest_introductions(topics, documents)`: groups documents by topic
  (skipping outlier topic -1), picks the earliest doc per topic by
  `sort_string`, and tallies each earliest doc's parties into yearly and total
  country counts. This is everything `TopicIntroduction.__init__` did after
  computing `topic_info`, minus the sanity-check report file write.
- `_country_topic_year_triples(topics, documents, start_year, end_year)`: builds
  the (country, topic, year) triples that back `TopicDiversity`, skipping
  outlier topic -1 and documents outside the closed year window.

`TopicDiversity._diversity_within` was already a separate, parameter-driven
method operating only on `self._country_topic_years`, so it is exercised
directly against a synthetic triples list via `object.__new__(TopicDiversity)`,
without running `__init__` (and therefore without touching `get_wp_bertopic()`).
"""
from antarctic_ladder_metrics.topic_introduction import (
    _country_topic_year_triples,
    _earliest_introductions,
    TopicDiversity,
)


def _doc(sort_string, year, parties):
    return {"sort_string": sort_string, "year": year, "parties": parties}


# ------------------------------------------------------------- _earliest_introductions

def test_outlier_topic_excluded_entirely():
    """A topic -1 (outlier) document contributes no earliest doc and no counts."""
    docs = [_doc("2001-A", 2001, ["France"])]
    result = _earliest_introductions([-1], docs)
    assert result["earliest_docs"] == []
    assert result["yearly_topic_introduction_count"] == {}
    assert result["topic_introduction_count"] == {}


def test_ties_broken_by_earlier_sort_string():
    """Two docs share a topic; the one with the lexicographically earlier
    `sort_string` is picked as having introduced it, regardless of list order."""
    docs = [
        _doc("2005-B", 2005, ["France"]),
        _doc("2001-A", 2001, ["Germany"]),
    ]
    result = _earliest_introductions([0, 0], docs)
    assert result["earliest_docs"] == [docs[1]]
    assert result["yearly_topic_introduction_count"] == {(2001, "germany"): 1}
    assert result["topic_introduction_count"] == {"germany": 1}


def test_multiple_parties_on_earliest_doc_each_credited():
    """Every (normalized) party on the earliest doc gets its own credit."""
    docs = [_doc("2001-A", 2001, ["France", "Germany", "USA"])]
    result = _earliest_introductions([0], docs)
    assert result["yearly_topic_introduction_count"] == {
        (2001, "france"): 1,
        (2001, "germany"): 1,
        (2001, "united states"): 1,
    }
    assert result["topic_introduction_count"] == {
        "france": 1, "germany": 1, "united states": 1,
    }


def test_yearly_and_total_counts_are_consistent():
    """Summing a country's yearly counts across every year reproduces its total,
    for a country that introduces topics in more than one year."""
    docs = [
        _doc("2001-A", 2001, ["France"]),  # topic 0
        _doc("2005-A", 2005, ["France"]),  # topic 1
    ]
    result = _earliest_introductions([0, 1], docs)
    assert result["yearly_topic_introduction_count"] == {
        (2001, "france"): 1,
        (2005, "france"): 1,
    }
    assert result["topic_introduction_count"] == {"france": 2}

    recomputed_totals = {}
    for (_year, country), count in result["yearly_topic_introduction_count"].items():
        recomputed_totals[country] = recomputed_totals.get(country, 0) + count
    assert recomputed_totals == result["topic_introduction_count"]


def test_each_topic_picks_its_own_earliest_doc():
    docs = [
        _doc("2001-A", 2001, ["France"]),   # topic 0, earliest for topic 0
        _doc("2003-A", 2003, ["Germany"]),  # topic 0, later, should lose the tie
        _doc("2000-A", 2000, ["USA"]),      # topic 1, its only doc
    ]
    result = _earliest_introductions([0, 0, 1], docs)
    assert docs[0] in result["earliest_docs"]
    assert docs[2] in result["earliest_docs"]
    assert docs[1] not in result["earliest_docs"]
    assert len(result["earliest_docs"]) == 2


# ------------------------------------------------------- _country_topic_year_triples

def test_country_triples_outlier_topic_excluded():
    docs = [_doc("A", 2010, ["France"])]
    assert _country_topic_year_triples([-1], docs, 2000, 2025) == []


def test_country_triples_year_window_boundaries_are_inclusive():
    docs = [
        _doc("A", 1999, ["France"]),  # just below the window
        _doc("B", 2000, ["France"]),  # at start_year, included
        _doc("C", 2025, ["France"]),  # at end_year, included
        _doc("D", 2026, ["France"]),  # just above the window
    ]
    triples = _country_topic_year_triples([0, 0, 0, 0], docs, 2000, 2025)
    years = sorted(year for _, _, year in triples)
    assert years == [2000, 2025]


def test_country_triples_splits_and_normalizes_parties():
    """`parties` entries may be '|'-joined and use alias spellings; both the
    split and the normalization are applied before triples are built."""
    docs = [_doc("A", 2010, ["France|Germany", "USA"])]
    triples = _country_topic_year_triples([3], docs, 2000, 2025)
    countries = sorted(country for country, _, _ in triples)
    assert countries == ["france", "germany", "united states"]
    assert all(topic == 3 and year == 2010 for _, topic, year in triples)


# ---------------------------------------------------------------- _diversity_within

def _diversity_instance(triples):
    """A `TopicDiversity` with `_country_topic_years` set directly, bypassing
    `__init__` (and therefore `get_wp_bertopic()`) entirely."""
    instance = object.__new__(TopicDiversity)
    instance._country_topic_years = triples
    return instance


def test_diversity_within_does_not_itself_filter_outlier_topics():
    """Outlier exclusion is `_country_topic_year_triples`'s job upstream; once a
    triple has been let through, `_diversity_within` counts it like any other."""
    triples = [("france", 5, 2010), ("france", -1, 2011)]
    instance = _diversity_instance(triples)
    assert instance._diversity_within() == {"france": 2}


def test_diversity_within_year_window_boundaries_are_inclusive():
    triples = [
        ("france", 0, 1999),
        ("france", 1, 2000),
        ("france", 2, 2009),
        ("france", 3, 2010),
    ]
    instance = _diversity_instance(triples)
    assert instance._diversity_within(2000, 2009) == {"france": 2}


def test_diversity_within_counts_distinct_topics_not_occurrences():
    """A country working the same topic across multiple years/documents counts
    once -- diversity is a distinct-topic count, not a document count."""
    triples = [
        ("france", 5, 2001),
        ("france", 5, 2005),
        ("france", 5, 2010),
    ]
    instance = _diversity_instance(triples)
    assert instance._diversity_within() == {"france": 1}


def test_diversity_within_full_window_matches_unrestricted_computation():
    """_diversity_within(None, None) -- the "Full" window used to seed
    self.countries_to_topics in __init__ -- must equal a plain distinct-topics-
    per-country count taken over every triple, with no year filtering at all."""
    triples = [
        ("france", 1, 2001), ("france", 2, 2010),
        ("germany", 1, 2001), ("germany", 1, 2015),
        ("usa", 3, 2020),
    ]
    instance = _diversity_instance(triples)

    expected = {}
    for country, topic, _year in triples:
        expected.setdefault(country, set()).add(topic)
    expected = {c: len(s) for c, s in expected.items()}

    assert instance._diversity_within(None, None) == expected
    assert instance._diversity_within() == expected

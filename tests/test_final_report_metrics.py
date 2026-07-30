"""Tests for the pure/near-pure helpers in
antarctic_ladder_metrics/final_report_metrics.py.

This file is the largest and messiest of the nine ladder metrics: LLM prompt
construction, sqlite storage, fuzzy string matching, and a multiprocessing pipeline
all live in one module. Two small refactors were made here, both behavior-preserving:

- `classify_intervening_parties` used to fuse the network call with reply parsing.
  The parsing tail (regex-extract the bracketed list, raise ValueError if absent,
  match countries case-insensitively by raw or display name) is now a standalone
  `_parse_intervention_reply(reply, countries)`, called by
  `classify_intervening_parties` right after it gets `reply` from the network. This
  is just naming the existing tail of the function -- the computation is identical.

- `_get_connection` hardcoded the module-level `METRICS_DB_PATH`. It (and
  `get_document`/`set_document`/`check_document_exists`/`get_country_figures`) now
  take an optional `db_path` parameter defaulting to `METRICS_DB_PATH`, mirroring the
  path-as-argument convention `enrich_measure_data` already uses in
  ACTM_Measure_Scraper/src/MeasureEnricher.py. Every call site in the real pipeline
  omits the argument, so the default keeps current behavior byte-for-byte; tests
  pass a tmp_path db instead.

Out of scope, and why:
- `openai.OpenAI` / the actual HTTP call in `classify_intervening_parties`, and
  `warm_intervention_cache` -- both make a real network call.
- `FinalReportBaker.__init__` -- runs a multiprocessing pool and the full OCR/scrape
  pipeline; not reachable without touching the network and real files.
- `FinalReportMentionFigures` / `FinalReportInterventionFigures` -- both subclass
  `FinalReportBaker` and their `__init__` calls `super().__init__()`, so they inherit
  the same network/multiprocessing dependency.
"""
import inspect

import pytest

from antarctic_ladder_metrics import final_report_metrics as frm
from antarctic_ladder_metrics.final_report_metrics import (
    FinalReportBaker,
    _build_intervention_messages,
    _format_country_name,
    _parse_intervention_reply,
    _render_few_shot_example,
    _strip_unicode_artifacts,
    check_document_exists,
    fuzzy_term_coincidence_checker,
    get_country_figures,
    get_document,
    set_document,
)


# ------------------------------------------------------------- _strip_unicode_artifacts

@pytest.mark.parametrize("text", [
    "Normal ASCII sentence.",
    "Text with accented letters like déjà vu and some 中文 characters.",
    "",
])
def test_strip_unicode_artifacts_leaves_normal_text_unchanged(text):
    assert _strip_unicode_artifacts(text) == text


def test_strip_unicode_artifacts_removes_private_use_area_glyph():
    """PDF extraction sometimes bakes dingbats/bullets into a custom font that maps
    to the Private Use Area (category 'Co') with no real Unicode meaning."""
    assert _strip_unicode_artifacts("BadGlyph") == "BadGlyph"


def test_strip_unicode_artifacts_removes_non_whitespace_control_chars():
    """A control char outside the kept whitespace set (category 'Cc') is stripped --
    contrasts with the next test, where \\n/\\r/\\t of the same general 'control
    character' flavor are explicitly preserved."""
    assert _strip_unicode_artifacts("A\x0bB") == "AB"


@pytest.mark.parametrize("char", ["\n", "\r", "\t"])
def test_strip_unicode_artifacts_preserves_newline_tab_and_carriage_return(char):
    text = f"Line one{char}Line two"
    assert _strip_unicode_artifacts(text) == text


# --------------------------------------------------------------- _format_country_name

@pytest.mark.parametrize("country,expected", [
    ("united kingdom", "United Kingdom"),
    ("new zealand", "New Zealand"),
    ("czech republic", "Czech Republic"),
    ("chile", "Chile"),
    ("china", "China"),
])
def test_format_country_name_title_cases_each_space_separated_word(country, expected):
    assert _format_country_name(country) == expected


# ------------------------------------------------------ fuzzy_term_coincidence_checker

def test_fuzzy_term_coincidence_checker_finds_exact_substring_match():
    result = fuzzy_term_coincidence_checker("China raised the issue.", ["china", "japan"])
    assert result == ["china"]


def test_fuzzy_term_coincidence_checker_finds_fuzzy_typo_match():
    """'Kizgdom' is 'Kingdom' with a single letter substituted (n -> z), embedded in
    a full sentence -- a realistic OCR/typo error, not a literal substring of
    'united kingdom'. thefuzz.fuzz.partial_ratio on a single-character edit out of a
    14-character term clears the 90 threshold comfortably."""
    sentence = "The delegation from United Kizgdom raised a formal objection."
    result = fuzzy_term_coincidence_checker(sentence, ["united kingdom"])
    assert result == ["united kingdom"]


def test_fuzzy_term_coincidence_checker_returns_empty_for_unrelated_sentence():
    sentence = "Completely unrelated meteorological observations were recorded."
    result = fuzzy_term_coincidence_checker(sentence, ["china", "japan", "chile"])
    assert result == []


def test_extract_mentions_delegates_to_the_checker_with_class_level_countries():
    """`FinalReportBaker.extract_mentions` only reads `self.COUNTRIES` (a class
    attribute) and calls the module-level checker -- it never touches instance
    state. Re-reading the method body confirms this, so it can be called with the
    class object itself standing in for `self` (which does have `.COUNTRIES`),
    without running `FinalReportBaker.__init__` (which would scrape/OCR/network)."""
    result = FinalReportBaker.extract_mentions(FinalReportBaker, "China and Japan discussed the matter.")
    assert set(result) == {"china", "japan"}


# ------------------------------------------------------------- _render_few_shot_example

def test_render_few_shot_example_matches_expected_shape():
    example = {
        "sentence": "Foo said something notable.",
        "parties": ["Foo", "Bar"],
        "intervening": ["Foo"],
    }
    rendered = _render_few_shot_example(example)
    assert rendered == (
        "Sentence: Foo said something notable.\n"
        "Parties: [Foo, Bar].\n"
        "Answer: [Foo]"
    )


def test_render_few_shot_example_handles_empty_intervening_list():
    example = {"sentence": "Bystander sentence.", "parties": ["Foo"], "intervening": []}
    rendered = _render_few_shot_example(example)
    assert rendered.endswith("Answer: []")


# --------------------------------------------------------------- _render_few_shot_block

def test_render_few_shot_block_contains_every_examples_sentence():
    """Deliberately checked against the real module-level FEW_SHOT_EXAMPLES rather
    than hardcoded text, since the examples list is expected to grow over time and
    this test shouldn't need updating when it does."""
    block = frm._render_few_shot_block()
    assert block
    assert "More examples:" in block
    for example in frm.FEW_SHOT_EXAMPLES:
        assert example["sentence"] in block


def test_render_few_shot_block_is_empty_string_when_no_examples(monkeypatch):
    monkeypatch.setattr(frm, "FEW_SHOT_EXAMPLES", [])
    assert frm._render_few_shot_block() == ""


# ----------------------------------------------------------- _build_intervention_messages

def test_build_intervention_messages_structure_and_cache_control():
    messages = _build_intervention_messages("China said hi.", ["China", "Japan"])

    assert len(messages) == 1
    message = messages[0]
    assert message["role"] == "user"

    cached_block, per_call_block = message["content"]

    assert cached_block["text"] == frm.INTERVENTION_CACHED_PREFIX
    assert cached_block["cache_control"] == {"type": "ephemeral"}

    assert "cache_control" not in per_call_block
    assert per_call_block["text"] == "Problem:\nSentence: China said hi.\nParties: [China, Japan].\n"


def test_build_intervention_messages_strips_unicode_artifacts_from_sentence():
    """The sentence goes through `_strip_unicode_artifacts` before being interpolated
    into the per-call block, so a stray PDF-extraction glyph must not survive."""
    dirty_sentence = "China said hi."
    messages = _build_intervention_messages(dirty_sentence, ["China"])
    per_call_text = messages[0]["content"][1]["text"]

    assert "" not in per_call_text
    assert "China said hi." in per_call_text


# --------------------------------------------------------------- _parse_intervention_reply

def test_parse_intervention_reply_well_formed():
    reply = "[China, Norway]"
    result = _parse_intervention_reply(reply, ["china", "ukraine", "norway"])
    assert result == ["china", "norway"]


def test_parse_intervention_reply_raises_value_error_without_brackets():
    with pytest.raises(ValueError):
        _parse_intervention_reply("Sorry, I can't help with that.", ["china"])


def test_parse_intervention_reply_matching_is_case_insensitive():
    """The LLM reply may echo either the raw lowercase country key or its
    capitalized display form; matching lowercases both sides so either survives.
    ('CHINA' exercises the raw-key path in a different case; 'united kingdom' is
    matched via its own lowercase key here since `_format_country_name(...).lower()`
    is always identical to `country.lower()` -- capitalizing then lowercasing is a
    no-op -- so the two match branches are equivalent once both sides are lowered.)"""
    reply = "[CHINA, united kingdom]"
    result = _parse_intervention_reply(reply, ["china", "united kingdom", "japan"])
    assert set(result) == {"china", "united kingdom"}
    assert "japan" not in result


def test_parse_intervention_reply_strips_quoted_names():
    reply = '''["China", 'Norway']'''
    result = _parse_intervention_reply(reply, ["china", "norway"])
    assert set(result) == {"china", "norway"}


def test_parse_intervention_reply_excludes_country_absent_from_reply():
    reply = "[China]"
    result = _parse_intervention_reply(reply, ["china", "japan"])
    assert result == ["china"]


# ------------------------------------------------------------------------- sqlite helpers

@pytest.mark.parametrize("func", [
    frm._get_connection,
    get_document,
    set_document,
    check_document_exists,
    get_country_figures,
])
def test_db_path_defaults_still_point_at_the_real_metrics_db(func):
    """The path-parameterization refactor must not change default behavior for the
    real pipeline: every function's new `db_path` parameter defaults to the
    module-level METRICS_DB_PATH, so every existing call site (which omits the
    argument) keeps hitting the same real database file as before."""
    sig = inspect.signature(func)
    assert sig.parameters["db_path"].default == frm.METRICS_DB_PATH


def test_set_document_then_get_document_round_trips_all_fields(tmp_path):
    db_path = tmp_path / "metrics.sqlite3"
    set_document(
        "doc1", 0, "Some chunk text.",
        mentioned=["china"], intervening=["china"], year=2001,
        db_path=db_path,
    )

    doc = get_document("doc1", 0, db_path=db_path)

    assert doc == {
        "chunk": "Some chunk text.",
        "mentioned": ["china"],
        "intervening": ["china"],
    }


def test_get_document_returns_none_for_a_document_never_written(tmp_path):
    db_path = tmp_path / "metrics.sqlite3"
    assert get_document("missing", 0, db_path=db_path) is None


def test_set_document_defaults_mentioned_and_intervening_to_empty_lists(tmp_path):
    db_path = tmp_path / "metrics.sqlite3"
    set_document("doc1", 0, "chunk with no country mentions", year=1999, db_path=db_path)

    doc = get_document("doc1", 0, db_path=db_path)
    assert doc["mentioned"] == []
    assert doc["intervening"] == []


def test_check_document_exists_false_before_and_true_after_set_document(tmp_path):
    db_path = tmp_path / "metrics.sqlite3"
    assert check_document_exists("doc1", 0, db_path=db_path) is False

    set_document("doc1", 0, "chunk", db_path=db_path)

    assert check_document_exists("doc1", 0, db_path=db_path) is True


def test_check_document_exists_is_keyed_on_document_and_chunk_num_together(tmp_path):
    db_path = tmp_path / "metrics.sqlite3"
    set_document("doc1", 0, "chunk", db_path=db_path)

    assert check_document_exists("doc1", 1, db_path=db_path) is False
    assert check_document_exists("doc2", 0, db_path=db_path) is False


def test_get_country_figures_filters_by_intervention_flag(tmp_path):
    db_path = tmp_path / "metrics.sqlite3"
    set_document("doc1", 0, "chunk1", mentioned=["china"], intervening=["china"], year=2000, db_path=db_path)
    set_document("doc1", 1, "chunk2", mentioned=["china", "japan"], intervening=["japan"], year=2001, db_path=db_path)

    assert get_country_figures("china", must_be_intervention=False, db_path=db_path) == 2
    assert get_country_figures("china", must_be_intervention=True, db_path=db_path) == 1
    assert get_country_figures("japan", must_be_intervention=True, db_path=db_path) == 1
    assert get_country_figures("japan", must_be_intervention=False, db_path=db_path) == 1


def test_get_country_figures_filters_by_year_range(tmp_path):
    db_path = tmp_path / "metrics.sqlite3"
    set_document("doc1", 0, "chunk1", mentioned=["china"], year=2000, db_path=db_path)
    set_document("doc1", 1, "chunk2", mentioned=["china"], year=2010, db_path=db_path)

    assert get_country_figures("china", False, min_year=2005, db_path=db_path) == 1
    assert get_country_figures("china", False, max_year=2005, db_path=db_path) == 1
    assert get_country_figures("china", False, min_year=2000, max_year=2010, db_path=db_path) == 2
    assert get_country_figures("china", False, min_year=2001, max_year=2009, db_path=db_path) == 0


def test_get_country_figures_matches_an_exact_json_entry(tmp_path):
    db_path = tmp_path / "metrics.sqlite3"
    set_document("doc1", 0, "chunk1", mentioned=["china"], year=2000, db_path=db_path)

    assert get_country_figures("china", False, db_path=db_path) == 1


def test_get_country_figures_like_pattern_is_anchored_by_json_quotes(tmp_path):
    """`get_country_figures` filters with `LIKE '%"<country>"%'` against the
    JSON-encoded mentioned/intervening column. On its face this looks like it could
    false-positive when one country's name is a substring of another entry's text
    -- but `json.dumps` always wraps each list entry in double quotes immediately
    before and after the string, so the pattern's own literal quote characters
    anchor the match to a whole JSON string, not a loose substring. 'china' is a
    substring of 'south china sea group', but `["south china sea group"]` never
    contains the literal text `"china"` (quote immediately before AND after) --
    only `\" china \"`-style boundaries with a space, not a quote, next to it. So
    this is not a real risk for the current COUNTRIES list, and this test pins
    down why."""
    db_path = tmp_path / "metrics.sqlite3"
    set_document("doc1", 0, "chunk1", mentioned=["south china sea group"], year=2000, db_path=db_path)

    assert get_country_figures("china", False, db_path=db_path) == 0

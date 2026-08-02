"""Tests for the shared punkt sentence splitter.

`chunk_sentences` sets the unit of LLM censorship, so its grouping and its
empty-chunk handling both affect cache keys downstream.
"""
import pytest

from sentence_splitter import (chunk_sentence_spans, chunk_sentences,
                               split_sentences)


def test_splits_on_terminal_punctuation():
    assert split_sentences("One here. Two follows! Three?") == [
        "One here.", "Two follows!", "Three?"]


def test_single_sentence_stays_whole():
    assert split_sentences("Just the one sentence.") == ["Just the one sentence."]


def test_empty_text_yields_no_sentences():
    assert split_sentences("") == []


def test_needs_no_network_once_punkt_is_installed():
    """Regression: `_ensure_punkt` used to call `nltk.download` unconditionally, which
    fetched the package index from raw.githubusercontent.com through urllib with no
    timeout -- a stalled connection hung the whole pipeline indefinitely.

    Monkeypatching the download to raise proves the happy path never reaches it.
    """
    import nltk
    import sentence_splitter

    original_download = nltk.download
    original_ready = sentence_splitter._punkt_ready
    def explode(*args, **kwargs):
        raise AssertionError("nltk.download called despite punkt_tab being present")

    nltk.download = explode
    sentence_splitter._punkt_ready = False   # force the guard to re-run
    try:
        assert split_sentences("Offline works.") == ["Offline works."]
    finally:
        nltk.download = original_download
        sentence_splitter._punkt_ready = original_ready


def test_groups_sentences_into_fixed_size_chunks():
    text = "A one. B two. C three. D four."
    assert chunk_sentences(text, 2) == ["A one. B two.", "C three. D four."]


def test_final_chunk_may_be_short():
    text = "A one. B two. C three."
    assert chunk_sentences(text, 2) == ["A one. B two.", "C three."]


def test_chunk_size_larger_than_the_text_gives_one_chunk():
    assert chunk_sentences("A one. B two.", 10) == ["A one. B two."]


def test_empty_text_yields_no_chunks():
    assert chunk_sentences("", 3) == []


def test_whitespace_only_text_yields_no_chunks():
    """Empty chunks are dropped rather than passed on as blank prompts."""
    assert chunk_sentences("   \n  ", 3) == []


# ------------------------------------------------------------------ chunk_sentence_spans

_LAID_OUT = (
    "ANTARCTIC TREATY\n\n"
    "The United Kingdom proposes a measure.\n"
    "It also notes the matter.\n\n"
    "    A third point follows here.\n"
)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 6])
def test_span_chunks_are_byte_identical_to_chunk_sentences(chunk_size):
    """Load-bearing: the chunk text is the LLM phrase-detection cache key
    (`sha256(author + "\\n" + chunk)`). If these two ever disagree, every cached chunk misses and
    the next censorship run re-detects the whole corpus with live, paid LLM calls."""
    assert [c for _s, _e, c in chunk_sentence_spans(_LAID_OUT, chunk_size)] \
        == chunk_sentences(_LAID_OUT, chunk_size)


def test_spans_point_at_the_source_text():
    for start, end, chunk in chunk_sentence_spans(_LAID_OUT, 1):
        span = _LAID_OUT[start:end]
        # The span is the source region; the chunk is that region with its inter-sentence
        # whitespace collapsed, so they agree once whitespace is normalised.
        assert " ".join(span.split()) == " ".join(chunk.split())


def test_spans_are_ordered_and_non_overlapping():
    spans = [(s, e) for s, e, _ in chunk_sentence_spans(_LAID_OUT, 2)]
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert prev_end <= next_start


def test_text_between_spans_is_the_layout():
    """What sits between the spans is exactly the structure the chunk text has lost -- rebuilding
    a document from spans plus these gaps reproduces the source."""
    spans = chunk_sentence_spans(_LAID_OUT, 1)
    rebuilt, cursor = [], 0
    for start, end, _chunk in spans:
        rebuilt.append(_LAID_OUT[cursor:start])
        rebuilt.append(_LAID_OUT[start:end])
        cursor = end
    rebuilt.append(_LAID_OUT[cursor:])
    assert "".join(rebuilt) == _LAID_OUT


@pytest.mark.parametrize("text", ["", "   \n\n  ", "Single sentence.", "No terminator"])
def test_degenerate_inputs_agree_with_chunk_sentences(text):
    assert [c for _s, _e, c in chunk_sentence_spans(text, 6)] == chunk_sentences(text, 6)

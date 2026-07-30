"""Tests for the shared punkt sentence splitter.

`chunk_sentences` sets the unit of LLM censorship and of the semantic filter, so its
grouping and its empty-chunk handling both affect cache keys downstream.
"""
from sentence_splitter import chunk_sentences, split_sentences


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

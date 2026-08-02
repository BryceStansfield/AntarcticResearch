"""Tests for how LLM censorship rebuilds a document.

The benchmark trains one classifier per censorship arm and attributes the difference between them
to censorship. That only holds if censorship is the *only* difference from the raw text. It wasn't:
the LLM arm reassembled the document from its sentence chunks, which are sentences joined by single
spaces, so every newline that separated two sentences was replaced by a space. Measured over 300
real papers that dropped 8029 of 54677 newlines (~15%) -- precisely the inter-sentence and
inter-paragraph ones. The naive arm is a plain `re.sub` over the document and loses none, so the
two arms were not comparable.

The fix substitutes into the *source span* of each chunk and copies the text between spans through
untouched. Two things have to stay true, and both are pinned here:

* the chunk text handed to phrase detection is unchanged, because it is the cache key -- if it
  moves, the entire corpus is re-detected with live paid LLM calls;
* what gets redacted is unchanged; only the surrounding layout is.

Phrase detection is stubbed throughout, so nothing here makes a call.
"""
import pytest

import embeddings.working_paper_censorship as wc
from sentence_splitter import chunk_sentences


LAID_OUT = (
    "ANTARCTIC TREATY\n"
    "AGENDA ITEM 12\n\n"
    "The United Kingdom proposes a measure.\n"
    "It was submitted by Norway earlier.\n\n"
    "    A third point follows.\n"
)


@pytest.fixture
def no_llm(monkeypatch):
    """Fail loudly if anything reaches for the live detector."""
    monkeypatch.setattr(wc, "detect_revealing_phrases",
                        lambda *a, **k: pytest.fail("must not call the LLM"))


def _censor(monkeypatch, text, phrases_by_chunk):
    """Run llm_censor_text with phrase detection served from a dict, recording the chunks asked for."""
    asked = []

    def _fake(author, chunk):
        asked.append(chunk)
        return phrases_by_chunk.get(chunk, [])

    monkeypatch.setattr(wc, "get_or_detect_phrases", _fake)
    return wc.llm_censor_text(text, "United Kingdom"), asked


def test_the_chunks_sent_for_detection_are_unchanged(monkeypatch, no_llm):
    """The cache-key guarantee. These strings are hashed into the phrase cache, so if the rebuild
    changed them the next run would re-detect every chunk in the corpus at cost."""
    _out, asked = _censor(monkeypatch, LAID_OUT, {})
    assert asked == chunk_sentences(LAID_OUT, wc.LLM_CHUNK_SENTENCES)


def test_layout_survives_when_nothing_is_censored(monkeypatch, no_llm):
    """With no phrases to remove, censorship is the identity -- it used to be a reflow."""
    out, _ = _censor(monkeypatch, LAID_OUT, {})
    assert out == LAID_OUT


def test_a_phrase_is_still_redacted(monkeypatch, no_llm):
    chunks = chunk_sentences(LAID_OUT, wc.LLM_CHUNK_SENTENCES)
    out, _ = _censor(monkeypatch, LAID_OUT, {chunks[0]: ["United Kingdom"]})

    assert "United Kingdom" not in out
    assert wc.PLACEHOLDER in out


def test_redacting_preserves_the_surrounding_layout(monkeypatch, no_llm):
    """The regression, in miniature: the blank lines and the indent must still be there after a
    redaction, and only the phrase itself may differ from the source."""
    chunks = chunk_sentences(LAID_OUT, wc.LLM_CHUNK_SENTENCES)
    out, _ = _censor(monkeypatch, LAID_OUT, {chunks[0]: ["United Kingdom"]})

    assert out == LAID_OUT.replace("United Kingdom", wc.PLACEHOLDER)
    assert out.count("\n") == LAID_OUT.count("\n")
    assert "\n\n" in out and "    A third point" in out


def test_a_phrase_spanning_a_line_break_is_still_matched(monkeypatch, no_llm):
    """Chunks collapse a newline inside a phrase to a space, so the phrase list can name a phrase
    that appears across two lines in the source. `_censor_pattern` joins words with `\\s+`, which
    is what lets it still match there."""
    text = "It was proposed by the United\nKingdom of interest. Nothing else.\n"
    chunks = chunk_sentences(text, wc.LLM_CHUNK_SENTENCES)
    out, _ = _censor(monkeypatch, text, {chunks[0]: ["United Kingdom"]})

    assert wc.PLACEHOLDER in out
    assert "United" not in out and "Kingdom" not in out


def test_censorship_is_scoped_to_the_chunk_that_revealed_it(monkeypatch, no_llm):
    """Segment-wide censorship: a phrase revealed in one chunk must not censor another. This is
    unchanged by the rebuild, but it is the property the span arithmetic could most easily break."""
    text = "Alpha by Norway here. " * 3 + "\n\n" + "Beta by Norway there. " * 3
    monkeypatch.setattr(wc, "LLM_CHUNK_SENTENCES", 3)
    chunks = chunk_sentences(text, 3)
    assert len(chunks) >= 2

    out, _ = _censor(monkeypatch, text, {chunks[0]: ["Norway"]})
    assert out.count(wc.PLACEHOLDER) == 3, "only the first chunk's occurrences"
    assert "Norway" in out, "the second chunk's must survive"


def test_text_punkt_skips_between_sentences_is_preserved(monkeypatch, no_llm):
    """Headers and page furniture fall outside every chunk span and must be copied through."""
    text = "HEADER LINE\n\n\nA sentence here.\n\n\nFOOTER LINE"
    out, _ = _censor(monkeypatch, text, {})
    assert out == text


@pytest.mark.parametrize("text", ["", "   \n\n  "])
def test_empty_documents_come_back_unchanged(monkeypatch, no_llm, text):
    out, _ = _censor(monkeypatch, text, {})
    assert out == text


# --------------------------------------------------------------- the cache-only variant

def test_cached_variant_returns_none_on_a_miss(monkeypatch, no_llm):
    """Its contract: never fall back to a live call. A single undetected chunk aborts."""
    monkeypatch.setattr(wc, "_phrase_cache", {})
    assert wc.llm_censor_text_cached(LAID_OUT, "United Kingdom") is None


def test_cached_variant_preserves_layout_too(monkeypatch, no_llm):
    chunks = chunk_sentences(LAID_OUT, wc.LLM_CHUNK_SENTENCES)
    monkeypatch.setattr(wc, "_phrase_cache",
                        {wc._cache_key("United Kingdom", c): ["United Kingdom"] for c in chunks})

    out = wc.llm_censor_text_cached(LAID_OUT, "United Kingdom")
    assert out == LAID_OUT.replace("United Kingdom", wc.PLACEHOLDER)


def test_the_two_variants_agree(monkeypatch, no_llm):
    """They differ only in where phrases come from, so on a full cache they must produce the same
    document -- the exports read through the cached one and the benchmark through the other."""
    chunks = chunk_sentences(LAID_OUT, wc.LLM_CHUNK_SENTENCES)
    phrases = {c: ["Norway"] for c in chunks}
    monkeypatch.setattr(wc, "_phrase_cache",
                        {wc._cache_key("United Kingdom", c): ["Norway"] for c in chunks})

    live, _ = _censor(monkeypatch, LAID_OUT, phrases)
    assert live == wc.llm_censor_text_cached(LAID_OUT, "United Kingdom")

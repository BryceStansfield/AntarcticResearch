"""Shared punkt-based sentence splitter.

Wraps nltk's sentence tokenizer so callers don't each have to manage the one-time
``punkt_tab`` download.
"""
import nltk
from nltk.tokenize import sent_tokenize

_punkt_ready = False


def _ensure_punkt() -> None:
    global _punkt_ready
    if not _punkt_ready:
        # Only reach for the network when the corpus is genuinely absent. nltk.download
        # otherwise still fetches its index from raw.githubusercontent.com just to
        # confirm an already-installed package is current, and it does so through
        # urllib with no timeout -- so a stalled connection hangs the whole pipeline
        # indefinitely rather than failing.
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab")
        _punkt_ready = True


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences using nltk's punkt tokenizer."""
    _ensure_punkt()
    return sent_tokenize(text)


def chunk_sentences(text: str, chunk_size: int) -> list[str]:
    """Group ``text``'s sentences into chunks of ``chunk_size`` sentences each (joined by
    spaces). Empty chunks are dropped."""
    return [chunk for _start, _end, chunk in chunk_sentence_spans(text, chunk_size)]


def chunk_sentence_spans(text: str, chunk_size: int) -> list[tuple[int, int, str]]:
    """``chunk_sentences``, but also locating each chunk in the source text.

    Returns ``(start, end, chunk)`` per chunk, where ``chunk`` is byte-for-byte what
    ``chunk_sentences`` yields and ``text[start:end]`` is the region those sentences came from.

    The two differ, and that is the point. A chunk is its sentences joined by single spaces, so
    whatever separated those sentences in the source -- a newline, a blank line, an indent -- is
    already gone from it; the span still has it. Callers that need a cache key (the chunk *is* the
    key for LLM phrase detection) want the chunk, and callers rebuilding a censored document want
    the span, because writing reflowed chunks back out in place of the original is a second,
    uncontrolled difference from the source text on top of whatever was censored.

    Sentences are located by scanning forward, so text punkt skips between sentences -- page
    furniture, headers, blank lines -- simply falls outside every span and is left to the caller.
    """
    sentences = split_sentences(text)

    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence in sentences:
        found = text.find(sentence, cursor)
        if found == -1:
            # punkt returns substrings of its input, so this should not happen; degrade to an
            # empty span at the cursor rather than raising or silently shifting every later span.
            spans.append((cursor, cursor))
            continue
        spans.append((found, found + len(sentence)))
        cursor = found + len(sentence)

    chunks = []
    for i in range(0, len(sentences), chunk_size):
        group = sentences[i:i + chunk_size]
        chunk = " ".join(group).strip()
        if not chunk:
            continue
        group_spans = spans[i:i + chunk_size]
        chunks.append((group_spans[0][0], group_spans[-1][1], chunk))
    return chunks

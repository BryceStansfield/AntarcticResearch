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
        nltk.download("punkt_tab")
        _punkt_ready = True


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences using nltk's punkt tokenizer."""
    _ensure_punkt()
    return sent_tokenize(text)

"""Tests for `utils.split_parties`.

Every WP/IP authorship figure runs its author lists through this, so a quiet
mis-split here shows up as country credit landing in the wrong place.
"""
import numpy as np
import pytest

from utils import split_parties


def test_strips_and_lowercases_each_party():
    assert split_parties(["  Australia ", "CHILE"]) == ["australia", "chile"]


def test_splits_pipe_joined_entries():
    assert split_parties(["Australia|Chile"]) == ["australia", "chile"]


def test_splits_and_normalises_together():
    assert split_parties([" Australia | New Zealand "]) == ["australia", "new zealand"]


def test_flattens_a_mix_of_joined_and_single_entries():
    assert split_parties(["Australia|Chile", "Norway"]) == ["australia", "chile",
                                                            "norway"]


def test_empty_input_yields_empty_output():
    assert split_parties([]) == []


def test_accepts_a_numpy_array():
    """The real parties column arrives from parquet as an object ndarray, not a list."""
    parties = np.array(["Chile", "Argentina"], dtype=object)
    assert split_parties(parties) == ["chile", "argentina"]


def test_preserves_duplicates():
    """Deduplication is the caller's job; the live callers count occurrences."""
    assert split_parties(["Chile", "Chile"]) == ["chile", "chile"]


def test_preserves_input_order():
    assert split_parties(["Norway|Chile", "Australia"]) == ["norway", "chile",
                                                            "australia"]


def test_a_trailing_pipe_emits_an_empty_party():
    """Characterises current behaviour: no empty-string filtering.

    An empty party would become its own country key downstream. No row of the live
    corpus produces one, so this is latent rather than active.
    """
    assert split_parties(["Australia|"]) == ["australia", ""]


def test_a_bare_string_is_rejected():
    """Regression: a plain string used to be iterated per character, silently turning
    "Chile" into five single-letter 'parties' instead of failing.

    `measure_wp_introduction` calls this on `representation.get("parties", [])`, so a
    representation storing a bare string would have corrupted country credit quietly.
    """
    with pytest.raises(TypeError, match="bare str"):
        split_parties("Chile")


def test_a_single_party_still_works_when_wrapped():
    """The fix must not make the legitimate one-party case awkward."""
    assert split_parties(["Chile"]) == ["chile"]

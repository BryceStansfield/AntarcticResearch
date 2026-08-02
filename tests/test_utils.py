"""Tests for `utils.split_parties`.

Every WP/IP authorship figure runs its author lists through this, so a quiet
mis-split here shows up as country credit landing in the wrong place.
"""
import numpy as np
import pytest

import pathlib
import time

from utils import line_buffer_stdout, split_parties


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


# --------------------------------------------------------------------- line_buffer_stdout

def test_line_buffer_stdout_makes_writes_appear_immediately(tmp_path):
    """The property that matters: a redirected script's progress reaches the file as it prints.

    Python block-buffers stdout when it is not a terminal, so a long run's `print` output sits in
    an ~8KB buffer while stderr warnings flush past it -- the log looks stalled and a working run
    is indistinguishable from a hung one. Exercised in a subprocess because it is a property of a
    real redirected stdout, which pytest's capture replaces.
    """
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "prints.py"
    log = tmp_path / "out.log"
    # Print, then block. If stdout were block-buffered the line would still be unwritten.
    script.write_text(textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(pathlib.Path.cwd())!r})
        from utils import line_buffer_stdout
        line_buffer_stdout()
        print("progress line")
        time.sleep(30)
    """))

    proc = subprocess.Popen([sys.executable, str(script)], stdout=log.open("w"), stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 15
        while time.time() < deadline and "progress line" not in log.read_text():
            time.sleep(0.2)
        assert "progress line" in log.read_text(), "line should be flushed before the process exits"
    finally:
        proc.kill()
        proc.wait()


def test_line_buffer_stdout_is_idempotent():
    """Called from every long-running entry point; calling it twice must not raise."""
    line_buffer_stdout()
    line_buffer_stdout()

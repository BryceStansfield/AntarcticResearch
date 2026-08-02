import sys


def line_buffer_stdout() -> None:
    """Emit progress output as it happens, even when stdout is a file or a pipe.

    Python block-buffers stdout whenever it is not a terminal, so a script redirected to a log
    accumulates roughly 8KB of ``print`` output before any of it is written. Warnings go to stderr
    and are not buffered the same way, so a long run's log fills with library warnings while every
    progress line it actually prints stays invisible -- the run looks hung when it is working
    normally, and there is no way to tell it apart from one that really has stalled.

    Fixed here rather than by launching with ``python -u`` or ``PYTHONUNBUFFERED=1``, because the
    property wanted is "this script reports progress", which should not depend on remembering a
    flag at every call site. Being in-process, it also applies to a run already queued behind
    another one.

    Costs nothing worth measuring: these scripts print a handful of lines per minute, not a stream.
    """
    sys.stdout.reconfigure(line_buffering=True)


def split_parties(parties: list[str]) -> list[str]:
    """Flatten a list of party strings, splitting any '|'-joined entries into
    their individual parties, and normalize (strip + lowercase) each one.

    A bare string is rejected rather than accepted: iterating one yields its
    characters, which would quietly produce single-letter "parties" and credit
    authorship to nonexistent countries instead of failing.
    """
    if isinstance(parties, str):
        raise TypeError(
            f"split_parties expects a sequence of party strings, got a bare str: "
            f"{parties!r}. Wrap it in a list if it is a single party."
        )
    return [s.strip().lower() for p in parties for s in p.split('|')]

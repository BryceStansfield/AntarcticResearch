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

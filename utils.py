def split_parties(parties: list[str]) -> list[str]:
    """Flatten a list of party strings, splitting any '|'-joined entries into
    their individual parties, and normalize (strip + lowercase) each one."""
    return [s.strip().lower() for p in parties for s in p.split('|')]

from typing import Any


def _fold(key):
    """Lowercase string keys, pass everything else through untouched."""
    return key.lower() if isinstance(key, str) else key


class CaseInsensitiveDict(dict):
    """A dict whose string keys compare case-insensitively.

    Every mutating entry point has to be overridden, not just ``__setitem__``. ``dict.__init__``
    and ``dict.update`` are implemented in C and store keys directly rather than routing through
    ``__setitem__``, so ``CaseInsensitiveDict({"USA": 1})`` used to keep the key as ``"USA"`` --
    after which ``d["usa"]`` raised ``KeyError`` on a dict whose whole purpose is that it should
    not. The classmethod ``from_dict`` was the only constructor that folded keys, so the
    difference was invisible until someone used the type's own constructor.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        for other in args:
            items = other.items() if hasattr(other, "items") else other
            for key, value in items:
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def __setitem__(self, key, value):
        super().__setitem__(_fold(key), value)

    def __getitem__(self, key):
        return super().__getitem__(_fold(key))

    def __delitem__(self, key):
        super().__delitem__(_fold(key))

    def __contains__(self, key):
        return super().__contains__(_fold(key))

    def get(self, key, default=None) -> Any:
        return super().get(_fold(key), default)

    def pop(self, key, *default):
        return super().pop(_fold(key), *default)

    def setdefault(self, key, default=None):
        return super().setdefault(_fold(key), default)

    @classmethod
    def from_dict(cls, d):
        result = cls()
        for k, v in d.items():
            if k in result:
                raise ValueError(f"Duplicate case-insensitive key: {k!r}")
            result[k] = v
        return result

country_alternative_names = CaseInsensitiveDict.from_dict({
    "Republic of Korea": ["South Korea", "Korea", "S Korea", "Korea (ROK)"],
    "Czechia": ["Czech Republic"],
    "Russia": ["Russian Federation"],
    "United States": ["United States of America", "USA", "US"],
    "New Zealand": ["NZ"],
    "United Kingdom": ["UK"],
    "Turkey": ["türkiye"],
    "Ivory Coast": ["cote d'ivoire"],
    "Argentina": ["argentino"]  # Grammatical variation.
})

alternative_names_to_countries = CaseInsensitiveDict.from_dict({
    alt: c1 for c1, alts in country_alternative_names.items() for alt in alts
})

def all_names_for(country_name):
    """Every spelling of a country, whichever spelling you start from.

    ``country_alternative_names`` is keyed by canonical name only, so looking an alias up in it
    returns nothing: ``country_alternative_names.get("USA")`` is empty while
    ``.get("United States")`` lists three aliases. Callers that expand a name through it directly
    therefore only see the full alias set when they happen to hold the canonical spelling.

    That is not a hypothetical split. ``get_list_of_country_names()`` returns canonical names *and*
    every alias mixed together, so a caller iterating it asks about "USA" as readily as "United
    States" -- and for "USA" would miss a dict holding its value under "United States", reporting a
    spurious zero. Normalising to the canonical name first and expanding from there makes every
    spelling of a country resolve to the same set.
    """
    canonical = alternative_names_to_countries.get(country_name, country_name)
    return [canonical] + list(country_alternative_names.get(canonical, []))


def get_country_value_from_dict(country_dict, country_name, missing=0):
    """Sum a country's entries, returning `missing` if it matches no key at all.

    `missing` defaults to 0 because most figures are counts, where a country absent
    from the dict genuinely scored zero. Figures whose values are averages must pass
    float("nan") instead: for those, 0 is not an absence but the best attainable
    score, so defaulting would silently rank an unmeasured country top.
    """
    country_dict = CaseInsensitiveDict.from_dict(country_dict)

    # The canonical name and its alternatives are deduplicated before summing. An
    # alias list that repeats the canonical name would otherwise match the same dict
    # entry twice and silently double that country's figure.
    names = all_names_for(country_name)
    seen = set()
    values = []
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        if name in country_dict:
            values.append(country_dict[name])

    return sum(values) if values else missing

def check_dict_coverage(country_dict, countries):
    country_dict = CaseInsensitiveDict.from_dict(country_dict)
    matched_keys = set()
    not_found = []

    # Matches are recorded lowercased. country_dict is case-insensitive and so holds
    # lowercased keys; recording the caller's original casing here would make the
    # unused_keys comparison below never match, reporting every key as unused.
    for country in countries:
        found = False
        # Every spelling, resolved through the canonical name -- see all_names_for. Asking about
        # an alias used to check only that alias, so a dict keyed on the canonical name reported
        # the alias as not found while also reporting its own key as unused.
        for name in all_names_for(country):
            if name in country_dict:
                matched_keys.add(name.lower())
                found = True
        if not found:
            not_found.append(country)

    unused_keys = [k for k in country_dict if k not in matched_keys]
    return unused_keys, not_found

def get_list_of_country_names():
    with open("data/country_names.txt", "r") as f:
        return set([s.strip() for s in f.readlines()]) | set(country_alternative_names.keys()) | set([c for l in country_alternative_names.values() for c in l])

def get_list_of_country_subunits():
    with open("data/country_subunits.txt", "r") as f:
        d = {}
        for l in f.readlines():
            k, v = l.split(';')
            d[k.strip()] = v.strip()
        return d

def get_list_of_country_affiliations():
    with open("data/country_institutions.txt", "r") as f:
        d = {}
        for l in f.readlines():
            k, v = l.split(';')
            d[k.strip()] = v.strip()
        return d

def normalize_country_name(c):
    base_name = alternative_names_to_countries.get(c, None)
    if base_name is not None:
        return base_name.lower()
    return c.lower()
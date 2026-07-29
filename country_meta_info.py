from typing import Any


class CaseInsensitiveDict(dict):
    def __setitem__(self, key, value):
        super().__setitem__(key.lower() if isinstance(key, str) else key, value)

    def __getitem__(self, key):
        return super().__getitem__(key.lower() if isinstance(key, str) else key)

    def __contains__(self, key):
        return super().__contains__(key.lower() if isinstance(key, str) else key)

    def get(self, key, default=None) -> Any:
        return super().get(key.lower() if isinstance(key, str) else key, default)

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
    names = [country_name] + list(country_alternative_names.get(country_name, []))
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

    for country in countries:
        found = False
        if country in country_dict:
            matched_keys.add(country)
            found = True
        for alt_name in country_alternative_names.get(country, []):
            if alt_name in country_dict:
                matched_keys.add(alt_name)
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
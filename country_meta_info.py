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
    "United States": ["United States of America", "USA", "United States", "US"],
    "New Zealand": ["NZ"],
    "United Kingdom": ["UK"],
    "Turkey": ["türkiye"],
    "Ivory Coast": ["cote d'ivoire"],
    "Argentina": ["argentino"]  # Grammatical variation.
})

def get_country_value_from_dict(country_dict, country_name):
    country_dict = CaseInsensitiveDict.from_dict(country_dict)
    s = 0

    # First we try the country name as-is.
    if country_name in country_dict:
        s += country_dict[country_name]

    # Next we try any alternative names.
    for alt_name in country_alternative_names.get(country_name, []):
        if alt_name in country_dict:
            s += country_dict[alt_name]

    return s

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

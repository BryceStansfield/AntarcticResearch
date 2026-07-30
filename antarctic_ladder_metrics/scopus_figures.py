from itertools import groupby

import pandas as pd
import country_meta_info
from antarctic_ladder_metrics.constants import END_YEAR

# The Scopus export is capped at 20k rows, which truncates its earliest year: 2012
# holds 49 documents against a ~1400/year norm, so including it would read as a
# collapse in output rather than an export artifact. 2013 is the first complete
# year. The upper bound tracks the ladder's END_YEAR, which has complete data
SCOPUS_START_YEAR = 2013


def _build_keys_by_length(str_to_country: dict) -> list:
    """Group str_to_country's keys by string length, longest first.

    Matching must try longer, more specific keys before shorter, more general ones,
    or a short key can win a match it shouldn't: "indian river state college"
    contains both "indian river" and "india", and without length precedence the
    affiliation would resolve to India instead of going unmatched/ambiguous as
    the specific "indian river" key intends.

    Returns a list of lists: each inner list holds all keys of a given length, and
    the outer list is ordered by that length descending.
    """
    keys = sorted(str_to_country.keys(), key=len, reverse=True)
    return [list(g) for _, g in groupby(keys, key=len)]


def _resolve_affiliation_country(affiliation: str, keys_by_length: list, str_to_country: dict):
    """Resolve a single lowercased affiliation string to at most one country.

    Tries each length group in order (longest keys first, see
    `_build_keys_by_length`); the first group with any hit decides the outcome,
    even if that hit turns out ambiguous -- matching does not fall through to
    shorter keys once a longer group has matched something.

    Returns (matched_country, was_ambiguous):
      - (country, False) if exactly one distinct country's key(s) were found.
      - (None, True) if the deciding length group's keys resolved to more than one
        distinct country (ambiguous match, not counted for any country).
      - (None, False) if no length group matched at all (unresolved).
    """
    for length_group in keys_by_length:
        matched_countries = {str_to_country[s] for s in length_group if s in affiliation}

        if not matched_countries:
            continue

        if len(matched_countries) == 1:
            return next(iter(matched_countries)), False

        return None, True

    return None, False


def _build_str_to_country() -> dict:
    """Build the lowercased-string -> canonical-country lookup from country_meta_info.

    Split out of ScopusFigures.__init__ purely to separate the country_meta_info IO
    from the matching logic above; behaviour is unchanged from the inline version.
    """
    country_names = [c.lower() for c in country_meta_info.get_list_of_country_names()]

    # Map each name to its canonical country so grammatical variants and alternative
    # names (e.g. "argentino" -> "argentina") resolve to a single country value.
    str_to_country = dict([(c.lower(), country_meta_info.normalize_country_name(c),) for c in country_names if len(c) > 3])   # Very short names screw with the matching.
    str_to_country |= dict([(k.lower(), v.lower(),) for k, v in country_meta_info.get_list_of_country_subunits().items()])
    str_to_country |= dict([(k.lower(), v.lower(),) for k, v in country_meta_info.get_list_of_country_affiliations().items()])
    return str_to_country


class ScopusFigures():
    def __init__(self, report: bool = False, scopus_table: pd.DataFrame = None,
                 str_to_country: dict = None) -> None:
        # scopus_table and str_to_country are optional purely so tests can inject
        # small synthetic fixtures; when omitted (the real pipeline's call shape)
        # behaviour is identical to the pre-refactor inline version.
        if scopus_table is None:
            scopus_table = pd.read_csv("data/scopus_export.csv")
        scopus_table = scopus_table[(scopus_table["Year"] >= SCOPUS_START_YEAR) & (scopus_table["Year"] <= END_YEAR)]

        if str_to_country is None:
            str_to_country = _build_str_to_country()

        # Finally, we need to sort by key length descending, otherwise we get some bad matches
        # E.g. indian river state college => india.
        keys_by_length = _build_keys_by_length(str_to_country)

        # Keyed (publication_year, country)
        self.country_counts_by_year = {}

        unresolved = []
        ambiguous = []

        # keys, vals and affiliations assumed lower.
        # Count each country at most once per document to avoid double counting a
        # document that lists several affiliations from the same country.
        for year, affiliations in zip(scopus_table["Year"].tolist(), scopus_table["Affiliations"].fillna('').tolist()):
            document_countries = set()

            for affiliation in [a.lower() for a in affiliations.split(';') if a != '']:
                country, is_ambiguous = _resolve_affiliation_country(affiliation, keys_by_length, str_to_country)

                if country is not None:
                    document_countries.add(country)
                elif is_ambiguous:
                    ambiguous.append(affiliation)
                else:
                    unresolved.append(affiliation)

            for country in document_countries:
                key = (year, country)
                self.country_counts_by_year[key] = self.country_counts_by_year.get(key, 0) + 1

        self.country_counts = {}
        for (_, country), count in self.country_counts_by_year.items():
            self.country_counts[country] = self.country_counts.get(country, 0) + count

        if report:
            print(f"Unresolved affiliations ({len(unresolved)}):")
            print(unresolved)
            print(f"Ambiguous affiliations ({len(ambiguous)}):")
            print(ambiguous)

    def country_dict(self) -> dict:
        return self.country_counts

    def figure_title(self) -> str:
        return "Affiliated Research Items"

    def save_full_figures(self, path: str):
        yearly_figures = sorted(
            ({"year": int(k[0]), "country": k[1], "value": v} for k, v in self.country_counts_by_year.items()),
            key=lambda r: (r["year"], r["country"]))
        pd.DataFrame(yearly_figures).to_csv(path)

if __name__ == "__main__":
    ScopusFigures()
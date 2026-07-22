from itertools import groupby

import pandas as pd
import country_meta_info
from antarctic_ladder_metrics.constants import END_YEAR

# The Scopus export is capped at 20k rows, which truncates its earliest year: 2012
# holds 49 documents against a ~1400/year norm, so including it would read as a
# collapse in output rather than an export artifact. 2013 is the first complete
# year. The upper bound tracks the ladder's END_YEAR, which has complete data
SCOPUS_START_YEAR = 2013

class ScopusFigures():
    def __init__(self, report: bool = False) -> None:
        scopus_table = pd.read_csv("data/scopus_export.csv")
        scopus_table = scopus_table[(scopus_table["Year"] >= SCOPUS_START_YEAR) & (scopus_table["Year"] <= END_YEAR)]
        country_names = [c.lower() for c in country_meta_info.get_list_of_country_names()]
        
        # Map each name to its canonical country so grammatical variants and alternative
        # names (e.g. "argentino" -> "argentina") resolve to a single country value.
        str_to_country = dict([(c.lower(), country_meta_info.normalize_country_name(c),) for c in country_names if len(c) > 3])   # Very short names screw with the matching.
        str_to_country |= dict([(k.lower(), v.lower(),) for k, v in country_meta_info.get_list_of_country_subunits().items()])
        str_to_country |= dict([(k.lower(), v.lower(),) for k, v in country_meta_info.get_list_of_country_affiliations().items()])
        
        # Finally, we need to sort by key length descending, otherwise we get some bad matches
        # E.g. indian river state college => india.
        # keys_by_length is a list of lists: each inner list holds all keys of a given
        # length, and the outer list is sorted by key length descending.
        keys = sorted(str_to_country.keys(), key=len, reverse=True)
        keys_by_length = [list(g) for _, g in groupby(keys, key=len)]

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
                matched = False

                for length_group in keys_by_length:
                    matched_countries = {str_to_country[s] for s in length_group if s in affiliation}

                    if not matched_countries:
                        continue

                    # Only count unambiguous matches; if a length group matches more than
                    # one country the affiliation is ambiguous, so flag it without counting.
                    if len(matched_countries) == 1:
                        document_countries.add(next(iter(matched_countries)))
                    else:
                        ambiguous.append(affiliation)

                    matched = True
                    break

                if not matched:
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
from itertools import groupby

import pandas as pd
import country_meta_info

class ScopusFigures():
    def __init__(self, report: bool = False) -> None:
        scopus_table = pd.read_csv("data/scopus_export.csv")
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

        self.country_counts = {}

        unresolved = []
        ambiguous = []

        # keys, vals and affiliations assumed lower.
        # Count each country at most once per document to avoid double counting a
        # document that lists several affiliations from the same country.
        for affiliations in scopus_table["Affiliations"].fillna('').tolist():
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
                self.country_counts[country] = self.country_counts.get(country, 0) + 1

        if report:
            print(f"Unresolved affiliations ({len(unresolved)}):")
            print(unresolved)
            print(f"Ambiguous affiliations ({len(ambiguous)}):")
            print(ambiguous)

    def country_dict(self) -> dict:
        return self.country_counts

    def figure_title(self) -> str:
        return "Affiliated Research Items"

if __name__ == "__main__":
    ScopusFigures()
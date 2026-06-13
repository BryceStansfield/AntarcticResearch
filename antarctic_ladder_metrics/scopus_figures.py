import pandas as pd
import country_meta_info

class ScopusFigures():
    def __init__(self) -> None:
        scopus_table = pd.read_csv("data/scopus_export.csv")
        country_names = [c.lower() for c in country_meta_info.get_list_of_country_names()]
        
        str_to_country = dict([(c.lower(), c.lower(),) for c in country_names if len(c) > 3])   # Very short names screw with the matching.
        str_to_country |= dict([(k.lower(), v.lower(),) for k, v in country_meta_info.get_list_of_country_subunits().items()])
        str_to_country |= dict([(k.lower(), v.lower(),) for k, v in country_meta_info.get_list_of_country_affiliations().items()])
        
        # Finally, we need to sort by key length descending, otherwise we get some bad matches
        # E.g. indian river state college => india.
        keys_by_length = list(str_to_country.keys())
        keys_by_length.sort(reverse=True, key=lambda s: len(s))

        self.country_counts = {}

        all_affiliations = [a.lower() for la in scopus_table["Affiliations"].fillna('').tolist() for a in la.split(';') if a != '']
        unresolved = []

        # keys, vals and affiliations assumed lower
        for affiliation in all_affiliations:
            matched = False

            for s in keys_by_length:
                if s in affiliation:
                    country = str_to_country[s]

                    if country in self.country_counts:
                        self.country_counts[country] += 1
                    else:
                        self.country_counts[country] = 1
                    matched = True
                    break

            if not matched:
                unresolved.append(affiliation)

    def country_dict(self) -> dict:
        return self.country_counts

    def figure_title(self) -> str:
        return "Affiliated Research Items"

if __name__ == "__main__":
    ScopusFigures()
import pandas as pd
import country_meta_info
from collections import Counter

class GetScopusFigures():
    def __init__(self) -> None:
        scopus_table = pd.read_csv("data/scopus_export.csv")
        country_names = [c.lower() for c in country_meta_info.get_list_of_country_names()]
        
        str_to_country = dict([(c.lower(), c.lower(),) for c in country_names])
        str_to_country |= dict([(k.lower(), v.lower(),) for k, v in country_meta_info.get_list_of_country_subunits().items()])
        str_to_country |= dict([(k.lower(), v.lower(),) for k, v in country_meta_info.get_list_of_country_affiliations().items()])

        self.country_counts = {}

        all_affiliations = [a.lower() for la in scopus_table["Affiliations"].fillna('').tolist() for a in la.split(';') if a != '']
        unresolved = []

        # keys, vals and affiliations assumed lower
        for affiliation in all_affiliations:
            matched = False

            for s in str_to_country:
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
        
        print(Counter(unresolved))
    

if __name__ == "__main__":
    GetScopusFigures()
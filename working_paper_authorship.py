from pyarrow import parquet as pq
import pandas as pd
from collections import Counter


class WorkingPaperAuthorship():
    def __init__(self) -> None:
        wp_authorship_table = pd.read_parquet("data/antarctic-db/processed/document-summary.parquet")
        wp_authorship_table = wp_authorship_table[wp_authorship_table["meeting_type"] == "ATCM"][["parties", "meeting_year"]]
        wp_authorship_table = wp_authorship_table[(wp_authorship_table["meeting_year"] >= 2000) & (wp_authorship_table["meeting_year"] <= 2024)]
        wp_authorship_table["First Author"] = wp_authorship_table["parties"].map(lambda l: l[0])
        
        first_authors = list(wp_authorship_table["First Author"])
        first_authors = list(map(lambda s: s.lower(), filter(lambda s: '|' not in s, first_authors)))     # Entries containing a | are sorted in alphabetical author and have no real first authorship information.
        
        self.first_authorships = Counter(first_authors)
    
    def country_dict(self) -> dict:
        return dict(self.first_authorships)

    def figure_title(self) -> str:
        return "Working Papers"

if __name__ == "__main__":
    WorkingPaperAuthorship()
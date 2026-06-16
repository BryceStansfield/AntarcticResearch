import pandas as pd

from antarctic_ladder_metrics.constants import *

from utils import split_parties

class InformationPaperAuthorship():
    def __init__(self) -> None:
        ip_authorship_table = pd.read_parquet("data/antarctic-db/processed/document-summary.parquet")
        ip_authorship_table = ip_authorship_table[(ip_authorship_table["meeting_type"] == "ATCM") & (ip_authorship_table["party_type"] == "ip")][["parties", "meeting_year", "paper_id"]]
        ip_authorship_table = ip_authorship_table[(ip_authorship_table["meeting_year"] >= START_YEAR) & (ip_authorship_table["meeting_year"] <= END_YEAR)]
        ip_authorship_table = ip_authorship_table.drop_duplicates(subset="paper_id", keep="first")

        authors = list(ip_authorship_table["parties"].map(split_parties))

        self.country_authorships = {}
        for pl in authors:
            for p in pl:
                if p in self.country_authorships:
                    self.country_authorships[p] += 1/len(pl)
                else:
                    self.country_authorships[p] = 1/len(pl)

    def country_dict(self) -> dict:
        return dict(self.country_authorships)

    def figure_title(self) -> str:
        return "Information Paper Authorship"

if __name__ == "__main__":
    InformationPaperAuthorship()

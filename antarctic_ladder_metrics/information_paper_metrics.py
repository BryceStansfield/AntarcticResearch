import pandas as pd

from antarctic_ladder_metrics.constants import *

from utils import split_parties

class InformationPaperAuthorship():
    def __init__(self) -> None:
        ip_authorship_table = pd.read_parquet("data/antarctic-db/processed/document-summary.parquet")
        ip_authorship_table = ip_authorship_table[(ip_authorship_table["meeting_type"] == "ATCM") & (ip_authorship_table["party_type"] == "ip")][["parties", "meeting_year", "paper_id"]]
        ip_authorship_table = ip_authorship_table[(ip_authorship_table["meeting_year"] >= START_YEAR) & (ip_authorship_table["meeting_year"] <= END_YEAR)]
        ip_authorship_table = ip_authorship_table.drop_duplicates(subset="paper_id", keep="first")

        self.yearly_country_authorships = {}
        for year in range(START_YEAR, END_YEAR+1):
            authors = list(ip_authorship_table["parties"].map(split_parties))
            
            for pl in authors:
                for p in pl:
                    self.yearly_country_authorships[(year, p)] = self.yearly_country_authorships.get((year, p), 0) + 1/len(pl)

        self.country_authorships = {}
        for k in self.yearly_country_authorships:
            self.country_authorships[k[1]] = self.country_authorships.get(k[1], 0) + self.yearly_country_authorships[k]

    def country_dict(self) -> dict:
        return dict(self.country_authorships)

    def figure_title(self) -> str:
        return "Information Paper Authorship"
    
    def save_full_figures(self, path:str):
        yearly_figures = [{"year": k[0], "country": k[1], "value": v} for k,v in self.yearly_country_authorships.items()]
        pd.DataFrame(yearly_figures).to_csv(path)

if __name__ == "__main__":
    InformationPaperAuthorship()

import pandas as pd

from antarctic_ladder_metrics.constants import *

from utils import split_parties
import country_meta_info


def compute_ip_country_authorships(ip_authorship_table: pd.DataFrame,
                                    start_year: int = START_YEAR,
                                    end_year: int = END_YEAR) -> tuple[dict, dict]:
    """Aggregate information-paper authorship counts per (year, country) and per country.

    Takes the raw document-summary table -- needing only 'meeting_type', 'party_type',
    'parties', 'meeting_year' and 'paper_id' -- plus a closed [start_year, end_year]
    window, rather than reading the parquet path itself. That is what lets this be
    driven over a small synthetic frame in tests without touching the real corpus;
    see enrich_measure_data in ACTM_Measure_Scraper/src/MeasureEnricher.py for the
    same path-out-of-the-function pattern.

    Returns (yearly_country_authorships, country_authorships): the first keyed by
    (year, country), the second by country alone and equal to summing the first
    over year.
    """
    ip_authorship_table = ip_authorship_table[(ip_authorship_table["meeting_type"] == "ATCM") & (ip_authorship_table["party_type"] == "ip")][["parties", "meeting_year", "paper_id"]]
    ip_authorship_table = ip_authorship_table[(ip_authorship_table["meeting_year"] >= start_year) & (ip_authorship_table["meeting_year"] <= end_year)]
    ip_authorship_table = ip_authorship_table.drop_duplicates(subset="paper_id", keep="first")

    yearly_country_authorships = {}
    for year in range(start_year, end_year+1):
        authors = list(ip_authorship_table[ip_authorship_table["meeting_year"] == year]["parties"].map(split_parties))

        for pl in authors:
            for p in pl:
                p = country_meta_info.normalize_country_name(p)
                yearly_country_authorships[(year, p)] = yearly_country_authorships.get((year, p), 0) + 1

    country_authorships = {}
    for k in yearly_country_authorships:
        country_authorships[k[1]] = country_authorships.get(k[1], 0) + yearly_country_authorships[k]

    return yearly_country_authorships, country_authorships


class InformationPaperAuthorship():
    def __init__(self, parquet_path: str = "data/antarctic-db/processed/document-summary.parquet",
                 start_year: int = START_YEAR, end_year: int = END_YEAR) -> None:
        ip_authorship_table = pd.read_parquet(parquet_path)
        self.yearly_country_authorships, self.country_authorships = compute_ip_country_authorships(
            ip_authorship_table, start_year, end_year)

    def country_dict(self) -> dict:
        return dict(self.country_authorships)

    def figure_title(self) -> str:
        return "Information Paper Authorship"
    
    def save_full_figures(self, path:str):
        yearly_figures = [{"year": k[0], "country": k[1], "value": v} for k,v in self.yearly_country_authorships.items()]
        pd.DataFrame(yearly_figures).to_csv(path)

if __name__ == "__main__":
    InformationPaperAuthorship()

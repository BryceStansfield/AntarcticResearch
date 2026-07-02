import csv

from antarctic_ladder_metrics.constants import *
import pandas as pd

class ScarLeadershipFigures():
    def __init__(self):
        self.country_counts_by_years = {}

        with open("data/SCAR_Leadership.csv", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i < 2:  # skip the two header rows
                    continue
                year = int(row[0])
                if year < START_YEAR or year > END_YEAR:
                    continue
                for cell in row[1:]:  # skip first column (Year)
                    if cell == "AustraliaFrance":
                        cell = "Australia & France" # Typo, sic.
                    for country in cell.split("&"):
                        country = country.strip()
                        if country:
                            self.country_counts_by_years[(year, country,)] = self.country_counts_by_years.get((year, country), 0) + 1

        self._counts = {}
        for k in self.country_counts_by_years:
                self._counts += self._counts.get(k[1], 0) + self.country_counts_by_years[k]

    def country_dict(self) -> dict:
        return self._counts

    def figure_title(self) -> str:
        return "Scar Leadership Positions"
    
    def save_full_figures(self, path: str):
        yearly_figures = [{"year": k[0], "country": k[1], "value": v} for k,v in self.country_counts_by_years.items()]
        pd.DataFrame(yearly_figures).to_csv(path)

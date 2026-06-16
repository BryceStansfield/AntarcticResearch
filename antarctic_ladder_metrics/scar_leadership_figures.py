import csv

from antarctic_ladder_metrics.constants import *

class ScarLeadershipFigures():
    def __init__(self):
        country_counts = {}

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
                            country_counts[country] = country_counts.get(country, 0) + 1

        self._counts = country_counts

    def country_dict(self) -> dict:
        return self._counts

    def figure_title(self) -> str:
        return "Scar Leadership Positions"
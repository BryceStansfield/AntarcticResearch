import pandas as pd
import pathlib
from collections import defaultdict
import country_meta_info

class FacilityFigures:
    def __init__(self) -> None:
        self.facilities = pd.read_csv(pathlib.Path(__file__).parent / "data" /"Facilities_Nov2024.csv", encoding="ISO-8859-1")
        
        # First we clean peak population. 
        self.facilities["Peak Population"] = self.facilities["Peak Population"].replace(',','', regex=True).fillna(0).astype(int)

        # Here we treat 4 seasonal facilities as being equivialent to 1 year-round facility.
        self.facilities["Seasonal Adjusted Peak Population"] = self.facilities["Peak Population"] * self.facilities["Seasonality"].map({"Year-Round": 1, "Seasonal": 0.25})

        self.country_sums = defaultdict(int)

        for _, row in self.facilities.iterrows():
            if str(row["Operator (additional)"]) != "nan":
                self.country_sums[row["Operator (primary)"]] += 0.75 * row["Seasonal Adjusted Peak Population"]
                self.country_sums[row["Operator (additional)"]] += 0.25 * row["Seasonal Adjusted Peak Population"]
            else:
                self.country_sums[row["Operator (primary)"]] += row["Seasonal Adjusted Peak Population"]
    
    def get_country_score(self, country: str) -> int:
        return country_meta_info.get_country_value_from_dict(self.country_sums, country)
    
    def figure_title(self) -> str:
        return "Facility Population"

class VesselCrewFigures:
    def __init__(self) -> None:
        self.vessels = pd.read_csv(pathlib.Path(__file__).parent / "data" / "Vessels+in+operation_Nov2024.csv", encoding="ISO-8859-1")

        # Data cleaning
        # NOTE: It might be good to loosen this restriction.
        self.vessels = self.vessels[self.vessels["Status"] == "In Service"]

        def take_first_figure(x):
            if isinstance(x, (int, float)):
                return x
            
            # The Ukrainian vessel 'Noosfera' cites two figures for maximum capacity. We choose the first.
            return x.split('-')[0].strip()

        self.vessels["Maximum Passenger"] = self.vessels["Maximum Passenger"].map(take_first_figure).fillna(0).astype(int)
        self.vessels["Maximum Crew"] = self.vessels["Maximum Crew"].map(take_first_figure).fillna(0).astype(int)
        self.vessels["Total Capacity"] = self.vessels["Maximum Passenger"] + self.vessels["Maximum Crew"]

        self.country_sums = self.vessels.groupby("Country")["Total Capacity"].sum().to_dict()
    
    def get_country_score(self, country: str) -> int:
        return country_meta_info.get_country_value_from_dict(self.country_sums, country)

    def figure_title(self) -> str:
        return "Vessel Crew"

if __name__ == "__main__":
    FacilityFigures()
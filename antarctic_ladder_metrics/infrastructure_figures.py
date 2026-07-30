import pandas as pd
import pathlib
from collections import defaultdict

# Kept as module-level constants (rather than recomputed inline) so the
# constructors below can default to them while still accepting an override
# path for testing. `__file__` here is still this module's own path, so the
# resulting path is identical to what was previously built inline.
DEFAULT_FACILITIES_PATH = pathlib.Path(__file__).parent.parent / "data" / "Facilities_Nov2024.csv"
DEFAULT_VESSELS_PATH = pathlib.Path(__file__).parent.parent / "data" / "Vessels+in+operation_Nov2024.csv"


def take_first_figure(x):
    """Return a single capacity figure from a cell that may cite a range.

    Numeric cells (including pandas' float NaN standing in for a blank) are
    returned unchanged. String cells -- e.g. the Ukrainian vessel 'Noosfera',
    which cites two figures for maximum capacity -- are reduced to the
    substring before the first "-".
    """
    if isinstance(x, (int, float)):
        return x

    return x.split('-')[0].strip()


class FacilityFigures:
    def __init__(self, facilities_path=None) -> None:
        if facilities_path is None:
            facilities_path = DEFAULT_FACILITIES_PATH

        self.facilities = pd.read_csv(facilities_path, encoding="ISO-8859-1")

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

    def country_dict(self) -> dict:
        return dict(self.country_sums)

    def figure_title(self) -> str:
        return "Facility Population"

class VesselCrewFigures:
    def __init__(self, vessels_path=None) -> None:
        if vessels_path is None:
            vessels_path = DEFAULT_VESSELS_PATH

        self.vessels = pd.read_csv(vessels_path, encoding="ISO-8859-1")

        # Data cleaning
        # NOTE: It might be good to loosen this restriction.
        self.vessels = self.vessels[self.vessels["Status"] == "In Service"]

        self.vessels["Maximum Passenger"] = self.vessels["Maximum Passenger"].map(take_first_figure).fillna(0).astype(int)
        self.vessels["Maximum Crew"] = self.vessels["Maximum Crew"].map(take_first_figure).fillna(0).astype(int)
        self.vessels["Total Capacity"] = self.vessels["Maximum Passenger"] + self.vessels["Maximum Crew"]

        self.country_sums = self.vessels.groupby("Country")["Total Capacity"].sum().to_dict()

    def country_dict(self) -> dict:
        return dict(self.country_sums)

    def figure_title(self) -> str:
        return "Vessel Crew"

if __name__ == "__main__":
    FacilityFigures()

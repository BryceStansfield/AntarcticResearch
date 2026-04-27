import pandas as pd

def base_population_sum():
    facilities = pd.read_csv("Facilities_Nov2024.csv", encoding="ISO-8859-1")
    facilities["Peak Population"] = facilities["Peak Population"].str.replace(",", "").astype(float)
    return facilities.groupby("Operator (primary)")["Peak Population"].sum()

def vessel_tonnage_sum():
    vessels = pd.read_csv("Vessels+in+operation_Nov2024.csv", encoding="ISO-8859-1")
    vessels["Maximum Passenger"] = vessels["Maximum Passenger"].str.replace(",", "").astype(float)
    return vessels.groupby("Country")["Maximum Passenger"].sum()

if __name__ == "__main__":
    print(vessel_tonnage_sum())
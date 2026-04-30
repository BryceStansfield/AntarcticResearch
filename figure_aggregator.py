import pandas as pd

from infrastructure_figures import FacilityFigures

def aggregate_all_figures():
    from infrastructure_figures import FacilityFigures, VesselCrewFigures

    countries = ["Argentina", "Australia", "Belgium", "Brazil", "Bulgaria", "Chile", "China", "Czechia", "Ecuador", "Finland", "France", "Germany", "India", "Italy", "Japan", "Republic of Korea", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Russia", "South Africa", "Spain", "Sweden", "United Kingdom", "United States", "Uruguay"]
    figures = [FacilityFigures(), VesselCrewFigures()]

    results = pd.DataFrame(columns=["Country"] + [figure.figure_title() for figure in figures])
    for country in countries:
        row = {"Country": country}
        for figure in figures:
            row[figure.figure_title()] = figure.get_country_score(country)
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
    return results

if __name__ == "__main__":
    print(aggregate_all_figures())
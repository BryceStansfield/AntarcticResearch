import pandas as pd

import country_meta_info
from infrastructure_figures import FacilityFigures, VesselCrewFigures
from final_report_metrics import FinalReportMentionFigures, FinalReportInterventionFigures
from scar_leadership_figures import ScarLeadershipFigures
from scopus_figures import ScopusFigures

def aggregate_all_figures():
    countries = ["Argentina", "Australia", "Belgium", "Brazil", "Bulgaria", "Chile", "China", "Czechia", "Ecuador", "Finland", "France", "Germany", "India", "Italy", "Japan", "Republic of Korea", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Russia", "South Africa", "Spain", "Sweden", "United Kingdom", "United States", "Uruguay"]
    figures = [FacilityFigures(), VesselCrewFigures(), FinalReportMentionFigures(), FinalReportInterventionFigures(), ScarLeadershipFigures(), ScopusFigures()]
    figure_dicts = [figure.country_dict() for figure in figures]

    # TODO: Next session, investigate gap between these figures and Parsas for Influence.
    results = pd.DataFrame(columns=["Country"] + [figure.figure_title() for figure in figures])
    for country in countries:
        row = {"Country": country}
        for figure, cdict in zip(figures, figure_dicts):
            row[figure.figure_title()] = country_meta_info.get_country_value_from_dict(cdict, country)
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

    print("\n--- Coverage Check ---")
    for figure, cdict in zip(figures, figure_dicts):
        unused, not_found = country_meta_info.check_dict_coverage(cdict, countries)
        print(f"\n{figure.figure_title()}:")
        print(f"  Unused dict keys:    {unused}")
        print(f"  Countries not found: {not_found}")

    return results

if __name__ == "__main__":
    print(aggregate_all_figures())
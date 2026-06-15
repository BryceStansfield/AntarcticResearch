import pandas as pd

import country_meta_info
from antarctic_ladder_metrics.infrastructure_figures import FacilityFigures, VesselCrewFigures
from antarctic_ladder_metrics.final_report_metrics import FinalReportMentionFigures, FinalReportInterventionFigures
from antarctic_ladder_metrics.scar_leadership_figures import ScarLeadershipFigures
from antarctic_ladder_metrics.scopus_figures import ScopusFigures
from antarctic_ladder_metrics.ratification_speed import RatificationSpeed
from antarctic_ladder_metrics.working_paper_metrics import WorkingPaperAuthorship, WPCollaborationGraphCentrality
from antarctic_ladder_metrics.topic_introduction import TopicIntroduction
from downloaders.download_all import download_and_extract_all

def aggregate_all_figures():
    download_and_extract_all()
    
    # TODO: Make sure that all figures are over the same time period.
    countries = ["Argentina", "Australia", "Belgium", "Brazil", "Bulgaria", "Chile", "China", "Czechia", "Ecuador", "Finland", "France", "Germany", "India", "Italy", "Japan", "Republic of Korea", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Russia", "South Africa", "Spain", "Sweden", "United Kingdom", "United States", "Uruguay"]
    figures = [FacilityFigures(), VesselCrewFigures(), FinalReportMentionFigures(), FinalReportInterventionFigures(), ScarLeadershipFigures(), ScopusFigures(), RatificationSpeed(), WorkingPaperAuthorship(), WPCollaborationGraphCentrality(), TopicIntroduction()]
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

    results.to_csv("data/ladder_results.csv")
    return results

if __name__ == "__main__":
    print(aggregate_all_figures())